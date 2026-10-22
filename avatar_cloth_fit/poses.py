"""Humanoid bone discovery and Blender-friendly FBT pose probes.

No ``bpy`` import is required. The public functions accept an Armature object
or any object exposing ``data.bones``. Standard roles use ``.L``/``.R`` keys;
detected custom deform roles use ``secondary.<kind>[.<side>][.<index>]``.

Rotations are Euler XYZ deltas in radians, keyed by bone name. The primary
suite contains combined full-body poses suitable for testing weight behavior;
it assumes a broadly T-shaped humanoid rest pose and Blender armature axes
(X across, -Y forward, Z up). Rest-basis conversion handles bone roll, while
unusual avatar axes or rest poses may still need custom mapping.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict


_CENTER_ROLES = {
    "hips": ("hips", "pelvis", "hip"),
    "spine": ("spine", "torso", "abdomen"),
    "chest": ("upperchest", "chest", "spine2", "spine03", "spine3"),
    "neck": ("neck",),
    "head": ("head",),
}
_SIDE_ROLES = {
    "shoulder": ("shoulder", "clavicle"),
    "upper_arm": ("upperarm", "arm"),
    "lower_arm": ("lowerarm", "forearm", "elbow"),
    "hand": ("hand", "wrist"),
    "upper_leg": ("upperleg", "thigh", "upleg"),
    "lower_leg": ("lowerleg", "knee", "leg", "shin", "calf"),
    "foot": ("foot", "ankle"),
    "toe": ("toebase", "toe"),
}
# Ordered: the first matching kind wins, so "Belly_physbone" is a belly and
# "Thigh_Jiggle_L" a thigh. "hip" only reaches here for side bones that are
# not the primary hips/upper legs (e.g. Booth-style Hip_L butt bones).
_SECONDARY_TERMS = {
    "belly": ("belly", "stomach", "abdomen", "tummy"),
    "breast": ("breast", "boob", "bust", "oppai"),
    "butt": ("butt", "glute", "hip"),
    "thigh": ("thigh",),
    "tail": ("tail",),
    "ear": ("ear",),
    "cloth": ("cloth", "skirt", "dress", "cape", "robe", "ribbon", "frill", "hem", "garment"),
    "physics": ("jiggle", "phys", "bounce", "dynamic", "spring"),
}
# PhysBone/helper bones often reuse limb words (Thigh_Jiggle_L, UpperArm_Twist,
# Head_end, Leg_IK.L). They never take a primary humanoid role.
_HELPER_TERMS = ("jiggle", "phys", "bounce", "dynamic", "spring", "twist", "helper",
                 "collision", "collider")
_HELPER_TOKENS = {"end", "ik", "pole", "target", "ctrl"}


def _is_helper(tokens):
    compact = "".join(tokens)
    return any(term in compact for term in _HELPER_TERMS) or any(token in _HELPER_TOKENS for token in tokens)


def _bones(armature):
    data = getattr(armature, "data", None)
    bones = getattr(data, "bones", None)
    if bones is None:
        bones = getattr(armature, "bones", ())
    try:
        return list(bones)
    except TypeError:
        return []


def _tokens(name):
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    value = value.replace("\\", " ").replace("/", " ").replace(":", " ")
    return [part.lower() for part in re.findall(r"[A-Za-z0-9]+", value)]


def _side(tokens):
    for token in tokens:
        if token == "l" or token.startswith("left"):
            return "L"
        if token == "r" or token.startswith("right"):
            return "R"
        # Support compact lowercase names such as lupperarm/rupperleg.
        if re.match(r"^l(?=(upper|lower|arm|leg|hand|foot|shoulder|clavicle|forearm|thigh|knee|elbow|ankle|wrist))", token):
            return "L"
        if re.match(r"^r(?=(upper|lower|arm|leg|hand|foot|shoulder|clavicle|forearm|thigh|knee|elbow|ankle|wrist))", token):
            return "R"
    return None


def _has_term(tokens, term):
    compact = "".join(tokens)
    term = term.lower()
    if term in compact:
        return True
    match = re.fullmatch(r"([a-z]+)0?(\d+)", term)
    if match:
        base, number = match.groups()
        return any(token == base + number or token == base + "0" + number for token in tokens)
    return False


def _has_secondary_term(tokens, term):
    # Short words must begin or end a token: "ear" is not part of "forearm",
    # yet "Ear_L", "PonyTail" and "ponytail" still match. "ear" and "hip"
    # must begin one ("Gear", "Whip" are not ears or hips).
    if len(term) >= 5:
        return term in "".join(tokens)
    if term in ("ear", "hip"):
        return any(token.startswith(term) for token in tokens)
    return any(token.startswith(term) or token.endswith(term) for token in tokens)


def _candidate_score(tokens, role, terms):
    if _is_helper(tokens) or not any(_has_term(tokens, term) for term in terms):
        return None
    compact = "".join(tokens)
    if role == "spine" and re.search(r"spine0?[2-9]", compact):
        return None
    if role == "upper_arm" and any(x in compact for x in ("forearm", "lowerarm", "elbow")):
        return None
    if role == "lower_arm" and "upperarm" in compact:
        return None
    if role == "upper_leg" and any(x in compact for x in ("lowerleg", "shin", "calf", "knee")):
        return None
    if role == "lower_leg" and any(x in compact for x in ("upperleg", "upleg", "thigh")):
        return None
    if role == "foot" and any(x in compact for x in ("toebase", "toe")):
        return None
    score = max(len(term) for term in terms if _has_term(tokens, term))
    if role == "spine" and "spine" in compact and not re.search(r"spine0?\d+", compact):
        score += 2
    if role == "chest" and "upperchest" in compact:
        score += 8
    if role == "chest" and re.search(r"spine0?[2-9]", compact):
        score += 5
    if role == "toe" and "toebase" in compact:
        score += 4
    return score


def _pick(ranked):
    # Highest confidence first, then a stable case-insensitive name order.
    return sorted(ranked, key=lambda item: (-item[0], item[1]))[0][2] if ranked else None


def identify_bones(armature):
    """Return common humanoid and custom deform roles mapped to bone names.

    Unity Humanoid, Mixamo, VRM, ``Bip``, CATS/VRChat (``Left leg``/``Left
    knee``/``Left elbow``) names, and separator/namespace variants are
    supported. Left/right may be written as words, single-letter tokens, or
    compact lowercase prefixes such as ``lupperarm``. PhysBone/helper bones
    (jiggle, twist, IK, end bones) never take primary roles. An upper limb that
    is not an ancestor of its lower limb yields to the lower limb's same-side
    parent.
    """
    bones = sorted(_bones(armature), key=lambda bone: str(getattr(bone, "name", "")).casefold())
    result = {}
    used = set()

    for role, terms in _CENTER_ROLES.items():
        ranked = []
        for bone in bones:
            name = str(getattr(bone, "name", ""))
            if name in used:
                continue
            score = _candidate_score(_tokens(name), role, terms)
            if score is not None:
                ranked.append((score, name.casefold(), name))
        chosen = _pick(ranked)
        if chosen:
            result[role] = chosen
            used.add(chosen)

    for role, terms in _SIDE_ROLES.items():
        for side in ("L", "R"):
            ranked = []
            for bone in bones:
                name = str(getattr(bone, "name", ""))
                tokens = _tokens(name)
                if name in used or _side(tokens) != side:
                    continue
                score = _candidate_score(tokens, role, terms)
                if score is not None:
                    ranked.append((score, name.casefold(), name))
            chosen = _pick(ranked)
            if chosen:
                result[role + "." + side] = chosen
                used.add(chosen)

    # Hierarchy repair: CATS "Left leg" is the upper leg of "Left knee", and a
    # thigh jiggle bone named "Thigh_L" is not an ancestor of the knee.
    by_name = {str(getattr(bone, "name", "")): bone for bone in bones}
    for side in ("L", "R"):
        for upper, lower in (("upper_leg", "lower_leg"), ("upper_arm", "lower_arm")):
            upper_key, lower_name = upper + "." + side, result.get(lower + "." + side)
            if lower_name is None:
                continue
            current = result.get(upper_key)
            ancestors, parent = [], getattr(by_name[lower_name], "parent", None)
            while parent is not None:
                ancestors.append(str(getattr(parent, "name", "")))
                parent = getattr(parent, "parent", None)
            if not ancestors or current in ancestors:
                continue
            candidate = ancestors[0]
            tokens = _tokens(candidate)
            if candidate in used or _side(tokens) != side or _is_helper(tokens):
                continue
            if current is not None:
                used.discard(current)
            result[upper_key] = candidate
            used.add(candidate)

    found = defaultdict(list)
    for bone in bones:
        name = str(getattr(bone, "name", ""))
        if name in used or not getattr(bone, "use_deform", True):
            continue
        tokens = _tokens(name)
        compact = "".join(tokens)
        if "collision" in compact or "collider" in compact:
            continue
        for kind, terms in _SECONDARY_TERMS.items():
            if any(_has_secondary_term(tokens, term) for term in terms):
                found[(kind, _side(tokens))].append(name)
                break

    for (kind, side), names in sorted(found.items(), key=lambda item: (item[0][0], item[0][1] or "")):
        label = "secondary." + kind + ("." + side if side else "")
        for index, name in enumerate(names, 1):
            key = label if len(names) == 1 else label + "." + str(index)
            result[key] = name
    return result


def _local_axis_for_world(bone, world_axis):
    """Express an armature-space axis in the bone's rest local basis."""
    matrix = getattr(bone, "matrix_local", None)
    if matrix is None:
        return tuple(float(v) for v in world_axis)
    try:
        # Rest-matrix columns are local basis vectors expressed in armature space.
        value = tuple(sum(float(matrix[row][col]) * world_axis[row] for row in range(3)) for col in range(3))
        length = math.sqrt(sum(component * component for component in value))
        if length > 1e-8:
            return tuple(component / length for component in value)
    except (IndexError, TypeError, ValueError, AttributeError):
        pass
    return tuple(float(v) for v in world_axis)


def _bone_point(bone, endpoint):
    for attr in (endpoint + "_local", endpoint):
        value = getattr(bone, attr, None)
        if value is not None:
            try:
                return tuple(float(value[index]) for index in range(3))
            except (IndexError, TypeError, ValueError):
                pass
    return None


def _add(a, b):
    return tuple(a[i] + b[i] for i in range(3))


def _sub(a, b):
    return tuple(a[i] - b[i] for i in range(3))


def _scale(v, factor):
    return tuple(component * factor for component in v)


def _dot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _unit(v):
    if v is None:
        return None
    length = math.sqrt(_dot(v, v))
    if length <= 1e-8:
        return None
    return _scale(v, 1.0 / length)


def _pair_axis(matches, bones_by_name, role):
    left = bones_by_name.get(matches.get(role + ".L"))
    right = bones_by_name.get(matches.get(role + ".R"))
    if left is None or right is None:
        return None
    left_head, right_head = _bone_point(left, "head"), _bone_point(right, "head")
    if left_head is not None and right_head is not None:
        delta = _sub(left_head, right_head)
        if _dot(delta, delta) > 1e-10:
            return delta
    # Some rigs put both upper-bone heads at the same center; tails still show
    # the anatomical left/right direction.
    left_tail, right_tail = _bone_point(left, "tail"), _bone_point(right, "tail")
    if left_tail is not None and right_tail is not None:
        return _sub(left_tail, right_tail)
    return None


def _anatomical_basis(matches, bones_by_name):
    """Return armature-space basis vectors for anatomical left, back, and up."""
    hips = bones_by_name.get(matches.get("hips"))
    anchor = _bone_point(hips, "head") if hips is not None else None
    if anchor is None:
        spine = bones_by_name.get(matches.get("spine"))
        anchor = _bone_point(spine, "head") if spine is not None else None

    up = None
    if anchor is not None:
        candidates = []
        for role in ("head", "neck", "chest", "spine"):
            bone = bones_by_name.get(matches.get(role))
            point = _bone_point(bone, "tail") if bone is not None else None
            if point is not None:
                delta = _sub(point, anchor)
                candidates.append((_dot(delta, delta), delta))
        if candidates:
            up = _unit(max(candidates, key=lambda item: item[0])[1])
    if up is None:
        up = (0.0, 0.0, 1.0)

    left = _unit(_pair_axis(matches, bones_by_name, "upper_arm"))
    if left is None:
        left = _unit(_pair_axis(matches, bones_by_name, "upper_leg"))
    if left is None:
        left = (1.0, 0.0, 0.0)

    # Gram-Schmidt: X=left, Z=up, Y=up cross left (back).
    left = _unit(_sub(left, _scale(up, _dot(left, up))))
    if left is None:
        fallback = min(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                       key=lambda axis: abs(_dot(axis, up)))
        left = _unit(_sub(fallback, _scale(up, _dot(fallback, up))))
    back = _unit(_cross(up, left))
    if back is None:
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    left = _unit(_cross(back, up))
    return (left, back, up)


def _to_armature_axis(axis, basis):
    return tuple(axis[0] * basis[0][i] + axis[1] * basis[1][i] + axis[2] * basis[2][i]
                 for i in range(3))


def _axis_angle_euler(axis, degrees):
    """Convert a local axis-angle rotation exactly to an XYZ Euler tuple."""
    half = math.radians(degrees) * 0.5
    sine = math.sin(half)
    x, y, z = (component * sine for component in axis)
    w = math.cos(half)
    # Quaternion to XYZ Euler, equivalent to Blender's default XYZ order.
    ex = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_y = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    ey = math.asin(sin_y)
    ez = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return (ex, ey, ez)


def _local_motion(bone, anatomical_axis, degrees, basis=None):
    """Local rest-basis rotation axis and signed angle for an anatomical axis."""
    basis = basis or ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    armature_axis = _to_armature_axis(anatomical_axis, basis)
    return (_local_axis_for_world(bone, armature_axis), float(degrees))


def motion_spec(name, motion, **extra):
    """Pose spec whose Euler rotations come from local axis-angle motion."""
    spec = {"name": name, "motion": dict(motion),
            "rotations": {bone: _axis_angle_euler(axis, degrees) for bone, (axis, degrees) in motion.items()}}
    spec.update(extra)
    return spec


def _swing_motion(bone, basis, degrees):
    """Swing a secondary bone like a PhysBone rather than twisting it.

    Uses the first anatomical axis (left, then up, then back) that is clearly
    perpendicular to the bone, made exactly perpendicular. Forward breast,
    belly and back-pointing butt bones bounce up/down; hanging thigh, skirt
    and tail bones and upright ears swing forward/back. A fixed per-kind axis
    along the bone would only spin it in place.
    """
    head, tail = _bone_point(bone, "head"), _bone_point(bone, "tail")
    direction = _unit(_sub(tail, head)) if head is not None and tail is not None else None
    axis = basis[0]
    if direction is not None:
        for candidate in (basis[0], basis[2], basis[1]):
            perpendicular = _sub(candidate, _scale(direction, _dot(candidate, direction)))
            if _dot(perpendicular, perpendicular) > 0.25:
                axis = _unit(perpendicular)
                break
    return (_local_axis_for_world(bone, axis), float(degrees))


def joint_lines(armature, movable, bend_degrees=60.0, twist_degrees=35.0, chain_degrees=45.0):
    """Generic motion lines for a large, rig-agnostic pose library.

    Each movable bone bends about its rest-local X and Z axes and twists about
    its own Y axis, both signs. Chain lines bend a movable bone together with
    each movable child about the same local axis, sampling combined
    multi-joint motion. No bone names, body regions or avatar proportions
    are assumed; the caller chooses movable bones structurally. Each line is
    ``{id, kind, motion: {bone: (local_axis, max_degrees)}}`` and is sampled
    at fractions of its maximum angle by ``line_sample``.
    """
    bones_by_name = {str(getattr(bone, "name", "")): bone for bone in _bones(armature)}
    movable = sorted(name for name in movable if name in bones_by_name)
    axes = (("x", (1.0, 0.0, 0.0), bend_degrees), ("z", (0.0, 0.0, 1.0), bend_degrees),
            ("y", (0.0, 1.0, 0.0), twist_degrees))
    lines = []
    for name in movable:
        for label, axis, degrees in axes:
            if degrees <= 0:
                continue
            for sign, suffix in ((1.0, "+"), (-1.0, "-")):
                lines.append({"id": f"joint.{name}.{label}{suffix}", "kind": "joint",
                              "motion": {name: (axis, degrees * sign)}})
    chosen = set(movable)
    for name in movable:
        children = sorted(str(getattr(child, "name", "")) for child in getattr(bones_by_name[name], "children", ())
                          if str(getattr(child, "name", "")) in chosen)
        for child in children:
            for label, axis, _ in axes[:2]:
                for sign, suffix in ((1.0, "+"), (-1.0, "-")):
                    lines.append({"id": f"chain.{name}>{child}.{label}{suffix}", "kind": "chain",
                                  "motion": {name: (axis, chain_degrees * sign),
                                             child: (axis, chain_degrees * sign)}})
    return lines


def core_line(spec):
    """A core probe as a line: intermediate fractions scale every joint angle."""
    return {"id": spec["name"], "kind": "core", "motion": spec["motion"]}


def line_sample(line, fraction):
    """Pose spec for a fraction (0, 1] of a line's maximum angles."""
    motion = {bone: (axis, degrees * fraction) for bone, (axis, degrees) in line["motion"].items()}
    name = line["id"] if line["kind"] == "core" and fraction == 1.0 else f"{line['id']}@{round(fraction * 100)}%"
    return motion_spec(name, motion, line=line["id"], fraction=float(fraction))


def adaptive_fractions(samples, min_step, peak=0.25, single_joint=False, sweep_step=None):
    """New fractions to sample along one line, given evaluated samples.

    ``samples`` are ``(fraction, severity, kinds, path_near)`` including the
    rest end ``(0, 0, (), False)``; ``kinds`` is the set of problem kinds at
    that sample and ``path_near`` marks moving body geometry that may pass
    within range of the garment between this sample and the previous one.

    A line sampled only at its end gains a midpoint when the body may sweep
    through range on the way, or when that end shows a problem on a
    multi-joint line (``single_joint`` lines bend one joint about one axis by
    at most 60 degrees, where linear-blend strain, folds and own-surface
    contact grow monotonically with the angle). Afterwards an interval is
    bisected when a problem present at its start is gone at its end, when the
    body may sweep through range between its samples, or next to a sample
    whose severity peaks above both neighbors: these are the signatures of
    non-monotonic failures that end samples miss. Intervals narrower than
    ``min_step`` are final; sweep evidence alone splits only intervals wider
    than ``sweep_step`` (default ``min_step``).
    """
    samples = sorted(samples, key=lambda row: row[0])
    sweep_step = min_step if sweep_step is None else max(sweep_step, min_step)
    new = set()

    def split(a, b, step=min_step):
        if b[0] - a[0] > step * 1.0001:
            new.add(round((a[0] + b[0]) * .5, 6))

    if len(samples) == 2:
        if samples[1][2] and not single_joint:
            split(samples[0], samples[1])
        elif samples[1][3]:
            split(samples[0], samples[1], sweep_step)
    else:
        for a, b in zip(samples, samples[1:]):
            if set(a[2]) - set(b[2]):
                split(a, b)
            elif b[3]:
                split(a, b, sweep_step)
        for a, b, c in zip(samples, samples[1:], samples[2:]):
            if b[1] > max(a[1], c[1]) * (1. + peak) + 1.:
                split(a, b)
                split(b, c)
    known = {round(row[0], 6) for row in samples}
    return sorted(new - known)


def pose_specs(armature, include_secondary=True, forward_sign=1.0):
    """Return combined and isolated FBT pose probes as ``{name, rotations, motion}``.

    ``motion`` maps each bone to its local rotation axis and signed angle in
    degrees, so ``core_line``/``line_sample`` can sample intermediate angles.

    The combined suite has at most fourteen poses: arm lowering/reaching/raising,
    elbow bends, two stride directions, a side step, squat, torso bend/twist,
    and a crouch/reach combination. Pose angles range from about 20 to 65
    degrees to expose meaningful deformation. Up to eighteen isolated probes
    then move one joint at a time (elbows, knees, hip flexion, shoulder lower
    and forward raise, spine, neck); elbows, knees and hips get an
    intermediate angle besides a deep, sitting-like bend, so transition
    artifacts are attributed to a single joint and sampled where linear
    blending is not yet at its extreme. The rest
    pose is intentionally omitted because the caller adds it separately. Hips
    are never translated or rotated as a global control; their mapped
    upper-leg joints supply hip motion. ``forward_sign`` defaults to 1.0 and
    may be -1.0 to reverse the inferred forward direction for flexion and
    reach probes. Up to sixteen optional secondary probes swing deform-bone
    groups by +/-8 degrees per bone about an axis perpendicular to each bone,
    as PhysBones move them; collision bones and non-deform bones are ignored.
    """
    matches = identify_bones(armature)
    bones_by_name = {str(getattr(bone, "name", "")): bone for bone in _bones(armature)}
    basis = _anatomical_basis(matches, bones_by_name)
    forward_sign = float(forward_sign)
    specs = []

    def add_pose(name, moves, forward_roles=()):
        motion = {}
        for role, axis, degrees, side_sign in moves:
            bone_name = matches.get(role)
            bone = bones_by_name.get(bone_name)
            if bone is not None:
                direction_sign = forward_sign if axis == (1, 0, 0) or role in forward_roles else 1.0
                motion[bone_name] = _local_motion(bone, axis, degrees * side_sign * direction_sign, basis)
        if motion:
            specs.append(motion_spec(name, motion))

    def bilateral(role, axis, degrees, left_sign=1.0, right_sign=1.0):
        return [(role + ".L", axis, degrees, left_sign), (role + ".R", axis, degrees, right_sign)]

    # Signs follow the inferred anatomical frame: +Y lowers the left T-pose
    # arm; -Z rotates each arm/forearm toward the avatar's forward direction.
    add_pose("arms_down", bilateral("upper_arm", (0, 1, 0), 55, 1, -1))
    add_pose("reach_forward", bilateral("upper_arm", (0, 0, 1), 50, -1, 1),
             forward_roles=("upper_arm.L", "upper_arm.R"))
    add_pose("arms_raised", bilateral("upper_arm", (0, 1, 0), 65, -1, 1))
    add_pose("elbows_bent", bilateral("lower_arm", (0, 0, 1), 60, -1, 1),
             forward_roles=("lower_arm.L", "lower_arm.R"))

    add_pose("stride_left", [
        ("upper_leg.L", (1, 0, 0), 32, -1), ("lower_leg.L", (1, 0, 0), 38, 1),
        ("upper_leg.R", (1, 0, 0), 22, 1),
    ])
    add_pose("stride_right", [
        ("upper_leg.R", (1, 0, 0), 32, -1), ("lower_leg.R", (1, 0, 0), 38, 1),
        ("upper_leg.L", (1, 0, 0), 22, 1),
    ])
    add_pose("side_step", bilateral("upper_leg", (0, 1, 0), 24, -1, 1))
    add_pose("squat", [
        ("upper_leg.L", (1, 0, 0), 42, -1), ("upper_leg.R", (1, 0, 0), 42, -1),
        ("lower_leg.L", (1, 0, 0), 58, 1), ("lower_leg.R", (1, 0, 0), 58, 1),
    ])
    add_pose("torso_bend_forward", [("spine", (1, 0, 0), 32, 1), ("chest", (1, 0, 0), 20, 1)])
    add_pose("torso_bend_back", [("spine", (1, 0, 0), 22, -1), ("chest", (1, 0, 0), 15, -1)])
    add_pose("torso_twist_left", [("spine", (0, 0, 1), 30, 1), ("chest", (0, 0, 1), 22, 1)])
    add_pose("torso_twist_right", [("spine", (0, 0, 1), 30, -1), ("chest", (0, 0, 1), 22, -1)])
    add_pose("crouch_reach", [
        ("upper_leg.L", (1, 0, 0), 34, -1), ("upper_leg.R", (1, 0, 0), 34, -1),
        ("lower_leg.L", (1, 0, 0), 48, 1), ("lower_leg.R", (1, 0, 0), 48, 1),
        ("upper_arm.L", (0, 0, 1), 38, -1), ("upper_arm.R", (0, 0, 1), 38, 1),
        ("spine", (1, 0, 0), 18, 1),
    ], forward_roles=("upper_arm.L", "upper_arm.R"))

    # Isolated single-joint probes. Signs mirror the combined poses above.
    for side, arm_sign in (("L", -1), ("R", 1)):
        elbow = "lower_arm." + side
        add_pose("isolated_elbow_half." + side, [(elbow, (0, 0, 1), 45, arm_sign)], forward_roles=(elbow,))
        add_pose("isolated_elbow." + side, [(elbow, (0, 0, 1), 100, arm_sign)], forward_roles=(elbow,))
        knee = "lower_leg." + side
        add_pose("isolated_knee_half." + side, [(knee, (1, 0, 0), 45, 1)])
        add_pose("isolated_knee." + side, [(knee, (1, 0, 0), 100, 1)])
        add_pose("isolated_hip_flex_half." + side, [("upper_leg." + side, (1, 0, 0), 45, -1)])
        add_pose("isolated_hip_flex." + side, [("upper_leg." + side, (1, 0, 0), 90, -1)])
        shoulder = "upper_arm." + side
        add_pose("isolated_shoulder_lower_half." + side, [(shoulder, (0, 1, 0), 28, -arm_sign)])
        add_pose("isolated_shoulder_forward." + side, [(shoulder, (0, 0, 1), 45, arm_sign)],
                 forward_roles=(shoulder,))
    add_pose("isolated_spine_half", [("spine", (1, 0, 0), 16, 1)])
    add_pose("isolated_neck", [("neck", (1, 0, 0), 25, 1)])

    if include_secondary:
        # Probe whole detected chains together so nearby linked deform bones
        # are evaluated as a coherent feature. Two signs per kind gives at
        # most sixteen secondary pose specs across the eight supported kinds.
        groups = defaultdict(list)
        for role, bone_name in matches.items():
            if not role.startswith("secondary."):
                continue
            bone = bones_by_name.get(bone_name)
            if bone is None or not getattr(bone, "use_deform", True):
                continue
            compact_name = "".join(_tokens(bone_name))
            if "collision" in compact_name or "collider" in compact_name:
                continue
            kind = role.split(".", 2)[1]
            groups[kind].append((bone_name, bone))
        for kind in sorted(groups):
            for sign, suffix in ((-1, "-"), (1, "+")):
                motion = {bone_name: _swing_motion(bone, basis, 8.0 * sign) for bone_name, bone in groups[kind]}
                specs.append(motion_spec("secondary." + kind + suffix, motion))
    return specs
