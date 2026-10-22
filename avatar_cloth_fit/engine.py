"""Transactional Blender integration and a yielding pose/refinement pipeline."""
import json
import math
import time
import re
from collections import Counter, OrderedDict, defaultdict
from contextlib import contextmanager, nullcontext
import bpy
import numpy as np
from mathutils import Euler, Matrix
from .geometry import (ContactSamples, ProximityGrid, Surface, layer_couplings, mesh_arrays, near_faces,
                       self_intersections)
from .solver import project, fit_pose, smooth_update, to_dense
from .poses import adaptive_fractions, core_line, identify_bones, joint_lines, line_sample, pose_specs
from .topology import SurfaceGraph
from . import assisted, parts, quality, screening

CORRECTION_KEY = 'Cloth Fit Correction'
ASSISTED_KEY = 'Cloth Fit Assisted'
# Object properties naming the shape keys this add-on wrote.
FIT_KEY_PROPERTIES = ('acf_correction_key', 'acf_assisted_key')


def armature_for(body, explicit=None):
    if explicit:
        return explicit
    rigs = {m.object for m in body.modifiers if m.type == 'ARMATURE' and m.show_viewport and m.object}
    if len(rigs) == 1:
        return rigs.pop()
    if len(rigs) > 1:
        return None
    rigs = {m.object for m in body.modifiers if m.type == 'ARMATURE' and m.object}
    if len(rigs) == 1:
        return rigs.pop()
    return body.parent if body.parent and body.parent.type == 'ARMATURE' else None


def groups_metadata(obj):
    return [(g.name, g.lock_weight) for g in obj.vertex_groups]


def restore_groups(obj, metadata):
    obj.vertex_groups.clear()
    for name, locked in metadata:
        obj.vertex_groups.new(name=name).lock_weight = locked


def store_review_markers(obj, indices):
    """Keep diagnostics outside the deformation/Weight Paint group namespace.

    Creating a vertex group changes Blender's active group. A binary review
    mask then looks like scattered skin weights when entering Weight Paint.
    A point attribute has no skinning or paint-group meaning.
    """
    owner_name = obj.get('acf_review_attribute')
    attribute = obj.data.attributes.get(owner_name) if owner_name else None
    if attribute is None or attribute.domain != 'POINT' or attribute.data_type != 'BOOLEAN':
        # Blender shares attribute names with vertex groups and returns None
        # on a group-name collision. Also avoid replacing user-owned attributes.
        name = 'acf_review_mask'
        suffix = 1
        while name in obj.data.attributes or name in obj.vertex_groups:
            name = f'acf_review_mask.{suffix:03d}'
            suffix += 1
        attribute = obj.data.attributes.new(name=name, type='BOOLEAN', domain='POINT')
        if attribute is None:
            raise ValueError('Could not create review markers; switch the clothing to Object Mode')
    values = np.zeros(len(obj.data.vertices), dtype=bool)
    values[list(indices)] = True
    attribute.data.foreach_set('value', values)
    obj['acf_review_attribute'] = attribute.name
    # Only remove our tagged legacy diagnostic group, and never a group which
    # the user has since connected to a bone, modifier, or shape key.
    legacy_name = obj.get('acf_review_group')
    legacy = obj.vertex_groups.get(legacy_name) if legacy_name else None
    referenced = legacy_name and (
        any(m.type == 'ARMATURE' and m.object and legacy_name in m.object.data.bones for m in obj.modifiers)
        or group_referenced(obj, legacy_name))
    if legacy and not referenced:
        obj.vertex_groups.remove(legacy)
        obj.pop('acf_review_group', None)
    obj.data.update()
    return attribute.name


def review_vertices(obj):
    """Read current markers; older .blend files still support their tagged group."""
    name = obj.get('acf_review_attribute')
    attribute = obj.data.attributes.get(name) if name else None
    if attribute and attribute.domain == 'POINT' and attribute.data_type == 'BOOLEAN':
        values = np.empty(len(obj.data.vertices), dtype=bool)
        attribute.data.foreach_get('value', values)
        return set(int(i) for i in np.flatnonzero(values))
    legacy_name = obj.get('acf_review_group')
    legacy = obj.vertex_groups.get(legacy_name) if legacy_name else None
    if legacy:
        return {v.index for v in obj.data.vertices if any(g.group == legacy.index and g.weight > .5 for g in v.groups)}
    return set()


def restore_active_group(obj, rig, preferred=None, diagnostic_name=None):
    """Return the paint selection to its prior group, with a bone fallback."""
    choices = [preferred] if preferred and preferred != diagnostic_name else []
    if rig and rig.data.bones.active:
        choices.append(rig.data.bones.active.name)
    if rig:
        choices.append(identify_bones(rig).get('hips'))
    choices.extend(g.name for g in obj.vertex_groups if not rig or g.name in rig.data.bones)
    for name in choices:
        group = obj.vertex_groups.get(name) if name else None
        if group and group.name != diagnostic_name:
            obj.vertex_groups.active_index = group.index
            return group.name
    return None


def group_referenced(obj, name):
    """A modifier mask or shape key uses this vertex group."""
    return (any(getattr(m, 'vertex_group', '') == name for m in obj.modifiers)
            or bool(obj.data.shape_keys and any(k.vertex_group == name for k in obj.data.shape_keys.key_blocks)))


def is_clothing_armature(obj):
    return obj is not None and obj.type == 'ARMATURE' and 'acf_clothing_armature' in obj


def remove_unused_clothing_armature(rig):
    """Delete an armature this add-on created once no object uses it."""
    if any(o.parent == rig or any(m.type == 'ARMATURE' and m.object == rig for m in o.modifiers)
           for o in bpy.data.objects if o != rig):
        return False
    data = rig.data
    bpy.data.objects.remove(rig, do_unlink=True)
    if data.users == 0:
        bpy.data.armatures.remove(data)
    return True


def restore_backup(obj, backup):
    if backup is None or backup.type != 'MESH':
        raise ValueError('The saved backup object is no longer available')
    created_rigs = {m.object for m in obj.modifiers if m.type == 'ARMATURE' and is_clothing_armature(m.object)}
    if is_clothing_armature(obj.parent):
        created_rigs.add(obj.parent)
    restore_groups(obj, groups_metadata(backup))
    obj.data = backup.data.copy()
    obj.vertex_groups.active_index = backup.vertex_groups.active_index
    obj.active_shape_key_index = backup.active_shape_key_index
    # Undo a later "Create clothing armature" or the binding of an imported
    # garment's empty Armature modifier: return the deform target and parent
    # to the backup's, keeping the object where it is.
    for modifier in obj.modifiers:
        saved = backup.modifiers.get(modifier.name)
        if (modifier.type == 'ARMATURE' and saved and saved.type == 'ARMATURE'
                and saved.object != modifier.object
                and (is_clothing_armature(modifier.object) or saved.object is None)):
            modifier.object = saved.object
    if is_clothing_armature(obj.parent) and backup.parent != obj.parent:
        world = obj.matrix_world.copy()
        obj.parent = backup.parent
        obj.parent_type = backup.parent_type
        obj.parent_bone = backup.parent_bone
        obj.matrix_parent_inverse = backup.matrix_parent_inverse.copy()
        obj.matrix_world = world
    created_modifier = obj.get('acf_added_modifier', '')
    if created_modifier and obj.modifiers.get(created_modifier) and not backup.modifiers.get(created_modifier):
        obj.modifiers.remove(obj.modifiers[created_modifier])
    obj.pop('acf_added_modifier', None)
    # Reports and markers describe the backup's weights; keep them together.
    for key in ('acf_report', 'acf_review_group', 'acf_review_attribute', 'acf_part_attribute') + FIT_KEY_PROPERTIES:
        if key in backup:
            obj[key] = backup[key]
        else:
            obj.pop(key, None)
    for rig in created_rigs:
        remove_unused_clothing_armature(rig)


class Transaction:
    def __init__(self, obj):
        self.obj = obj
        self.original_data = obj.data
        self.groups = groups_metadata(obj)
        self.active_group = obj.vertex_groups.active_index
        self.active_key = obj.active_shape_key_index
        self.original_backup = obj.acf_backup
        self.metadata = {key: obj.get(key) for key in ('acf_report', 'acf_review_group', 'acf_review_attribute',
                                                      'acf_part_attribute', 'acf_added_modifier') + FIT_KEY_PROPERTIES}
        self.added_modifier = None
        self.bound_modifier = None
        self.committed = False
        collection = bpy.data.collections.get('Avatar Cloth Fit Backups')
        if collection is None:
            collection = bpy.data.collections.new('Avatar Cloth Fit Backups')
            bpy.context.scene.collection.children.link(collection)
        self.backup = obj.copy()
        self.backup.data = obj.data.copy()
        self.backup.name = obj.name + ' [Cloth Fit Backup]'
        self.backup.acf_backup = None
        collection.objects.link(self.backup)
        self.backup.hide_render = True
        bpy.context.view_layer.update()
        # Users may exclude the backup collection from the active view layer.
        if self.backup.name in bpy.context.view_layer.objects:
            self.backup.hide_set(True)
        self.backup.hide_select = True
        obj.data = obj.data.copy()

    def commit(self):
        self.obj.acf_backup = self.backup
        if self.added_modifier:
            self.obj['acf_added_modifier'] = self.added_modifier.name
        self.committed = True

    def rollback(self):
        if self.committed:
            return
        work_data = self.obj.data
        restore_groups(self.obj, self.groups)
        self.obj.data = self.original_data
        self.obj.vertex_groups.active_index = self.active_group
        self.obj.active_shape_key_index = self.active_key
        self.obj.acf_backup = self.original_backup
        for key, value in self.metadata.items():
            if value is None:
                self.obj.pop(key, None)
            else:
                self.obj[key] = value
        if self.added_modifier:
            self.obj.modifiers.remove(self.added_modifier)
        if self.bound_modifier:
            self.bound_modifier.object = None
        backup_data = self.backup.data
        bpy.data.objects.remove(self.backup, do_unlink=True)
        if backup_data.users == 0:
            bpy.data.meshes.remove(backup_data)
        if work_data.users == 0:
            bpy.data.meshes.remove(work_data)


class Profile:
    """Wall-clock seconds and counters per operation, plus per-pose seconds.

    Operations may nest (actual validation includes mesh extraction and BVH
    queries), so operation seconds are inclusive and need not sum to the
    elapsed time; ``phase_seconds`` in the report remain exclusive phases.
    """
    def __init__(self):
        self.seconds = defaultdict(float)
        self.counters = Counter()
        self.poses = defaultdict(lambda: defaultdict(float))

    @contextmanager
    def time(self, operation, pose=None):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.seconds[operation] += elapsed
            if pose is not None:
                self.poses[pose][operation] += elapsed

    def update(self):
        with self.time('depsgraph_update'):
            bpy.context.view_layer.update()
        self.counters['depsgraph_updates'] += 1

    def summary(self):
        return {'operation_seconds': {k: round(v, 4) for k, v in sorted(self.seconds.items())},
                'counters': dict(sorted(self.counters.items())),
                'pose_seconds': {pose: {k: round(v, 4) for k, v in values.items()}
                                 for pose, values in self.poses.items()}}


def _update(profile=None):
    if profile is None:
        bpy.context.view_layer.update()
    else:
        profile.update()


class RigState:
    """Isolate direct deform-bone tests from animation and control constraints."""
    def __init__(self, rig, objects, profile=None):
        self.profile = profile
        self.written = set()
        self.rig = rig
        self.position = rig.data.pose_position
        self.bones = [(pb, pb.rotation_mode, pb.matrix_basis.copy()) for pb in rig.pose.bones]
        self.identity = Matrix.Identity(4)
        self.constraints = [(c, c.mute) for pb in rig.pose.bones for c in pb.constraints]
        self.drivers = [(d, d.mute) for d in rig.animation_data.drivers] if rig.animation_data else []
        self.anim = rig.animation_data
        self.action = self.anim.action if self.anim else None
        self.slot = getattr(self.anim, 'action_slot', None) if self.anim else None
        self.nla = self.anim.use_nla if self.anim else False
        self.modifiers = [(m, m.show_viewport) for obj in objects for m in obj.modifiers if m.type != 'ARMATURE']

    def __enter__(self):
        self.rig.data.pose_position = 'POSE'
        if self.anim:
            self.anim.action = None
            self.anim.use_nla = False
        for d, _ in self.drivers:
            d.mute = True
        for c, _ in self.constraints:
            c.mute = True
        for m, _ in self.modifiers:
            m.show_viewport = False
        self.set_pose({'rotations': {}})
        return self

    def set_pose(self, spec):
        bases = {name: Euler(angles, 'XYZ').to_matrix().to_4x4()
                 for name, angles in spec['rotations'].items() if self.rig.pose.bones.get(name)}
        self.write(bases, reset_all=True)
        _update(self.profile)

    def write(self, bases, reset_all=False):
        """Assign pose bases; bones written earlier and absent now return to identity."""
        bones = self.rig.pose.bones
        if reset_all:
            for pb, _, _ in self.bones:
                pb.matrix_basis = self.identity
        else:
            for name in self.written - bases.keys():
                bones[name].matrix_basis = self.identity
        for name, basis in bases.items():
            bones[name].matrix_basis = basis
        self.written = set(bases)

    def __exit__(self, *_):
        for pb, mode, basis in self.bones:
            pb.rotation_mode = mode
            pb.matrix_basis = basis
        for c, mute in self.constraints:
            c.mute = mute
        for d, mute in self.drivers:
            d.mute = mute
        for m, visible in self.modifiers:
            m.show_viewport = visible
        if self.anim:
            self.anim.action = self.action
            if self.slot and self.action:
                self.anim.action_slot = self.slot
            self.anim.use_nla = self.nla
        self.rig.data.pose_position = self.position
        bpy.context.view_layer.update()


def rig_links(source, target):
    """Match primary bones and map unsupported source bones to a supported ancestor.

    Clothing rigs often contain hips/legs plus exclusive strap/string chains.
    Keeping that rig avoids destroying those chains or adding a second modifier.
    """
    target_names = {b.name for b in target.data.bones if b.use_deform}
    links = {name: name for name in target_names if source.data.bones.get(name)}
    clean = lambda name: re.sub(r'[^a-z0-9]', '', name.rsplit(':', 1)[-1].lower())
    by_clean = {}
    for b in source.data.bones:
        by_clean.setdefault(clean(b.name), []).append(b.name)
    for name in target_names - links.keys():
        matches = by_clean.get(clean(name), [])
        if len(matches) == 1:
            links[name] = matches[0]
    source_roles, target_roles = identify_bones(source), identify_bones(target)
    for role, name in target_roles.items():
        if not role.startswith('secondary.') and name in target_names and name not in links and role in source_roles:
            links[name] = source_roles[role]
    reverse = {source_name: target_name for target_name, source_name in links.items()}
    weights_map = {}
    for b in source.data.bones:
        ancestor = b
        while ancestor and ancestor.name not in reverse:
            ancestor = ancestor.parent
        if ancestor:
            weights_map[b.name] = reverse[ancestor.name]
    return links, weights_map


class RigKinematics:
    """Analytic armature-space pose matrices for rotation-only pose bases.

    Sampling mutes constraints, drivers and actions (RigState), so a pose is
    determined by the bone hierarchy alone. ``Bone.convert_local_to_pose``
    applies each bone's own inheritance options like Blender's evaluation,
    so a whole pose can be solved, assigned and evaluated with one
    dependency-graph update instead of one update per linked bone.
    """
    def __init__(self, rig):
        self.rig = rig
        bones = rig.data.bones
        self.order = sorted((b.name for b in bones), key=lambda name: (len(bones[name].parent_recursive), name))
        self.bones = {name: bones[name] for name in self.order}
        self.parent = {name: (b.parent.name if b.parent else None) for name, b in self.bones.items()}
        self.rest = {name: b.matrix_local.copy() for name, b in self.bones.items()}
        self.rest_inverse = {name: matrix.inverted() for name, matrix in self.rest.items()}
        self._rest_rows = {}
        self.identity = Matrix.Identity(4)
        self.world = rig.matrix_world.copy()
        self.world_inverse = self.world.inverted()
        self.rest_pose = self.pose({})

    def _convert(self, name, matrix, poses, invert=False):
        bone, parent = self.bones[name], self.parent[name]
        if parent is None:
            return bone.convert_local_to_pose(matrix, self.rest[name], invert=invert)
        return bone.convert_local_to_pose(matrix, self.rest[name], parent_matrix=poses[parent],
                                          parent_matrix_local=self.rest[parent], invert=invert)

    def pose(self, bases, desired=None):
        """Pose matrices from bases; ``desired`` pose matrices of some bones
        are honored instead, and their solved bases are added to ``bases``.
        Only bones below a changed bone are recomputed."""
        rest_pose = getattr(self, 'rest_pose', None)
        desired = desired or {}
        poses, changed = {}, set()
        for name in self.order:
            parent = self.parent[name]
            if name in desired:
                poses[name] = desired[name]
                bases[name] = self._convert(name, desired[name], poses, invert=True)
                changed.add(name)
            elif name in bases or rest_pose is None or parent in changed:
                poses[name] = self._convert(name, bases.get(name, self.identity), poses)
                changed.add(name)
            else:
                poses[name] = rest_pose[name]
        return poses

    def deformation(self, poses, names):
        """World-space skinning matrices (vertex world -> posed world).

        Bones still at their rest pose matrix reuse a cached row."""
        key = tuple(names)
        cache = self._rest_rows.get(key)
        if cache is None:
            cache = self._rest_rows[key] = np.array([self._skinning(self.rest_pose[name], name) for name in names])
        result = cache.copy()
        for k, name in enumerate(names):
            if poses[name] is not self.rest_pose[name]:
                result[k] = self._skinning(poses[name], name)
        return result

    def _skinning(self, pose, name):
        return self.world @ pose @ self.rest_inverse[name] @ self.world_inverse


class PoseSolution:
    """Analytic bases for both rigs and the resulting skinning matrices."""
    def __init__(self, source_bases, target_bases, source_poses, target_poses):
        self.source_bases, self.target_bases = source_bases, target_bases
        self.source_poses, self.target_poses = source_poses, target_poses


class PoseState:
    """Sample an avatar and optional independent garment rig in the same pose."""
    def __init__(self, rig, clothing_rig, objects, links, profile=None):
        self.profile = profile
        self.source = RigState(rig, objects, profile)
        self.target = RigState(clothing_rig, (), profile) if clothing_rig != rig else None
        self.links = links
        self.rig, self.clothing_rig = rig, clothing_rig
        self.source_kinematics = RigKinematics(rig)
        self.target_kinematics = RigKinematics(clothing_rig) if self.target else self.source_kinematics
        self.source_rest_inverse = {name: rig.data.bones[name].matrix_local.inverted()
                                    for name in links.values()}

    def __enter__(self):
        self.source.__enter__()
        try:
            if self.target:
                self.target.__enter__()
        except BaseException:
            self.source.__exit__(*__import__('sys').exc_info())
            raise
        self.set_pose({'rotations': {}})
        return self

    def solve(self, spec):
        """Bases and pose matrices for a spec without touching Blender data.

        Linked garment bones copy the avatar bone's world deformation; their
        children see the solved parent pose, like the former sequential
        per-bone assignment with an update after every bone. Garment-only
        rotations then compose onto the solved bases.
        """
        source_kin, target_kin = self.source_kinematics, self.target_kinematics
        source_bases = {name: Euler(angles, 'XYZ').to_matrix().to_4x4()
                        for name, angles in spec['rotations'].items() if name in source_kin.bones}
        clothing = {name: Euler(angles, 'XYZ').to_matrix().to_4x4()
                    for name, angles in spec.get('clothing_rotations', {}).items() if name in target_kin.bones}
        if not self.target:
            for name, rotation in clothing.items():
                source_bases[name] = source_bases.get(name, source_kin.identity) @ rotation
            poses = source_kin.pose(source_bases)
            return PoseSolution(source_bases, source_bases, poses, poses)
        source_poses = source_kin.pose(source_bases)
        source_world, target_world = self.rig.matrix_world, self.clothing_rig.matrix_world
        to_target = target_world.inverted() @ source_world
        to_source = source_world.inverted() @ target_world
        desired = {}
        for name, source_name in self.links.items():
            # Target pose = deformation (in target armature space) @ target rest.
            deformation = source_poses[source_name] @ self.source_rest_inverse[source_name]
            desired[name] = to_target @ deformation @ to_source @ target_kin.rest[name]
        target_bases = {}
        target_poses = target_kin.pose(target_bases, desired)
        if clothing:
            for name, rotation in clothing.items():
                target_bases[name] = target_bases.get(name, target_kin.identity) @ rotation
            target_poses = target_kin.pose(dict(target_bases))
        return PoseSolution(source_bases, target_bases, source_poses, target_poses)

    def apply(self, solution):
        """Assign a solved pose and evaluate it with one dependency-graph update."""
        profile = self.profile
        with (profile.time('pose_sync') if profile else nullcontext()):
            self.source.write(solution.source_bases)
            if self.target:
                self.target.write(solution.target_bases)
        if profile:
            profile.counters['pose_syncs'] += 1
        _update(profile)

    def set_pose(self, spec):
        solution = self.solve(spec)
        self.apply(solution)
        return solution

    def __exit__(self, *exc):
        try:
            if self.target:
                self.target.__exit__(*exc)
        finally:
            self.source.__exit__(*exc)


def bone_matrices(rig, names):
    world = rig.matrix_world
    inverse = world.inverted()
    return np.array([world @ rig.pose.bones[name].matrix @ rig.data.bones[name].matrix_local.inverted() @ inverse for name in names])


def sparse_weights(obj, names, totals=None):
    """Normalized deform rows per vertex; ``totals`` (a list) receives each
    vertex's raw deform-group weight sum, which decides whether Blender's
    Armature modifier moves the vertex at all."""
    group_to_bone = {g.index: names.index(g.name) for g in obj.vertex_groups if g.name in names}
    result = []
    for v in obj.data.vertices:
        raw = [(group_to_bone[g.group], g.weight) for g in v.groups if g.group in group_to_bone]
        row = {bone: weight for bone, weight in raw if weight > 1e-8}
        total = sum(row.values())
        result.append({k: w / total for k, w in row.items()} if total else {})
        if totals is not None:
            totals.append(sum(weight for _, weight in raw))
    return result


def source_correspondence_faces(body, triangles, rows, source_names, links):
    """Exclude disconnected, fallback-only surfaces without skin evidence.

    Ancestor mapping is still useful for unsupported skin secondary bones.
    It is not evidence that an independently skinned appendage is a clothing
    contact surface. Direct matches anchor source islands; shared raw weights
    or material datablock identity conservatively retain split skin islands.
    Neither names nor spatial/bone-position heuristics classify the surfaces.
    Directly supported accessories, connected surfaces and reused-material
    ambiguities remain eligible rather than guessing the garment's intent.
    """
    components = parts.islands(body.data)
    component_id = np.empty(len(body.data.vertices), dtype=np.int32)
    direct_names = set(links.values())
    direct = {i for i, name in enumerate(source_names) if name in direct_names}
    influences = []
    for index, component in enumerate(components):
        component_id[component] = index
        influences.append({bone for vertex in component for bone in rows[vertex]})
    materials = [set() for _ in components]
    for face in body.data.polygons:
        # Missing material slots provide no semantic evidence. Do not equate
        # all unassigned surfaces through the default material index zero.
        slot = face.material_index
        if slot < len(body.data.materials) and body.data.materials[slot] is not None:
            materials[component_id[face.vertices[0]]].add(body.data.materials[slot])
    triangle_islands = component_id[triangles[:, 0]]
    surface_islands = set(triangle_islands)
    anchored = [i for i, bones in enumerate(influences) if i in surface_islands and bones & direct]
    skin_bones = set().union(*(influences[i] for i in anchored))
    skin_materials = set().union(*(materials[i] for i in anchored))
    retained = np.array([not anchored or bool(bones & direct or bones & skin_bones or
                                             materials[i] & skin_materials)
                         for i, bones in enumerate(influences)], dtype=bool)
    faces = np.flatnonzero(retained[triangle_islands]).astype(np.int32)
    stats = {'islands': len(components), 'anchored_islands': len(anchored),
             'retained_islands': int(retained.sum()),
             'excluded_islands': int(np.count_nonzero(~retained)),
             'body_triangles': len(triangles), 'transfer_triangles': len(faces),
             'excluded_triangles': len(triangles) - len(faces)}
    return faces, stats


def frame(triangles):
    x = triangles[:, 1] - triangles[:, 0]
    x /= np.maximum(np.linalg.norm(x, axis=1)[:, None], 1e-12)
    z = np.cross(x, triangles[:, 2] - triangles[:, 0])
    z /= np.maximum(np.linalg.norm(z, axis=1)[:, None], 1e-12)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def write_weights(obj, names, candidates, weights, preserve_names=()):
    groups = []
    for name in names:
        g = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
        groups.append(g)
        if not g.lock_weight and g.name not in preserve_names:
            g.remove(list(range(len(obj.data.vertices))))
    # Bucket quantized values: avoids tens of thousands of individual RNA calls.
    buckets = {}
    for i in range(len(weights)):
        for candidate, value in zip(candidates[i], weights[i]):
            if value > 1e-7 and not groups[candidate].lock_weight and groups[candidate].name not in preserve_names:
                quantized = round(float(value), 7)
                buckets.setdefault((int(candidate), quantized), []).append(i)
    for (bone, value), indices in buckets.items():
        groups[bone].add(indices, value, 'REPLACE')
    obj.data.update()
    bpy.context.view_layer.update()


def apply_part_weights(context, settings):
    """Repair marked parts/current hardware without re-transferring the garment."""
    clothing, body = settings.clothing, settings.body
    if not clothing or clothing.type != 'MESH' or clothing.mode != 'OBJECT':
        raise ValueError('Choose clothing and switch to Object Mode')
    rig = armature_for(clothing)
    source_rig = armature_for(body, settings.armature) if body else None
    if rig is None or source_rig is None:
        raise ValueError('Choose the avatar body and clothing armatures first')
    names = [b.name for b in rig.data.bones if b.use_deform]
    _, mapping = rig_links(source_rig, rig)
    body_names = [b.name for b in source_rig.data.bones if b.use_deform]
    supported = {mapping.get(body_names[i]) for row in sparse_weights(body, body_names) for i in row}
    protected = {g.name for g in clothing.vertex_groups if g.lock_weight}
    if settings.preserve_clothing_bones:
        protected.update(name for name in names if name not in supported)
    transaction = Transaction(clothing)
    try:
        dense = np.zeros((len(clothing.data.vertices), len(names)))
        group_map = {g.index: names.index(g.name) for g in clothing.vertex_groups if g.name in names}
        for vertex in clothing.data.vertices:
            for assignment in vertex.groups:
                if assignment.group in group_map:
                    dense[vertex.index, group_map[assignment.group]] = assignment.weight
        xyz, _, _ = mesh_arrays(clothing)
        body_xyz, _, _ = mesh_arrays(body)
        height = max(float(np.ptp(body_xyz, axis=0).max()), 1e-6)
        rows, stats = parts.constraints(clothing, xyz, dense, names, protected,
                                       settings.attach_hardware, settings.influence_limit,
                                       height * settings.strap_smoothing / 100)
        if not rows:
            raise ValueError('No editable marked parts or metal islands found')
        indices = list(rows)
        active = clothing.vertex_groups.active.name if clothing.vertex_groups.active else None
        for bone, name in enumerate(names):
            if name in protected:
                continue
            group = clothing.vertex_groups.get(name)
            if group:
                group.remove(indices)
            values = {}
            for index, row in rows.items():
                if row[bone] > 1e-8:
                    values.setdefault(float(row[bone]), []).append(index)
            if values:
                group = group or clothing.vertex_groups.new(name=name)
                for value, vertices in values.items():
                    group.add(vertices, value, 'REPLACE')
        legacy = clothing.get('acf_review_group')
        if legacy:
            store_review_markers(clothing, set())
        restore_active_group(clothing, rig, active, legacy)
        # Collision markers/report were measured for the previous weights.
        clothing.pop('acf_report', None)
        attribute_name = clothing.pop('acf_review_attribute', None)
        if attribute_name and clothing.data.attributes.get(attribute_name):
            clothing.data.attributes.remove(clothing.data.attributes[attribute_name])
        clothing.data.update()
        context.view_layer.update()
        transaction.commit()
        return stats
    finally:
        transaction.rollback()


def clothing_armature_plan(meshes, rig, threshold, estimated=None):
    """Choose the bones a standalone clothing armature needs.

    A deform bone is needed when its normalized weight reaches ``threshold``
    on any vertex of any mesh, so PhysBone chains (breasts, butt, belly,
    thighs, tails, skirts) stay exactly when the garment follows them. Every
    ancestor stays too, so bone paths still match the avatar for VRCFury
    Armature Link and similar merge tools. A lighter bone merges into its
    nearest kept ancestor unless its own or the receiving group is locked;
    then it stays and its protected weights are unchanged. ``estimated``
    peaks (from ``body_weight_peaks`` before a fit) count like weights.
    """
    bones = rig.data.bones
    deform = {b.name for b in bones if b.use_deform}
    peak, locked = dict(estimated or {}), set()
    for obj in meshes:
        names = {g.index: g.name for g in obj.vertex_groups if g.name in deform}
        locked.update(g.name for g in obj.vertex_groups if g.lock_weight and g.name in deform)
        for vertex in obj.data.vertices:
            row = [(names[a.group], a.weight) for a in vertex.groups if a.group in names and a.weight > 0.]
            total = sum(weight for _, weight in row)
            for name, weight in row:
                peak[name] = max(peak.get(name, 0.), weight / total)
    if not peak:
        raise ValueError(f'The clothing has no weights for deform bones of {rig.name}; choose the avatar body '
                         'or run Fit & refine first')
    needed = {name for name, value in peak.items() if value >= threshold} | (locked & peak.keys())

    def nearest_kept(name, kept):
        parent = bones[name].parent
        while parent and parent.name not in kept:
            parent = parent.parent
        return parent.name if parent else None

    while True:
        kept = set()
        for name in needed:
            bone = bones[name]
            while bone and bone.name not in kept:
                kept.add(bone.name)
                bone = bone.parent
        merges = {name: nearest_kept(name, kept) for name in peak if name not in kept}
        blocked = {name for name, into in merges.items() if into in locked}
        if not blocked:
            break
        needed |= blocked
    return {'kept': [b.name for b in bones if b.name in kept],
            'removed': [b.name for b in bones if b.name not in kept],
            'merged': {name: {'into': into, 'max_weight': round(peak[name], 6)}
                       for name, into in sorted(merges.items())}}


def _remove_bones(context, rig, names):
    """Delete bones, descendants first, in Edit Mode; restore the selection."""
    if not names:
        return
    depth = {name: len(rig.data.bones[name].parent_recursive) for name in names}
    view_layer = context.view_layer
    active = view_layer.objects.active
    selected = [o for o in view_layer.objects if o.select_get()]
    for obj in selected:
        obj.select_set(False)
    view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    try:
        edit_bones = rig.data.edit_bones
        for name in sorted(names, key=lambda name: (-depth[name], name)):
            edit_bones.remove(edit_bones[name])
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')
        rig.select_set(False)
        for obj in selected:
            obj.select_set(True)
        view_layer.objects.active = active


def _merge_removed_weights(obj, removed, merged):
    """Add removed bones' weights to their kept ancestors; drop their groups.

    Groups that a modifier mask or shape key uses are kept, but without a bone
    they no longer deform. Unrelated groups are untouched.
    """
    groups = obj.vertex_groups
    removed_groups = {g.index: g for g in groups if g.name in removed}
    buckets = {}
    for vertex in obj.data.vertices:
        for assignment in vertex.groups:
            group = removed_groups.get(assignment.group)
            into = merged.get(group.name, {}).get('into') if group else None
            if into and assignment.weight > 0.:
                buckets.setdefault((into, assignment.weight), []).append(vertex.index)
    for (into, weight), indices in buckets.items():
        receiver = groups.get(into) or groups.new(name=into)
        receiver.add(indices, weight, 'ADD')
    for group in removed_groups.values():
        if not group_referenced(obj, group.name):
            groups.remove(group)


def body_weight_peaks(context, body, meshes, rig):
    """Estimate, before any fit, the bones unfitted garments will follow.

    Uses the fit's own body correspondence: up to four distinct nearby body
    surfaces per garment vertex within the same search radius, sampled in
    rest pose. Transfer smoothing only blends these rows, so a bone whose
    interpolated, normalized body weight stays below a threshold everywhere
    here stays below it in the initial transfer. A bone pruned on this basis
    transfers into its kept parent, exactly like merged weights.
    """
    names = [b.name for b in rig.data.bones if b.use_deform]
    rows = sparse_weights(body, names)
    used = sorted({bone for row in rows for bone in row})
    if not used:
        raise ValueError(f'{body.name} has no deform weights for {rig.name}')
    column = {bone: k for k, bone in enumerate(used)}
    with RigState(rig, (body, *meshes)):
        depsgraph = context.evaluated_depsgraph_get()
        source, source_tri, _ = mesh_arrays(body, depsgraph)
        garments = [mesh_arrays(obj, depsgraph) for obj in meshes]
    if len(source) != len(rows):
        raise ValueError('An active deformation changed the body topology')
    dense = np.zeros((len(source), len(used)))
    for vertex, row in enumerate(rows):
        for bone, value in row.items():
            dense[vertex, column[bone]] = value
    face_groups = np.empty(len(source_tri), dtype=np.int64)
    for begin in range(0, len(source_tri), 8192):
        block = dense[source_tri[begin:begin + 8192]].sum(axis=1)
        face_groups[begin:begin + 8192] = np.where(block.max(axis=1) > 0, block.argmax(axis=1), -1)
    surface = Surface(source, source_tri, face_groups=face_groups)
    height = max(float(np.ptp(source, axis=0).max()), 1e-6)
    peak = np.zeros(len(used))
    for xyz, _, normals in garments:
        for i in range(len(xyz)):
            *_, nearby = surface.correspond(xyz[i], normals[i], height * .018, 4)
            for _, _, ref, _, bary in nearby:
                row = bary @ dense[source_tri[ref]]
                total = row.sum()
                if total > 1e-12:
                    np.maximum(peak, row / total, out=peak)
    return {names[bone]: float(peak[k]) for bone, k in column.items() if peak[k] > 0.}


def create_clothing_armature(context, settings):
    """Give clothing its own pruned copy of the avatar (or its current) armature.

    Mirrors the manual VRCFury workflow: duplicate the avatar armature,
    parent and deform the clothing with the copy, then delete every bone the
    clothing does not need. Works before a fit: clothing without a valid
    Armature target is bound to a copy of the avatar armature, and its bones
    come from the body weights under it (``body_weight_peaks``). Fitted
    clothing uses its actual weights. Other selected meshes share the copy.
    The avatar armature and body are never modified; each mesh is backed up
    and Restore clothing backup returns it to its previous binding.
    """
    clothing, body = settings.clothing, settings.body
    if context.mode != 'OBJECT':
        raise ValueError('Switch to Object Mode')
    if not clothing or clothing.type != 'MESH':
        raise ValueError('Choose the clothing mesh')
    others = (o for o in context.selected_objects if o.type == 'MESH' and o not in (clothing, body)
              and ' [Cloth Fit Backup]' not in o.name)
    meshes = list(dict.fromkeys(clothing_objects(settings) + sorted(others, key=lambda o: o.name)))
    if not meshes:
        raise ValueError('Choose clothing rather than the avatar body or a backup')
    avatar_rig = armature_for(body, settings.armature) if body else settings.armature
    rig, modifiers = None, []
    for obj in meshes:
        if obj.library or obj.override_library:
            raise ValueError('Use local editable clothing; linked/library override objects are not supported')
        enabled = [m for m in obj.modifiers if m.type == 'ARMATURE' and m.show_viewport]
        if len(enabled) > 1:
            raise ValueError(f'{obj.name} must use at most one enabled Armature modifier')
        modifier = enabled[0] if enabled else None
        target = modifier.object if modifier else None
        if target is not None and target.type != 'ARMATURE':
            raise ValueError(f"{obj.name}'s Armature modifier targets a non-armature object")
        target = target or avatar_rig
        if target is None or target.type != 'ARMATURE':
            raise ValueError(f'{obj.name} has no armature yet; choose the avatar body (or avatar armature) first')
        if rig not in (None, target):
            if not avatar_rig:
                raise ValueError('Choose the avatar body to combine clothing armatures')
            rig = avatar_rig
        elif rig is None:
            rig = target
        modifiers.append(modifier)
    # Independently created clothing copies can share a fresh avatar copy.
    # Require identical rest bones before rebinding: otherwise the meshes need
    # retargeting by Fit & refine first.
    for obj, modifier in zip(meshes, modifiers):
        target = modifier.object if modifier and modifier.object else avatar_rig
        if target != rig:
            for bone in target.data.bones:
                if not bone.use_deform:
                    continue
                match = rig.data.bones.get(bone.name)
                if match is None or not np.allclose(np.array(target.matrix_world @ bone.matrix_local),
                                                    np.array(rig.matrix_world @ match.matrix_local), atol=1e-5):
                    raise ValueError(f'{obj.name}: {bone.name} has a different rest skeleton; fit to a shared rig first')
    deform = {b.name for b in rig.data.bones if b.use_deform}

    def unfitted(obj, modifier):
        if modifier is None or modifier.object is None:
            return True
        groups = {g.index for g in obj.vertex_groups if g.name in deform}
        return not any(a.group in groups and a.weight > 0. for v in obj.data.vertices for a in v.groups)

    estimate_from = [obj for obj, modifier in zip(meshes, modifiers) if unfitted(obj, modifier)]
    estimated = None
    if estimate_from:
        if not body or body.type != 'MESH' or armature_for(body, settings.armature) != rig:
            raise ValueError(f'Choose the avatar body deformed by {rig.name} to pick bones before fitting, '
                             'or run Fit & refine first')
        estimated = body_weight_peaks(context, body, estimate_from, rig)
    plan = clothing_armature_plan(meshes, rig, settings.bone_weight_threshold, estimated)
    removed_deform = deform & set(plan['removed'])
    view_layer = context.view_layer
    transactions, relinked, created = [], [], None
    try:
        for obj in meshes:
            transactions.append(Transaction(obj))
        created = rig.copy()
        created.data = rig.data.copy()
        created.name = created.data.name = clothing.name + ' Armature'
        for key in [key for key in created.keys() if key.startswith('acf_')]:
            del created[key]
        created['acf_clothing_armature'] = rig.name
        for collection in clothing.users_collection:
            collection.objects.link(created)
        view_layer.update()
        if created.name not in view_layer.objects:
            raise ValueError("Include the clothing's collection in the active view layer")
        created.hide_viewport = False
        created.hide_set(False)
        _remove_bones(context, created, plan['removed'])
        for transaction, obj, modifier in zip(transactions, meshes, modifiers):
            active = obj.vertex_groups.active.name if obj.vertex_groups.active else None
            _merge_removed_weights(obj, removed_deform, plan['merged'])
            restore_active_group(obj, created, active)
            if modifier is None:
                # Removed again by rollback and by Restore clothing backup.
                modifier = obj.modifiers.new('Avatar Cloth Fit Armature', 'ARMATURE')
                transaction.added_modifier = modifier
            world = obj.matrix_world.copy()
            relinked.append((obj, modifier, modifier.object, obj.parent, obj.parent_type,
                             obj.parent_bone, obj.matrix_parent_inverse.copy(), world))
            modifier.object = created
            obj.parent = created
            obj.parent_type = 'OBJECT'
            obj.matrix_parent_inverse = created.matrix_world.inverted()
            obj.matrix_world = world
            obj.data.update()
        view_layer.update()
        report = {'armature': created.name, 'source_armature': rig.name, 'meshes': [o.name for o in meshes],
                  'bones_estimated_from_body_for': [o.name for o in estimate_from],
                  'bone_weight_threshold': settings.bone_weight_threshold, 'kept_bones': plan['kept'],
                  'removed_bones': plan['removed'], 'merged_weights': plan['merged'],
                  'groups_without_bones': {o.name: sorted(g.name for g in o.vertex_groups
                                                          if g.name not in created.data.bones)[:40]
                                           for o in meshes}}
        created['acf_bone_report'] = json.dumps(report)
        bpy.data.texts.new('Cloth Fit Armature - ' + created.name).write(json.dumps(report, indent=2))
        for transaction in transactions:
            transaction.commit()
        return report
    except BaseException:
        for obj, modifier, target, parent, parent_type, parent_bone, inverse, world in reversed(relinked):
            modifier.object = target
            obj.parent, obj.parent_type, obj.parent_bone = parent, parent_type, parent_bone
            obj.matrix_parent_inverse = inverse
            obj.matrix_world = world
        if created is not None:
            data = created.data
            bpy.data.objects.remove(created, do_unlink=True)
            if data.users == 0:
                bpy.data.armatures.remove(data)
        raise
    finally:
        for transaction in transactions:
            transaction.rollback()


def fit_key(obj, prop, default):
    """The shape key an earlier run wrote for ``prop``, if it still exists."""
    keys = obj.data.shape_keys
    return keys.key_blocks.get(obj.get(prop) or default) if keys else None


def write_fit_key(obj, prop, default, world_delta, key=None):
    """Store a world-space displacement as a key relative to the reference key.

    An existing key is overwritten rather than stacked, so reruns never apply
    a fit twice; the Basis and every other key keep their data.
    """
    if obj.data.shape_keys is None:
        obj.shape_key_add(name='Basis')
    reference = obj.data.shape_keys.reference_key
    if key is None:
        key = obj.shape_key_add(name=default, from_mix=False)
    key.relative_key = reference
    inverse = np.array(obj.matrix_world.inverted())[:3, :3]
    basis = np.empty(len(reference.data) * 3)
    reference.data.foreach_get('co', basis)
    key.data.foreach_set('co', (basis.reshape(-1, 3) + world_delta @ inverse.T).ravel())
    key.value = 1.
    key.mute = False
    obj[prop] = key.name
    return key


def protected_islands(obj, labels, triangles, auto_hardware):
    """Accessory islands that must keep their shape: explicit hardware marks,
    plus small detached islands made only of metal materials.

    Metallic fabric is common, so automatic detection never applies to the
    largest island or to an island with 15% or more of its area, and a
    material the largest island uses is not metal evidence (imports often give
    every material, fabric and straps included, Metallic 1).
    """
    roles = parts.roles(obj)
    chosen = set(np.unique(labels[roles == parts.HARDWARE]).tolist())
    materials = obj.data.materials
    if auto_hardware and len(materials) and len(triangles):
        metal = np.array([parts.metal_material(m) for m in materials] + [False])
        slot = np.empty(len(obj.data.loop_triangles), dtype=np.int64)
        obj.data.loop_triangles.foreach_get('material_index', slot)
        slot = np.where(slot < len(materials), slot, len(materials))
        corners = obj.data.vertices
        xyz = np.empty(len(corners) * 3)
        corners.foreach_get('co', xyz)
        xyz = xyz.reshape(-1, 3)
        area = .5 * np.linalg.norm(np.cross(xyz[triangles[:, 1]] - xyz[triangles[:, 0]],
                                            xyz[triangles[:, 2]] - xyz[triangles[:, 0]]), axis=1)
        island = labels[triangles[:, 0]]
        keys, inverse = np.unique(island, return_inverse=True)
        island_area = np.bincount(inverse, area, len(keys))
        metal[np.unique(slot[inverse == int(np.argmax(island_area))])] = False
        all_metal = np.bincount(inverse, (~metal[slot]).astype(np.float64), len(keys)) == 0
        small = island_area < .15 * island_area.max(initial=0.)
        chosen.update(int(k) for k in keys[all_metal & small])
    return [np.flatnonzero(labels == k) for k in sorted(chosen)]


def _rest_rigs(obj, settings):
    rigs = {armature_for(obj)}
    if settings is not None and settings.body:
        rigs.add(armature_for(settings.body, settings.armature))
    return [rig for rig in rigs if rig is not None and rig.type == 'ARMATURE']


class AssistSession:
    """One Assisted Fit adjustment, written to the ``Cloth Fit Assisted`` key.

    The adjustment edits the current rest-pose shape-key mix of ``obj`` and
    stores the change in its own shape key relative to the Basis, so the
    Basis, other keys, UVs, materials and vertex groups never change. The
    first adjustment after a fit creates an Avatar Cloth Fit backup; later
    adjustments keep that pre-adjustment backup. ``cancel`` restores the
    object exactly; ``confirm`` keeps the result.
    """

    def __init__(self, obj, settings=None, pivot='SELECTION', orientation='GLOBAL', cursor=(0., 0., 0.),
                 show_rest=False):
        if obj is None or obj.type != 'MESH':
            raise ValueError('Make the clothing mesh the active object')
        if obj.mode != 'OBJECT':
            raise ValueError('Switch the clothing to Object Mode')
        if obj.library or obj.override_library:
            raise ValueError('Use a local editable mesh; linked/library override objects are not supported')
        if abs(obj.matrix_world.determinant()) < 1e-12:
            raise ValueError('The clothing needs a nonzero scale')
        if obj.data.shape_keys and not obj.data.shape_keys.use_relative:
            raise ValueError('Absolute shape keys are not supported')
        if not len(obj.data.polygons):
            raise ValueError('The clothing needs faces')
        count = len(obj.data.vertices)
        selected = np.zeros(count, dtype=bool)
        obj.data.vertices.foreach_get('select', selected)
        if not selected.any():
            raise ValueError('Select the region to adjust in Edit Mode first')
        self.obj = obj
        self.transaction = None
        self.original_data = None
        self.active_key = obj.active_shape_key_index
        self.rest = []
        prior = obj.acf_backup
        if prior is not None and prior.get('acf_assisted_backup'):
            # Keep the backup from before the first adjustment; edit a copy
            # so cancelling restores the mesh exactly.
            self.original_data = obj.data
            obj.data = obj.data.copy()
        else:
            self.transaction = Transaction(obj)
            self.transaction.backup['acf_assisted_backup'] = True
        try:
            self._prepare(settings, selected, pivot, orientation, cursor)
            if show_rest:
                # Edit and display in the rest pose the adjustment is defined in.
                for rig in _rest_rigs(obj, settings):
                    if rig.data.pose_position != 'REST':
                        self.rest.append((rig.data, rig.data.pose_position))
                        rig.data.pose_position = 'REST'
        except BaseException:
            self.cancel()
            raise

    def _prepare(self, settings, selected, pivot, orientation, cursor):
        obj, mesh = self.obj, self.obj.data
        count = len(mesh.vertices)
        if mesh.shape_keys is None:
            obj.shape_key_add(name='Basis', from_mix=False)
        key = fit_key(obj, 'acf_assisted_key', ASSISTED_KEY)
        if key is None or key == mesh.shape_keys.reference_key:
            key = obj.shape_key_add(name=ASSISTED_KEY, from_mix=False)
        obj['acf_assisted_key'] = key.name
        if key.mute or key.value < 1e-3:
            key.mute = False
            key.value = 1.
        self.key_name = key.name
        self.value = key.value
        self.key_before = np.empty(count * 3)
        key.data.foreach_get('co', self.key_before)
        self.key_before = self.key_before.reshape(-1, 3)
        # The current rest shape: Blender's own evaluation of the key mix.
        solo = obj.show_only_shape_key
        obj.show_only_shape_key = False
        temporary = obj.shape_key_add(name='Cloth Fit temporary mix', from_mix=True)
        mix = np.empty(count * 3)
        temporary.data.foreach_get('co', mix)
        obj.shape_key_remove(temporary)
        obj.show_only_shape_key = solo
        self.mix = mix.reshape(-1, 3)
        matrix = np.array(obj.matrix_world, dtype=np.float64)
        self.linear, self.offset = matrix[:3, :3], matrix[:3, 3]
        self.inverse = np.linalg.inv(self.linear)
        world = self.mix @ self.linear.T + self.offset
        mesh.calc_loop_triangles()
        triangles = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
        mesh.loop_triangles.foreach_get('vertices', triangles)
        triangles = triangles.reshape(-1, 3)
        edges = np.empty(len(mesh.edges) * 2, dtype=np.int64)
        mesh.edges.foreach_get('vertices', edges)
        edges = edges.reshape(-1, 2)
        from .topology import connected_labels
        labels = connected_labels(edges, count)
        pinned = np.zeros(count, dtype=bool)
        group = obj.vertex_groups.get(settings.assist_pin_group) if settings is not None else None
        if group is not None:
            for vertex in mesh.vertices:
                for assignment in vertex.groups:
                    if assignment.group == group.index and assignment.weight > 0.:
                        pinned[vertex.index] = True
        auto = settings.attach_hardware if settings is not None else True
        rigid = protected_islands(obj, labels, triangles, auto)
        self.region = assisted.Region(world, triangles, edges, selected, pinned, rigid)
        self.pinned_count = int(pinned.sum())
        self.protected_count = len(rigid)
        if pivot == 'CURSOR':
            self.pivot = np.asarray(cursor, dtype=np.float64)
        else:
            self.pivot = world[selected].mean(axis=0)
        if orientation == 'LOCAL':
            # The object's own axes, orthonormalized (scale removed).
            q, r = np.linalg.qr(self.linear)
            self.axes = q * np.sign(np.diag(r))
        elif orientation == 'SELECTION':
            self.axes = assisted.principal_axes(world[selected])
        else:
            self.axes = np.eye(3)

    def control(self, translation=(0., 0., 0.), rotation=None, scale=(1., 1., 1.)):
        return assisted.Control(self.pivot, translation, rotation, scale, self.axes)

    def _key(self):
        return self.obj.data.shape_keys.key_blocks[self.key_name]

    def _write(self, world):
        local = (world - self.offset) @ self.inverse.T
        self._key().data.foreach_set('co', (self.key_before + (local - self.mix) / self.value).ravel())
        self.obj.data.update()

    def preview(self, control, radius, profile):
        """Interactive update: closed-form deformation only."""
        weight = self.region.weights(radius, profile)
        self._write(self.region.deform(control, weight))
        return weight

    def confirm(self, control, radius, profile, relax_iterations):
        """Correct the transition band once, store the key and keep the backup."""
        try:
            weight = self.region.weights(radius, profile)
            positions = self.region.deform(control, weight)
            positions, stats = self.region.relax(positions, control, weight, relax_iterations)
            self._write(positions)
            moved = np.linalg.norm(positions - self.region.xyz, axis=1) > 1e-7 * self.region.size
            stats.update(moved_vertices=int(moved.sum()), pinned_vertices=self.pinned_count,
                         protected_islands=self.protected_count)
            obj = self.obj
            obj.active_shape_key_index = list(obj.data.shape_keys.key_blocks).index(self._key())
            # Collision diagnostics were measured on the previous shape.
            obj.pop('acf_report', None)
            attribute_name = obj.pop('acf_review_attribute', None)
            if attribute_name and obj.data.attributes.get(attribute_name):
                obj.data.attributes.remove(obj.data.attributes[attribute_name])
            obj.data.update()
        except BaseException:
            self.cancel()
            raise
        self._restore_pose()
        if self.transaction:
            self.transaction.commit()
        elif self.original_data is not None:
            name = self.original_data.name
            if self.original_data.users == 0:
                bpy.data.meshes.remove(self.original_data)
                self.obj.data.name = name
            self.original_data = None
        return stats

    def cancel(self):
        """Restore the object exactly as it was before the session."""
        self._restore_pose()
        if self.transaction:
            self.transaction.rollback()
            self.transaction = None
        elif self.original_data is not None:
            work = self.obj.data
            self.obj.data = self.original_data
            self.obj.active_shape_key_index = self.active_key
            if work.users == 0:
                bpy.data.meshes.remove(work)
            self.original_data = None

    def _restore_pose(self):
        for data, position in self.rest:
            data.pose_position = position
        self.rest = []


def prepare_assisted_key(obj):
    """Make the ``Cloth Fit Assisted`` key the one Sculpt Mode edits.

    Creates the pre-adjustment backup once (as the first Assisted Fit
    adjustment would), the Basis and the key if needed. The Basis then stays
    the untouched original that Shape Relax restores detail from.
    """
    if obj.mode != 'OBJECT':
        raise ValueError('Switch the clothing to Object Mode')
    if obj.data.shape_keys and not obj.data.shape_keys.use_relative:
        raise ValueError('Absolute shape keys are not supported')
    prior = obj.acf_backup
    if prior is None or not prior.get('acf_assisted_backup'):
        transaction = Transaction(obj)
        transaction.backup['acf_assisted_backup'] = True
        transaction.commit()
    if obj.data.shape_keys is None:
        obj.shape_key_add(name='Basis', from_mix=False)
    key = fit_key(obj, 'acf_assisted_key', ASSISTED_KEY)
    if key is None or key == obj.data.shape_keys.reference_key:
        key = obj.shape_key_add(name=ASSISTED_KEY, from_mix=False)
    obj['acf_assisted_key'] = key.name
    key.mute, key.value = False, 1.
    obj.active_shape_key_index = list(obj.data.shape_keys.key_blocks).index(key)
    return key


def clear_assisted(obj):
    """Remove the Assisted Fit key; the backup still holds the earlier shape."""
    key = fit_key(obj, 'acf_assisted_key', ASSISTED_KEY)
    if key is None or key == obj.data.shape_keys.reference_key:
        raise ValueError('This mesh has no Assisted Fit adjustments')
    obj.shape_key_remove(key)
    obj.pop('acf_assisted_key', None)
    keys = obj.data.shape_keys
    if keys and len(keys.key_blocks) == 1:
        obj.shape_key_remove(keys.reference_key)
    obj.data.update()


def validate_input(body, clothing, rig):
    if not body or not clothing or body.type != 'MESH' or clothing.type != 'MESH':
        raise ValueError('Choose a body mesh and a clothing mesh')
    if body == clothing:
        raise ValueError('Body and clothing must be different objects')
    if not rig or rig.type != 'ARMATURE':
        raise ValueError('Choose the avatar armature (or a body with an Armature modifier)')
    clothing_rig = armature_for(clothing) or rig
    for obj in {body, clothing, rig, clothing_rig}:
        if obj.library or obj.override_library:
            raise ValueError('Use local editable objects; linked/library override objects are not supported')
        if abs(obj.matrix_world.determinant()) < 1e-12:
            raise ValueError('Objects must have nonzero scale')
    for obj in (body, clothing):
        armatures = [m for m in obj.modifiers if m.type == 'ARMATURE' and m.show_viewport]
        if len(armatures) > 1:
            raise ValueError('Each mesh must use at most one enabled Armature modifier')
        for m in armatures:
            # An imported garment's empty Armature modifier is bound to the
            # avatar (or clothing) armature by the fit itself.
            if (m.object is None and obj == body) or (m.object is not None and m.object.type != 'ARMATURE'):
                raise ValueError('Each enabled Armature modifier needs a valid armature target')
            if m.vertex_group or m.use_bone_envelopes or not m.use_vertex_groups or m.use_multi_modifier:
                raise ValueError('Armature modifiers must use unmasked vertex groups, without envelopes or multi-modifier')
    if not any(m.type == 'ARMATURE' and m.object == rig and m.show_viewport for m in body.modifiers):
        raise ValueError('Body needs an enabled Armature modifier with existing deform weights')
    if not len(body.data.polygons) or not len(clothing.data.polygons):
        raise ValueError('Body and clothing need faces')
    if clothing.data.shape_keys and not clothing.data.shape_keys.use_relative:
        raise ValueError('Absolute clothing shape keys are not supported')
    if any(b.bbone_segments > 1 for r in {rig, clothing_rig} for b in r.data.bones if b.use_deform):
        raise ValueError('Segmented B-Bone deformation is not supported by the refinement solver')
    links, _ = rig_links(rig, clothing_rig)
    if not links:
        raise ValueError('Clothing rig has no matching avatar bones; use common names or humanoid bone conventions')


def clothing_objects(settings):
    """An explicit saved set; changing the primary pointer returns to single mode."""
    primary = settings.clothing
    members = [item.object for item in getattr(settings, 'clothing_items', ()) if item.object]
    if primary not in members:
        members = [primary] if primary else []
    return list(dict.fromkeys(o for o in members if o != settings.body and o.type == 'MESH'
                              and ' [Cloth Fit Backup]' not in o.name))


def validate_set(settings):
    meshes = clothing_objects(settings)
    if not meshes:
        raise ValueError('Choose a clothing mesh or use selected clothing')
    rig = armature_for(settings.body, settings.armature) if settings.body else None
    for obj in meshes:
        validate_input(settings.body, obj, rig)
    return meshes, rig


def run(context, settings):
    from . import clothing_set
    meshes, _ = validate_set(settings)
    if len(meshes) > 1:
        return (yield from clothing_set.run(context, settings, meshes))
    return (yield from run_single(context, settings))


def run_single(context, settings):
    """Yield progress messages. Closing this generator restores an unfinished run."""
    body, clothing = settings.body, settings.clothing
    rig = armature_for(body, settings.armature) if body else None
    validate_input(body, clothing, rig)
    clothing_rig = armature_for(clothing) or rig
    names = [b.name for b in clothing_rig.data.bones if b.use_deform]
    links, weights_map = rig_links(rig, clothing_rig)
    if not names:
        raise ValueError('Avatar armature has no deform bones')
    start = time.perf_counter()
    profile = Profile()
    transaction = Transaction(clothing)
    try:
        modifier = next((m for m in clothing.modifiers if m.type == 'ARMATURE' and m.show_viewport), None)
        if modifier is None:
            modifier = clothing.modifiers.new('Avatar Cloth Fit Armature', 'ARMATURE')
            modifier.object = clothing_rig
            transaction.added_modifier = modifier
        elif modifier.object is None:
            modifier.object = clothing_rig
            transaction.bound_modifier = modifier
        source_names = [b.name for b in rig.data.bones if b.use_deform]
        source_totals = []
        source_rows = sparse_weights(body, source_names, source_totals)
        name_indices = {name: i for i, name in enumerate(names)}
        source_weights = []
        unmatched_source = set()
        for row in source_rows:
            mapped = {}
            for index, value in row.items():
                source_name = source_names[index]
                target_name = weights_map.get(source_name)
                if target_name:
                    target_index = name_indices[target_name]
                    mapped[target_index] = mapped.get(target_index, 0.) + value
                else:
                    unmatched_source.add(source_name)
            source_weights.append(mapped)
        body_bones = {bone for row in source_weights for bone in row}
        warnings = []
        if clothing_rig != rig:
            warnings.append('Separate clothing rig retained; matching avatar bone deformation was copied into it for pose tests.')
        if unmatched_source:
            warnings.append('Body bones without a supported clothing-rig ancestor: ' + ', '.join(sorted(unmatched_source)))
        if any(m.show_viewport and m.type != 'ARMATURE' for obj in (body, clothing) for m in obj.modifiers):
            warnings.append('Non-armature modifiers were excluded from the pose analysis; inspect the final modifier stack.')
        if modifier.use_deform_preserve_volume:
            warnings.append('Preserve Volume uses an approximate linear solver; acceptance was checked with Blender deformation.')
        with PoseState(rig, clothing_rig, (body, clothing), links, profile) as state:
            depsgraph = context.evaluated_depsgraph_get()
            source, source_tri, _ = mesh_arrays(body, depsgraph)
            # A correction key from an earlier run describes an earlier shape.
            # Fit from the shape without it and overwrite it, never stack.
            correction_key = fit_key(clothing, 'acf_correction_key', CORRECTION_KEY)
            if settings.correct_shape and correction_key is not None and correction_key.value != 0.:
                correction_key.value = 0.
                _update(profile)
            xyz, triangles, normals = mesh_arrays(clothing, depsgraph)
            if len(source) != len(body.data.vertices) or len(xyz) != len(clothing.data.vertices):
                raise ValueError('An active deformation changed the mesh topology')
            height = max(float(np.ptp(source, axis=0).max()), 1e-6)
            # All distances are a proportion of avatar size, independent of Blender units.
            clearance = settings.clearance * height / 100
            max_fit = settings.max_correction * height / 100
            reach = settings.search_distance * height / 100
            contact_samples = ContactSamples(triangles)
            transfer_faces, source_selection = source_correspondence_faces(
                body, source_tri, source_rows, source_names, links)
            if source_selection['excluded_triangles']:
                warnings.append(f"{source_selection['excluded_islands']} disconnected source islands "
                                f"({source_selection['excluded_triangles']} triangles) use only unsupported "
                                "bone chains without shared skin weights or materials; excluded from weight "
                                "correspondence, but retained for all body collision checks.")
            count = len(xyz)
            candidates = np.zeros((count, 12), dtype=np.int32)
            weights = np.zeros((count, 12))
            fixed = np.zeros((count, 12), dtype=bool)
            fixed_values = np.zeros((count, 12))
            locked_groups = {g.index: names.index(g.name) for g in clothing.vertex_groups if g.lock_weight and g.name in names}
            preserved_groups = dict(locked_groups)
            if settings.preserve_clothing_bones:
                preserved_groups.update({g.index: names.index(g.name) for g in clothing.vertex_groups if g.name in names and names.index(g.name) not in body_bones})
            preserved_names = {clothing.vertex_groups[index].name for index in preserved_groups}
            edges = np.array([tuple(e.vertices) for e in clothing.data.edges], dtype=np.int32).reshape(-1, 2)
            graph = SurfaceGraph(xyz, triangles, edges)
            transition = settings.transition_width * height / 100
            # Dense body weights over the editable target bones only. Locked
            # and preserved bones never receive transferred weight.
            columns = np.array(sorted(body_bones - set(preserved_groups.values())), dtype=np.int64)
            if not len(columns):
                raise ValueError('All candidate deform groups are locked with zero weights')
            column_of = {int(bone): k for k, bone in enumerate(columns)}
            source_dense = np.zeros((len(source), len(columns)))
            for vertex, row in enumerate(source_weights):
                for bone, value in row.items():
                    if bone in column_of:
                        source_dense[vertex, column_of[bone]] = value
            face_groups = np.empty(len(source_tri), dtype=np.int64)
            for begin in range(0, len(source_tri), 8192):
                block = source_dense[source_tri[begin:begin + 8192]].sum(axis=1)
                face_groups[begin:begin + 8192] = np.where(block.max(axis=1) > 0, block.argmax(axis=1), -1)
            surface = Surface(source, source_tri, correspondence_faces=transfer_faces, face_groups=face_groups)
            # Up to four distinct nearby body surfaces per garment vertex.
            choices = 4
            alt_ref = np.zeros((count, choices), dtype=np.int64)
            alt_bary = np.zeros((count, choices, 3))
            alt_point = np.zeros((count, choices, 3))
            alt_score = np.full((count, choices), np.inf)
            alt_distance = np.full((count, choices), np.inf)
            locks = []
            for i in range(count):
                *_, nearby = surface.correspond(xyz[i], normals[i], height * 0.018, choices)
                for k, (score, distance, ref, point, bc) in enumerate(nearby):
                    alt_ref[i, k], alt_bary[i, k], alt_point[i, k] = ref, bc, point
                    alt_score[i, k], alt_distance[i, k] = score, distance
                existing_locks = {preserved_groups[g.group]: g.weight for g in clothing.data.vertices[i].groups if g.group in preserved_groups and g.weight > 1e-8}
                if sum(existing_locks.values()) > 1 + 1e-5:
                    raise ValueError(f'Locked or preserved clothing-bone weights exceed 1 at vertex {i}')
                if len(existing_locks) > 11:
                    raise ValueError('More than 11 protected influences at a vertex exceed the sparse candidate budget')
                locks.append(existing_locks)
                if i % 512 == 0:
                    yield f'Matching clothing to body: {i}/{count}'
            valid = np.isfinite(alt_score)
            rows_index = np.arange(count)

            def alternative_rows(begin, end):
                block = np.zeros((end - begin, choices, len(columns)))
                for k in range(choices):
                    use = valid[begin:end, k]
                    corner_rows = source_dense[source_tri[alt_ref[begin:end, k][use]]]
                    block[use, k] = np.einsum('vi,vib->vb', alt_bary[begin:end, k][use], corner_rows)
                total = block.sum(axis=2, keepdims=True)
                return block / np.where(total > 1e-12, total, 1.)

            def tolerance(distance):
                return np.maximum(.35 * distance, height * .0025)

            def select(chosen, consensus=None):
                """Rows, ambiguity and nearby-surface bones for chosen hits.

                With a consensus, first re-choose each vertex's hit by score
                plus disagreement with its confident garment neighborhood.
                """
                chosen = chosen.copy()
                rows = np.zeros((count, len(columns)))
                ambiguity = np.zeros(count)
                nearby = np.zeros((count, len(columns)))
                for begin in range(0, count, 4096):
                    end = min(count, begin + 4096)
                    block = alternative_rows(begin, end)
                    local = np.arange(end - begin)
                    score = alt_score[begin:end]
                    if consensus is not None:
                        spread = tolerance(alt_distance[begin:end, 0])[:, None]
                        disagreement = .5 * np.abs(block - consensus[begin:end, None]).sum(axis=2)
                        # A weightless surface never wins a consensus vote.
                        disagreement[block.sum(axis=2) <= 1e-12] = 1.
                        cost = np.where(valid[begin:end], ((score - score[:, :1]) / spread) ** 2 + 3. * disagreement, np.inf)
                        best = np.argmin(cost, axis=1)
                        # Switch only for a clear gain; near-ties between
                        # neighboring blends would just jitter the transition.
                        current = cost[local, chosen[begin:end]]
                        chosen[begin:end] = np.where(current - cost[local, best] > .5, best, chosen[begin:end])
                    pick = chosen[begin:end]
                    rows[begin:end] = block[local, pick]
                    difference = .5 * np.abs(block - rows[begin:end, None]).sum(axis=2)
                    spread = tolerance(alt_distance[begin:end][local, pick])[:, None]
                    closeness = np.exp(-((score - score[local, pick][:, None]) / spread) ** 2)
                    closeness[~valid[begin:end]] = 0.
                    closeness[local, pick] = 0.
                    ambiguity[begin:end] = (closeness * difference).max(axis=1)
                    nearby[begin:end] = np.where(valid[begin:end, :, None], block, 0.).max(axis=1)
                return chosen, rows, ambiguity, nearby

            def confidence_for(chosen, ambiguity):
                """Geometric trust in a correspondence, 0.01..1.

                Distance falls off over a few percent of avatar size; facing
                uses |dot| so double-sided or folded fabric is not penalized;
                ambiguity is the weight disagreement of equally close surfaces.
                """
                distance = alt_distance[rows_index, chosen]
                ref = alt_ref[rows_index, chosen]
                near = 1. / (1. + (distance / (height * .03)) ** 4)
                alignment = np.abs(np.einsum('ij,ij->i', normals, surface.normals[ref]))
                facing = np.clip((alignment - .2) / .5, 0., 1.)
                facing = 1. - np.clip(distance / (height * .005), 0., 1.) * (1. - facing)
                return np.clip(near * facing * (1. - ambiguity), .01, 1.)

            yield 'Checking correspondence consistency'
            chosen, transfer_rows, ambiguity, nearby_rows = select(np.zeros(count, dtype=np.int64))
            nearest_rows = transfer_rows
            confidence = confidence_for(chosen, ambiguity)
            for _ in range(2 if transition > 0 else 0):
                # Each vertex is judged by its neighbors' opinion, not its own:
                # an ambiguous strip must not vote for itself. Neighbors that
                # are themselves ambiguous get little say.
                vote = confidence * (1. - ambiguity) ** 2 * (transfer_rows.sum(axis=1) > 0)
                field = graph.smooth(transfer_rows, vote, 2 * transition)
                total_vote = graph.neighbor_sum(vote)
                consensus = graph.neighbor_sum(field * vote[:, None])
                consensus = np.where(total_vote[:, None] > 1e-12,
                                     consensus / np.maximum(total_vote, 1e-300)[:, None], field)
                chosen, transfer_rows, ambiguity, nearby_rows = select(chosen, consensus)
                confidence = confidence_for(chosen, ambiguity)
            refs = alt_ref[rows_index, chosen].astype(np.int32)
            bary = alt_bary[rows_index, chosen]
            points = alt_point[rows_index, chosen]
            distances = alt_distance[rows_index, chosen]
            # Stacked layers (pockets, linings, cords, folds, a hood on the back)
            # share smoothing and updates with the garment surface beneath them.
            # Uses the consistent correspondences, so a sleeve against the torso
            # (opposite body normals) is never merged with the torso panel.
            couple_a, couple_b, couple_w = layer_couplings(
                xyz, triangles, normals, points, surface.normals[refs], height * .03)
            graph = graph.coupled(couple_a, couple_b, couple_w)
            # Body transition sharpness under each garment vertex, for review.
            used_refs, ref_inverse = np.unique(refs, return_inverse=True)
            reference_rate = quality.body_transition_rate(source, source_tri[used_refs], source_dense)[ref_inverse.ravel()]
            del source_dense
            yield 'Smoothing transferred weights'
            # Confidence-weighted, geometry-aware smoothing: reliable contacts
            # keep their body weights; loose, distant or ambiguous fabric is
            # inpainted from its own reliable neighborhood along the surface.
            # Skin-tight fabric must follow the skin's own creases: a blend
            # wider than its body offset cuts inside concave folds (armpit,
            # hip front, belly) when the joint bends.
            smoothing_length = transition * np.clip(distances / max(transition, 1e-12), .2, 1.)
            has_row = transfer_rows.sum(axis=1) > 1e-12
            smoothed = graph.smooth(transfer_rows, confidence * has_row, smoothing_length) if transition > 0 else transfer_rows.copy()
            smoothed = np.maximum(smoothed, 0.)
            totals = smoothed.sum(axis=1, keepdims=True)
            smoothed = np.where(totals > 1e-12, smoothed / np.where(totals > 1e-12, totals, 1.), 0.)
            # Candidate support spreads over the same geodesic neighborhood that
            # smoothing and refinement use, so neighbors share bone slots.
            support = graph.propagate(smoothed, max(2 * transition, height * .005))
            ranking = smoothed + 1e-3 * support + 1e-5 * nearby_rows
            order = np.argsort(-ranking, axis=1, kind='stable')[:, :12]
            for i in range(count):
                existing_locks = locks[i]
                free = [int(columns[k]) for k in order[i] if ranking[i, k] > 1e-12][:12 - len(existing_locks)]
                chosen_bones = list(existing_locks) + free
                if not chosen_bones:
                    raise ValueError('All candidate deform groups are locked with zero weights')
                if smoothed[i].sum() <= 0 and sum(existing_locks.values()) < 1 - 1e-5:
                    raise ValueError(f'Body correspondence for clothing vertex {i} has no deform weights')
                # Padding uses a duplicate index with a fixed zero, never an extra influence.
                candidates[i] = chosen_bones + [chosen_bones[0]] * (12 - len(chosen_bones))
                fixed[i, len(chosen_bones):] = True
                for k, bone in enumerate(chosen_bones):
                    if bone in existing_locks:
                        fixed[i, k] = True
                        fixed_values[i, k] = existing_locks[bone]
                    else:
                        weights[i, k] = smoothed[i, column_of[bone]]
            far = int(np.count_nonzero(distances > reach))
            correspondence_stats = {
                'mean_confidence': round(float(confidence.mean()), 4),
                'low_confidence_vertices': int(np.count_nonzero(confidence < .25)),
                'ambiguous_vertices': int(np.count_nonzero(ambiguity > .5)),
                'consensus_switches': int(np.count_nonzero(chosen != 0)),
                'mean_candidate_bones': round(float(np.mean([len(set(row)) for row in candidates.tolist()])), 2),
                'layer_coupled_vertices': len(np.unique(couple_a)),
            }
            floor_rate = 1. / (.05 * height)
            transition_stats = {'nearest': quality.abrupt_transitions(graph, nearest_rows, reference_rate, floor_rate)[0],
                                'transfer': quality.abrupt_transitions(graph, smoothed, reference_rate, floor_rate)[0]}
            del nearest_rows, transfer_rows, nearby_rows, support, ranking
            if far:
                warnings.append(f'{far} vertices are farther from the body than the correspondence search range; inspect loose or misaligned regions.')
            corrected = 0
            correction = np.zeros_like(xyz)
            ref_normals = surface.normals[refs]
            if settings.correct_shape:
                signed = np.einsum('ij,ij->i', xyz - points, ref_normals)
                push = (distances <= reach) & (signed < clearance)
                correction[push] = ref_normals[push] * np.minimum(clearance - signed[push], max_fit)[:, None]
            offsets = xyz + correction - points

            def coherent(update):
                # One implicit smoothing step along the fabric (at least one
                # typical edge long) so straps, hoods and sleeves move as smooth
                # sheets rather than single-vertex spikes. Its short reach
                # leaves distant loose structures in place; later passes
                # re-push any clipping that smoothing diluted.
                if not np.any(update):
                    return update
                return graph.smooth(update, np.ones(count), max(max_fit, graph.reference_length))

            if settings.correct_shape:
                offsets -= correction
                correction = coherent(correction)
                correction *= np.minimum(1, max_fit / np.maximum(np.linalg.norm(correction, axis=1), 1e-12))[:, None]
                parts.correct_rigid_shape(clothing, correction, settings.attach_hardware)
                offsets += correction
                # Correct face-level rest clipping too, including coarse faces
                # whose corners all sit outside the body. Never exceed the
                # user's per-vertex displacement bound.
                degree = np.zeros(count)
                for k in range(3):
                    np.add.at(degree, triangles[:, k], 4 / 3)
                for fit_pass in range(4):
                    fitted = xyz + correction
                    vertex_force = np.zeros_like(xyz)
                    for begin in range(0, count, 2048):
                        end = min(count, begin + 2048)
                        vertex_force[begin:end], _ = surface.collisions(fitted[begin:end], clearance, reach)
                        yield f'Fitting rest shape {fit_pass + 1}/4: {end}/{count}'
                    sample_xyz = contact_samples.positions(fitted)
                    sample_force = np.zeros_like(sample_xyz)
                    for begin in range(0, len(sample_xyz), 2048):
                        end = min(len(sample_xyz), begin + 2048)
                        sample_force[begin:end], _ = surface.collisions(sample_xyz[begin:end], clearance, reach)
                        yield f'Fitting face clearance {fit_pass + 1}/4: {end}/{len(sample_xyz)}'
                    distributed = contact_samples.distribute(sample_force, count)
                    proposal = correction + coherent(vertex_force + distributed / np.maximum(degree[:, None], 1))
                    length = np.linalg.norm(proposal, axis=1)
                    proposal *= np.minimum(1, max_fit / np.maximum(length, 1e-12))[:, None]
                    parts.correct_rigid_shape(clothing, proposal, settings.attach_hardware)
                    offsets += proposal - correction
                    correction = proposal
                corrected = int(np.count_nonzero(np.linalg.norm(correction, axis=1) > 1e-10))
            if corrected:
                # Relative to Basis so current expressions and existing keys stay independent.
                write_fit_key(clothing, 'acf_correction_key', CORRECTION_KEY, correction, correction_key)
                xyz += correction
                bpy.context.view_layer.update()
            required_slots = np.count_nonzero(fixed_values > 1e-8, axis=1) + (fixed_values.sum(axis=1) < 1 - 1e-5)
            influence_limits = np.maximum(settings.influence_limit, required_slots)
            expanded_vertices = int(np.count_nonzero(influence_limits > settings.influence_limit))
            if expanded_vertices:
                warnings.append(f'{expanded_vertices} vertices need up to {int(influence_limits.max())} influences to keep all protected weights and fill the remaining body weight; check Unity skin-weight quality.')
            weights = project(weights, fixed, fixed_values, influence_limits)
            dense = np.zeros((count, len(names)))
            vertex_ids = np.arange(count)
            for k in range(weights.shape[1]):
                np.add.at(dense, (vertex_ids, candidates[:, k]), weights[:, k])
            part_rows, part_stats = parts.constraints(clothing, xyz, dense, names, preserved_names,
                                                     settings.attach_hardware, settings.influence_limit,
                                                     height * settings.strap_smoothing / 100)
            parts.pin_sparse(part_rows, candidates, weights, fixed, fixed_values)
            if part_stats['skipped_protected_hardware']:
                warnings.append(f"{part_stats['skipped_protected_hardware']} hardware islands retain varying protected weights or have no compatible free fabric influence; those assignments were retained.")
            if part_stats['skipped_unattached_hardware']:
                warnings.append(f"{part_stats['skipped_unattached_hardware']} hardware islands have no supported fabric attachment; retained their surface-transferred weights.")
            if part_stats['skipped_belts']:
                warnings.append('Some marked belt vertices could not be smoothed without changing protected weights.')
            baseline = weights.copy()
            # Attachment refresh can change sparse bone slots. Restore their
            # indices as well as weights if actual-pose acceptance rejects it.
            part_indices = np.fromiter(part_rows, dtype=np.intp)
            baseline_part_candidates = candidates[part_indices]
            baseline_part_fixed = fixed[part_indices]
            baseline_part_values = fixed_values[part_indices]
            baseline_part_stats = dict(part_stats)
            has_hardware = bool(part_stats['attached_hardware_islands'])

            def refresh_hardware():
                if not has_hardware:
                    return
                dense.fill(0.)
                for k in range(weights.shape[1]):
                    np.add.at(dense, (vertex_ids, candidates[:, k]), weights[:, k])
                rows, stats = parts.constraints(clothing, xyz, dense, names, preserved_names,
                                               settings.attach_hardware, settings.influence_limit,
                                               smooth_belts=False)
                parts.pin_sparse(rows, candidates, weights, fixed, fixed_values)
                for key in ('attached_hardware_islands', 'blended_hardware_islands',
                            'skipped_protected_hardware', 'skipped_unattached_hardware'):
                    part_stats[key] = stats[key]
            deformation = quality.Deformation(xyz, triangles)
            strain_edges = deformation.edges[deformation.valid_edges]
            strain_rest = deformation.rest_length[deformation.valid_edges]
            rest_frame = frame(source[source_tri[refs]])
            local_offsets = np.einsum('vij,vj->vi', rest_frame, offsets)
            thorough = settings.validation_mode == 'THOROUGH'
            forward = -1.0 if settings.reverse_forward else 1.0
            core_specs = [{'name': 'Rest', 'rotations': {}}] + pose_specs(rig, settings.secondary_bones, forward)
            if settings.secondary_bones:
                # Unknown garment-only names (e.g. Bone.l) still deserve probes.
                # Rotate each exclusive root; descendants inherit naturally.
                exclusive_names = {name for name in preserved_names if name_indices[name] not in body_bones}
                roots = [name for name in exclusive_names if
                         (not clothing_rig.data.bones[name].parent or clothing_rig.data.bones[name].parent.name not in exclusive_names)]
                for name in sorted(roots)[:6]:
                    for sign in (-1, 1):
                        core_specs.append({'name': 'clothing.' + name + ('+' if sign > 0 else '-'), 'rotations': {}, 'clothing_rotations': {name: (math.radians(8) * sign, 0., 0.)}})
            matches = identify_bones(rig)
            phase_seconds = {'transfer_and_rest_fit': time.perf_counter() - start}
            phase_start = time.perf_counter()
            del surface
            profile.counters['body_triangles'] = len(source_tri)
            source_kin, target_kin = state.source_kinematics, state.target_kinematics
            body_modifier = next(m for m in body.modifiers if m.type == 'ARMATURE' and m.object == rig and m.show_viewport)
            # Exactly linear skinning on both meshes makes linear-blend predictions
            # match Blender and the convex candidate envelope bound every position.
            linear = not modifier.use_deform_preserve_volume and not body_modifier.use_deform_preserve_volume
            used_bones = np.unique(candidates[~fixed | (fixed_values > 0)])
            body_skin = screening.Skin.from_rows(source_rows, len(source_names), source_totals)
            envelope = screening.EnvelopeIndex(xyz, candidates, used_bones, part_indices)
            ref_vertices = np.zeros(len(source), dtype=bool)
            ref_vertices[source_tri[refs].ravel()] = True
            margin = height * 1e-5
            threshold = max(clearance * .1, height * 1e-5)
            follow_limit = max(height * .01, 4 * clearance)
            rest_crossings = self_intersections(xyz, triangles)[1] if settings.self_collision else set()
            if rest_crossings:
                warnings.append(f'{len(rest_crossings)} self-intersections already exist in rest pose; weight changes cannot remove rest intersections.')
            meter = screening.PoseMeter(triangles, contact_samples, deformation, clearance, reach, threshold,
                                        rest_crossings, settings.self_collision, source_tri, profile)
            near_rest = near_faces(source, source_tri, xyz, reach)
            rest_spheres = screening.triangle_spheres(source, source_tri)
            rest_view = screening.BodyView(source, rest_spheres, source_tri)
            body_radius = float(np.percentile(rest_spheres[1], 95))
            screener = screening.Screener(deformation, source_tri[refs], bary, distances, xyz, clearance, threshold,
                                          reach, max(clearance, height * .005), body_radius, profile)
            rest_proximity = screening.PointProximity(xyz, reach + height * 1e-5, body_radius)
            signature_vertices = np.unique(np.concatenate((source_tri[refs].ravel(), source_tri[near_rest].ravel())))

            slot_cache = {}

            def basis_for(matrices, slots=None):
                # Transform the garment once per distinct candidate bone, then
                # gather per slot; avoids a dense vertex x all-bones product.
                slots = candidates if slots is None else slots
                cached = slot_cache.get(id(slots))
                if cached is None or not np.array_equal(cached[0], slots):
                    bones, inverse = np.unique(slots, return_inverse=True)
                    cached = slot_cache[id(slots)] = (slots.copy(), bones, inverse.reshape(slots.shape))
                _, bones, inverse = cached
                transforms = matrices[bones]
                moved = np.matmul(xyz, transforms[:, :3, :3].transpose(0, 2, 1)) + transforms[:, None, :3, 3]
                return moved[inverse, np.arange(count)[:, None]]

            def blend(matrices, values=None, slots=None):
                values = weights if values is None else values
                return (values[:, :, None] * basis_for(matrices, slots)).sum(axis=1)

            def solve(spec):
                with profile.time('kinematics', spec['name']):
                    solution = state.solve(spec)
                    solution.source_D = source_kin.deformation(solution.source_poses, source_names)
                    solution.target_D = target_kin.deformation(solution.target_poses, names)
                    # Keep only bases and skinning matrices for large libraries.
                    solution.source_poses = solution.target_poses = None
                return solution

            rest_solution = solve(core_specs[0])

            def moved_rows(matrices, rest_matrices):
                return np.any(matrices != rest_matrices, axis=(1, 2))

            moved_cache = OrderedDict()

            def predicted_body(solution):
                """Avatar surface by linear blend; unmoved vertices keep actual Rest values.

                Moved vertices and faces depend only on which bones moved, so
                consecutive lines of one joint share them (small LRU)."""
                bones = moved_rows(solution.source_D, rest_solution.source_D)
                if not np.any(bones):
                    return rest_view
                with profile.time('body_prediction'):
                    key = bones.tobytes()
                    entry = moved_cache.get(key)
                    if entry is None:
                        mask = body_skin.vertices_of(bones)
                        entry = moved_cache[key] = (np.flatnonzero(mask), screening.moved_faces(source_tri, mask))
                        if len(moved_cache) > 8:
                            moved_cache.popitem(last=False)
                    else:
                        moved_cache.move_to_end(key)
                    moved, faces = entry
                    if not len(moved):
                        return rest_view
                    return screening.BodyView(source, rest_spheres, source_tri, moved,
                                              body_skin.positions(source, solution.source_D, moved), faces)

            ref_corners = source_tri[refs]

            def follow_target(posed_body):
                """Body-following garment targets from a body array or view."""
                if isinstance(posed_body, screening.BodyView):
                    posed_tri = posed_body.positions_at(ref_corners.ravel()).reshape(-1, 3, 3)
                else:
                    posed_tri = posed_body[ref_corners]
                return np.einsum('vi,vij->vj', bary, posed_tri) + np.einsum('vi,vij->vj', local_offsets, frame(posed_tri))

            def follow_vertices(positions, target, limit=None):
                """Confident vertices deviating from body-following by more than
                ``limit`` (default 1% of avatar size)."""
                error = np.linalg.norm(positions - target, axis=1)
                return (error > (follow_limit if limit is None else limit)) & (confidence > .25) & ~np.all(fixed, axis=1)

            # ---------------------------------------------------------------
            # Pose library: humanoid core probes plus a generic joint library.
            # Movable bones are chosen structurally: deform bones with a parent
            # whose subtree carries body weight, excluding global controls whose
            # subtree holds every weighted bone (they only move the avatar rigidly).
            source_index = {name: i for i, name in enumerate(source_names)}
            weighted = set(int(b) for b in np.unique(body_skin.bones[body_skin.weights > 0]))
            descendants = {}
            for name in reversed(source_kin.order):
                below = {name}
                for child in rig.data.bones[name].children:
                    below |= descendants[child.name]
                descendants[name] = below
            subtree_weighted = {name: {source_index[n] for n in below if n in source_index} & weighted
                                for name, below in descendants.items()}
            movable = [name for name in source_names if rig.data.bones[name].parent is not None
                       and subtree_weighted[name] and subtree_weighted[name] != weighted]
            lines = [core_line(spec) for spec in core_specs[1:] if 'motion' in spec]
            lines += joint_lines(rig, movable)
            if len(core_specs) == 1:
                warnings.append('No standard bone names detected: humanoid combined probes were skipped; '
                                'the generic joint library still tests every movable bone.')
            # Structural line proof: a line that moves no garment bone and whose
            # moving body vertices stay inside a ball around the rotating joint
            # that is out of range of the static garment is irrelevant at every
            # angle (positions under nested rotations about joint heads remain
            # within |p - h| + 2 sum |c_j - h| of the top joint head h).
            heads = {name: np.array(rig.matrix_world @ rig.data.bones[name].head_local) for name in source_names}
            target_children = {name: [c.name for c in clothing_rig.data.bones[name].children] for name in target_kin.order}
            used_names = {names[int(b)] for b in used_bones}
            linked_by_source = defaultdict(list)
            for target_name, source_name in links.items():
                linked_by_source[source_name].append(target_name)

            def line_proven_irrelevant(line):
                if not linear:
                    return False
                bones = list(line['motion'])
                tops = [b for b in bones if not any(b != o and b in descendants[o] for o in bones)]
                if len(tops) != 1 or any(b not in source_index for b in bones):
                    return False
                top = tops[0]
                moving = descendants[top]
                if clothing_rig == rig:
                    affected = set(moving)
                else:
                    affected, stack = set(), [t for s in moving for t in linked_by_source.get(s, ())]
                    while stack:
                        name = stack.pop()
                        if name not in affected:
                            affected.add(name)
                            stack.extend(target_children.get(name, ()))
                if affected & used_names:
                    return False
                mask = np.zeros(len(source_names), dtype=bool)
                mask[[source_index[n] for n in moving if n in source_index]] = True
                vertices = body_skin.vertices_of(mask)
                if np.any(vertices & ref_vertices):
                    return False
                faces = screening.moved_faces(source_tri, vertices)
                if not len(faces):
                    return True
                corners = np.unique(source_tri[faces])
                h = heads[top]
                radius = float(np.linalg.norm(source[corners] - h, axis=1).max())
                radius += sum(2 * float(np.linalg.norm(heads[b] - h)) for b in bones if b != top)
                gap = float(np.sqrt(np.min(np.sum((xyz - h) ** 2, axis=1))))
                return gap > radius + reach + margin

            def rest_equivalent(solution, view):
                """Proof: same garment, targets and bounded queries as Rest."""
                if not linear or np.any(moved_rows(solution.target_D[used_bones], rest_solution.target_D[used_bones])):
                    return False
                if np.any(ref_vertices[view.moved]):
                    return False
                _, rest_part, posed_part = view.change_from_rest()
                return not rest_proximity.touches((rest_part, posed_part))

            def body_difference(view, other_view):
                """Triangles whose positions differ between two posed bodies,
                with their bounding spheres in both."""
                a, b = view.positions(), other_view.positions()
                candidates_ = np.union1d(view.moved, other_view.moved)
                differs = np.zeros(len(source), dtype=bool)
                differs[candidates_[np.any(a[candidates_] != b[candidates_], axis=1)]] = True
                faces = screening.moved_faces(source_tri, differs)
                return differs, (screening.triangle_spheres(a, source_tri[faces]),
                                 screening.triangle_spheres(b, source_tri[faces]))

            def equivalent_pair(solution, other):
                """Proof against another evaluated pose with identical garment bones."""
                if not np.array_equal(solution.target_D[used_bones], other['solution'].target_D[used_bones]):
                    return False
                if not linear:
                    return (np.array_equal(solution.target_D, other['solution'].target_D) and
                            np.array_equal(solution.source_D, other['solution'].source_D))
                differs, spheres = body_difference(predicted_body(solution), predicted_body(other['solution']))
                if np.any(differs & ref_vertices):
                    return False
                return not screening.faces_near(envelope.points(solution.target_D), reach + margin, spheres)

            records = []
            by_name = {}
            # Candidate equivalence partners share garment bone transforms and
            # every body vertex near the Rest garment or under a correspondence;
            # only the first pose with that signature is checked by full proof.
            signatures = {}
            pred_rest_positions = blend(rest_solution.target_D)
            pred_rest = screener.screen(pred_rest_positions, rest_view,
                                        quality.blended_linear(weights, candidates, rest_solution.target_D),
                                        True, pose='Rest')

            def evaluate(spec, line=None, fraction=1.0, core=False, views=None):
                """Prove equivalence or screen one pose by linear-blend prediction."""
                name = spec['name']
                if name in by_name:
                    return by_name[name]
                solution = solve(spec)
                with profile.time('relevance', name):
                    garment_moved = bool(np.any(moved_rows(solution.target_D[used_bones],
                                                           rest_solution.target_D[used_bones])))
                    view = predicted_body(solution)
                    record = {'name': name, 'spec': spec, 'line': line, 'fraction': fraction, 'core': core,
                              'solution': solution, 'equivalent': None, 'garment_moved': garment_moved}
                    if name != 'Rest' and not garment_moved and rest_equivalent(solution, view):
                        record['equivalent'] = 'Rest'
                    else:
                        key = solution.target_D[used_bones].tobytes() + view.positions_at(signature_vertices).tobytes()
                        other = signatures.setdefault(key, record)
                        if other is not record and equivalent_pair(solution, other):
                            record['equivalent'] = other['name']
                profile.counters['poses_screened_kinematic'] += 1
                if record['equivalent'] is None:
                    with profile.time('predicted_screening', name):
                        if garment_moved:
                            positions = blend(solution.target_D)
                            linear_part = quality.blended_linear(weights, candidates, solution.target_D)
                        else:
                            positions, linear_part = pred_rest_positions, None
                        measured = screener.screen(positions, view, linear_part, garment_moved,
                                                   reference=pred_rest, pose=name)
                        follow = (follow_vertices(positions, follow_target(view)) if garment_moved else None)
                    record['follow_count'] = 0 if follow is None else int(np.count_nonzero(follow))
                    record['kinds'], record['severity'] = screening.problem_summary(
                        measured.row, pred_rest.row, record['follow_count'])
                    record['predicted'] = measured
                    profile.counters['poses_screened_predicted'] += 1
                else:
                    record['kinds'], record['severity'] = (), 0.
                if views is not None:
                    views[name] = view
                records.append(record)
                by_name[name] = record
                return record

            body_span = 2. * float(np.percentile(rest_spheres[1], 95))
            path_cell = height * .02
            path_subset = screening.voxel_subset(xyz, path_cell)
            path_slack = float(np.sqrt(3.) * path_cell)

            def garment_of(rec):
                """Predicted garment positions for a screened or equivalent pose."""
                while rec['equivalent'] is not None and rec['equivalent'] != 'Rest':
                    rec = by_name[rec['equivalent']]
                if rec['equivalent'] == 'Rest' or not rec['garment_moved']:
                    return pred_rest_positions
                return blend(rec['solution'].target_D)

            interval_cache = {}

            def path_near(line_records, views):
                """Moving body or garment may enter range between adjacent samples.

                Sampling heuristic, not a proof: single-axis rotation arcs
                between samples stay within 0.52 of their chord (steps up to 60
                degrees) of an endpoint, and chords bound combined motion the
                same way between close samples. The garment is represented by
                one vertex per 2%-of-size voxel and the body by moved vertices,
                inflated by the voxel diagonal and a typical triangle span.
                """
                result = []
                previous = None
                for rec in line_records:
                    key = (previous['name'] if previous else 'Rest', rec['name'])
                    if key not in interval_cache:
                        before_view = views.get(previous['name'], rest_view) if previous else rest_view
                        view = views.get(rec['name']) or predicted_body(rec['solution'])
                        moved = np.union1d(view.moved, before_view.moved)
                        a, b = view.positions_at(moved), before_view.positions_at(moved)
                        differs = np.any(a != b, axis=1)
                        near = False
                        if np.any(differs):
                            a, b = a[differs], b[differs]
                            chord = float(np.linalg.norm(a - b, axis=1).max())
                            garment = garment_of(rec)
                            previous_garment = garment_of(previous) if previous else pred_rest_positions
                            garment_chord = float(np.linalg.norm(garment - previous_garment, axis=1).max())
                            inflation = .52 * (chord + garment_chord)
                            points = np.concatenate((garment[path_subset], previous_garment[path_subset]))
                            body_points = np.concatenate((a, b))
                            wide = ProximityGrid(points, reach + inflation + body_span + path_slack)
                            if np.any(wide.query(body_points)):
                                near = not np.any(ProximityGrid(points, reach).query(body_points))
                        interval_cache[key] = near
                    result.append(interval_cache[key])
                    previous = rec
                return result

            for spec in core_specs:
                evaluate(spec, line=spec['name'] if 'motion' in spec else None, core=True)
            # Each line starts at its full angle; adaptive bisection adds 50%
            # and finer angles where problems start or stop, the body may sweep
            # through range between samples, or severity curves.
            initial = (1.,) if not thorough else tuple(k / 6 for k in range(1, 7))
            min_step = .125 if not thorough else 1. / 24
            relevant_lines, irrelevant_lines = [], 0
            for line in lines:
                if line['kind'] != 'core' and line_proven_irrelevant(line):
                    irrelevant_lines += 1
                    continue
                relevant_lines.append(line)
            line_records = {}
            adaptive_added = 0
            yield f'Screening {len(relevant_lines)} relevant motion lines ({irrelevant_lines} proven irrelevant)'
            for number, line in enumerate(relevant_lines):
                fractions = list(initial)
                rows, views = {}, {}
                while fractions:
                    for fraction in fractions:
                        spec = line_sample(line, fraction)
                        rows[fraction] = evaluate(spec, line=line['id'], fraction=fraction,
                                                  core=line['kind'] == 'core' and fraction == 1., views=views)
                    ordered = [rows[f] for f in sorted(rows)]
                    with profile.time('adaptive_sampling'):
                        near = path_near(ordered, views)
                    evidence = [(0., 0., (), False)] + [(rec['fraction'], rec['severity'], rec['kinds'], flag)
                                                       for rec, flag in zip(ordered, near)]
                    fractions = adaptive_fractions(evidence, min_step, single_joint=line['kind'] == 'joint',
                                                   sweep_step=None if thorough else .5)
                    adaptive_added += len(fractions)
                line_records[line['id']] = [rows[f] for f in sorted(rows)]
                if number % 8 == 0:
                    yield f'Screening poses: line {number + 1}/{len(relevant_lines)}'
            interval_cache.clear()
            phase_seconds['pose_screening'] = time.perf_counter() - phase_start
            phase_start = time.perf_counter()

            # ---------------------------------------------------------------
            # Selection. Core probes keep actual validation as before. In
            # Automatic mode a library pose joins them only when it reveals
            # something the core poses do not: a predicted problem kind none of
            # them shows, or a severity above all of them (worst per line).
            # Thorough validates every relevant library pose.
            core_named = [by_name[spec['name']] for spec in core_specs]
            validated = [core_named[0]] + [rec for rec in core_named[1:] if rec['equivalent'] is None]
            chosen = {rec['name'] for rec in validated}
            library = [rec for rec in records if rec['name'] not in chosen and rec['equivalent'] is None]
            if thorough:
                promoted = sorted(library, key=lambda rec: (-rec['severity'], rec['name']))
            else:
                core_kinds = set().union(*(set(rec['kinds']) for rec in validated))
                core_worst = max((rec['severity'] for rec in validated), default=0.)
                worst = {}
                for rec in library:
                    if set(rec['kinds']) - core_kinds or rec['severity'] > core_worst:
                        key = rec['line'] or rec['name']
                        if key not in worst or (rec['severity'], rec['name']) > (worst[key]['severity'], worst[key]['name']):
                            worst[key] = rec
                promoted = sorted(worst.values(), key=lambda rec: (-rec['severity'], rec['name']))[:6]
            validated += promoted
            chosen |= {rec['name'] for rec in promoted}
            screened = [rec for rec in library if rec['name'] not in chosen]
            named = core_named + promoted
            max_memory = count * 12 * 3 * 8 + len(source) * 3 * 8 * len(validated) + count * 3 * 8 * len(validated) * 2
            if max_memory > 650 * 1024 * 1024:
                raise ValueError('Pose data would exceed the 650 MiB working budget; split the garment or reduce mesh density')
            yield f'Sampling {len(validated)} avatar poses with Blender deformation'
            prediction = {'body': 0., 'garment': 0., 'worst_pose': None}

            def note_prediction(kind, actual, predicted, pose):
                error = float(np.abs(actual - predicted).max(initial=0.))
                if error > prediction[kind]:
                    prediction[kind] = error
                    if error >= max(prediction['body'], prediction['garment']):
                        prediction['worst_pose'] = pose

            for rec in validated:
                state.apply(rec['solution'])
                with profile.time('mesh_extraction', rec['name']):
                    posed_source, _, _ = mesh_arrays(body, depsgraph)
                profile.counters['mesh_extractions'] += 1
                note_prediction('body', posed_source, predicted_body(rec['solution']).positions(), rec['name'])
                if np.array_equal(posed_source, source):
                    posed_source = source
                rec['actual_body'] = posed_source
                rec['target'] = follow_target(posed_source)
                yield 'Sampled ' + rec['name']
            phase_seconds['pose_sampling'] = time.perf_counter() - phase_start
            # Bounded cache of envelope collision surfaces for refined poses.
            surface_cache = OrderedDict()
            surface_budget = 128 * 1024 * 1024

            def envelope_surface(rec):
                key = id(rec['actual_body'])
                if key in surface_cache:
                    surface_cache.move_to_end(key)
                    return surface_cache[key][0]
                surf = meter.region(rec['actual_body'], envelope.points(rec['solution'].target_D))
                size = max(1, surf.face_count * 256)
                while surface_cache and sum(v[1] for v in surface_cache.values()) + size > surface_budget:
                    surface_cache.popitem(last=False)
                surface_cache[key] = (surf, size)
                profile.counters['collision_surfaces_built'] += 1
                return surf

            def weight_transitions():
                dense_weights, _, _ = to_dense(weights, candidates)
                return quality.abrupt_transitions(graph, dense_weights, reference_rate, floor_rate)

            refine_names = set()

            def measure_actual(label, reference=None):
                """Blender's actual deformation in every validated pose.

                Query results are reused only for points whose actual position
                did not change and which no moved body triangle can reach:
                against Rest for a first measurement, against an earlier
                measurement of the same pose (``reference``) afterwards."""
                results = []
                write_weights(clothing, names, candidates, weights, preserved_names)
                transitions, abrupt = weight_transitions()
                for index, rec in enumerate(validated):
                    name = rec['name']
                    state.apply(rec['solution'])
                    with profile.time('mesh_extraction', name):
                        actual, _, _ = mesh_arrays(clothing, depsgraph)
                    profile.counters['mesh_extractions'] += 1
                    profile.counters['actual_validations'] += 1
                    linear_part = quality.blended_linear(weights, candidates, rec['solution'].target_D)
                    surface = envelope_surface(rec) if name in refine_names else None
                    previous = reference[index] if reference is not None else (results[0] if index else None)
                    with profile.time('actual_validation', name):
                        measured = meter.measure(actual, rec['actual_body'], linear_part, reference=previous,
                                                 surface=surface, pose=name)
                    if linear and reference is None:
                        note_prediction('garment', actual, blend(rec['solution'].target_D), name)
                    results.append(measured)
                    yield label + ': ' + name
                if meter.capped and 'Self-intersection pair budget reached; report may undercount intersections.' not in warnings:
                    warnings.append('Self-intersection pair budget reached; report may undercount intersections.')
                return results, transitions, abrupt

            def named_rows(results):
                metrics = {rec['name']: dict(m.row, pose=rec['name']) for rec, m in zip(validated, results)}
                # Preserve every named row and its contribution to acceptance.
                # Only geometrically proven equivalent probes are reused.
                return [dict(metrics[rec['equivalent'] or rec['name']], pose=rec['name']) for rec in named]

            def problem_matrix(results, abrupt):
                # Column 0: static weight transitions; then one column per pose.
                problems = np.zeros((count, len(results) + 1))
                problems[list(abrupt), 0] += 1
                for k, measured in enumerate(results):
                    problems[:, k + 1] = measured.problems
                return problems

            def residual_of(results, abrupt):
                residual = set(abrupt)
                for measured in results:
                    residual.update(measured.residual)
                return residual

            def screen_after(baseline_weights, baseline_slots):
                """Re-check screened poses incrementally with the new weights.

                Uses the same screening subset and checks as the baseline
                screening, so screened rows compare like with like."""
                rows = []
                for rec in screened:
                    solution = rec['solution']
                    before_m = rec['predicted']
                    if not rec['garment_moved']:
                        # The garment shape is weight-independent in this pose.
                        rows.append(before_m)
                        continue
                    positions = blend(solution.target_D)
                    previous = blend(solution.target_D, baseline_weights, baseline_slots)
                    if np.array_equal(positions, previous):
                        rows.append(before_m)
                        continue
                    with profile.time('predicted_revalidation', rec['name']):
                        linear_part = quality.blended_linear(weights, candidates, solution.target_D)
                        rows.append(screener.screen(positions, predicted_body(solution), linear_part, True,
                                                    pose=rec['name']))
                    profile.counters['poses_revalidated_predicted'] += 1
                return rows

            def regressions(after, before, transitions_after, transitions_before):
                """Named non-regression failures; empty means accept."""
                failed = []
                if (sum(row['body_depth_squared'] for row in after) >
                        sum(row['body_depth_squared'] for row in before) + height * height * 1e-9):
                    failed.append('total body clearance')
                if sum(row['self_crossings'] for row in after) > sum(row['self_crossings'] for row in before):
                    failed.append('total self-intersections')
                # Clearance must not be bought with worse deformation: excessive
                # strain, collapsed/folded faces or sharper weight transitions.
                if (sum(row['excess_strain'] for row in after) >
                        sum(row['excess_strain'] for row in before) * 1.02 + 1e-9):
                    failed.append('total excess strain')
                if (transitions_after['abrupt_edges'] >
                        transitions_before['abrupt_edges'] + max(2, graph.mesh_edges // 2000)):
                    failed.append('abrupt weight transitions')
                # Also guard individual poses: aggregate improvement can conceal a new failure.
                return failed + pose_regressions(zip(after, before))

            faces_slack = max(2, len(triangles) // 2000)

            def pose_regressions(pairs):
                failed = []
                for a, b in pairs:
                    if a['body_depth_squared'] > b['body_depth_squared'] * 1.10 + height * height * 1e-7:
                        failed.append('body clearance: ' + a['pose'])
                    # Totals stay strict; one pose may wobble by a few pairs
                    # among many pre-existing layer contacts. Screened poses
                    # carry no crossing count.
                    if (a['self_crossings'] is not None and b['self_crossings'] is not None and
                            a['self_crossings'] > b['self_crossings'] + max(2, b['self_crossings'] // 50)):
                        failed.append('self-intersections: ' + a['pose'])
                    if a['excess_strain'] > b['excess_strain'] * 1.15 + 1e-7:
                        failed.append('excess strain: ' + a['pose'])
                    if a['collapsed_faces'] + a['folded_faces'] > b['collapsed_faces'] + b['folded_faces'] + faces_slack:
                        failed.append('collapsed/folded faces: ' + a['pose'])
                return failed

            def screened_regressions(rows):
                return pose_regressions((dict(a.row, pose='screened ' + rec['name']), rec['predicted'].row)
                                        for rec, a in zip(screened, rows))

            phase_start = time.perf_counter()
            baseline_results, baseline_transitions, baseline_abrupt = yield from measure_actual('Baseline')
            transition_stats['baseline'] = baseline_transitions
            before = named_rows(baseline_results)
            before_problems = problem_matrix(baseline_results, baseline_abrupt)
            before_residual = residual_of(baseline_results, baseline_abrupt)
            phase_seconds['baseline'] = time.perf_counter() - phase_start
            phase_start = time.perf_counter()

            # ---------------------------------------------------------------
            # Localized refinement: only poses whose actual problems weights can
            # change, only an active region grown geodesically around them.
            baseline_candidates = candidates.copy()
            refine = []
            useful = np.zeros(count, dtype=bool)
            # Rest problems (clipping, crossings) exist in every pose and no
            # weight can change the rest shape; target what each pose adds.
            # Clipping, crossings, collapsed/folded faces and body-following
            # error start refinement; strain forces then act in that region
            # and the strain checks still cover the whole garment.
            rest_counts = baseline_results[0].hard
            rest_problems = rest_counts > 0
            crossing_poses = set()
            for index, rec in enumerate(validated[1:], 1):
                measured = baseline_results[index]
                if measured.row['self_crossings']:
                    crossing_poses.add(index)
                # Library poses guard acceptance; optimizing weights for their
                # extreme angles would trade away quality in ordinary poses.
                if not rec['core']:
                    continue
                follow = follow_vertices(measured.positions, rec['target'])
                # Meaningful body-following error (3% of avatar size) opens a
                # region; the 1% count still ranks poses.
                problem = (measured.hard > rest_counts + 1e-9) | (follow & follow_vertices(
                    measured.positions, rec['target'], 3 * follow_limit))
                if not np.any(problem):
                    continue
                basis = basis_for(rec['solution'].target_D)
                spread = np.where(fixed, 0., np.linalg.norm(basis - measured.positions[:, None], axis=2)).max(axis=1)
                fixable = problem & (spread > height * 1e-6)
                if not np.any(fixable):
                    continue
                kinds, severity = screening.problem_summary(measured.row, baseline_results[0].row,
                                                            int(np.count_nonzero(follow)))
                refine.append((int(np.count_nonzero(fixable & ~rest_problems)) + severity, index, fixable))
            refine.sort(key=lambda row: (-row[0], row[1]))
            if not thorough:
                refine = refine[:16]
            refine.sort(key=lambda row: row[1])
            for _, index, fixable in refine:
                useful |= fixable
                refine_names.add(validated[index]['name'])
            spread_radius = max(2 * transition, height * .005)
            active = graph.propagate(useful[:, None].astype(float), spread_radius)[:, 0] > 0
            active &= ~np.all(fixed, axis=1)
            halo = graph.propagate(active[:, None].astype(float),
                                   max(spread_radius, 2 * float(np.max(smoothing_length, initial=0.))))[:, 0] > 0
            free_rows = np.ones(count, dtype=bool)
            free_rows[part_indices] = False
            outside = ~halo & free_rows
            active_rows = np.flatnonzero(active)
            active_triangles = np.any(active[triangles], axis=1)
            active_samples = meter.sample_mask(active)
            iterations_used = 0
            last_change = 0.
            previous_residual = None
            for iteration in range(settings.iterations if len(active_rows) else 0):
                previous_weights = weights.copy()
                residual = 0.
                for _, index, _ in refine:
                    rec = validated[index]
                    with profile.time('refinement_step', rec['name']):
                        matrices = rec['solution'].target_D
                        basis = basis_for(matrices)
                        positions = np.einsum('vk,vkj->vj', weights, basis)
                        surf = envelope_surface(rec)
                        body_force = np.zeros_like(positions)
                        with profile.time('bvh_query', rec['name']):
                            body_force[active_rows] = surf.collisions(positions[active_rows], clearance, reach,
                                                                      profile.counters)[0]
                            sample_positions = contact_samples.positions(positions)
                            sample_force = np.zeros_like(sample_positions)
                            sample_rows = np.flatnonzero(active_samples)
                            sample_force[sample_rows] = surf.collisions(sample_positions[sample_rows], clearance,
                                                                        reach, profile.counters)[0]
                        body_force += contact_samples.distribute(sample_force, count) * .25
                        self_force = np.zeros_like(positions)
                        # Separating forces only where the pose had new
                        # crossings; acceptance still checks every pose.
                        if settings.self_collision and index in crossing_poses:
                            with profile.time('self_intersection', rec['name']):
                                self_force = self_intersections(positions, triangles, rest_crossings,
                                                                changed=active_triangles, counters=profile.counters)[0]
                        strain = quality.strain_forces(positions, strain_edges, strain_rest)
                        rows = active_rows
                        residual += float(np.linalg.norm(body_force[rows], axis=1).sum() +
                                          np.linalg.norm(self_force[rows], axis=1).sum())
                        weights[rows] = fit_pose(weights[rows], baseline[rows], basis[rows], rec['target'][rows],
                                                 body_force[rows], self_force[rows], fixed[rows], fixed_values[rows],
                                                 settings.strength, height, influence_limits[rows],
                                                 follow=confidence[rows], strain=strain[rows])
                    profile.counters['refinement_steps'] += 1
                    yield f'Refining {iteration + 1}/{settings.iterations}: {rec["name"]}'
                with profile.time('smoothing'):
                    weights = smooth_update(weights, baseline, candidates, graph, fixed, fixed_values, smoothing_length, influence_limits)
                # Changes stay inside the geodesic halo; everything else keeps the
                # exact initial transfer, so validation reuse stays exact.
                weights[outside] = baseline[outside]
                refresh_hardware()
                iterations_used += 1
                # Hardware rows are rebuilt (with new slots) from the fabric.
                last_change = float(np.abs(weights[free_rows] - previous_weights[free_rows]).max(initial=0.))
                # Converged when weights stop moving or the collision/crossing
                # residual the passes work against no longer drops by 3%.
                stalled = previous_residual is not None and residual > .97 * previous_residual
                previous_residual = residual
                if last_change < 1e-3 or stalled:
                    break
            phase_seconds['refinement'] = time.perf_counter() - phase_start
            phase_start = time.perf_counter()
            refined_any = not np.array_equal(weights, baseline) or not np.array_equal(candidates, baseline_candidates)
            if refined_any:
                after_results, result_transitions, result_abrupt = yield from measure_actual('Checking result', baseline_results)
            else:
                after_results, result_transitions, result_abrupt = baseline_results, baseline_transitions, baseline_abrupt
            transition_stats['result'] = result_transitions
            after = named_rows(after_results)
            residual = residual_of(after_results, result_abrupt)
            phase_seconds['result_check'] = time.perf_counter() - phase_start
            # Accept using Blender's actual modifier deformation, not just LBS predictions.
            failed = regressions(after, before, result_transitions, baseline_transitions)
            partial_failed = []
            reverted_vertices = None
            worse = np.zeros(count, dtype=bool)
            problems = problem_matrix(after_results, result_abrupt)
            refined = weights
            phase_start = time.perf_counter()

            def revert_regions(regions, label):
                """Return neighborhoods smoothly to the initial transfer and
                re-check every validated pose."""
                nonlocal weights
                revert = np.clip(graph.propagate(regions[:, None].astype(float),
                                                 max(2 * transition, height * .005))[:, 0], 0., 1.)
                # Attachment rows are rebuilt from the blended fabric below.
                revert[part_indices] = 0.
                weights = project(revert[:, None] * baseline + (1. - revert[:, None]) * refined,
                                  fixed, fixed_values, influence_limits)
                weights[outside] = baseline[outside]
                refresh_hardware()
                results, transitions, abrupt = yield from measure_actual(label, baseline_results)
                rows = named_rows(results)
                return revert, results, transitions, abrupt, rows, regressions(rows, before, transitions, baseline_transitions)

            for attempt in range(2 if failed else 0):
                # Keep refinement where it helped: return only neighborhoods that
                # measured worse to the initial transfer, blending smoothly, and
                # re-check every aggregate and per-pose criterion. A second
                # round also reverts regions the first blend made worse.
                grown = worse | np.any(problems > before_problems + 1e-9, axis=1)
                if np.array_equal(grown, worse):
                    break
                worse = grown
                revert, partial_results, partial_transitions, partial_abrupt, partial_rows, partial_failed = \
                    yield from revert_regions(worse, f'Checking partially reverted result {attempt + 1}')
                problems = problem_matrix(partial_results, partial_abrupt)
                if not partial_failed:
                    after, transition_stats['result'] = partial_rows, partial_transitions
                    after_results = partial_results
                    residual = residual_of(partial_results, partial_abrupt)
                    reverted_vertices = int(np.count_nonzero(revert > .5))
                    break
            accepted = not failed or reverted_vertices is not None
            # Screened library poses guard the weights about to be kept, once:
            # regions they predict worse return to the initial transfer and
            # every validated pose is re-checked; any remaining failure rejects.
            screened_after = [rec['predicted'] for rec in screened]
            if accepted and refined_any and screened:
                with profile.time('screened_guard'):
                    screened_after = screen_after(baseline, baseline_candidates)
                screened_failed = screened_regressions(screened_after)
                if screened_failed:
                    for rec, measured in zip(screened, screened_after):
                        worse |= measured.problems(count) > rec['predicted'].problems(count) + 1e-9
                    revert, guard_results, guard_transitions, guard_abrupt, guard_rows, guard_failed = \
                        yield from revert_regions(worse, 'Checking result after screened-pose guard')
                    screened_after = screen_after(baseline, baseline_candidates)
                    guard_failed = guard_failed + screened_regressions(screened_after)
                    if guard_failed:
                        accepted, reverted_vertices = False, None
                        failed, partial_failed = failed or screened_failed, guard_failed
                    else:
                        after, transition_stats['result'] = guard_rows, guard_transitions
                        after_results = guard_results
                        residual = residual_of(guard_results, guard_abrupt)
                        reverted_vertices = int(np.count_nonzero(revert > .5))
            if reverted_vertices is not None:
                warnings.append(f'Refinement was kept except near {reverted_vertices} vertices whose actual-pose '
                                'or screened-pose checks got worse; those regions returned smoothly to the initial transfer.')
            if failed or reverted_vertices is not None:
                phase_seconds['partial_check'] = time.perf_counter() - phase_start
            rejected = None
            if not accepted:
                rejected = {'failed_checks': failed[:24], 'failed_check_count': len(failed),
                            'partial_revert_failed_checks': partial_failed[:24],
                            'measured': [{key: row[key] for key in ('pose', 'body_depth_squared', 'self_crossings',
                                                                    'excess_strain', 'collapsed_faces', 'folded_faces')}
                                         for row in after]}
                weights = baseline
                candidates[part_indices] = baseline_part_candidates
                fixed[part_indices] = baseline_part_fixed
                fixed_values[part_indices] = baseline_part_values
                part_stats = baseline_part_stats
                write_weights(clothing, names, candidates, weights, preserved_names)
                after = before
                residual = before_residual
                screened_after = [rec['predicted'] for rec in screened]
                transition_stats['result'] = transition_stats['baseline']
                warnings.append('Refinement failed the actual-pose non-regression checks (' + ', '.join(failed[:3]) +
                                (', ...' if len(failed) > 3 else '') + '); retained the initial surface transfer and optional fit correction.')
            defect_poses = [row['pose'] for row in after if row['collapsed_faces'] + row['folded_faces']]
            if defect_poses:
                warnings.append(f"Collapsed or folded-over faces remain in {len(defect_poses)} poses "
                                f"({', '.join(defect_poses[:6])}{', ...' if len(defect_poses) > 6 else ''}); they are marked for review.")
            screened_defects = [rec['name'] for rec, m in zip(screened, screened_after)
                                if m.row['collapsed_faces'] + m.row['folded_faces'] or m.row['self_crossings']]
            if screened_defects:
                warnings.append(f"{len(screened_defects)} screened library poses predict remaining collapsed/folded faces "
                                f"or self-intersections ({', '.join(screened_defects[:4])}{', ...' if len(screened_defects) > 4 else ''}); "
                                "they were checked by linear-blend prediction only. Use Thorough validation to check them in Blender.")
            prediction_error = max(prediction['body'], prediction['garment'])
            prediction_reliable = linear and prediction_error <= max(height * 1e-5, clearance * .05)
            if not linear:
                warnings.append('Preserve Volume is enabled: pose exclusion proofs were disabled and library screening '
                                'used approximate linear-blend predictions; validated poses used Blender deformation.')
            elif not prediction_reliable:
                warnings.append(f'Linear-blend predictions differed from Blender deformation by {prediction_error:.3g} units; '
                                'screened library poses are approximate. Use Thorough validation.')
            state.set_pose(core_specs[0])
            previous_name = transaction.groups[transaction.active_group][0] if 0 <= transaction.active_group < len(transaction.groups) else None
            legacy_diagnostic = clothing.get('acf_review_group')
            store_review_markers(clothing, residual)
            restore_active_group(clothing, clothing_rig, previous_name, legacy_diagnostic)
            redundant_poses = {rec['name']: rec['equivalent'] for rec in core_named if rec['equivalent']}
            equivalent_library = [rec for rec in records if rec['equivalent'] and not rec['core']]
            library_stats = {
                'motion_lines': len(lines), 'lines_proven_irrelevant': irrelevant_lines,
                'lines_sampled': len(relevant_lines), 'movable_bones': len(movable),
                'generated_poses': len(records) + irrelevant_lines * 2,
                'sampled_poses': len(records), 'adaptive_samples': adaptive_added,
                'proven_equivalent': sum(1 for rec in records if rec['equivalent']),
                'screened_by_prediction': len(screened),
                'promoted_to_actual': len(promoted),
                'actual_validated': len(validated),
                'refined': len(refine),
                'refinement_active_vertices': int(len(active_rows)),
                'refinement_iterations': iterations_used,
                'refinement_last_change': round(last_change, 6),
                'linear_prediction_max_error': prediction,
                'predictions_exact': bool(prediction_reliable),
                'validation_mode': 'THOROUGH' if thorough else 'AUTO',
            }
            report = {'body': body.name, 'clothing': clothing.name, 'armature': rig.name, 'clothing_armature': clothing_rig.name, 'bone_links': links, 'preserved_groups': sorted(preserved_names), 'parts': part_stats, 'vertices': count, 'poses': len(named), 'mapped_bones': matches, 'refinement_accepted': bool(accepted), 'shape_corrected_vertices': int(corrected), 'distant_vertices': int(far), 'review_vertices': len(residual), 'before': before, 'after': after, 'warnings': warnings, 'elapsed_seconds': round(time.perf_counter() - start, 2)}
            report.update(evaluated_poses=len(validated), redundant_poses=redundant_poses,
                          pose_library=library_stats,
                          screened_poses=[{'pose': rec['name'], 'severity': rec['severity'], 'problems': list(rec['kinds']),
                                           'self_crossings': m.row['self_crossings'],
                                           'collapsed_folded': m.row['collapsed_faces'] + m.row['folded_faces'],
                                           'validation': 'linear prediction'}
                                          for rec, m in zip(screened, screened_after)],
                          promoted_poses=[{'pose': rec['name'], 'line': rec['line'], 'fraction': rec['fraction'],
                                           'severity': rec['severity'], 'problems': list(rec['kinds'])}
                                          for rec in promoted],
                          equivalent_library_poses=len(equivalent_library),
                          refined_poses=[validated[index]['name'] for _, index, _ in refine],
                          source_correspondence=source_selection,
                          correspondence=correspondence_stats,
                          weight_transitions=transition_stats,
                          rejected_refinement=rejected,
                          refinement_reverted_vertices=reverted_vertices,
                          collision_surfaces_built=profile.counters['collision_surfaces_built'],
                          unique_surface_samples=len(triangles) + len(contact_samples.edges),
                          phase_seconds={name: round(seconds, 3) for name, seconds in phase_seconds.items()},
                          profile=profile.summary())
            clothing['acf_report'] = json.dumps(report)
            text = bpy.data.texts.new('Cloth Fit Report - ' + clothing.name)
            text.write(json.dumps(report, indent=2))
        transaction.commit()
        settings.status = (f'Done: {len(named)} poses validated, {len(records)} screened, '
                           f'{len(residual)} vertices to review')
        return report
    finally:
        transaction.rollback()
