"""Sculpt Mode tools: Smart Grab, Fit to Body and Relax brushes, Region Resize.

Toolbar tools (``WorkSpaceTool``) map the left mouse button to modal
operators. A stroke reads the coordinates Sculpt Mode edits (the active shape
key, or the mesh when it has none), changes them with ``brushes`` and writes
them back; Blender records one undo step per stroke. The math lives in
``brushes.py``; this module only talks to Blender.
"""
import itertools
import math
import traceback
import bpy
import numpy as np
from bpy.props import EnumProperty
from bpy_extras.view3d_utils import (location_3d_to_region_2d, region_2d_to_location_3d, region_2d_to_origin_3d,
                                     region_2d_to_vector_3d)
from mathutils import Vector
from . import brushes, engine
from .assisted import Accessories

REFERENCE = 'acf_reference'
_TOPOLOGY = {}
_BODY = {}


# ------------------------------------------------------------------ mesh access
def topology(obj):
    """Connectivity of ``obj`` (cached until its topology or datablock changes)."""
    mesh = obj.data
    key = (mesh.as_pointer(), len(mesh.vertices), len(mesh.edges), len(mesh.loops))
    cached = _TOPOLOGY.get(obj.name)
    if cached and cached[0] == key:
        return cached[1]
    mesh.calc_loop_triangles()
    triangles = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
    mesh.loop_triangles.foreach_get('vertices', triangles)
    polygon = np.empty(len(mesh.loop_triangles), dtype=np.int64)
    mesh.loop_triangles.foreach_get('polygon_index', polygon)
    edges = np.empty(len(mesh.edges) * 2, dtype=np.int64)
    mesh.edges.foreach_get('vertices', edges)
    result = brushes.Topology(len(mesh.vertices), triangles.reshape(-1, 3), edges.reshape(-1, 2))
    result.polygon = polygon
    _TOPOLOGY[obj.name] = (key, result)
    return result


def group_weights(obj, name, count):
    """Per-vertex weights of vertex group ``name`` (all 1 without a group, as Blender does)."""
    group = obj.vertex_groups.get(name) if name else None
    if group is None:
        return np.ones(count)
    weights = np.zeros(count)
    for vertex in obj.data.vertices:
        for assignment in vertex.groups:
            if assignment.group == group.index:
                weights[vertex.index] = assignment.weight
    return weights


def shape_mix(obj, edited):
    """Rest shape Blender shows (relative key mix, local) and ``edited``'s per-vertex share of it.

    Moving a vertex of ``edited`` by d moves the shown shape by share * d.
    """
    keys = obj.data.shape_keys
    count = len(obj.data.vertices)
    coordinates = {}

    def co(block):
        if block.name not in coordinates:
            values = np.empty(count * 3)
            block.data.foreach_get('co', values)
            coordinates[block.name] = values.reshape(-1, 3)
        return coordinates[block.name]

    if obj.show_only_shape_key:
        return co(edited).copy(), np.ones(count)
    mix = co(keys.reference_key).copy()
    share = np.zeros(count)
    for block in keys.key_blocks:
        if block == keys.reference_key or block.mute or not block.value:
            continue
        weight = block.value * group_weights(obj, block.vertex_group, count)
        mix += weight[:, None] * (co(block) - co(block.relative_key))
        if block == edited:
            share = weight
    return mix, share


class Edit:
    """The coordinates Sculpt Mode edits.

    Sculpting a shape key other than the Basis: brushes see the rest shape
    Blender shows (the mix of all keys, e.g. the Basis plus Cloth Fit
    Correction plus the sculpted key), and ``write`` stores the change in the
    sculpted key, divided by its share of the mix (as Adjust Region does).
    Otherwise the active key (the Basis) or the mesh itself.
    """

    def __init__(self, obj):
        mesh = obj.data
        keys = mesh.shape_keys
        if keys and not keys.use_relative:
            raise ValueError('Absolute shape keys are not supported')
        if getattr(obj, 'use_dynamic_topology_sculpting', False):
            raise ValueError('Turn off Dyntopo: Cloth Fit tools keep the clothing topology')
        if any(m.type == 'MULTIRES' for m in obj.modifiers):
            raise ValueError('Cloth Fit tools edit the base mesh; remove or apply the Multiresolution modifier first')
        if abs(obj.matrix_world.determinant()) < 1e-12:
            raise ValueError('The clothing needs a nonzero scale')
        self.obj, self.mesh = obj, mesh
        self.key = obj.active_shape_key if keys else None
        self.basis = self.key is not None and self.key == keys.reference_key
        own = np.empty(len(mesh.vertices) * 3)
        (self.key.data if self.key else mesh.vertices).foreach_get('co', own)
        self.own = own.reshape(-1, 3)
        self.local, self.share = self.own, np.ones(len(mesh.vertices))
        if self.key is not None and not self.basis:
            self.local, self.share = shape_mix(obj, self.key)
            if not self.movable.any():
                raise ValueError(f'Shape key "{self.key.name}" is muted or at value 0, so sculpting it shows no '
                                 'change; raise its value or click Sculpt Clothing')
        matrix = np.array(obj.matrix_world, dtype=np.float64)
        self.linear, self.offset = matrix[:3, :3], matrix[:3, 3]
        self.inverse = np.linalg.inv(self.linear)
        self.start = self.to_world(self.local)

    @property
    def movable(self):
        """Vertices the sculpted key can move (its share of the shown shape is not ~0)."""
        return np.abs(self.share) > 1e-3

    def to_world(self, local):
        return local @ self.linear.T + self.offset

    def to_local(self, world):
        return (world - self.offset) @ self.inverse.T

    def write(self, world):
        if world is self.start:
            own = self.own
        else:
            share = np.where(self.movable, self.share, 1.)
            own = self.own + (self.to_local(world) - self.local) / share[:, None]
        own = own.ravel()
        if self.key is not None:
            self.key.data.foreach_set('co', own)
        if self.key is None or self.basis:
            self.mesh.vertices.foreach_set('co', own)
        self.mesh.update()

    def mirror_point(self, point, signs):
        return self.to_world(self.to_local(np.asarray(point)[None])[0] * signs)

    def mirror_vector(self, vector, signs):
        return ((np.asarray(vector) @ self.inverse.T) * signs) @ self.linear.T


def reference(obj, edit):
    """World-space reference shape for Relax, or None.

    Sculpting a non-Basis key: the Basis. Otherwise a stored ``acf_reference``
    point attribute (Store Reference, or captured automatically before the
    first Cloth Fit stroke on a mesh without shape keys), else the Avatar
    Cloth Fit backup mesh when it has the same vertices.
    """
    keys = obj.data.shape_keys
    count = len(obj.data.vertices)
    if keys and edit.key is not None and not edit.basis:
        local = np.empty(count * 3)
        keys.reference_key.data.foreach_get('co', local)
        return edit.to_world(local.reshape(-1, 3))
    attribute = obj.data.attributes.get(REFERENCE)
    if attribute is not None and attribute.domain == 'POINT' and attribute.data_type == 'FLOAT_VECTOR' \
            and len(attribute.data) == count:
        local = np.empty(count * 3)
        attribute.data.foreach_get('vector', local)
        return edit.to_world(local.reshape(-1, 3))
    backup = obj.acf_backup
    if backup is not None and backup.type == 'MESH' and len(backup.data.vertices) == count \
            and len(backup.data.edges) == len(obj.data.edges):
        local = np.empty(count * 3)
        backup.data.vertices.foreach_get('co', local)
        return edit.to_world(local.reshape(-1, 3))
    return None


def store_reference(obj, local):
    attribute = obj.data.attributes.get(REFERENCE)
    if attribute is not None and (attribute.domain != 'POINT' or attribute.data_type != 'FLOAT_VECTOR'):
        obj.data.attributes.remove(attribute)
        attribute = None
    if attribute is None:
        attribute = obj.data.attributes.new(REFERENCE, 'FLOAT_VECTOR', 'POINT')
    attribute.data.foreach_set('vector', np.asarray(local, dtype=np.float64).ravel())


def freedom(obj, settings, topo, edit=None, protect=True):
    """Share each vertex may move (1 - sculpt mask, 0 for pins, vertices ``edit`` cannot move and, with
    ``protect``, accessories)."""
    mesh = obj.data
    count = len(mesh.vertices)
    result = np.ones(count)
    mask = mesh.attributes.get('.sculpt_mask')
    if mask is not None and mask.domain == 'POINT' and len(mask.data) == count:
        values = np.empty(count)
        mask.data.foreach_get('value', values)
        result = 1. - np.clip(values, 0., 1.)
    group = obj.vertex_groups.get(settings.assist_pin_group)
    if group is not None:
        index = group.index
        for vertex in mesh.vertices:
            for assignment in vertex.groups:
                if assignment.group == index and assignment.weight > 0.:
                    result[vertex.index] = 0.
    mesh.calc_loop_triangles()
    islands = engine.protected_islands(obj, topo.labels, topo.triangles, settings.attach_hardware)
    for members in islands if protect else ():
        result[members] = 0.
    if edit is not None:
        result[~edit.movable] = 0.
    return result, islands


# A deform bone counts as posed only beyond these (import rounding such as
# 1e-5 m offsets or 0.999999 scale is common on avatar rigs and harmless).
POSED_DEGREES, POSED_SHIFT, POSED_SCALE = .5, 1e-3, 5e-3


def pose_deviation(rig):
    """Largest visible deviation of a deform bone from its rest placement.

    Uses the evaluated pose (constraints and drivers included), so it reports
    what actually deforms the meshes. Returns (degrees, shift as a share of
    the rig size, scale change, bone name).
    """
    heads = np.array([bone.head_local[:] for bone in rig.data.bones]) if len(rig.data.bones) else np.zeros((1, 3))
    size = max(float(np.ptp(heads, axis=0).max()), 1e-6)
    worst = (0., 0., 0., '')
    for bone in rig.pose.bones:
        if not bone.bone.use_deform:
            continue
        rest, now = bone.bone.matrix_local, bone.matrix
        _, rotation, scale = (rest.inverted_safe() @ now).decompose()
        row = (math.degrees(rotation.angle) % 360., (now.translation - rest.translation).length / size,
               max(abs(value - 1.) for value in scale), bone.name)
        row = (min(row[0], 360. - row[0]),) + row[1:]
        if (row[0] / POSED_DEGREES, row[1] / POSED_SHIFT, row[2] / POSED_SCALE) > \
                (worst[0] / POSED_DEGREES, worst[1] / POSED_SHIFT, worst[2] / POSED_SCALE):
            worst = row
    return worst


def check_rest(obj, settings):
    """Fitting happens in the rest pose; refuse to edit a visibly posed avatar."""
    for rig in engine._rest_rigs(obj, settings):
        if rig.data.pose_position != 'POSE':
            continue
        degrees, shift, scale, name = pose_deviation(rig)
        if degrees > POSED_DEGREES or shift > POSED_SHIFT or scale > POSED_SCALE:
            raise ValueError(f'"{rig.name}" is posed: bone "{name}" is {degrees:.1f}° / {shift:.1%} of the rig size / '
                             f'{scale:.1%} scale away from rest. In Pose Mode select all bones (A, also in hidden '
                             'collections) and clear with Alt+G, Alt+R, Alt+S, or set Pose Position to Rest Position')


def body_surface(context, settings, garment):
    body = settings.body
    if body is None or body.type != 'MESH':
        raise ValueError('Choose the avatar Body in the Cloth Fit panel first')
    if body == garment:
        raise ValueError('The Body cannot be the clothing being sculpted')
    evaluated = body.evaluated_get(context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        xyz = np.empty(len(mesh.vertices) * 3)
        mesh.vertices.foreach_get('co', xyz)
        mesh.calc_loop_triangles()
        triangles = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
        mesh.loop_triangles.foreach_get('vertices', triangles)
    finally:
        evaluated.to_mesh_clear()
    matrix = np.array(body.matrix_world, dtype=np.float64)
    xyz = xyz.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]
    triangles = triangles.reshape(-1, 3)
    if np.linalg.det(matrix[:3, :3]) < 0:
        triangles = triangles[:, [0, 2, 1]]
    key = hash(xyz.tobytes())
    cached = _BODY.get(body.name)
    if cached and cached[0] == key:
        return cached[1]
    surface = brushes.Body(xyz, triangles)
    _BODY[body.name] = (key, surface)
    return surface


def symmetry(mesh):
    """Sign vectors of Sculpt Mode's mirror passes (first = no mirror)."""
    axes = [k for k, name in enumerate(('use_mirror_x', 'use_mirror_y', 'use_mirror_z')) if getattr(mesh, name)]
    result = []
    for combo in itertools.product((1., -1.), repeat=len(axes)):
        signs = np.ones(3)
        for axis, sign in zip(axes, combo):
            signs[axis] = sign
        result.append(signs)
    return result


def world_radius(region, view, mouse, depth, pixels):
    a = region_2d_to_location_3d(region, view, mouse, Vector(depth))
    b = region_2d_to_location_3d(region, view, (mouse[0] + pixels, mouse[1]), Vector(depth))
    return max((b - a).length, 1e-9)


def view_ray(region, view, mouse):
    return np.array(region_2d_to_origin_3d(region, view, mouse)), np.array(region_2d_to_vector_3d(region, view, mouse))


def cast(surface, region, view, mouse):
    return surface.ray(*view_ray(region, view, mouse))


def mirrored_hits(edit, surface, labels, point, triangle, radius, signs_list, ray, connected):
    """Hit per symmetry pass as (point, triangle or None, signs).

    Candidates for a mirrored pass are the mirrored view ray's hit and the
    nearest surface, both within the radius of the mirrored point; one on the
    same connected part as the original wins (the pants, not a strap or lining
    lying nearest), else the ray hit. Without matching surface a
    Topology-masked pass is dropped; otherwise the brush still works around
    the mirrored point (no triangle), as Blender's symmetry does.
    """
    hits = [(point, triangle, signs_list[0])]
    part = labels[surface.triangles[triangle, 0]]
    origin, direction = ray
    for signs in signs_list[1:]:
        spot = edit.mirror_point(point, signs)
        found = [surface.ray(edit.mirror_point(origin, signs), edit.mirror_vector(direction, signs)),
                 surface.nearest(spot, radius)]
        found = [f for f in found if f is not None and np.linalg.norm(f[0] - spot) <= radius]
        found.sort(key=lambda f: labels[surface.triangles[f[2], 0]] != part)
        if found:
            hits.append((found[0][0], found[0][2], signs))
        elif not connected:
            hits.append((spot, None, signs))
    return hits


# ------------------------------------------------------------------ brush stroke
BRUSHES = (('GRAB', 'Smart Grab', 'Pull fabric like an elastic grab; folds and spacing keep their shape'),
           ('FIT', 'Fit to Body', 'Move fabric toward a set clearance from the avatar body'),
           ('RELAX', 'Relax', 'Even out lumps and spacing while keeping folds, silhouette and the gap to the body'))
# Dabs are placed every SPACING brush radii along the mouse path; one mouse
# event places at most MAX_DABS (farther apart, with matching strength), so a
# fast stroke never queues more work than the viewport can show.
SPACING, MAX_DABS = .15, 3


class ACF_OT_brush_stroke(bpy.types.Operator):
    bl_idname = 'acf.brush_stroke'
    bl_label = 'Cloth Fit Brush'
    bl_description = 'Paint with the active Cloth Fit brush (one undo step per stroke)'
    bl_options = {'REGISTER', 'UNDO'}
    brush: EnumProperty(items=BRUSHES, default='GRAB')

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (context.mode == 'SCULPT' and obj is not None and obj.type == 'MESH'
                and not context.scene.acf_settings.running)

    def invoke(self, context, event):
        settings = context.scene.acf_settings
        obj = context.active_object
        self.region, self.view = context.region, context.region_data
        if self.view is None:
            return {'CANCELLED'}
        self.mouse = (event.mouse_region_x, event.mouse_region_y)
        try:
            check_rest(obj, settings)
            self.edit = Edit(obj)
            self.topo = topology(obj)
            # Blender's Topology automasking: reach only the connected part under the cursor;
            # accessories then follow their fabric. Without it the brush moves them directly (kept rigid).
            self.connected = context.scene.tool_settings.sculpt.use_automasking_topology
            self.freedom, islands = freedom(obj, settings, self.topo, self.edit, protect=self.connected)
            self.surface = brushes.Surface(self.topo, self.edit.start)
            self.body = body_surface(context, settings, obj) if self.brush == 'FIT' or (
                self.brush == 'RELAX' and settings.body is not None and settings.body.type == 'MESH'
                and settings.body != obj) else None
            self.accessories = Accessories(self.edit.start, self.topo.triangles, islands) if islands else None
            self.signs = symmetry(obj.data)
        except ValueError as exc:
            self.report({'ERROR'}, str(exc))
            settings.status = str(exc)
            return {'CANCELLED'}
        self.settings = settings
        self.positions = self.edit.start.copy()
        self.reliable = self.ambiguous = 0
        if self.brush != 'RELAX' and self.edit.key is None and obj.data.attributes.get(REFERENCE) is None:
            # Remember the shape before the first Cloth Fit edit for Relax.
            store_reference(obj, self.edit.local)
        if self.brush == 'GRAB':
            ray = view_ray(self.region, self.view, self.mouse)
            hit = self.surface.ray(*ray)
            if hit is None:
                return {'PASS_THROUGH'}
            point, _, triangle = hit
            self.depth = point
            radius = world_radius(self.region, self.view, self.mouse, point, settings.brush_radius)
            hits = mirrored_hits(self.edit, self.surface, self.topo.labels, point, triangle, radius, self.signs, ray, self.connected)
            self.grab_signs = [signs for _, _, signs in hits]
            self.grab = brushes.Grab(self.topo, self.edit.start, [(p, t) for p, t, _ in hits], radius,
                                     self.freedom, settings.grab_stiffness, self.connected)
            self.start_point = np.array(region_2d_to_location_3d(self.region, self.view, self.mouse, Vector(point)))
        else:
            self.lengths = self.topo.lengths(self.edit.start)
            self.reference = reference(obj, self.edit) if self.brush == 'RELAX' else None
            self.last = None
            self._dab(self.mouse, settings.brush_strength)
            self._write()
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _write(self):
        if self.accessories is not None:
            if self.connected:
                self.accessories.follow(self.positions)
            else:
                self.accessories.rigidify(self.positions)
        self.edit.write(self.positions)

    def _stroke_to(self, mouse):
        """Dabs along the mouse path since the last dab; one write for all of them."""
        spacing = max(2., SPACING * self.settings.brush_radius)
        if self.last is None:
            if self._dab(mouse, self.settings.brush_strength):
                self._write()
            return
        length = math.dist(mouse, self.last)
        needed = int(length // spacing)
        if not needed:
            return
        count = min(needed, MAX_DABS)
        # Fewer, farther dabs apply the strength the skipped ones would have.
        strength = 1. - (1. - self.settings.brush_strength) ** (needed / count)
        start = self.last
        changed = False
        for k in range(1, count + 1):
            share = k * needed * spacing / (count * length)
            spot = (start[0] + (mouse[0] - start[0]) * share, start[1] + (mouse[1] - start[1]) * share)
            changed |= self._dab(spot, strength)
            self.last = spot
        if changed:
            self._write()

    def _dab(self, mouse, strength):
        """One dab at ``mouse`` (region pixels); True when it moved something."""
        settings = self.settings
        ray = view_ray(self.region, self.view, mouse)
        hit = self.surface.ray(*ray)
        if hit is None:
            return False
        self.last = mouse
        point, _, triangle = hit
        radius = world_radius(self.region, self.view, mouse, point, settings.brush_radius)
        hits = mirrored_hits(self.edit, self.surface, self.topo.labels, point, triangle, radius, self.signs, ray, self.connected)
        weight = np.zeros(len(self.positions))
        # The BVH holds the stroke-start surface; follow each hit to where that
        # spot of fabric is now (a pass without surface mirrors the tracked hit).
        primary = brushes.track(point, triangle, self.edit.start, self.positions, self.topo.triangles)
        tracked = [(brushes.track(p, t, self.edit.start, self.positions, self.topo.triangles) if t is not None
                    else self.edit.mirror_point(primary, s), t) for p, t, s in hits]
        for w, _ in brushes.brush_weights(self.topo, self.positions, self.lengths, tracked, radius,
                                          settings.brush_falloff, self.freedom, self.connected):
            weight = np.maximum(weight, w)
        members = np.flatnonzero(weight > 0.)
        if not len(members):
            return False
        if self.brush == 'FIT':
            normals = brushes.vertex_normals(self.positions, brushes.triangles_touching(self.topo, weight > 0.))
            move, reliable = brushes.fit_to_body(self.topo, self.positions, members, normals, self.body,
                                                  settings.fit_clearance, settings.fit_mode, settings.fit_reach)
            self.positions[members] += move * (strength * weight[members])[:, None]
            self.reliable += int(reliable.sum())
            self.ambiguous += int((~reliable).sum())
        else:
            target = brushes.relax(self.topo, self.positions, weight, self.freedom, self.reference)
            before = self.positions[members]
            self.positions[members] += (target[members] - before) * (strength * weight[members])[:, None]
            if self.body is not None:
                brushes.keep_clearance(self.body, before, self.positions, members, settings.fit_clearance,
                                       settings.fit_reach)
        return True

    def modal(self, context, event):
        try:
            if event.type == 'INBETWEEN_MOUSEMOVE':
                # Blender turns every queued mouse move but the newest into
                # INBETWEEN_MOUSEMOVE; handling only the newest keeps the brush
                # with the cursor instead of working through a backlog.
                return {'RUNNING_MODAL'}
            if event.type == 'MOUSEMOVE':
                mouse = (event.mouse_region_x, event.mouse_region_y)
                if self.brush == 'GRAB':
                    now = np.array(region_2d_to_location_3d(self.region, self.view, mouse, Vector(self.depth)))
                    move = now - self.start_point
                    self.positions = self.grab.update([self.edit.mirror_vector(move, s) for s in self.grab_signs])
                    self._write()
                else:
                    self._stroke_to(mouse)
                return {'RUNNING_MODAL'}
            if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
                if self.brush == 'GRAB':
                    self.positions = self.grab.finish()
                    self._write()
                self._report()
                return {'FINISHED'}
            if event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
                self.edit.write(self.edit.start)
                return {'CANCELLED'}
        except Exception as exc:
            traceback.print_exc()
            self.edit.write(self.edit.start)
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def _report(self):
        if self.brush == 'FIT' and self.ambiguous:
            share = self.ambiguous / max(self.reliable + self.ambiguous, 1)
            self.settings.status = (f'Fit to Body: {share:.0%} of brushed vertices had no clear body surface '
                                    'under them and followed their neighbors instead')


# ------------------------------------------------------------------ Region Resize
def face_set_region(obj, topo, triangle, freedom):
    """Vertices of the face set under ``triangle`` (connected part), else its unmasked island."""
    mesh = obj.data
    attribute = mesh.attributes.get('.sculpt_face_set')
    region = np.zeros(topo.count, dtype=bool)
    if attribute is not None and attribute.domain == 'FACE' and len(attribute.data) == len(mesh.polygons):
        sets = np.empty(len(mesh.polygons), dtype=np.int64)
        attribute.data.foreach_get('value', sets)
        triangle_set = sets[topo.polygon]
        same = triangle_set == triangle_set[triangle]
        if not same.all():
            from .topology import connected_labels
            chosen = topo.triangles[same]
            region[chosen.ravel()] = True
            sides = chosen[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
            labels = connected_labels(sides, topo.count)
            region &= labels == labels[topo.triangles[triangle][0]]
    if not region.any():
        region = topo.labels == topo.labels[topo.triangles[triangle][0]]
    return region & (freedom > .5)


class ACF_OT_region_resize(bpy.types.Operator):
    bl_idname = 'acf.region_resize'
    bl_label = 'Region Resize'
    bl_description = ('Click a face set (or a separate piece) and change its width along its own centerline; '
                      'cross-section handles reshape it along a curved path')
    bl_options = {'REGISTER', 'UNDO'}
    _handle = None

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (context.mode == 'SCULPT' and obj is not None and obj.type == 'MESH'
                and not context.scene.acf_settings.running)

    def invoke(self, context, event):
        settings = context.scene.acf_settings
        obj = context.active_object
        self.region, self.view, self.area = context.region, context.region_data, context.area
        if self.view is None:
            return {'CANCELLED'}
        mouse = (event.mouse_region_x, event.mouse_region_y)
        try:
            check_rest(obj, settings)
            self.edit = Edit(obj)
            self.topo = topology(obj)
            self.freedom, islands = freedom(obj, settings, self.topo, self.edit)
            surface = brushes.Surface(self.topo, self.edit.start)
            hit = cast(surface, self.region, self.view, mouse)
            if hit is None:
                return {'PASS_THROUGH'}
            self.area_mask = face_set_region(obj, self.topo, hit[2], self.freedom)
            self.accessories = Accessories(self.edit.start, self.topo.triangles, islands) if islands else None
            self.settings = settings
            self._build()
        except ValueError as exc:
            self.report({'ERROR'}, str(exc))
            settings.status = str(exc)
            return {'CANCELLED'}
        self.active = None
        self.hover = None
        self.mode = 'IDLE'
        self.mouse = Vector(mouse)
        self._handle = bpy.types.SpaceView3D.draw_handler_add(self._draw, (), 'WINDOW', 'POST_PIXEL')
        context.window_manager.modal_handler_add(self)
        self._preview(context)
        return {'RUNNING_MODAL'}

    def _build(self, handles=None):
        s = self.settings
        previous = handles
        self.resize = brushes.Resize(self.topo, self.edit.start, self.area_mask, self.freedom, s.resize_transition,
                                     s.brush_falloff, s.resize_pin_edges, s.resize_keep_detail)
        if previous is not None:
            self.resize.handles = previous

    def _preview(self, context):
        self.positions = self.resize.deform()
        if self.accessories is not None:
            self.accessories.follow(self.positions)
        self.edit.write(self.positions)
        unit = context.scene.unit_settings
        width = bpy.utils.units.to_string(unit.system, 'LENGTH', self.settings.resize_transition * unit.scale_length,
                                          precision=3)
        target = 'section' if self.active is not None else 'all sections'
        scale = (self.resize.handles[self.active].scale if self.active is not None
                 else float(np.mean([h.scale for h in self.resize.handles])))
        self.area.header_text_set(
            f'Region Resize: width x{scale:.3f} ({target}), {len(self.resize.handles)} handles, fade {width}'
            f'{", edges pinned" if self.settings.resize_pin_edges else ""}'
            f'{", fold depth kept" if self.settings.resize_keep_detail else ""}')
        context.workspace.status_text_set(
            'Drag empty space: width  |  Click handle: select, drag: move  |  S scale  G move  A add  X delete  '
            '|  Wheel fade  E pin edges  D fold depth  |  Enter confirm  Esc cancel')
        self.area.tag_redraw()

    def _handle_points(self):
        handles = self.resize.handles
        centers, _ = self.resize.curve(np.array([h.t for h in handles]))
        return [Vector(c + h.offset) for c, h in zip(centers, handles)]

    def _screen(self, point):
        return location_3d_to_region_2d(self.region, self.view, point)

    def _pick(self, mouse):
        best, index = 14., None
        for k, point in enumerate(self._handle_points()):
            screen = self._screen(point)
            if screen is not None and (screen - mouse).length < best:
                best, index = (screen - mouse).length, k
        return index

    def _draw(self):
        if bpy.context.region != self.region:
            return
        import gpu
        from gpu_extras.batch import batch_for_shader
        resize = self.resize
        ts = np.linspace(0., 1., 48)
        centers, tangents = resize.curve(ts)
        _, offsets = resize._fields(ts)
        line = [self._screen(Vector(c)) for c in centers + offsets]
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.blend_set('ALPHA')
        gpu.state.line_width_set(2.)
        shader.uniform_float('color', (1., .62, .15, .9))
        points = [tuple(p) for p in line if p is not None]
        if len(points) > 1:
            batch_for_shader(shader, 'LINE_STRIP', {'pos': points}).draw(shader)
        gpu.state.line_width_set(1.)
        handles = resize.handles
        handle_t = np.array([h.t for h in handles])
        radius = resize.section_radius(handle_t)
        centers_h, tangents_h = resize.curve(handle_t)
        for k, h in enumerate(handles):
            tangent = tangents_h[k]
            side = np.cross(tangent, [0., 0., 1.] if abs(tangent[2]) < .9 else [1., 0., 0.])
            side /= np.linalg.norm(side)
            other = np.cross(tangent, side)
            ring = []
            for a in np.linspace(0., math.tau, 33):
                p = centers_h[k] + h.offset + radius[k] * h.scale * (math.cos(a) * side + math.sin(a) * other)
                screen = self._screen(Vector(p))
                if screen is not None:
                    ring.append(tuple(screen))
            selected = k == self.active
            shader.uniform_float('color', (1., .85, .3, .95) if selected else (1., 1., 1., .5))
            if len(ring) > 1:
                batch_for_shader(shader, 'LINE_STRIP', {'pos': ring}).draw(shader)
            screen = self._screen(Vector(centers_h[k] + h.offset))
            if screen is not None:
                size = 7. if selected or k == self.hover else 5.
                square = [(screen.x - size, screen.y - size), (screen.x + size, screen.y - size),
                          (screen.x + size, screen.y + size), (screen.x - size, screen.y + size)]
                batch_for_shader(shader, 'TRIS', {'pos': square}, indices=((0, 1, 2), (0, 2, 3))).draw(shader)
        gpu.state.blend_set('NONE')

    def _end(self, context):
        if self._handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            self._handle = None
        self.area.header_text_set(None)
        context.workspace.status_text_set(None)
        self.area.tag_redraw()

    def modal(self, context, event):
        try:
            return self._modal(context, event)
        except Exception as exc:
            traceback.print_exc()
            self._end(context)
            self.edit.write(self.edit.start)
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

    def _begin(self, mode):
        self.mode = mode
        self.start_mouse = self.mouse.copy()
        self.saved = [brushes.Handle(h.t, h.offset.copy(), h.scale) for h in self.resize.handles]

    def _modal(self, context, event):
        s = self.settings
        if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
            self.mouse = Vector((event.mouse_region_x, event.mouse_region_y))
            if self.mode == 'IDLE':
                hover = self._pick(self.mouse)
                if hover != self.hover:
                    self.hover = hover
                    self.area.tag_redraw()
                return {'RUNNING_MODAL'}
            self._apply_mouse()
            self._preview(context)
            return {'RUNNING_MODAL'}
        if event.type == 'LEFTMOUSE':
            if event.value == 'PRESS' and self.mode == 'IDLE':
                picked = self._pick(self.mouse)
                self.active = picked
                self._begin('MOVE' if picked is not None else 'WIDTH')
                self.dragging = True
                self._preview(context)
            elif event.value == 'RELEASE' and self.mode != 'IDLE' and getattr(self, 'dragging', False):
                self.mode, self.dragging = 'IDLE', False
            elif event.value == 'PRESS' and self.mode != 'IDLE':
                self.mode, self.dragging = 'IDLE', False
            return {'RUNNING_MODAL'}
        if event.value != 'PRESS':
            return {'RUNNING_MODAL'}
        if event.type in {'RET', 'NUMPAD_ENTER', 'SPACE'}:
            return self._confirm(context)
        if event.type in {'ESC', 'RIGHTMOUSE'}:
            if self.mode != 'IDLE':
                self.resize.handles, self.mode, self.dragging = self.saved, 'IDLE', False
                self._preview(context)
                return {'RUNNING_MODAL'}
            self._end(context)
            self.edit.write(self.edit.start)
            s.status = 'Region Resize cancelled; clothing unchanged'
            return {'CANCELLED'}
        if event.type == 'S':
            self._begin('SCALE')
            self.dragging = False
        elif event.type == 'G' and self.active is not None:
            self._begin('MOVE')
            self.dragging = False
        elif event.type == 'A':
            self._add()
        elif event.type in {'X', 'DEL'} and self.active is not None and len(self.resize.handles) > 2:
            del self.resize.handles[self.active]
            self.active = None
        elif event.type == 'TAB':
            count = len(self.resize.handles)
            self.active = 0 if self.active is None else (self.active + 1) % count
        elif event.type in {'WHEELUPMOUSE', 'PAGE_UP', 'NUMPAD_PLUS', 'WHEELDOWNMOUSE', 'PAGE_DOWN', 'NUMPAD_MINUS'}:
            grow = event.type in {'WHEELUPMOUSE', 'PAGE_UP', 'NUMPAD_PLUS'}
            s.resize_transition = s.resize_transition * 1.15 if grow else s.resize_transition / 1.15
            self._build(self.resize.handles)
        elif event.type == 'E':
            s.resize_pin_edges = not s.resize_pin_edges
            self._build(self.resize.handles)
        elif event.type == 'D':
            s.resize_keep_detail = not s.resize_keep_detail
            self._build(self.resize.handles)
        else:
            return {'RUNNING_MODAL'}
        self._preview(context)
        return {'RUNNING_MODAL'}

    def _targets(self):
        return [self.active] if self.active is not None else list(range(len(self.resize.handles)))

    def _apply_mouse(self):
        delta = self.mouse - self.start_mouse
        if self.mode == 'WIDTH':
            factor = math.exp(delta.x / 250.)
            for k in self._targets():
                self.resize.handles[k].scale = min(max(self.saved[k].scale * factor, .2), 5.)
        elif self.mode == 'SCALE':
            points = self._handle_points()
            anchor = self._screen(points[self.active]) if self.active is not None else None
            if anchor is None:
                anchor = self._screen(Vector(np.mean([np.array(p) for p in points], axis=0))) or self.start_mouse
            factor = (self.mouse - anchor).length / max((self.start_mouse - anchor).length, 1.)
            for k in self._targets():
                self.resize.handles[k].scale = min(max(self.saved[k].scale * factor, .2), 5.)
        elif self.mode == 'MOVE' and self.active is not None:
            base = self.saved[self.active]
            center, _ = self.resize.curve(np.array([base.t]))
            depth = Vector(center[0] + base.offset)
            a = region_2d_to_location_3d(self.region, self.view, self.start_mouse, depth)
            b = region_2d_to_location_3d(self.region, self.view, self.mouse, depth)
            self.resize.handles[self.active].offset = base.offset + np.array(b - a)

    def _add(self):
        ts = np.linspace(0., 1., 101)
        centers, _ = self.resize.curve(ts)
        scale, offset = self.resize._fields(ts)
        best, chosen = None, None
        for t, c, o in zip(ts, centers, offset):
            screen = self._screen(Vector(c + o))
            if screen is not None:
                distance = (screen - self.mouse).length
                if best is None or distance < best:
                    best, chosen = distance, t
        if chosen is None or any(abs(h.t - chosen) < .02 for h in self.resize.handles):
            return
        k = int(round(chosen * 100))
        self.resize.handles.append(brushes.Handle(chosen, offset[k], float(scale[k])))
        self.resize.handles.sort(key=lambda h: h.t)
        self.active = next(i for i, h in enumerate(self.resize.handles) if h.t == chosen)

    def _confirm(self, context):
        self._end(context)
        positions = self.resize.deform()
        positions, stats = self.resize.relax(positions, self.settings.assist_relax)
        if self.accessories is not None:
            self.accessories.follow(positions)
        self.edit.write(positions)
        moved = int(np.count_nonzero(np.linalg.norm(positions - self.edit.start, axis=1) > 1e-9))
        self.settings.status = (f'Resized {moved} vertices' + ('; fade zone smoothed' if stats.get('relaxed') else '')
                                + '. Rerun Fit & Refine Weights to re-check clipping')
        self.report({'INFO'}, self.settings.status)
        return {'FINISHED'}

    def cancel(self, context):
        self._end(context)
        self.edit.write(self.edit.start)


# ------------------------------------------------------------------ toolbar tools
def brush_cursor(context, _tool, xy):
    import gpu
    from gpu_extras.batch import batch_for_shader
    radius = context.scene.acf_settings.brush_radius
    ring = [(xy[0] + radius * math.cos(a), xy[1] + radius * math.sin(a)) for a in np.linspace(0., math.tau, 49)]
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    shader.uniform_float('color', (1., 1., 1., .6))
    batch_for_shader(shader, 'LINE_STRIP', {'pos': ring}).draw(shader)
    gpu.state.blend_set('NONE')


def draw_brush_settings(context, layout, brush):
    s = context.scene.acf_settings
    layout.prop(s, 'brush_radius')
    layout.prop(s, 'brush_strength')
    if brush == 'GRAB':
        layout.prop(s, 'grab_stiffness')
    else:
        layout.prop(s, 'brush_falloff', text='')
    if brush == 'FIT':
        layout.prop(s, 'fit_mode', text='')
        layout.prop(s, 'fit_clearance')
    elif brush == 'RELAX' and s.body is not None:
        layout.prop(s, 'fit_clearance')
    layout.prop(context.scene.tool_settings.sculpt, 'use_automasking_topology', text='Topology', toggle=True)
    obj = context.active_object
    if obj is not None and obj.type == 'MESH':
        row = layout.row(align=True)
        row.label(text='', icon='MOD_MIRROR')
        for axis in 'xyz':
            row.prop(obj.data, f'use_mirror_{axis}', text=axis.upper(), toggle=True)


def _keymap(brush):
    return (
        ('acf.brush_stroke', {'type': 'LEFTMOUSE', 'value': 'PRESS'}, {'properties': [('brush', brush)]}),
        ('acf.brush_stroke', {'type': 'LEFTMOUSE', 'value': 'PRESS', 'shift': True}, {'properties': [('brush', 'RELAX')]}),
        ('wm.radial_control', {'type': 'F', 'value': 'PRESS'},
         {'properties': [('data_path_primary', 'scene.acf_settings.brush_radius')]}),
        ('wm.radial_control', {'type': 'F', 'value': 'PRESS', 'shift': True},
         {'properties': [('data_path_primary', 'scene.acf_settings.brush_strength')]}),
    )


class ACF_TL_smart_grab(bpy.types.WorkSpaceTool):
    bl_space_type, bl_context_mode = 'VIEW_3D', 'SCULPT'
    bl_idname = 'acf.smart_grab'
    bl_label = 'Smart Grab'
    bl_description = ('Cloth Fit: pull fabric like an elastic grab. The grabbed spot follows the mouse and the pull '
                      'fades softly along the garment (past the brush circle); shape preservation keeps folds and '
                      'spacing. Shift: Relax')
    bl_icon = 'ops.transform.translate'
    bl_keymap = _keymap('GRAB')

    def draw_settings(context, layout, tool):
        draw_brush_settings(context, layout, 'GRAB')

    def draw_cursor(context, tool, xy):
        brush_cursor(context, tool, xy)


class ACF_TL_fit_body(bpy.types.WorkSpaceTool):
    bl_space_type, bl_context_mode = 'VIEW_3D', 'SCULPT'
    bl_idname = 'acf.fit_body'
    bl_label = 'Fit to Body'
    bl_description = ('Cloth Fit: paint fabric toward a set clearance from the avatar body (inward, outward, both, '
                      'or only out of clipping). Ambiguous spots follow their neighbors. Shift: Relax')
    bl_icon = 'ops.sculpt.cloth_filter'
    bl_keymap = _keymap('FIT')

    def draw_settings(context, layout, tool):
        draw_brush_settings(context, layout, 'FIT')

    def draw_cursor(context, tool, xy):
        brush_cursor(context, tool, xy)


class ACF_TL_relax(bpy.types.WorkSpaceTool):
    bl_space_type, bl_context_mode = 'VIEW_3D', 'SCULPT'
    bl_idname = 'acf.relax'
    bl_label = 'Shape Relax'
    bl_description = ('Cloth Fit: smooth lumps and uneven spacing while keeping the reference folds, deliberate '
                      'resizing and offsets, and never pulling fabric closer to the Body than the clearance')
    bl_icon = 'ops.sculpt.mesh_filter'
    bl_keymap = _keymap('RELAX')

    def draw_settings(context, layout, tool):
        draw_brush_settings(context, layout, 'RELAX')

    def draw_cursor(context, tool, xy):
        brush_cursor(context, tool, xy)


class ACF_TL_region_resize(bpy.types.WorkSpaceTool):
    bl_space_type, bl_context_mode = 'VIEW_3D', 'SCULPT'
    bl_idname = 'acf.region_resize_tool'
    bl_label = 'Region Resize'
    bl_description = ('Cloth Fit: click a face set (or a separate piece) to change its width along its centerline, '
                      'with cross-section handles for curved legs and a smooth fade into the rest')
    bl_icon = 'ops.transform.resize'
    bl_keymap = (('acf.region_resize', {'type': 'LEFTMOUSE', 'value': 'PRESS'}, None),)

    def draw_settings(context, layout, tool):
        s = context.scene.acf_settings
        layout.prop(s, 'resize_transition')
        layout.prop(s, 'brush_falloff', text='')
        layout.prop(s, 'resize_pin_edges')
        layout.prop(s, 'resize_keep_detail')


TOOLS = (ACF_TL_smart_grab, ACF_TL_fit_body, ACF_TL_relax, ACF_TL_region_resize)
CLASSES = (ACF_OT_brush_stroke, ACF_OT_region_resize)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    previous = None
    for tool in TOOLS:
        bpy.utils.register_tool(tool, after={previous} if previous else None, separator=previous is None)
        previous = tool.bl_idname


def unregister():
    for tool in reversed(TOOLS):
        bpy.utils.unregister_tool(tool)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    _TOPOLOGY.clear()
    _BODY.clear()
