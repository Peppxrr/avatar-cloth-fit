"""Avatar Cloth Fit — Blender 5.0 extension.

SPDX-License-Identifier: GPL-3.0-or-later
"""
import json
import math
import textwrap
import traceback
import time
import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty, FloatVectorProperty,
                       IntProperty, PointerProperty, StringProperty)
from mathutils import Euler, Matrix, Vector
from . import engine, sculpt

PROFILE_ITEMS = (
    ('SMOOTH', 'Smooth', 'Soft start and soft end; best default for fabric', 'SMOOTHCURVE', 0),
    ('SPHERE', 'Sphere', 'Stays strong for most of the radius, then drops quickly', 'SPHERECURVE', 1),
    ('LINEAR', 'Linear', 'Even fade over the radius', 'LINCURVE', 2),
    ('SHARP', 'Sharp', 'Concentrated near the selection, long faint tail', 'SHARPCURVE', 3),
    ('CONSTANT', 'Constant', 'Everything within the radius moves fully (hard edge at the radius)', 'NOCURVE', 4),
)
PIVOT_ITEMS = (
    ('SELECTION', 'Selection center', 'Rotate and scale about the center of the selected vertices'),
    ('CURSOR', '3D cursor', 'Rotate and scale about the 3D cursor'),
)
ORIENTATION_ITEMS = (
    ('GLOBAL', 'Global', 'X/Y/Z are the world axes'),
    ('LOCAL', 'Local', "X/Y/Z are the clothing object's axes"),
    ('SELECTION', 'Selection shape', 'X = longest extent of the selection, Z = thinnest. For a ring around a limb, '
                                     'Z runs along the limb, so Shift+Z scales the ring without lengthening it'),
)


def mesh_poll(_self, obj):
    return obj.type == 'MESH' and ' [Cloth Fit Backup]' not in obj.name


def wrapped(layout, text, context, icon='NONE'):
    scale = context.preferences.system.ui_scale if context.preferences else 1.
    width = max(20, int((context.region.width - 56 * scale) / (7.2 * scale))) if context.region else 40
    for number, line in enumerate(textwrap.wrap(text, width=width)):
        layout.label(text=line, icon=icon if number == 0 else 'BLANK1' if icon != 'NONE' else 'NONE')


def info_panel(layout, context, idname, title, lines=(), icon='INFO'):
    """Collapsed-by-default section: ``title`` in the header, wrapped ``lines`` when opened.

    Returns the body layout when open (for extra rows), else None.
    """
    header, body = layout.panel(idname, default_closed=True)
    header.label(text=title, icon=icon)
    if body is not None:
        for line in lines:
            wrapped(body, line, context)
    return body


class ACF_ClothingItem(bpy.types.PropertyGroup):
    object: PointerProperty(type=bpy.types.Object, poll=mesh_poll)


class ACF_Settings(bpy.types.PropertyGroup):
    body: PointerProperty(name='Avatar body', type=bpy.types.Object, poll=mesh_poll,
                          description='Weighted body surface; exclude clothing, eyes and separate accessories')
    clothing: PointerProperty(name='Clothing', type=bpy.types.Object, poll=mesh_poll,
                              description='Clothing mesh to weight. Use "Use selected clothing" for an outfit split into several meshes')
    clothing_items: CollectionProperty(type=ACF_ClothingItem)
    armature: PointerProperty(name='Avatar armature', type=bpy.types.Object, poll=lambda self, obj: obj.type == 'ARMATURE',
                              description='The body rig, not the separate clothing rig; optional when the body has an Armature modifier')
    # ------------------------------------------------------------ weighting
    correct_shape: BoolProperty(name='Correct rest clipping', default=False,
                                description='Push slightly clipping clothing outward in the rest pose, stored in a separate '
                                            '"Cloth Fit Correction" shape key (Basis unchanged). Bounded by Max correction')
    secondary_bones: BoolProperty(name='Probe secondary bones', default=True,
                                  description='Swing named belly/stomach, breast, butt/hip, thigh, tail, ear, clothing and '
                                              'jiggle/physbone bones like PhysBones; does not simulate Unity PhysBones')
    reverse_forward: BoolProperty(name='Reverse pose facing', default=False,
                                  description='Reverse generic forward/back pose directions for rigs facing the opposite anatomical convention')
    preserve_clothing_bones: BoolProperty(name='Preserve clothing-only bones', default=True,
                                          description='Keep existing weights on deform bones absent from the body source, such as dedicated skirt or cape chains')
    self_collision: BoolProperty(name='Check self-intersections', default=True,
                                 description='Check new nonadjacent triangle crossings in posed clothing; existing rest crossings and coplanar overlaps need manual review')
    attach_hardware: BoolProperty(name='Detect metal hardware', default=True,
                                  description='Treat small detached islands with metal material names or a constant metallic shader as hardware: '
                                              'they get their fabric attachment blend during weighting and stay rigid in Assisted Fit. '
                                              'Explicitly marked hardware always counts')
    strap_smoothing: FloatProperty(name='Strap smoothing (%)', default=1.8, min=.05, max=5., precision=2,
                                   description='Blend transferred weights over nearby vertices of each marked belt/strap, as a percentage of avatar size, '
                                               'including its thickness, while retaining joint motion and protected weights')
    clearance: FloatProperty(name='Clearance (%)', default=0.08, min=0.0, max=2.0, precision=3,
                             description='Desired gap between body and clothing as a percentage of the largest body dimension; '
                                         'default ~1.4 mm on a 1.8 m avatar. Raise slightly for thick fabric')
    max_correction: FloatProperty(name='Max correction (%)', default=0.5, min=0.01, max=3.0, precision=2,
                                  description='Largest rest-pose push of any vertex by Correct rest clipping, as a percentage of avatar size')
    search_distance: FloatProperty(name='Search range (%)', default=8.0, min=0.5, max=30.0,
                                   description='Range, as a percentage of avatar size, within which body collisions are measured and corrected; '
                                               'vertices farther from the body are reported as distant')
    influence_limit: IntProperty(name='Bone influences', default=4, min=1, max=8,
                                 description='Maximum deform bones per vertex (Unity/VRChat use 4); protected weights may require an extra influence, which is reported')
    iterations: IntProperty(name='Refinement passes', default=3, min=1, max=15,
                            description='Maximum refinement passes; refinement stops early once weights stop changing. More passes = slower, sometimes cleaner')
    validation_mode: EnumProperty(name='Pose validation', default='AUTO', items=(
        ('AUTO', 'Automatic', 'Screen a large joint library by exact linear-blend prediction, skip poses proven irrelevant, '
                              'and check core and problem poses with Blender deformation'),
        ('THOROUGH', 'Thorough', 'Denser joint-angle sampling; every relevant library pose is checked and refined with Blender deformation (much slower)')))
    strength: FloatProperty(name='Collision strength', default=1.5, min=0.1, max=5.0,
                            description='How strongly refinement pushes clipping clothing out of the body, relative to keeping weights smooth')
    transition_width: FloatProperty(name='Transition smoothing (%)', default=1.0, min=0.0, max=5.0, precision=2,
                                    description='Smoothing radius for transferred weights along the garment surface, as a percentage of avatar size; '
                                                'loose, distant or ambiguous fabric is smoothed proportionally further. 0 keeps raw nearest-surface weights')
    bone_weight_threshold: FloatProperty(name='Min bone weight', default=.01, min=0., max=.1, precision=3,
                                         description='A bone stays on the clothing armature when it carries at least this normalized weight on any vertex; '
                                                     'lighter bones merge into their nearest kept parent. Ancestors of kept bones always stay')
    # ------------------------------------------------------------ assisted fit
    assist_falloff: FloatProperty(name='Falloff radius', default=.05, min=0., soft_max=.5, subtype='DISTANCE', unit='LENGTH',
                                  description='How far the adjustment fades out along the garment surface beyond the selection. '
                                              '0 moves only the selection. Mouse wheel changes it while adjusting')
    assist_profile: EnumProperty(name='Falloff', items=PROFILE_ITEMS, default='SMOOTH',
                                 description='Shape of the fade from the selection to the falloff radius')
    assist_pivot: EnumProperty(name='Pivot', items=PIVOT_ITEMS, default='SELECTION')
    assist_orientation: EnumProperty(name='Axes', items=ORIENTATION_ITEMS, default='GLOBAL',
                                     description='Axes used by X/Y/Z constraints and per-axis scale')
    assist_relax: IntProperty(name='Smoothing passes', default=8, min=0, max=40,
                              description='After you confirm, even out stretching in the fade zone (as-rigid-as-possible solve) while '
                                          'keeping folds and detail. Kept only if it does not flip faces or add strain. 0 = off')
    assist_pin_group: StringProperty(name='Pin group', default='ACF Pinned',
                                     description='Vertex group whose vertices never move during Assisted Fit; nearby fabric fades smoothly into them')
    # ------------------------------------------------------------ sculpt brushes
    brush_radius: IntProperty(name='Radius', default=80, min=5, max=1000, subtype='PIXEL',
                              description='Brush radius in screen pixels (F to change in the viewport)')
    brush_strength: FloatProperty(name='Strength', default=.5, min=0., max=1., subtype='FACTOR',
                                  description='How much of the change each dab applies (Shift+F in the viewport). '
                                              'Smart Grab always follows the mouse fully')
    brush_falloff: EnumProperty(name='Falloff', items=PROFILE_ITEMS, default='SMOOTH',
                                description='Fade from the brush center (or region border) outward. Smart Grab uses '
                                            'its own elastic fade')
    grab_stiffness: FloatProperty(name='Shape preservation', default=.6, min=0., max=1., subtype='FACTOR',
                                  description='0 is a plain elastic grab; higher values keep the fabric\'s local shape '
                                              '(folds, spacing, triangle sizes) as it follows the pull, which then '
                                              'spreads a little farther instead of stretching')
    fit_mode: EnumProperty(name='Direction', default='BOTH', items=(
        ('BOTH', 'Fit', 'Move fabric toward the clearance from either side'),
        ('INWARD', 'Tighten', 'Only pull loose fabric in toward the clearance'),
        ('OUTWARD', 'Loosen', 'Only push fabric that is closer than the clearance outward'),
        ('RESOLVE', 'Resolve clipping', 'Only push fabric that is inside the body out to the clearance')))
    fit_clearance: FloatProperty(name='Clearance', default=.003, min=0., soft_max=.05, subtype='DISTANCE', unit='LENGTH',
                                 description='Target gap between fabric and skin for Fit to Body. Shape Relax never '
                                             'pulls fabric closer to the Body than this (or than it already was)')
    fit_reach: FloatProperty(name='Reach', default=.08, min=.001, soft_max=.5, subtype='DISTANCE', unit='LENGTH',
                             description='Fabric farther than this from the body is left alone by Fit to Body '
                                         '(and by the Shape Relax clearance check)')
    resize_transition: FloatProperty(name='Fade', default=.04, min=0., soft_max=.3, subtype='DISTANCE', unit='LENGTH',
                                     description='How far beyond (or with Pin edges, inside) the region the resize '
                                                 'fades out, measured along the fabric (mouse wheel while resizing)')
    resize_pin_edges: BoolProperty(name='Pin region edges', default=False,
                                   description='Keep the region border in place; the resize fades in from it (E)')
    resize_keep_detail: BoolProperty(name='Keep fold depth', default=True,
                                     description='Scale only the smooth width; folds and seams keep their depth instead '
                                                 'of growing or shrinking with the region (D)')
    status: StringProperty(default='Choose a weighted body and clothing')
    running: BoolProperty(default=False, options={'SKIP_SAVE'})


class ACF_OT_pick(bpy.types.Operator):
    bl_idname = 'acf.pick_selected'
    bl_label = 'Use selected clothing'
    bl_description = 'Use the selected clothing meshes as one outfit, keeping each object separate; excludes the avatar body'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.acf_settings
        obj = context.active_object
        selected = sorted((o for o in context.selected_objects if o != settings.body and mesh_poll(None, o)),
                          key=lambda o: o.name)
        if obj == settings.body or (obj and not mesh_poll(None, obj)):
            obj = selected[0] if selected else None
        if not obj or not mesh_poll(None, obj):
            self.report({'ERROR'}, 'Select a clothing mesh first')
            return {'CANCELLED'}
        settings.clothing = obj
        settings.clothing_items.clear()
        for mesh in [obj] + [o for o in selected if o != obj]:
            if mesh != settings.body and mesh_poll(None, mesh):
                settings.clothing_items.add().object = mesh
        return {'FINISHED'}


class ACF_OT_fit(bpy.types.Operator):
    bl_idname = 'acf.fit'
    bl_label = 'Fit & Refine Weights'
    bl_description = 'Back up the clothing, transfer body weights, test poses and keep refinement only if actual Blender collision checks pass'
    bl_options = {'REGISTER', 'UNDO'}
    _timer = None
    _job = None
    _scene = None

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and not context.scene.acf_settings.running

    def _finish(self, context, cancelled=False):
        if self._job:
            self._job.close()
            self._job = None
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        self._scene.acf_settings.running = False
        if cancelled:
            self._scene.acf_settings.status = 'Cancelled; original clothing and pose restored'
        context.window_manager.progress_end()
        for area in context.screen.areas if context.screen else []:
            area.tag_redraw()

    def execute(self, context):
        # Blocking execution supports background use and operator redo/testing.
        settings = context.scene.acf_settings
        self._scene = context.scene
        settings.running = True
        try:
            for message in engine.run(context, settings):
                settings.status = message
            self.report({'INFO'}, settings.status)
            return {'FINISHED'}
        except Exception as exc:
            settings.status = str(exc)
            traceback.print_exc()
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        finally:
            settings.running = False

    def invoke(self, context, _event):
        self._scene = context.scene
        settings = self._scene.acf_settings
        try:
            engine.validate_set(settings)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self._job = engine.run(context, settings)
        settings.running = True
        settings.status = 'Starting; Esc to cancel'
        self._timer = context.window_manager.event_timer_add(0.02, window=context.window)
        context.window_manager.modal_handler_add(self)
        context.window_manager.progress_begin(0, 100)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'ESC':
            self._finish(context, cancelled=True)
            return {'CANCELLED'}
        if event.type != 'TIMER':
            # Avoid edits to weights/meshes while the transaction is open.
            return {'RUNNING_MODAL'}
        try:
            # Progress yields are checkpoints, not mandatory 20 ms sleeps.
            # Batch inexpensive steps, returning regularly for redraw/Esc.
            deadline = time.perf_counter() + .012
            while True:
                self._scene.acf_settings.status = next(self._job)
                if time.perf_counter() >= deadline:
                    break
            for area in context.screen.areas:
                area.tag_redraw()
            return {'RUNNING_MODAL'}
        except StopIteration:
            self._finish(context)
            self.report({'INFO'}, self._scene.acf_settings.status)
            return {'FINISHED'}
        except Exception as exc:
            traceback.print_exc()
            self._finish(context)
            self._scene.acf_settings.status = str(exc)
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

    def cancel(self, context):
        self._finish(context, cancelled=True)


class ACF_OT_restore(bpy.types.Operator):
    bl_idname = 'acf.restore'
    bl_label = 'Restore Backup'
    bl_description = ('Restore mesh, shape keys and vertex groups from the most recent backup: the state before the last '
                      'weight fit, part repair or clothing armature, or before the first Assisted Fit adjustment since then')
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        s = context.scene.acf_settings
        return context.mode == 'OBJECT' and not s.running and s.clothing and s.clothing.acf_backup

    def execute(self, context):
        s = context.scene.acf_settings
        try:
            for obj in engine.clothing_objects(s):
                if obj.acf_backup:
                    engine.restore_backup(obj, obj.acf_backup)
            s.status = 'Original mesh, shape keys and weights restored'
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


class ACF_OT_review(bpy.types.Operator):
    bl_idname = 'acf.select_review'
    bl_label = 'Select Problems'
    bl_description = 'Select clothing vertices with remaining clipping, crossings, folds or abrupt weight changes and enter Edit Mode'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        s = context.scene.acf_settings
        return context.mode == 'OBJECT' and not s.running and s.clothing and 'acf_report' in s.clothing

    def execute(self, context):
        meshes = engine.clothing_objects(context.scene.acf_settings)
        bpy.ops.object.select_all(action='DESELECT')
        for obj in meshes:
            indices = engine.review_vertices(obj)
            obj.hide_set(False)
            obj.select_set(True)
            for v in obj.data.vertices:
                v.select = v.index in indices
            for edge in obj.data.edges:
                edge.select = False
            for face in obj.data.polygons:
                face.select = False
        context.view_layer.objects.active = context.scene.acf_settings.clothing
        context.tool_settings.mesh_select_mode = (True, False, False)
        bpy.ops.object.mode_set(mode='EDIT')
        return {'FINISHED'}


class ACF_OT_clothing_armature(bpy.types.Operator):
    bl_idname = 'acf.clothing_armature'
    bl_label = 'Create Clothing Armature'
    bl_description = ('Duplicate the avatar armature (or the one deforming the clothing), parent and deform the clothing '
                      '(and other selected meshes) with the copy, and delete bones they do not need, for separate export '
                      'and VRCFury Armature Link. Before a fit, bones come from the body weights under the clothing. '
                      'Backs up each mesh; the avatar is unchanged')
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        s = context.scene.acf_settings
        return context.mode == 'OBJECT' and not s.running and s.clothing

    def execute(self, context):
        s = context.scene.acf_settings
        try:
            report = engine.create_clothing_armature(context, s)
        except Exception as exc:
            s.status = str(exc)
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        kept = f"Created {report['armature']}: kept {len(report['kept_bones'])} bones, removed {len(report['removed_bones'])}"
        if report['bones_estimated_from_body_for']:
            s.status = kept + ', chosen from the body under the clothing. Now run Fit & Refine Weights'
        else:
            merged = [name for name, row in report['merged_weights'].items() if row['into']]
            s.status = kept + (f"; merged light weights of {', '.join(merged[:6])}{', ...' if len(merged) > 6 else ''} "
                               'into parents' if merged else '')
        self.report({'INFO'}, s.status)
        return {'FINISHED'}


class ACF_OT_mark_part(bpy.types.Operator):
    bl_idname = 'acf.mark_part'
    bl_label = 'Mark Clothing Part'
    bl_description = ('Mark entire linked mesh islands touched by the vertex selection. Belts get spatially smooth weights; '
                      'hardware/accessories follow their fabric attachment blend and stay rigid in Assisted Fit')
    bl_options = {'REGISTER', 'UNDO'}
    role: IntProperty(default=1, min=0, max=2)

    @classmethod
    def description(cls, context, properties):
        return ('Mark the selected linked islands as a flexible belt/strap: weights are blended along it',
                'Clear belt/hardware marks from the selected linked islands',
                'Mark the selected linked islands as hardware/accessory: one shared attachment blend for weights, '
                'rigid (never bent or scaled) in Assisted Fit')[(1, 0, 2).index(properties.role)]

    @classmethod
    def poll(cls, context):
        return (context.active_object and context.active_object.type == 'MESH'
                and context.mode == 'EDIT_MESH' and not context.scene.acf_settings.running
                and len(context.objects_in_mode) == 1)

    def execute(self, context):
        import bmesh
        from . import parts
        obj = context.active_object
        mesh = bmesh.from_edit_mesh(obj.data)
        mesh.verts.ensure_lookup_table()
        mesh.verts.index_update()
        selected = [v.index for v in mesh.verts if v.select]
        if not selected:
            self.report({'ERROR'}, 'Select the part with L first')
            return {'CANCELLED'}
        bpy.ops.object.mode_set(mode='OBJECT')
        try:
            count = parts.mark(obj, selected, self.role)
            context.scene.acf_settings.clothing = obj
            self.report({'INFO'}, f'Marked {count} linked parts')
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        finally:
            bpy.ops.object.mode_set(mode='EDIT')


class ACF_OT_apply_parts(bpy.types.Operator):
    bl_idname = 'acf.apply_parts'
    bl_label = 'Apply Belt & Hardware Weights'
    bl_description = 'Back up the current clothing and repair its marked belt and detached hardware weights; rerun Fit & Refine for a fresh collision report'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and not context.scene.acf_settings.running

    def execute(self, context):
        s = context.scene.acf_settings
        try:
            stats = engine.apply_part_weights(context, s)
            s.status = (f"Updated {stats['belt_islands']} belts and {stats['attached_hardware_islands']} hardware attachments; "
                        f"{stats['skipped_protected_hardware']} protected and {stats['skipped_unattached_hardware']} unattached "
                        'parts kept. Rerun Fit & Refine to check clipping.')
            self.report({'INFO'}, s.status)
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


# ---------------------------------------------------------------- Assisted Fit
def assisted_summary(stats):
    text = f"Adjusted {stats['moved_vertices']} vertices"
    before, after = stats['before'], stats['after']
    if stats['relaxed']:
        text += (f"; evened out {stats['band_vertices']} fade-zone vertices "
                 f"(max stretch {before['max_stretch']:.0%} -> {after['max_stretch']:.0%})")
    elif stats['band_vertices']:
        text += f"; fade zone max stretch {after['max_stretch']:.0%}"
    if after.get('flipped_faces'):
        text += f"; {after['flipped_faces']} faces folded over, reduce the move or widen the falloff"
    elif after['max_stretch'] > .5:
        text += '; widen the falloff to spread the stretch'
    return text + '. Rerun Fit & Refine Weights to re-check clipping'


def _active_mesh(context):
    obj = context.active_object
    return obj if obj is not None and obj.type == 'MESH' and mesh_poll(None, obj) else None


class ACF_OT_assisted_adjust(bpy.types.Operator):
    bl_idname = 'acf.assisted_adjust'
    bl_label = 'Adjust Region'
    bl_description = ('Move, rotate or scale the selected clothing region; the change fades out along the garment surface. '
                      'G/R/S switch move/rotate/scale, X/Y/Z constrain (Shift: plane), mouse wheel changes the falloff, '
                      'F cycles the falloff shape, Enter/click confirms, Esc/right-click cancels. '
                      'Stored in the "Cloth Fit Assisted" shape key')
    bl_options = {'REGISTER', 'UNDO'}

    translation: FloatVectorProperty(name='Move', size=3, subtype='TRANSLATION', unit='LENGTH')
    rotation: FloatVectorProperty(name='Rotate', size=3, subtype='EULER', unit='ROTATION')
    scale: FloatVectorProperty(name='Scale', size=3, default=(1., 1., 1.), min=.05, max=20., subtype='XYZ')
    falloff_radius: FloatProperty(name='Falloff radius', default=.05, min=0., soft_max=.5, subtype='DISTANCE', unit='LENGTH')
    falloff_profile: EnumProperty(name='Falloff', items=PROFILE_ITEMS, default='SMOOTH')
    pivot: EnumProperty(name='Pivot', items=PIVOT_ITEMS, default='SELECTION')
    orientation: EnumProperty(name='Axes', items=ORIENTATION_ITEMS, default='GLOBAL')
    relax_iterations: IntProperty(name='Smoothing passes', default=8, min=0, max=40)

    _session = None
    _handle = None

    @classmethod
    def poll(cls, context):
        obj = _active_mesh(context)
        return (obj is not None and context.mode in ('OBJECT', 'EDIT_MESH') and not context.scene.acf_settings.running
                and (context.mode == 'OBJECT' or len(context.objects_in_mode) == 1))

    # -------------------------------------------------------------- blocking
    def execute(self, context):
        obj = _active_mesh(context)
        edit = context.mode == 'EDIT_MESH'
        settings = context.scene.acf_settings
        if edit:
            bpy.ops.object.mode_set(mode='OBJECT')
        try:
            session = engine.AssistSession(obj, settings, self.pivot, self.orientation, context.scene.cursor.location)
            control = session.control(self.translation, Euler(self.rotation).to_matrix(), self.scale)
            stats = session.confirm(control, self.falloff_radius, self.falloff_profile, self.relax_iterations)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        finally:
            if edit:
                bpy.ops.object.mode_set(mode='EDIT')
        settings.status = assisted_summary(stats)
        self.report({'INFO'}, settings.status)
        return {'FINISHED'}

    # -------------------------------------------------------------- interactive
    def invoke(self, context, event):
        region = next((r for r in context.area.regions if r.type == 'WINDOW'), None) \
            if context.area is not None and context.area.type == 'VIEW_3D' else None
        if region is None or context.space_data.region_3d is None:
            return self.execute(context)
        settings = context.scene.acf_settings
        self.translation, self.rotation, self.scale = (0., 0., 0.), (0., 0., 0.), (1., 1., 1.)
        self.falloff_radius, self.falloff_profile = settings.assist_falloff, settings.assist_profile
        self.pivot, self.orientation = settings.assist_pivot, settings.assist_orientation
        self.relax_iterations = settings.assist_relax
        self._edit = context.mode == 'EDIT_MESH'
        if self._edit:
            bpy.ops.object.mode_set(mode='OBJECT')
        try:
            self._session = engine.AssistSession(_active_mesh(context), settings, self.pivot, self.orientation,
                                                 context.scene.cursor.location, show_rest=True)
        except Exception as exc:
            if self._edit:
                bpy.ops.object.mode_set(mode='EDIT')
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self._area, self._region, self._view = context.area, region, context.space_data.region_3d
        self._axes = self._session.axes
        self._rotation_matrix = Matrix.Identity(3)
        self._mode, self._axis, self._plane = 'MOVE', None, False
        self._mouse = Vector((event.mouse_x - region.x, event.mouse_y - region.y))
        self._begin()
        self._handle = bpy.types.SpaceView3D.draw_handler_add(self._draw, (context,), 'WINDOW', 'POST_PIXEL')
        context.window_manager.modal_handler_add(self)
        self._update(context)
        return {'RUNNING_MODAL'}

    def _begin(self):
        """Start a new drag segment from the current values."""
        self._start = self._mouse.copy()
        self._base = (Vector(self.translation), self._rotation_matrix.copy(), Vector(self.scale))

    def _pivot_2d(self):
        from bpy_extras.view3d_utils import location_3d_to_region_2d
        center = Vector(self._session.pivot) + Vector(self.translation)
        point = location_3d_to_region_2d(self._region, self._view, center)
        return point if point is not None else self._start

    def _axis_vector(self):
        return Vector(self._axes[:, self._axis])

    def _apply_mouse(self):
        from bpy_extras.view3d_utils import location_3d_to_region_2d, region_2d_to_location_3d
        translation, rotation, scale = self._base
        if self._mode == 'MOVE':
            depth = Vector(self._session.pivot) + translation
            delta = (region_2d_to_location_3d(self._region, self._view, self._mouse, depth) -
                     region_2d_to_location_3d(self._region, self._view, self._start, depth))
            if self._axis is not None:
                axis = self._axis_vector()
                delta = delta - axis * delta.dot(axis) if self._plane else axis * delta.dot(axis)
            self.translation = translation + delta
        elif self._mode == 'ROTATE':
            pivot = location_3d_to_region_2d(self._region, self._view, Vector(self._session.pivot) + translation)
            pivot = pivot if pivot is not None else self._start
            a, b = self._start - pivot, self._mouse - pivot
            angle = math.atan2(b.y, b.x) - math.atan2(a.y, a.x)
            toward_viewer = self._view.view_rotation @ Vector((0., 0., 1.))
            axis = toward_viewer if self._axis is None else self._axis_vector()
            if axis.dot(toward_viewer) < 0:
                angle = -angle
            self._rotation_matrix = Matrix.Rotation(angle, 3, axis) @ rotation
            self.rotation = self._rotation_matrix.to_euler('XYZ')
        else:
            pivot = self._pivot_2d()
            factor = (self._mouse - pivot).length / max((self._start - pivot).length, 1.)
            new = Vector(scale)
            for k in range(3):
                if self._axis is None or (self._plane and k != self._axis) or (not self._plane and k == self._axis):
                    new[k] = min(max(scale[k] * factor, .05), 20.)
            self.scale = new

    def _update(self, context):
        session = self._session
        control = session.control(self.translation, self._rotation_matrix, self.scale)
        session.preview(control, self.falloff_radius, self.falloff_profile)
        axis = '' if self._axis is None else (' not ' if self._plane else ' along ') + 'XYZ'[self._axis]
        if self._mode == 'MOVE':
            value = 'Move ' + ', '.join(f'{v:.4f}' for v in self.translation)
        elif self._mode == 'ROTATE':
            value = 'Rotate ' + ', '.join(f'{math.degrees(v):.1f}°' for v in self.rotation)
        else:
            value = 'Scale ' + ', '.join(f'{v:.3f}' for v in self.scale)
        unit = context.scene.unit_settings
        radius = bpy.utils.units.to_string(unit.system, 'LENGTH', self.falloff_radius * unit.scale_length, precision=3)
        self._area.header_text_set(f'Assisted Fit: {value}{axis}  |  falloff {radius} {self.falloff_profile.title()}')
        context.workspace.status_text_set('G move  R rotate  S scale  |  X/Y/Z axis, Shift+X/Y/Z plane  |  '
                                          'Wheel falloff  F falloff shape  |  Enter/Click confirm  Esc/Right-click cancel')
        self._area.tag_redraw()

    def _draw(self, context):
        if self._session is None or context.region != self._region:
            return
        import gpu
        from gpu_extras.batch import batch_for_shader
        from bpy_extras.view3d_utils import location_3d_to_region_2d
        center3 = Vector(self._session.pivot) + Vector(self.translation)
        center = location_3d_to_region_2d(self._region, self._view, center3)
        if center is None:
            return
        right = self._view.view_rotation @ Vector((1., 0., 0.))
        edge = location_3d_to_region_2d(self._region, self._view, center3 + right * self.falloff_radius)
        radius = (edge - center).length if edge is not None else 0.
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.blend_set('ALPHA')
        gpu.state.line_width_set(1.5)
        lines = [(center.x - 8, center.y), (center.x + 8, center.y), (center.x, center.y - 8), (center.x, center.y + 8),
                 tuple(center), tuple(self._mouse)]
        shader.uniform_float('color', (1., 1., 1., .7))
        batch_for_shader(shader, 'LINES', {'pos': lines}).draw(shader)
        if radius > 1.:
            ring = [(center.x + radius * math.cos(a * math.tau / 64), center.y + radius * math.sin(a * math.tau / 64))
                    for a in range(65)]
            shader.uniform_float('color', (1., .62, .15, .85))
            batch_for_shader(shader, 'LINE_STRIP', {'pos': ring}).draw(shader)
        gpu.state.line_width_set(1.)
        gpu.state.blend_set('NONE')

    def _end(self, context):
        if self._handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            self._handle = None
        self._area.header_text_set(None)
        context.workspace.status_text_set(None)
        self._area.tag_redraw()

    def modal(self, context, event):
        try:
            return self._modal(context, event)
        except Exception as exc:
            traceback.print_exc()
            self._cancel(context)
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

    def _modal(self, context, event):
        if event.type == 'MOUSEMOVE':
            self._mouse = Vector((event.mouse_x - self._region.x, event.mouse_y - self._region.y))
            self._apply_mouse()
            self._update(context)
        elif event.value != 'PRESS':
            pass
        elif event.type in {'LEFTMOUSE', 'RET', 'NUMPAD_ENTER', 'SPACE'}:
            return self._confirm(context)
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self._cancel(context)
            context.scene.acf_settings.status = 'Adjustment cancelled; clothing unchanged'
            return {'CANCELLED'}
        elif event.type in {'G', 'R', 'S'}:
            self._mode = {'G': 'MOVE', 'R': 'ROTATE', 'S': 'SCALE'}[event.type]
            self._begin()
            self._update(context)
        elif event.type in {'X', 'Y', 'Z'}:
            axis, plane = 'XYZ'.index(event.type), event.shift
            self._axis, self._plane = (None, False) if (axis, plane) == (self._axis, self._plane) else (axis, plane)
            self._apply_mouse()
            self._update(context)
        elif event.type in {'WHEELUPMOUSE', 'PAGE_UP', 'NUMPAD_PLUS'}:
            self.falloff_radius = max(self.falloff_radius * 1.15, .002 * self._session.region.size)
            self._update(context)
        elif event.type in {'WHEELDOWNMOUSE', 'PAGE_DOWN', 'NUMPAD_MINUS'}:
            self.falloff_radius = self.falloff_radius / 1.15 if self.falloff_radius > .002 * self._session.region.size else 0.
            self._update(context)
        elif event.type == 'F':
            names = [item[0] for item in PROFILE_ITEMS]
            self.falloff_profile = names[(names.index(self.falloff_profile) + 1) % len(names)]
            self._update(context)
        return {'RUNNING_MODAL'}

    def _confirm(self, context):
        settings = context.scene.acf_settings
        session, self._session = self._session, None
        self._end(context)
        context.window.cursor_set('WAIT')
        try:
            control = session.control(self.translation, self._rotation_matrix, self.scale)
            stats = session.confirm(control, self.falloff_radius, self.falloff_profile, self.relax_iterations)
        except Exception as exc:
            traceback.print_exc()
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        finally:
            context.window.cursor_set('DEFAULT')
            if self._edit:
                bpy.ops.object.mode_set(mode='EDIT')
        settings.assist_falloff, settings.assist_profile = self.falloff_radius, self.falloff_profile
        settings.status = assisted_summary(stats)
        self.report({'INFO'}, settings.status)
        return {'FINISHED'}

    def _cancel(self, context):
        session, self._session = self._session, None
        self._end(context)
        if session is not None:
            session.cancel()
        if self._edit:
            bpy.ops.object.mode_set(mode='EDIT')

    def cancel(self, context):
        self._cancel(context)


class ACF_OT_assisted_pins(bpy.types.Operator):
    bl_idname = 'acf.assisted_pins'
    bl_label = 'Assisted Fit Pins'
    bl_options = {'REGISTER', 'UNDO'}
    action: EnumProperty(items=(('PIN', 'Pin', 'Add the selected vertices to the pin group'),
                                ('UNPIN', 'Unpin', 'Remove the selected vertices from the pin group'),
                                ('SELECT', 'Select', 'Select the pinned vertices')))

    @classmethod
    def description(cls, context, properties):
        return {'PIN': 'Pin the selected vertices: Assisted Fit never moves them and fades nearby fabric into them',
                'UNPIN': 'Remove the selected vertices from the pin group',
                'SELECT': 'Select the pinned vertices'}[properties.action]

    @classmethod
    def poll(cls, context):
        return _active_mesh(context) is not None and context.mode == 'EDIT_MESH'

    def execute(self, context):
        obj = context.active_object
        name = context.scene.acf_settings.assist_pin_group.strip() or 'ACF Pinned'
        group = obj.vertex_groups.get(name)
        if group is None:
            if self.action != 'PIN':
                self.report({'WARNING'}, f'No "{name}" vertex group yet')
                return {'CANCELLED'}
            group = obj.vertex_groups.new(name=name)
            context.scene.acf_settings.assist_pin_group = name
        active = obj.vertex_groups.active_index
        obj.vertex_groups.active_index = group.index
        try:
            if self.action == 'PIN':
                weight = context.scene.tool_settings.vertex_group_weight
                context.scene.tool_settings.vertex_group_weight = 1.
                bpy.ops.object.vertex_group_assign()
                context.scene.tool_settings.vertex_group_weight = weight
            elif self.action == 'UNPIN':
                bpy.ops.object.vertex_group_remove_from()
            else:
                bpy.ops.mesh.select_all(action='DESELECT')
                bpy.ops.object.vertex_group_select()
        finally:
            # Keep Weight Paint on the group the user was painting.
            if active >= 0:
                obj.vertex_groups.active_index = active
        return {'FINISHED'}


class ACF_OT_assisted_clear(bpy.types.Operator):
    bl_idname = 'acf.assisted_clear'
    bl_label = 'Remove Adjustments'
    bl_description = 'Delete the "Cloth Fit Assisted" shape key of the active clothing mesh (Undo or Restore Backup bring it back)'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = _active_mesh(context)
        return (obj is not None and context.mode == 'OBJECT' and not context.scene.acf_settings.running
                and engine.fit_key(obj, 'acf_assisted_key', engine.ASSISTED_KEY) is not None)

    def execute(self, context):
        try:
            engine.clear_assisted(_active_mesh(context))
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        context.scene.acf_settings.status = 'Assisted Fit adjustments removed'
        return {'FINISHED'}


# ---------------------------------------------------------------- panels
class _Panel:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Cloth Fit'


class ACF_PT_panel(_Panel, bpy.types.Panel):
    bl_label = 'Avatar Cloth Fit'
    bl_idname = 'ACF_PT_panel'

    def draw(self, context):
        layout = self.layout
        s = context.scene.acf_settings
        layout.use_property_split = True
        layout.use_property_decorate = False
        col = layout.column()
        col.enabled = not s.running
        col.prop(s, 'body', text='Body')
        col.prop(s, 'armature', text='Armature')
        col.prop(s, 'clothing', text='Clothing')
        row = col.row()
        row.use_property_split = False
        row.operator('acf.pick_selected', icon='RESTRICT_SELECT_OFF')
        meshes = engine.clothing_objects(s)
        if len(meshes) > 1:
            body = info_panel(col, context, 'acf_outfit', f'Outfit: {len(meshes)} pieces', icon='OUTLINER_OB_GROUP_INSTANCE')
            if body is not None:
                flow = body.column(align=True)
                for obj in meshes:
                    flow.label(text=obj.name, icon='MESH_DATA')
        layout.separator()
        col = layout.column(align=True)
        col.enabled = not s.running
        big = col.row(align=True)
        big.scale_y = 1.5
        big.operator('acf.fit', icon='MOD_ARMATURE')
        row = col.row(align=True)
        row.operator('acf.restore', icon='LOOP_BACK')
        row.operator('acf.select_review', icon='VERTEXSEL')
        if s.running:
            # Progress stays visible while a fit runs.
            box = layout.box()
            wrapped(box, s.status, context, icon='CANCEL')
            box.label(text='Esc cancels and restores everything')
        elif s.status:
            info_panel(layout, context, 'acf_status', s.status, (s.status,))
        if s.clothing and 'acf_report' in s.clothing:
            try:
                report = json.loads(s.clothing['acf_report'])
                warnings = report['warnings']
                title = f'Report: {report["review_vertices"]} problems'
                if warnings:
                    title += f', {len(warnings)} warning' + ('s' if len(warnings) > 1 else '')
                body = info_panel(layout, context, 'acf_report', title, icon='ERROR' if warnings else 'CHECKMARK')
                if body is not None:
                    wrapped(body, f'{report["poses"]} poses checked, {report["review_vertices"]} problem vertices',
                            context)
                    library = report.get('pose_library')
                    if library:
                        wrapped(body, f'{library["sampled_poses"]} library poses: {library["proven_equivalent"]} '
                                      f'proven equivalent, {library["screened_by_prediction"]} screened', context)
                    for warning in warnings:
                        wrapped(body, warning, context, icon='ERROR')
            except (ValueError, KeyError):
                pass


class ACF_OT_sculpt_setup(bpy.types.Operator):
    bl_idname = 'acf.sculpt_setup'
    bl_label = 'Sculpt Clothing'
    bl_description = ('Back up the active clothing (once), sculpt on its "Cloth Fit Assisted" shape key so the Basis '
                      'stays the original for Shape Relax, enter Sculpt Mode and pick Smart Grab')
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (_active_mesh(context) is not None and context.mode in ('OBJECT', 'EDIT_MESH', 'SCULPT')
                and not context.scene.acf_settings.running)

    def execute(self, context):
        obj = _active_mesh(context)
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        try:
            key = engine.prepare_assisted_key(obj)
        except ValueError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        bpy.ops.object.mode_set(mode='SCULPT')
        if context.area is not None and context.area.type == 'VIEW_3D':
            bpy.ops.wm.tool_set_by_id(name='acf.smart_grab')
        context.scene.acf_settings.status = f'Sculpting "{key.name}"; the Basis stays the original shape'
        return {'FINISHED'}


class ACF_OT_store_reference(bpy.types.Operator):
    bl_idname = 'acf.store_reference'
    bl_label = 'Store Reference'
    bl_description = ('Remember the current shape as the reference whose folds and spacing Shape Relax restores. '
                      'Used when sculpting without shape keys (sculpting the Cloth Fit key uses the Basis)')
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _active_mesh(context) is not None and context.mode in ('OBJECT', 'SCULPT')

    def execute(self, context):
        obj = _active_mesh(context)
        try:
            sculpt.store_reference(obj, sculpt.Edit(obj).local)
        except ValueError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        context.scene.acf_settings.status = 'Reference stored; Shape Relax restores its local shape'
        return {'FINISHED'}


def _reference_label(obj):
    keys = obj.data.shape_keys
    if keys and obj.active_shape_key and obj.active_shape_key != keys.reference_key:
        return f'Relax reference: {keys.reference_key.name}', 'CHECKMARK'
    if obj.data.attributes.get(sculpt.REFERENCE) is not None:
        return 'Relax reference: stored shape', 'CHECKMARK'
    backup = obj.acf_backup
    if backup is not None and backup.type == 'MESH' and len(backup.data.vertices) == len(obj.data.vertices):
        return 'Relax reference: Cloth Fit backup', 'CHECKMARK'
    return 'No reference yet: Relax smooths without it', 'ERROR'


class ACF_PT_assisted(_Panel, bpy.types.Panel):
    bl_label = 'Assisted Fit'
    bl_idname = 'ACF_PT_assisted'
    bl_parent_id = 'ACF_PT_panel'

    def draw_header(self, context):
        self.layout.label(icon='MOD_SIMPLEDEFORM')

    def draw(self, context):
        layout = self.layout
        s = context.scene.acf_settings
        obj = _active_mesh(context)
        layout.enabled = not s.running
        if context.mode == 'SCULPT':
            self.draw_sculpt(context, layout, s, obj)
            return
        row = layout.row(align=True)
        row.scale_y = 1.5
        row.operator('acf.sculpt_setup', icon='SCULPTMODE_HLT')
        row.operator('acf.assisted_adjust', icon='ORIENTATION_GIMBAL')
        if context.mode != 'EDIT_MESH':
            info_panel(layout, context, 'acf_assisted_tips', 'How to start',
                       ('Sculpt Clothing: brushes in Sculpt Mode. Adjust Region: select a region in Edit Mode first.',))
        layout.use_property_split = True
        layout.use_property_decorate = False
        col = layout.column(heading='Adjust Region')
        col.prop(s, 'assist_falloff')
        col.prop(s, 'assist_profile')
        col.prop(s, 'assist_pivot')
        col.prop(s, 'assist_orientation')
        col.prop(s, 'assist_relax', text='Smoothing')
        self.draw_pins(context, layout, s, obj)
        self.draw_key(layout, obj)

    def draw_sculpt(self, context, layout, s, obj):
        tools = context.workspace.tools.from_space_view3d_mode('SCULPT', create=False)
        active = tools.idname if tools else ''
        col = layout.column(align=True)
        col.scale_y = 1.3
        for idname, label, icon in (('acf.smart_grab', 'Smart Grab', 'VIEW_PAN'),
                                    ('acf.fit_body', 'Fit to Body', 'MOD_SHRINKWRAP'),
                                    ('acf.relax', 'Shape Relax', 'MOD_SMOOTH'),
                                    ('acf.region_resize_tool', 'Region Resize', 'FULLSCREEN_ENTER')):
            col.operator('wm.tool_set_by_id', text=label, icon=icon, depress=active == idname).name = idname
        layout.use_property_split = True
        layout.use_property_decorate = False
        col = layout.column()
        col.prop(s, 'brush_radius')
        col.prop(s, 'brush_strength')
        if active == 'acf.smart_grab':
            col.prop(s, 'grab_stiffness')
        else:
            col.prop(s, 'brush_falloff')
        if active == 'acf.relax' and s.body is not None:
            col.prop(s, 'fit_clearance')
            col.prop(s, 'fit_reach')
        elif active == 'acf.fit_body':
            col.prop(s, 'fit_mode')
            col.prop(s, 'fit_clearance')
            col.prop(s, 'fit_reach')
            if s.body is None:
                wrapped(layout, 'Choose the avatar Body above first.', context, icon='ERROR')
        elif active == 'acf.region_resize_tool':
            col.prop(s, 'resize_transition')
            col.prop(s, 'resize_pin_edges', text='Pin edges')
            col.prop(s, 'resize_keep_detail', text='Fold depth')
            col.prop(s, 'assist_relax', text='Smoothing')
        if obj is not None:
            row = col.row(align=True, heading='Mirror')
            for axis in 'xyz':
                row.prop(obj.data, f'use_mirror_{axis}', text=axis.upper(), toggle=True)
            key = engine.fit_key(obj, 'acf_assisted_key', engine.ASSISTED_KEY)
            if key is None or obj.active_shape_key != key:
                wrapped(layout, 'Not sculpting the Cloth Fit key: edits go into the current shape.', context, icon='INFO')
                layout.operator('acf.sculpt_setup', text='Sculpt on Cloth Fit Key', icon='SHAPEKEY_DATA')
            text, icon = _reference_label(obj)
            row = layout.row()
            row.label(text=text, icon=icon)
            row.operator('acf.store_reference', text='', icon='IMPORT')
        info_panel(layout, context, 'acf_sculpt_tips', 'Brush tips',
                   ('Masked vertices and the pin group never move. Shift+drag: Shape Relax. F / Shift+F: radius / '
                    'strength.', 'Smart Grab pulls like an elastic grab: the pull fades softly past the brush circle.',
                    'Shape Relax keeps the gap to the Body (Clearance) when a Body is set.'))
        self.draw_pins(context, layout, s, obj)
        self.draw_key(layout, obj)

    def draw_pins(self, context, layout, s, obj):
        col = layout.column()
        if obj is not None and s.assist_pin_group in obj.vertex_groups:
            col.prop_search(s, 'assist_pin_group', obj, 'vertex_groups', text='Pin group')
        else:
            # A free name: Pin creates the group on first use.
            col.prop(s, 'assist_pin_group', text='Pin group')
        if context.mode == 'EDIT_MESH':
            row = layout.row(align=True)
            row.operator('acf.assisted_pins', text='Pin', icon='PINNED').action = 'PIN'
            row.operator('acf.assisted_pins', text='Unpin', icon='UNPINNED').action = 'UNPIN'
            row.operator('acf.assisted_pins', text='Select', icon='RESTRICT_SELECT_OFF').action = 'SELECT'

    def draw_key(self, layout, obj):
        key = engine.fit_key(obj, 'acf_assisted_key', engine.ASSISTED_KEY) if obj else None
        if key is not None:
            row = layout.row(align=True)
            row.use_property_split = False
            row.prop(key, 'value', text='Adjustment strength', slider=True)
            row.operator('acf.assisted_clear', text='', icon='TRASH')


class ACF_PT_weights(_Panel, bpy.types.Panel):
    bl_label = 'Weight Settings'
    bl_idname = 'ACF_PT_weights'
    bl_parent_id = 'ACF_PT_panel'

    def draw(self, context):
        s = context.scene.acf_settings
        layout = self.layout
        layout.enabled = not s.running
        col = layout.column(align=True)
        col.prop(s, 'influence_limit')
        col.prop(s, 'iterations')
        col.prop(s, 'clearance')
        row = layout.row()
        row.label(text='Pose validation')
        row.prop(s, 'validation_mode', text='')
        col = layout.column(align=True)
        col.prop(s, 'secondary_bones')
        col.prop(s, 'self_collision')
        col.prop(s, 'attach_hardware')
        col = layout.column(align=True)
        col.prop(s, 'correct_shape')
        sub = col.column(align=True)
        sub.active = s.correct_shape
        sub.prop(s, 'max_correction')


class ACF_PT_parts(_Panel, bpy.types.Panel):
    bl_label = 'Belts & Hardware'
    bl_idname = 'ACF_PT_parts'
    bl_parent_id = 'ACF_PT_panel'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        s = context.scene.acf_settings
        layout = self.layout
        layout.enabled = not s.running
        info_panel(layout, context, 'acf_parts_tips', 'How to mark', ('Edit Mode: hover a part and press L, then mark it.',))
        col = layout.column(align=True)
        col.operator('acf.mark_part', text='Mark Belt / Strap', icon='MOD_CURVE').role = 1
        col.operator('acf.mark_part', text='Mark Hardware / Accessory', icon='LOCKED').role = 2
        col.operator('acf.mark_part', text='Clear Marks', icon='X').role = 0
        layout.prop(s, 'strap_smoothing')
        layout.operator('acf.apply_parts', icon='MOD_VERTEX_WEIGHT')


class ACF_PT_export(_Panel, bpy.types.Panel):
    bl_label = 'Clothing Armature'
    bl_idname = 'ACF_PT_export'
    bl_parent_id = 'ACF_PT_panel'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        s = context.scene.acf_settings
        layout = self.layout
        layout.enabled = not s.running
        layout.prop(s, 'bone_weight_threshold')
        layout.operator('acf.clothing_armature', icon='ARMATURE_DATA')


class ACF_PT_advanced(_Panel, bpy.types.Panel):
    bl_label = 'Advanced'
    bl_idname = 'ACF_PT_advanced'
    bl_parent_id = 'ACF_PT_panel'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        s = context.scene.acf_settings
        layout = self.layout
        layout.enabled = not s.running
        col = layout.column(align=True)
        for name in ('transition_width', 'strength', 'search_distance'):
            col.prop(s, name)
        col = layout.column(align=True)
        col.prop(s, 'reverse_forward')
        col.prop(s, 'preserve_clothing_bones')
        info_panel(layout, context, 'acf_advanced_tips', 'Modifiers',
                   ('Non-armature modifiers are ignored during analysis and kept.',))


CLASSES = (ACF_ClothingItem, ACF_Settings, ACF_OT_pick, ACF_OT_fit, ACF_OT_restore, ACF_OT_review, ACF_OT_mark_part,
           ACF_OT_apply_parts, ACF_OT_clothing_armature, ACF_OT_assisted_adjust, ACF_OT_assisted_pins,
           ACF_OT_assisted_clear, ACF_OT_sculpt_setup, ACF_OT_store_reference, ACF_PT_panel, ACF_PT_assisted,
           ACF_PT_weights, ACF_PT_parts, ACF_PT_export, ACF_PT_advanced)


def menu_entry(self, _context):
    self.layout.separator()
    self.layout.operator('acf.assisted_adjust', text='Assisted Fit: Adjust Region', icon='ORIENTATION_GIMBAL')


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.acf_settings = PointerProperty(type=ACF_Settings)
    bpy.types.Object.acf_backup = PointerProperty(type=bpy.types.Object)
    bpy.types.VIEW3D_MT_edit_mesh_context_menu.append(menu_entry)
    sculpt.register()


def unregister():
    sculpt.unregister()
    bpy.types.VIEW3D_MT_edit_mesh_context_menu.remove(menu_entry)
    del bpy.types.Object.acf_backup
    del bpy.types.Scene.acf_settings
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
