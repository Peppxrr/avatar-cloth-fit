"""Weight a multi-piece outfit through one temporary shared surface."""
import json
from contextlib import ExitStack
import bpy
import numpy as np
from . import engine, parts
from .geometry import mesh_arrays

CORRECTION = ('acf_correction_key', engine.CORRECTION_KEY)


class SettingsView:
    def __init__(self, settings, **overrides):
        self.settings, self.overrides = settings, overrides

    def __getattr__(self, name):
        return self.overrides[name] if name in self.overrides else getattr(self.settings, name)

    def __setattr__(self, name, value):
        if name in ('settings', 'overrides'):
            object.__setattr__(self, name, value)
        elif name == 'status':
            self.settings.status = value
        else:
            self.overrides[name] = value


def run(context, settings, meshes):
    body = settings.body
    rig = engine.armature_for(body, settings.armature)
    rigs = [engine.armature_for(obj) or rig for obj in meshes]
    # A common source rig keeps garment-only chains when all pieces share it.
    common = rigs[0] if all(r == rigs[0] for r in rigs) else rig
    transactions, proxy, proxy_backup = [], None, None
    original_texts = set(bpy.data.texts)
    try:
        for obj in meshes:
            transactions.append(engine.Transaction(obj))
        with ExitStack() as stack:
            for obj, target in zip(meshes, rigs):
                links, _ = engine.rig_links(rig, target)
                stack.enter_context(engine.PoseState(rig, target, (body, obj), links))
            vertices, faces, edges, roles, bindings, silenced = [], [], [], [], [], []
            offset = 0
            for obj, target in zip(meshes, rigs):
                old_values = {}
                if settings.correct_shape:
                    key = engine.fit_key(obj, *CORRECTION)
                    if key:
                        old_values[key.name] = key.value
                        key.value = 0.
                silenced.append(old_values)
                engine._update()
                xyz, _, _ = mesh_arrays(obj, context.evaluated_depsgraph_get())
                if len(xyz) != len(obj.data.vertices):
                    raise ValueError(f'{obj.name}: active deformation changes topology')
                vertices.extend(xyz.tolist())
                reverse = obj.matrix_world.to_3x3().determinant() < 0.
                faces.extend([offset + v for v in (reversed(face.vertices) if reverse else face.vertices)]
                             for face in obj.data.polygons)
                edges.extend([offset + v for v in edge.vertices] for edge in obj.data.edges)
                role = parts.roles(obj)
                belts, hardware = parts.classified_parts(obj, settings.attach_hardware)
                # Preserve material evidence without turning automatic detection
                # into an explicit user hardware mark on the temporary mesh.
                for members in belts:
                    role[members] = parts.BELT
                roles.extend(role.tolist())
                # rig_links maps common bone names to target names for scattering.
                inverse, mapping = engine.rig_links(common, target)
                bindings.append((offset, len(xyz), mapping, inverse))
                offset += len(xyz)
            data = bpy.data.meshes.new('Cloth Fit temporary set')
            data.from_pydata(vertices, edges, faces)
            proxy = bpy.data.objects.new('Cloth Fit temporary set', data)
            context.scene.collection.objects.link(proxy)
            modifier = proxy.modifiers.new('Cloth Fit set armature', 'ARMATURE')
            modifier.object = common
            preserve_volume = {m.use_deform_preserve_volume for obj in meshes for m in obj.modifiers
                               if m.type == 'ARMATURE' and m.show_viewport}
            if len(preserve_volume) > 1:
                raise ValueError('Clothing pieces must use the same Preserve Volume setting')
            modifier.use_deform_preserve_volume = next(iter(preserve_volume), False)
            attr = data.attributes.new('acf_part_role', 'INT', 'POINT')
            attr.data.foreach_set('value', roles)
            proxy['acf_part_attribute'] = attr.name
            identifiers = np.concatenate([np.full(count, number, dtype=np.int32)
                                          for number, (_, count, _, _) in enumerate(bindings)])
            attr = data.attributes.new('acf_set_piece', 'INT', 'POINT')
            attr.data.foreach_set('value', identifiers)
            proxy['acf_set_piece_attribute'] = attr.name
            source_xyz, _, _ = mesh_arrays(body, context.evaluated_depsgraph_get())
            proxy['acf_set_body_size'] = max(float(np.ptp(source_xyz, axis=0).max()), 1e-6)
            material_offset = 0
            face_offset = 0
            for obj in meshes:
                for material in obj.data.materials:
                    data.materials.append(material)
                for face in obj.data.polygons:
                    data.polygons[face_offset + face.index].material_index = material_offset + face.material_index
                material_offset += len(obj.data.materials)
                face_offset += len(obj.data.polygons)
            # Preserve earlier correction keys so the single-mesh pipeline can
            # restore them if a new proposal is rejected. Their current
            # contribution is baked into one proxy key, including each piece's value.
            delta = np.zeros((offset, 3))
            found = False
            for obj, binding, old_values in zip(meshes, bindings, silenced):
                key = engine.fit_key(obj, *CORRECTION)
                if key and key.name in old_values:
                    found = True
                    raw = np.empty(len(obj.data.vertices) * 3)
                    base = np.empty_like(raw)
                    key.data.foreach_get('co', raw)
                    key.relative_key.data.foreach_get('co', base)
                    start, count, _, _ = binding
                    delta[start:start + count] = (raw - base).reshape(-1, 3) @ np.array(obj.matrix_world)[:3, :3].T * old_values[key.name]
            if found:
                engine.write_fit_key(proxy, *CORRECTION, delta)
            group_locks = {}
            for number, (obj, target, binding) in enumerate(zip(meshes, rigs, bindings)):
                start, count, mapping, inverse = binding
                for group in obj.vertex_groups:
                    deform = group.name in target.data.bones and target.data.bones[group.name].use_deform
                    name = inverse.get(group.name) if deform else f'piece{number}:{group.name}'
                    if name is None:
                        if any(a.group == group.index and a.weight > 0 for v in obj.data.vertices for a in v.groups):
                            raise ValueError(f'{obj.name}: clothing-only bone {group.name} needs a shared clothing rig')
                        name = f'piece{number}:{group.name}'
                    if deform and name in group_locks and group_locks[name] != group.lock_weight:
                        raise ValueError(f'Clothing pieces disagree on locked bone {group.name}; use matching locks')
                    group_locks[name] = group.lock_weight
                    destination = proxy.vertex_groups.get(name) or proxy.vertex_groups.new(name=name)
                    destination.lock_weight = group.lock_weight
                    for v in obj.data.vertices:
                        for a in v.groups:
                            if a.group == group.index:
                                destination.add([start + v.index], a.weight, 'REPLACE')
            view = SettingsView(settings, clothing=proxy, clothing_items=())
            report = yield from engine.run_single(context, view)
            proxy_backup = proxy.acf_backup
            review = engine.review_vertices(proxy)
            correction = None
            key = engine.fit_key(proxy, *CORRECTION)
            if key and key.value:
                basis = np.empty(offset * 3)
                coordinates = np.empty(offset * 3)
                proxy.data.shape_keys.reference_key.data.foreach_get('co', basis)
                key.data.foreach_get('co', coordinates)
                correction = (coordinates - basis).reshape(-1, 3)
            for obj, target, binding, old_values, transaction in zip(meshes, rigs, bindings, silenced, transactions):
                start, count, mapping, inverse = binding
                if correction is not None:
                    engine.write_fit_key(obj, *CORRECTION, correction[start:start + count],
                                         engine.fit_key(obj, *CORRECTION))
                else:
                    for name, value in old_values.items():
                        obj.data.shape_keys.key_blocks[name].value = value
                deform_names = {b.name for b in target.data.bones if b.use_deform}
                for group in list(obj.vertex_groups):
                    if group.name in deform_names and not group.lock_weight:
                        group.remove(list(range(count)))
                for v in proxy.data.vertices[start:start + count]:
                    for a in v.groups:
                        name = mapping.get(proxy.vertex_groups[a.group].name)
                        if name:
                            group = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
                            if not group.lock_weight:
                                group.add([v.index - start], a.weight, 'REPLACE')
                mod = next((m for m in obj.modifiers if m.type == 'ARMATURE' and m.show_viewport), None)
                if mod is None:
                    mod = obj.modifiers.new('Avatar Cloth Fit Armature', 'ARMATURE')
                    transaction.added_modifier = mod
                if mod.object is None:
                    mod.object = target
                    if transaction.added_modifier is None:
                        transaction.bound_modifier = mod
                local_report = dict(report, clothing=obj.name, vertices=count, clothing_set=[o.name for o in meshes])
                local_review = {i - start for i in review if start <= i < start + count}
                local_report['review_vertices'] = len(local_review)
                obj['acf_report'] = json.dumps(local_report)
                engine.store_review_markers(obj, local_review)
        for transaction in transactions:
            transaction.commit()
        settings.status = f"Done: {len(meshes)} clothing pieces; set weights validated"
        return report
    finally:
        for transaction in reversed(transactions):
            transaction.rollback()
        if proxy:
            proxy_backup = proxy.acf_backup or proxy_backup
            for owned in (proxy, proxy_backup):
                if owned:
                    data = owned.data
                    bpy.data.objects.remove(owned, do_unlink=True)
                    if data.users == 0:
                        bpy.data.meshes.remove(data)
        for text in set(bpy.data.texts) - original_texts:
            if text.name.startswith('Cloth Fit Report - Cloth Fit temporary set'):
                bpy.data.texts.remove(text)
