"""Explicit garment part roles and shared detached hardware attachments."""
import re
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

SURFACE, BELT, HARDWARE = 0, 1, 2
_METAL_WORDS = re.compile(r'(?:^|[\s_.-])(?:metal|metallic|chrome|steel|brass|iron|copper|aluminium|aluminum|titanium|gold|silver|buttons?|buckle|rivet|rings?)(?:$|[\s_.-])')


def metal_material(material):
    if material is None:
        return False
    label = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', material.name).lower()
    if _METAL_WORDS.search(label):
        return True
    if not material.use_nodes or not material.node_tree:
        return False
    # Only a direct, constant metallic surface is evidence. Linked textures,
    # mixed shaders and unused metallic nodes must not classify fabric as metal.
    output = next((n for n in material.node_tree.nodes
                   if n.type == 'OUTPUT_MATERIAL' and n.is_active_output), None)
    if output is None or not output.inputs['Surface'].is_linked:
        return False
    shader = output.inputs['Surface'].links[0].from_node
    if shader.type != 'BSDF_PRINCIPLED':
        return False
    metallic = shader.inputs.get('Metallic')
    return metallic is not None and not metallic.is_linked and metallic.default_value >= .5


def classified_parts(obj, auto_hardware):
    components = islands(obj.data)
    values = roles(obj)
    component_id = np.empty(len(obj.data.vertices), dtype=np.int32)
    for index, component in enumerate(components):
        component_id[component] = index
    materials = [set() for _ in components]
    set_ids = obj.data.attributes.get(obj.get('acf_set_piece_attribute', ''))
    set_areas = None
    if set_ids:
        # A shared fitting surface must not turn a shiny garment into a rigid
        # attachment to another garment. This filter applies only to set proxies;
        # the established single-mesh weighting behavior remains unchanged.
        obj.data.calc_loop_triangles()
        tri = np.array([tuple(t.vertices) for t in obj.data.loop_triangles], dtype=np.int64)
        xyz = np.array([tuple(v.co) for v in obj.data.vertices])
        area = .5 * np.linalg.norm(np.cross(xyz[tri[:, 1]] - xyz[tri[:, 0]], xyz[tri[:, 2]] - xyz[tri[:, 0]]), axis=1)
        set_areas = np.bincount(component_id[tri[:, 0]], area, minlength=len(components))
        piece = np.empty(len(obj.data.vertices), dtype=np.int32)
        set_ids.data.foreach_get('value', piece)
        maximum = {int(p): max((set_areas[k] for k, c in enumerate(components) if piece[c[0]] == p), default=0.)
                   for p in np.unique(piece)}
    for face in obj.data.polygons:
        materials[component_id[face.vertices[0]]].add(face.material_index)
    # A material the main fabric uses is not metal evidence (imports often give
    # every material, fabric and straps included, Metallic 1).
    fabric = materials[max(range(len(components)), key=lambda k: len(components[k]))] if components else set()
    belts, hardware = [], []
    for index, component in enumerate(components):
        automatic = (auto_hardware and materials[index] and
                     all(slot < len(obj.data.materials) and slot not in fabric
                         and metal_material(obj.data.materials[slot]) for slot in materials[index]))
        if automatic and set_areas is not None:
            automatic = (set_areas[index] < .15 * maximum[int(piece[component[0]])] and
                         np.ptp(xyz[component], axis=0).max(initial=0.) < .08 * obj['acf_set_body_size'])
        if np.any(values[component] == BELT):
            belts.append(component)
        elif np.any(values[component] == HARDWARE) or automatic:
            hardware.append(component)
    return belts, hardware


def correct_rigid_shape(obj, correction, auto_hardware=True):
    """A rigid rest-fit correction is a shared translation, never a bend.

    The mean stays within the existing per-vertex displacement bound. Flexible
    belts and ordinary fabric retain the solver's independent corrections.
    """
    _, hardware = classified_parts(obj, auto_hardware)
    for component in hardware:
        correction[component] = correction[component].mean(axis=0)


def islands(mesh):
    adjacency = [[] for _ in mesh.vertices]
    for edge in mesh.edges:
        a, b = edge.vertices
        adjacency[a].append(b)
        adjacency[b].append(a)
    result = []
    seen = set()
    for start in range(len(adjacency)):
        if start in seen:
            continue
        seen.add(start)
        todo, component = [start], []
        while todo:
            vertex = todo.pop()
            component.append(vertex)
            for neighbor in adjacency[vertex]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    todo.append(neighbor)
        result.append(np.array(component, dtype=np.int32))
    return result


def roles(obj):
    result = np.zeros(len(obj.data.vertices), dtype=np.int32)
    name = obj.get('acf_part_attribute')
    attr = obj.data.attributes.get(name) if name else None
    if attr and attr.domain == 'POINT' and attr.data_type == 'INT':
        attr.data.foreach_get('value', result)
    return result


def mark(obj, selected, role):
    values = roles(obj)
    selected = set(selected)
    count = 0
    for component in islands(obj.data):
        if any(int(i) in selected for i in component):
            values[component] = role
            count += 1
    if not count:
        raise ValueError('Select vertices on the belt or hardware first')
    name = obj.get('acf_part_attribute')
    attr = obj.data.attributes.get(name) if name else None
    if attr is None or attr.domain != 'POINT' or attr.data_type != 'INT':
        name, suffix = 'acf_part_role', 1
        while name in obj.data.attributes or name in obj.vertex_groups:
            name = f'acf_part_role.{suffix:03d}'
            suffix += 1
        attr = obj.data.attributes.new(name=name, type='INT', domain='POINT')
    if attr is None:
        raise ValueError('Mark garment parts in Object Mode')
    attr.data.foreach_set('value', values)
    obj['acf_part_attribute'] = attr.name
    obj.data.update()
    return count


def constraints(obj, xyz, dense, names, protected_names,
                auto_hardware=True, influence_limit=4, strap_radius=.01, *,
                smooth_belts=True):
    """Smooth marked flexible belts; give hardware a shared fabric anchor.

    Exact custom/locked weights take priority. Uniform protected mixtures can
    share an attachment; varying protected assignments cannot. Shared blended
    weights follow the fabric contact, but linear skinning may squash hardware.
    Material names are only a hint; users can explicitly mark any linked part.
    Set smooth_belts=False to refresh hardware from refined fabric weights
    without applying another belt-smoothing pass or returning belt pins.
    """
    belts, hardware = classified_parts(obj, auto_hardware)
    protected = np.zeros_like(dense)
    group_to_bone = {g.index: names.index(g.name) for g in obj.vertex_groups if g.name in protected_names and g.name in names}
    for vertex in obj.data.vertices:
        for assignment in vertex.groups:
            if assignment.group in group_to_bone:
                protected[vertex.index, group_to_bone[assignment.group]] = assignment.weight
    free = np.array([name not in protected_names for name in names])
    result = {}
    stats = {'belt_islands': 0, 'attached_hardware_islands': 0,
             'blended_hardware_islands': 0, 'skipped_protected_hardware': 0,
             'skipped_unattached_hardware': 0, 'skipped_belts': 0,
             'constrained_vertices': 0}

    def normalized_row(target, index):
        row = protected[index].copy()
        if row.sum() > 1. + 1e-5:
            raise ValueError(f'Protected clothing weights exceed 1 at vertex {index}')
        remaining = max(0., 1. - row.sum())
        if remaining < 1e-5:
            return row
        target = np.maximum(target, 0.) * free
        slots = max(1, influence_limit - int(np.count_nonzero(row > 1e-8)))
        target[np.argsort(-target)[slots:]] = 0.
        if target.sum() < 1e-12:
            return None
        return row + target * (remaining / target.sum())

    for component in (belts if smooth_belts else ()):
        tree = KDTree(len(component))
        for local_index, index in enumerate(component):
            tree.insert(Vector(xyz[index]), local_index)
        tree.balance()
        source = dense[component].copy()
        skipped = False
        for index in component:
            neighbors = tree.find_range(Vector(xyz[index]), strap_radius * 3)
            neighbor_indices = [i for _, i, _ in neighbors]
            factors = np.array([np.exp(-.5 * (distance / strap_radius) ** 2) for _, _, distance in neighbors])
            target = factors @ source[neighbor_indices] / factors.sum()
            row = normalized_row(target, index)
            if row is None:
                skipped = True
                continue
            dense[index] = row
            result[int(index)] = row
        stats['belt_islands'] += 1
        stats['skipped_belts'] += int(skipped)

    # Attach at the nearest actual contact, not the centroid of a hanging ring
    # or elongated fitting, which can lie over a different moving fabric panel.
    bvh = None
    triangles = np.empty((0, 3), dtype=np.int32)
    if hardware:
        obj.data.calc_loop_triangles()
        excluded = np.zeros(len(xyz), dtype=bool)
        for component in hardware:
            excluded[component] = True
        triangles = np.empty(len(obj.data.loop_triangles) * 3, dtype=np.int32)
        obj.data.loop_triangles.foreach_get('vertices', triangles)
        triangles = triangles.reshape(-1, 3)
        triangles = triangles[~np.any(excluded[triangles], axis=1)]
        if len(triangles):
            bvh = BVHTree.FromPolygons(xyz.tolist(), triangles.tolist(), all_triangles=True)
    from .geometry import barycentric
    for component in hardware:
        protected_rows = protected[component]
        if np.any(protected_rows != protected_rows[0]):
            stats['skipped_protected_hardware'] += 1
            continue
        contact = None
        if bvh:
            for index in component:
                hit, _, face, distance = bvh.find_nearest(Vector(xyz[index]))
                if hit is not None and (contact is None or (distance, int(index), face) < contact[:3]):
                    contact = (distance, int(index), face, hit)
        if contact is None:
            stats['skipped_unattached_hardware'] += 1
            continue
        tri = triangles[contact[2]]
        target = barycentric(np.array(contact[3]), xyz[tri]) @ dense[tri]
        # Preserve the contact's moving mixture rather than selecting one bone.
        # A shared row prevents varying-weight bends, not affine blend squash.
        row = normalized_row(target, int(component[0]))
        if row is None:
            key = 'skipped_protected_hardware' if np.any(protected_rows > 1e-8) else 'skipped_unattached_hardware'
            stats[key] += 1
            continue
        for index in component:
            dense[index] = row
            result[int(index)] = row
        stats['attached_hardware_islands'] += 1
        stats['blended_hardware_islands'] += int(np.count_nonzero(row > 1e-8) > 1)
    stats['constrained_vertices'] = len(result)
    return result, stats


def pin_sparse(rows, candidates, weights, fixed, fixed_values):
    for index, row in rows.items():
        bones = np.flatnonzero(row > 1e-8)
        if len(bones) > weights.shape[1]:
            raise ValueError('Part has more protected bone influences than the solver supports')
        candidates[index] = [int(i) for i in bones] + [int(bones[0])] * (weights.shape[1] - len(bones))
        weights[index] = 0.
        weights[index, :len(bones)] = row[bones]
        fixed[index] = True
        fixed_values[index] = weights[index]
