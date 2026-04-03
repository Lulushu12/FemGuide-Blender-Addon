"""
FemGuide Line Generator - Blender Addon
Ported from 3D Slicer module by Radu Josanu.

Generates surgical guidance geometry: a primary extruded polyline prism,
optional secondary/tertiary offset prisms, and optional cylinder pairs.

Install: Edit > Preferences > Add-ons > Install... > select this file > enable it.
Panel: 3D Viewport > Sidebar (N) > FemGuide tab.
"""

bl_info = {
    "name": "FemGuide Line Generator",
    "author": "Radu Josanu (ported to Blender)",
    "version": (1, 0, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > FemGuide",
    "description": "Generate surgical guidance geometry from a polyline with extrusion, secondary/tertiary lines, and cylinders.",
    "category": "Mesh",
}

import math
import numpy as np
import bpy


# ---------------------------------------------------------------------------
# Geometry helpers (mirror of the Slicer module logic, VTK-free)
# ---------------------------------------------------------------------------

def safe_normalize(v, fallback=(0.0, 0.0, 1.0)):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.array(fallback, dtype=float)
    return v / n


def generate_polyline_points(angle_increments, lengths):
    """Return list of (x,y,0) numpy arrays for the 2-D polyline."""
    current_angle = 0.0
    x, y = 0.0, 0.0
    pts = [np.array([x, y, 0.0])]
    for inc, length in zip(angle_increments, lengths):
        current_angle += math.radians(inc)
        x += length * math.cos(current_angle)
        y += length * math.sin(current_angle)
        pts.append(np.array([x, y, 0.0]))
    return pts


def compute_plane_normal(pts):
    """Compute the normal of the best-fit plane through the first 3 points."""
    if len(pts) < 3:
        return np.array([0.0, 0.0, 1.0])
    v1 = pts[1] - pts[0]
    v2 = pts[2] - pts[0]
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 1.0])
    return normal / norm


# ---------------------------------------------------------------------------
# Mesh builders
# ---------------------------------------------------------------------------

def build_primary_prism(pts, normal, thickness, extrusion, start_ext=None, end_ext=None):
    """
    Build a flat-faced prism along the polyline.

    For each segment a quad face is created (one side of the ribbon),
    then the whole mesh is extruded along *normal* by *extrusion* and
    centred on the base plane.
    """
    n_segs = len(pts) - 1
    if start_ext is None:
        start_ext = [0.0] * n_segs
    if end_ext is None:
        end_ext = [0.0] * n_segs

    verts = []
    faces = []

    for i in range(n_segs):
        p0 = pts[i].copy()
        p1 = pts[i + 1].copy()
        seg_vec = safe_normalize(p1 - p0)
        p0 = p0 - seg_vec * start_ext[i]
        p1 = p1 + seg_vec * end_ext[i]

        thick_vec = np.cross(normal, seg_vec)
        thick_vec = -safe_normalize(thick_vec) * thickness

        pA = p0
        pB = p0 + thick_vec
        pC = p1 + thick_vec
        pD = p1

        base = len(verts)
        verts += [pA, pB, pC, pD]
        faces.append((base, base + 1, base + 2, base + 3))

    return _extrude_flat_mesh(verts, faces, normal, extrusion)


def build_offset_prism(pts, normal, seg_index, length, width_inplane, extrude_outofplane, offset):
    """Build the secondary or tertiary prism offset from one segment."""
    if seg_index < 0 or seg_index >= len(pts) - 1:
        return None

    p0 = pts[seg_index]
    p1 = pts[seg_index + 1]
    seg_vec = safe_normalize(p1 - p0)
    plane_vec = safe_normalize(np.cross(normal, seg_vec))

    center = (p0 + p1) / 2.0 + plane_vec * offset
    half_len = seg_vec * (length / 2.0)
    half_w = plane_vec * (width_inplane / 2.0)

    pA = center - half_len - half_w
    pB = center + half_len - half_w
    pC = center + half_len + half_w
    pD = center - half_len + half_w

    verts = [pA, pB, pC, pD]
    faces = [(0, 1, 2, 3)]

    return _extrude_flat_mesh(verts, faces, normal, extrude_outofplane)


def _extrude_flat_mesh(verts_np, faces, normal, extrusion):
    """
    Given a list of numpy vert positions and face index tuples,
    create a closed solid by extruding along *normal* and centre it.
    Returns (vertices_list, faces_list) as plain Python lists of tuples.
    """
    offset = normal * (extrusion / 2.0)
    bottom = [v - offset for v in verts_np]
    top    = [v + offset for v in verts_np]
    n = len(verts_np)

    all_verts = [tuple(v) for v in bottom] + [tuple(v) for v in top]

    all_faces = []
    # bottom cap (reversed winding for outward normal)
    for f in faces:
        all_faces.append(tuple(reversed(f)))
    # top cap (shifted by n)
    for f in faces:
        all_faces.append(tuple(i + n for i in f))
    # side walls
    for f in faces:
        ring = list(f)
        for k in range(len(ring)):
            a = ring[k]
            b = ring[(k + 1) % len(ring)]
            all_faces.append((a, b, b + n, a + n))

    return all_verts, all_faces


def build_cylinder_pair(pts, normal, seg_index, offset_along, radius, height, offset_from_plane):
    """Return (verts, faces) for two cylinders above/below the segment point."""
    if seg_index < 0 or seg_index >= len(pts) - 1:
        return None

    p0 = pts[seg_index]
    p1 = pts[seg_index + 1]
    seg_vec = safe_normalize(p1 - p0)

    anchor = p0 + offset_along * (p1 - p0)

    y_local = seg_vec                                    # cylinder axis
    z_local = safe_normalize(normal)
    x_local = safe_normalize(np.cross(y_local, z_local))
    z_local = safe_normalize(np.cross(x_local, y_local))

    center1 = anchor + normal * offset_from_plane
    center2 = anchor - normal * offset_from_plane

    all_verts = []
    all_faces = []

    for center in (center1, center2):
        v, f = _make_cylinder(center, x_local, y_local, z_local, radius, height)
        base = len(all_verts)
        all_verts += v
        all_faces += [tuple(i + base for i in face) for face in f]

    return all_verts, all_faces


def _make_cylinder(center, x_local, y_local, z_local, radius, height, resolution=20):
    """Cylinder with axis along y_local, centred at *center*."""
    verts = []
    faces = []
    half_h = height / 2.0

    # bottom circle then top circle
    for side in (-1, 1):
        for k in range(resolution):
            angle = 2 * math.pi * k / resolution
            local_pos = (radius * math.cos(angle) * x_local +
                         side * half_h * y_local +
                         radius * math.sin(angle) * z_local)
            verts.append(tuple(center + local_pos))

    # side quads
    for k in range(resolution):
        a = k
        b = (k + 1) % resolution
        c = b + resolution
        d = a + resolution
        faces.append((a, b, c, d))

    # bottom cap
    bottom_center = len(verts)
    verts.append(tuple(center - half_h * y_local))
    for k in range(resolution):
        a = k
        b = (k + 1) % resolution
        faces.append((bottom_center, b, a))

    # top cap
    top_center = len(verts)
    verts.append(tuple(center + half_h * y_local))
    for k in range(resolution):
        a = k + resolution
        b = (k + 1) % resolution + resolution
        faces.append((top_center, a, b))

    return verts, faces


# ---------------------------------------------------------------------------
# Blender object helpers
# ---------------------------------------------------------------------------

def _upsert_mesh_object(name, verts, faces):
    """Create or replace a Blender mesh object with the given geometry."""
    # Remove old object/mesh if it exists
    if name in bpy.data.objects:
        old_obj = bpy.data.objects[name]
        bpy.data.objects.remove(old_obj, do_unlink=True)
    if name in bpy.data.meshes:
        bpy.data.meshes.remove(bpy.data.meshes[name])

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


# ---------------------------------------------------------------------------
# Scene properties (stored on the scene so values persist)
# ---------------------------------------------------------------------------

class FemGuideProperties(bpy.types.PropertyGroup):
    angle_increments: bpy.props.StringProperty(
        name="Angle Increments (deg)",
        description="Comma-separated angle increments for each segment",
        default="8,40,45,45,42",
    )
    lengths: bpy.props.StringProperty(
        name="Lengths",
        description="Comma-separated segment lengths",
        default="50,20.5,25,11,30",
    )
    start_extensions: bpy.props.StringProperty(
        name="Start Extensions",
        description="Comma-separated start extension per segment",
        default="",
    )
    end_extensions: bpy.props.StringProperty(
        name="End Extensions",
        description="Comma-separated end extension per segment",
        default="",
    )
    primary_thickness: bpy.props.FloatProperty(
        name="Primary Thickness",
        description="Half-width of primary ribbon (one direction)",
        default=1.0, min=0.1, max=50.0,
    )
    primary_extrusion: bpy.props.FloatProperty(
        name="Primary Extrusion Length",
        default=5.0, min=0.0, max=1000.0,
    )

    # Secondary
    use_secondary: bpy.props.BoolProperty(name="Generate Secondary Line", default=False)
    secondary_seg_index: bpy.props.IntProperty(name="Secondary Segment Index", default=0, min=0, max=100)
    secondary_length: bpy.props.FloatProperty(name="Secondary Length", default=20.0, min=0.1, max=500.0)
    secondary_extrusion: bpy.props.FloatProperty(name="Secondary Extrusion", default=2.0, min=0.1, max=100.0)
    secondary_offset: bpy.props.FloatProperty(name="Secondary Offset", default=0.0, min=-500.0, max=500.0)
    secondary_thickness: bpy.props.FloatProperty(name="Secondary Width (in-plane)", default=1.0, min=0.1, max=50.0)

    # Tertiary
    use_tertiary: bpy.props.BoolProperty(name="Generate Tertiary Line", default=False)
    tertiary_seg_index: bpy.props.IntProperty(name="Tertiary Segment Index", default=0, min=0, max=100)
    tertiary_length: bpy.props.FloatProperty(name="Tertiary Length", default=20.0, min=0.1, max=500.0)
    tertiary_extrusion: bpy.props.FloatProperty(name="Tertiary Extrusion", default=2.0, min=0.1, max=100.0)
    tertiary_offset: bpy.props.FloatProperty(name="Tertiary Offset", default=0.0, min=-500.0, max=500.0)
    tertiary_thickness: bpy.props.FloatProperty(name="Tertiary Width (in-plane)", default=1.0, min=0.1, max=50.0)

    # Cylinders
    use_cylinders: bpy.props.BoolProperty(name="Generate Cylinders", default=False)
    cylinder_seg_index: bpy.props.IntProperty(name="Cylinder Segment Index", default=0, min=0, max=100)
    cylinder_offset_along: bpy.props.FloatProperty(
        name="Cylinder Offset Along [0-1]",
        description="Fractional position along segment (0=start, 1=end)",
        default=0.5, min=0.0, max=1.0,
    )
    cylinder_offset_from_plane: bpy.props.FloatProperty(
        name="Cylinder Offset From Plane",
        default=5.0, min=0.0, max=500.0,
    )
    cylinder_radius: bpy.props.FloatProperty(name="Cylinder Radius", default=2.0, min=0.1, max=100.0)
    cylinder_height: bpy.props.FloatProperty(name="Cylinder Height", default=10.0, min=0.1, max=500.0)


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

class FEMGUIDE_OT_Generate(bpy.types.Operator):
    bl_idname = "femguide.generate"
    bl_label = "Generate Polyline"
    bl_description = "Generate the surgical guidance geometry"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.femguide

        # --- Parse inputs ---
        try:
            angles  = [float(a.strip()) for a in props.angle_increments.split(",") if a.strip()]
            lengths = [float(l.strip()) for l in props.lengths.split(",") if l.strip()]
        except ValueError:
            self.report({'ERROR'}, "Invalid angle or length values.")
            return {'CANCELLED'}

        if len(angles) != len(lengths) or not angles:
            self.report({'ERROR'}, "Angle and length lists must be the same non-zero length.")
            return {'CANCELLED'}

        n_segs = len(angles)

        try:
            start_ext = [float(v.strip()) for v in props.start_extensions.split(",") if v.strip()]
        except ValueError:
            start_ext = []
        if len(start_ext) != n_segs:
            start_ext = [0.0] * n_segs

        try:
            end_ext = [float(v.strip()) for v in props.end_extensions.split(",") if v.strip()]
        except ValueError:
            end_ext = []
        if len(end_ext) != n_segs:
            end_ext = [0.0] * n_segs

        # --- Geometry ---
        pts    = generate_polyline_points(angles, lengths)
        normal = compute_plane_normal(pts)

        # Primary
        verts, faces = build_primary_prism(
            pts, normal,
            props.primary_thickness,
            props.primary_extrusion,
            start_ext, end_ext,
        )
        _upsert_mesh_object("LineGeneratorModel", verts, faces)

        # Secondary
        if props.use_secondary:
            result = build_offset_prism(
                pts, normal,
                props.secondary_seg_index,
                props.secondary_length,
                props.secondary_thickness,
                props.secondary_extrusion,
                props.secondary_offset,
            )
            if result:
                _upsert_mesh_object("LineGeneratorSecondary", *result)
        else:
            # Remove stale object if unchecked
            if "LineGeneratorSecondary" in bpy.data.objects:
                bpy.data.objects.remove(bpy.data.objects["LineGeneratorSecondary"], do_unlink=True)

        # Tertiary
        if props.use_tertiary:
            result = build_offset_prism(
                pts, normal,
                props.tertiary_seg_index,
                props.tertiary_length,
                props.tertiary_thickness,
                props.tertiary_extrusion,
                props.tertiary_offset,
            )
            if result:
                _upsert_mesh_object("LineGeneratorTertiary", *result)
        else:
            if "LineGeneratorTertiary" in bpy.data.objects:
                bpy.data.objects.remove(bpy.data.objects["LineGeneratorTertiary"], do_unlink=True)

        # Cylinders
        if props.use_cylinders:
            result = build_cylinder_pair(
                pts, normal,
                props.cylinder_seg_index,
                props.cylinder_offset_along,
                props.cylinder_radius,
                props.cylinder_height,
                props.cylinder_offset_from_plane,
            )
            if result:
                _upsert_mesh_object("LineGeneratorCylinders", *result)
        else:
            if "LineGeneratorCylinders" in bpy.data.objects:
                bpy.data.objects.remove(bpy.data.objects["LineGeneratorCylinders"], do_unlink=True)

        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class FEMGUIDE_PT_Panel(bpy.types.Panel):
    bl_label = "FemGuide Line Generator"
    bl_idname = "FEMGUIDE_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "FemGuide"

    def draw(self, context):
        layout = self.layout
        props = context.scene.femguide

        # --- Primary ---
        box = layout.box()
        box.label(text="Primary Line", icon='CURVE_PATH')
        box.prop(props, "angle_increments")
        box.prop(props, "lengths")
        box.prop(props, "start_extensions")
        box.prop(props, "end_extensions")
        box.prop(props, "primary_thickness")
        box.prop(props, "primary_extrusion")

        # --- Secondary ---
        box = layout.box()
        box.prop(props, "use_secondary")
        if props.use_secondary:
            box.prop(props, "secondary_seg_index")
            box.prop(props, "secondary_length")
            box.prop(props, "secondary_thickness")
            box.prop(props, "secondary_extrusion")
            box.prop(props, "secondary_offset")

        # --- Tertiary ---
        box = layout.box()
        box.prop(props, "use_tertiary")
        if props.use_tertiary:
            box.prop(props, "tertiary_seg_index")
            box.prop(props, "tertiary_length")
            box.prop(props, "tertiary_thickness")
            box.prop(props, "tertiary_extrusion")
            box.prop(props, "tertiary_offset")

        # --- Cylinders ---
        box = layout.box()
        box.prop(props, "use_cylinders")
        if props.use_cylinders:
            box.prop(props, "cylinder_seg_index")
            box.prop(props, "cylinder_offset_along")
            box.prop(props, "cylinder_offset_from_plane")
            box.prop(props, "cylinder_radius")
            box.prop(props, "cylinder_height")

        layout.operator("femguide.generate", icon='MESH_DATA')


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

classes = (
    FemGuideProperties,
    FEMGUIDE_OT_Generate,
    FEMGUIDE_PT_Panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.femguide = bpy.props.PointerProperty(type=FemGuideProperties)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.femguide


if __name__ == "__main__":
    register()
