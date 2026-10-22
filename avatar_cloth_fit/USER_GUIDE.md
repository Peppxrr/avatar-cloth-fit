# Avatar Cloth Fit user guide

[Project overview](https://github.com/Peppxrr/avatar-cloth-fit) ·
[Download](https://github.com/Peppxrr/avatar-cloth-fit/releases/latest)

For Blender 5.0 and newer. Start with a weighted avatar body and clothing
placed roughly around it. Save your scene before fitting.

## Contents

- [Quick start](#quick-start)
- [Assisted Fit](#assisted-fit): sculpt brushes, Adjust Region, settings and recipes
- [Weight fitting](#weight-fitting): settings, troubleshooting, hardware and export rigs
- [Practical limits](#practical-limits)

## Quick start

1. Put the clothing roughly where it belongs on the avatar (Object Mode).
2. In **Cloth Fit**, set **Body** (the weighted body mesh, not eyes or
   accessories), **Armature** (optional when the body has an Armature
   modifier) and **Clothing**. For an outfit split into several meshes, select
   them all and click **Use selected clothing**.
3. Fix the shape with **Assisted Fit**: click **Sculpt Clothing** and use the
   brushes (a sleeve too narrow, a hem cutting into the thigh, lumps from a
   grab, a collar too high). See [Assisted Fit](#assisted-fit).
4. Click **Fit & Refine Weights**. **Esc** cancels and restores everything.
5. Read the status line and the **Report** line under it (click either to
   expand the details: poses checked, library poses, warnings), click
   **Select Problems** to see vertices that still clip or fold, fix them
   (Assisted Fit, part marks, settings) and run again. Hints in the panels
   are collapsed the same way.
6. For separate export, open **Clothing Armature** and click
   **Create Clothing Armature** (before or after fitting).

Every operation backs up the clothing first. **Restore Backup** returns to the
state before the last fit, part repair or clothing armature, or before the
first Assisted Fit adjustment since then. Blender's Undo works too.

## Assisted Fit

Assisted Fit is precise, non-destructive hand-fitting made for clothing. It
does not guess anatomy or resize the whole garment for you; you decide what
moves, and it keeps the result clean. There are two ways in:

* **Sculpt Clothing** (Sculpt Mode) — brushes, for most fitting work.
* **Adjust Region** (Edit Mode) — move, rotate or scale an exact selection.

### Sculpt Clothing (Sculpt Mode)

1. Select the clothing and click **Sculpt Clothing** (Cloth Fit tab →
   Assisted Fit). It backs up the mesh once, makes its **Cloth Fit Assisted**
   shape key the one you sculpt (so the Basis stays the original), enters
   Sculpt Mode and picks **Smart Grab**.
2. The four Cloth Fit tools sit at the bottom of the Sculpt Mode toolbar and
   in the panel. Blender's own brushes keep working on the same shape key, so
   you can mix them freely. The Cloth Fit brushes work on the shape you see —
   the Basis plus every active shape key, such as the **Cloth Fit Correction**
   from a weight fit — and store the change in the sculpted key only.
3. Every stroke is one Blender undo step (**Ctrl+Z**).

Common controls, as with Blender's brushes:

| Control | Action |
| --- | --- |
| **F** / **Shift+F** | Change radius / strength in the viewport |
| **Shift+drag** | Shape Relax with any Cloth Fit brush |
| **Mirror X/Y/Z** | Sculpt Mode's symmetry toggles (header or panel); mirrored strokes keep the symmetry plane intact |
| **Mask** (Blender's mask brush, **Ctrl** box/lasso mask) | Masked fabric does not move; partial mask moves partially |
| **Pin group** | Vertices in this vertex group never move |
| **Esc** / right click during a stroke | Cancel the stroke |

The **Topology** automasking toggle in Sculpt Mode (also in the Cloth Fit
tool header) controls brush reach:

* **On:** reach follows the surface of the connected piece under the cursor.
  Separate islands and nearby layers stay put. Protected hardware follows
  the fabric it is attached to and stays rigid.
* **Off:** reach uses straight-line distance, like Blender's brushes, so nearby
  layers and separate pieces can move together. Hardware in reach is brushed
  directly and kept rigid.

Masks and the pin group still restrict what can move. Adjust Region always
uses connected fabric, independently of this Sculpt Mode toggle.

#### Smart Grab

Behaves like Blender's Elastic Grab, made for clothing: the spot you grab
follows the mouse exactly and the pull fades softly with distance — about half
strength at the brush circle, with a gentle tail out to three radii. With
**Topology** on, distance follows the fabric; with it off, nearby layers and
separate pieces within reach can move too.
The result depends only on where the mouse is, so dragging back restores the
fabric. **Shape preservation** additionally solves the inner part
as-rigid-as-possible against the stroke-start shape, so folds and spacing
survive larger pulls and fewer triangles get stretched or flipped.

#### Fit to Body

Paint where the clothing should sit on the avatar. Each dab moves the brushed
fabric a share (**Strength**) of the way toward the target **Clearance** from
the **Body** chosen in the main panel, along the body's surface normal —
gently, not a shrinkwrap of the whole outfit.

* **Direction**: **Fit** (both ways), **Tighten** (only pull loose fabric in),
  **Loosen** (only push fabric that is too close out), **Resolve clipping**
  (only push fabric that is inside the body out).
* Fabric whose nearest body point is unclear — seen from the side, standing
  across the skin, or between two body parts such as the legs — is not snapped
  to it; it follows its reliable neighbors. The status bar reports how much of
  a stroke was ambiguous.
* **Reach**: fabric farther than this from the body is left alone.

#### Shape Relax

The everyday clean-up brush. It evens out lumps and uneven vertex spacing
while keeping folds, curvature and the silhouette. It compares the fabric with
a *reference* shape (stretched to the brush area's current size, so a sleeve
you widened stays wide) and smooths only the difference: lumps you added go,
the reference's folds stay, and smooth changes such as fabric pushed out from
the body by Fit to Body or Smart Grab are kept instead of pulled back in. It
is not Laplacian smoothing, which shrinks and flattens.

With a **Body** set it also never pulls fabric closer to the skin than the
**Clearance** (or than it already was), so relaxing skin-tight clothing does
not sink it into the body.

The reference is, in order: the Basis (when sculpting the Cloth Fit key — the
**Sculpt Clothing** setup), a shape stored with **Store Reference** (or
captured automatically before your first Cloth Fit stroke on a mesh without
shape keys), or the Cloth Fit backup. The panel shows which one is used.
Without any, it falls back to non-shrinking smoothing, which removes lumps but
cannot restore folds.

#### Region Resize

Change the width of a sleeve, pant leg, waistband or strap without scaling
the outfit. Click a **face set** (create them in Sculpt Mode with the Face
Set tools, e.g. **Face Sets → From Loose Parts**, or draw one); without face
sets the whole clicked piece is used, minus masked fabric. The region gets a
centerline (orange) with cross-section handles (squares and rings).

| Key / mouse | Action |
| --- | --- |
| Drag empty space left/right | Wider / narrower (all sections, or the selected one) |
| Click a handle, drag | Select it and move that section (bends the path, e.g. a digitigrade leg) |
| **S** | Scale the selected section (or all) by mouse distance; click to apply |
| **G** | Move the selected section |
| **A** | Add a section handle at the mouse |
| **X** / **Delete** | Remove the selected handle (two always remain) |
| **Tab** | Select the next handle |
| **Mouse wheel** | Wider / shorter fade into the surrounding fabric |
| **E** | Pin region edges: the border stays and the resize fades in from it |
| **D** | Keep fold depth on/off |
| **Enter** / **Space** | Confirm (the fade zone is then smoothed once) |
| **Esc** / right click | Undo the current drag, or cancel the whole resize |

**Keep fold depth** (default on) scales only the smooth width: folds, seams
and pleats keep their depth instead of growing with the region, which keeps
the silhouette believable. Turn it off to scale everything uniformly.
Lengthwise the region is never stretched.

#### Brush settings

| Setting | Default | What it does | When to change it |
| --- | --- | --- | --- |
| **Radius** | 80 px | Brush size on screen (**F**). | Larger for broad reshaping; smaller near seams and hardware. |
| **Strength** | 0.5 | Share of the change each dab applies (Fit to Body, Shape Relax) (**Shift+F**). Smart Grab always follows the mouse. | Lower for gentle, controllable passes. |
| **Falloff** | Smooth | Fade from the brush center (or region border for Region Resize). Smart Grab uses its own elastic fade. | Sphere for firmer dabs; Sharp for a precise tip. |
| **Shape preservation** (Smart Grab) | 0.6 | 0 = plain elastic grab; toward 1 the fabric keeps its local shape (folds, spacing, triangle sizes) as it follows the pull. | Lower for the plain elastic feel; raise on tight, dense or detailed fabric. |
| **Direction** (Fit to Body) | Fit | Which way fabric may move toward the clearance. | Resolve clipping for a safe clean-up pass; Tighten for baggy areas. |
| **Clearance** (Fit to Body, Shape Relax) | 3 mm | Target gap to the skin; Shape Relax never pulls fabric closer than this (or than it was). | 1–2 mm for skin-tight wear; 5–10 mm for thick fabric. |
| **Reach** (Fit to Body) | 8 cm | Fabric farther from the body is ignored. | Raise for loose garments you want to bring in. |
| **Fade** (Region Resize) | 4 cm | How far the resize fades into the fabric around the region (or inside it with Pin edges). | Longer for smooth transitions on large garments. |
| **Pin edges** / **Fold depth** (Region Resize) | off / on | See above. | |
| **Smoothing** (Region Resize, Adjust Region) | 8 passes | One as-rigid-as-possible pass over the fade zone after you confirm; kept only if it flips no faces. | 0 to keep the exact fade. |

Requirements: the avatar must be in **Rest Position** (fitting happens in the
rest pose); Dyntopo and Multiresolution are not supported (the tools keep the
mesh topology).

### Adjust Region (Edit Mode)

1. Select the clothing, press **Tab** for Edit Mode and select the region to
   reshape (vertices, edges or faces; **L** selects a whole linked piece).
   Selected vertices follow the control exactly.
2. Click **Adjust Region** in the **Assisted Fit** panel (also in the Edit Mode
   right-click menu as **Assisted Fit: Adjust Region**).
3. Move the mouse. The avatar is shown in its rest pose while you adjust.

| Key | Action |
| --- | --- |
| **G** / **R** / **S** | Move / rotate / scale (switch any time; earlier changes are kept) |
| **X**, **Y**, **Z** | Constrain to an axis (press again to clear) |
| **Shift+X/Y/Z** | Constrain to the plane without that axis (e.g. scale a ring wider without lengthening it) |
| **Mouse wheel**, **Page Up/Down**, **+/-** | Grow / shrink the falloff radius (the orange circle) |
| **F** | Cycle the falloff shape |
| **Enter**, **Space**, **left click** | Confirm |
| **Esc**, **right click** | Cancel (the clothing is restored exactly) |

4. After you confirm, a smoothing pass evens out stretching in the fade zone
   (the slow part runs once here, never while dragging). The status box reports
   how many vertices moved and the largest stretch in the fade zone.
5. Exact values can be typed afterwards in Blender's **Adjust Last Operation**
   panel (bottom left) — move, rotate, scale, falloff, shape, pivot, axes and
   smoothing passes all redo the adjustment.

#### Adjust Region settings

| Setting | Default | What it does | When to change it |
| --- | --- | --- | --- |
| **Falloff radius** | 5 cm | How far, measured *along the fabric*, the change fades out beyond the selection. 0 moves only the selection. | Larger for broad, smooth changes (loosening a whole sleeve); smaller for local fixes (a clipping hem). If the status reports a large stretch, widen it. |
| **Falloff** | Smooth | Fade curve. **Smooth**: soft start and end, best default. **Sphere**: stays strong, then drops. **Linear**: even. **Sharp**: concentrated near the selection. **Constant**: everything inside the radius moves fully (hard edge). | Sphere to move a big area almost rigidly; Sharp for a gentle nudge. |
| **Pivot** | Selection center | Center for rotate and scale. | **3D cursor** to rotate a collar about the neck or a sleeve about the shoulder: snap the cursor there first. |
| **Axes** | Global | What X/Y/Z mean for constraints and per-axis scale. **Local** uses the clothing object's axes. **Selection shape** uses the selection's own extents: X = longest, Z = thinnest. | Use **Selection shape** on a ring around a limb: Z then runs along the limb, so **S, Shift+Z** widens a cuff or sleeve without making it longer. |
| **Smoothing** | 8 passes | After confirming, an as-rigid-as-possible solve spreads stretch evenly through the fade zone while keeping folds and surface detail. It is kept only if it flips no faces and worsens neither average nor maximum stretch; otherwise the plain falloff result is kept. 0 = off. | 0 when you want the exact falloff curve; 15–25 for big moves on dense meshes. |
| **Pin group** | `ACF Pinned` | Vertices in this vertex group never move; fabric between the selection and the pins fades smoothly into them. Any existing group can be chosen. | Pin a waistband, collar seam or the part that already fits. |
| **Pin / Unpin / Select** | | Edit Mode: add the selected vertices to the pin group, remove them, or select the pinned vertices. | |
| **Adjustment strength** | 1 | Slider of the **Cloth Fit Assisted** shape key: 0 shows the original shape, 1 the adjusted one. The trash button deletes all adjustments. | Compare before/after, or tone everything down. |

### What Adjust Region protects

These isolation rules apply to Adjust Region. For Sculpt Mode brushes,
see the **Topology** toggle above.

* **Separate pieces.** The change spreads only through connected fabric. A
  lining, pocket, strap or second layer that is a separate mesh island stays
  put unless you select it too — even if it is a millimeter away. Select every
  island that should move (**L** on each).
* **Accessories.** Islands marked with **Mark Hardware / Accessory** (Belts &
  Hardware panel), or small detached metal pieces found by **Detect metal
  hardware**, never bend or scale. They move as one rigid piece with the fabric
  they sit on (or with the control when selected). To resize one on purpose,
  clear its mark first. A pinned vertex on an accessory keeps it fixed.
* **Detail and volume.** Rotation and scale are interpolated, not blended, so a
  partly rotated region keeps its distance from the pivot instead of shrinking,
  and folds keep their shape. Falloff distances follow the surface and do not
  depend on how evenly the mesh is triangulated.
* **Everything else.** The Basis, other shape keys, UVs, materials, vertex
  groups and weights never change. All adjustments accumulate in one
  **Cloth Fit Assisted** shape key relative to the Basis, which the weight fit
  then treats as the clothing's shape.

### Recipes

* **Lumpy tight clothing after grabbing:** Shape Relax over the lumps (or
  Shift+drag with any Cloth Fit brush), Strength 0.5, a radius a bit larger
  than the lumps.
* **Pulling fabric into place:** Smart Grab instead of Elastic Grab; mask or
  pin what already fits.
* **Clipping into the body:** Fit to Body with Direction **Resolve clipping**,
  then Shape Relax.
* **Leggings or a top that should hug the body:** Fit to Body, Direction
  **Fit**, Clearance 1–2 mm, light strokes.
* **Sleeve, pant leg or waistband too tight or loose:** Region Resize on its
  face set; drag sideways, or **S** on one handle for a local bulge.
* **Pants on digitigrade legs:** Region Resize on the pant leg, **A** to add
  handles at the knee/hock, then click-drag each handle to follow the leg.
* **Exact moves:** Adjust Region in Edit Mode — select a ring of loops
  (**Alt+click**), **Axes = Selection shape**, **S Shift+Z** widens a sleeve
  without lengthening it; snap the 3D cursor to the neck and **R** a collar.
* **Big change, ugly fade:** widen the falloff/fade, or make two smaller moves.

### Limits

Assisted Fit is a direct shaping tool, not a simulation or automatic fitter:
it does not detect anatomy or resize the whole garment, and only Fit to Body
looks at the avatar. Compressing fabric can fold it; the smoothing passes keep
only the part of a correction that folds nothing. Edits are made in the rest
pose; absolute shape keys and linked (library) meshes are not supported. Relative shape-key mixes are respected; a muted or zero-value sculpted key
must be enabled before brushing.
After reshaping, run **Fit & Refine Weights** again so clipping is re-checked.

## Weight fitting

### Buttons

* **Use selected clothing** — treat all selected clothing meshes as one outfit
  (each object stays separate). The body is excluded automatically.
* **Fit & Refine Weights** — back up, transfer weights, test poses, refine and
  report. **Esc** cancels and restores clothing, weights and rig pose. Dense
  meshes can take a minute or more. An empty Armature modifier on an imported
  garment is bound automatically.
* **Restore Backup** — see Quick start. Hidden backups live in the
  **Avatar Cloth Fit Backups** collection; keep them until you are happy.
* **Select Problems** — enter Edit Mode with the vertices that still clip,
  cross themselves, fold, collapse or change weight more abruptly than the body
  beneath them. Markers are a mesh attribute, not a vertex group, so Weight
  Paint stays on your active bone group.

### Settings reference

Distances in percent are relative to the avatar's largest dimension
(1% ≈ 1.8 cm on a 1.8 m avatar), so the defaults work regardless of units.

| Setting | Default | What it does | Try |
| --- | --- | --- | --- |
| **Bone influences** | 4 | Maximum bones per vertex. Protected or locked weights can require a higher limit on affected vertices; that is reported. | Keep 4 for Unity/VRChat. |
| **Refinement passes** | 3 | Maximum weight-refinement passes; stops early when weights settle. | 4–6 if clipping remains in poses; 1–2 for quick previews. |
| **Pose validation** | Automatic | **Automatic** screens a large joint library with an exact linear-blend prediction and checks core and problem poses with Blender's deformation. **Thorough** checks every relevant pose with Blender (much slower). | Thorough for the final check of an important garment. |
| **Clearance** | 0.08% | Desired gap between body and clothing (≈1.4 mm). | 0.15–0.3% for thick fabric, puffer jackets or when clipping shows through at a distance. |
| **Probe secondary bones** | On | Tests breast, butt, belly, thigh, tail, ear, clothing and jiggle/PhysBone chains with fixed-angle pose probes. Unity PhysBone simulation is not run. | Off only to save time when the avatar has none. |
| **Check self-intersections** | On | Tests posed clothing for newly crossing triangles and pushes them apart. | Off for quick previews only. |
| **Detect metal hardware** | On | Small detached islands with metal material names or a constant metallic shader count as hardware (shared attachment weights; rigid in Assisted Fit and the brushes). A material the garment's main piece also uses does not count (imports often make every material metallic). | Off if a shiny fabric piece is misdetected; mark real hardware explicitly instead. |
| **Correct rest clipping** | Off | Pushes slightly clipping clothing outward in the rest pose, into a separate **Cloth Fit Correction** shape key. | On when small rest-pose clipping remains; use Assisted Fit for larger shape problems. |
| **Max correction** | 0.5% | Largest push of any vertex by the correction (≈9 mm). | 0.2% to keep the design exact; 1–1.5% for sloppier imports. |
| **Strap smoothing** (Belts & Hardware) | 1.8% | Blend radius of weights along marked belts/straps. | 2.5–4% if a belt still kinks at the hips; lower if it slides too much. |
| **Min bone weight** (Clothing Armature) | 0.01 | Bones below this weight on every vertex are merged into their parent when creating the clothing armature. | 0.03–0.05 for fewer bones; 0 to keep every touched bone. |
| **Transition smoothing** (Advanced) | 1% | Smoothing radius of transferred weights along the garment. Loose or ambiguous fabric is smoothed further automatically; skin-tight fabric keeps the body's creases. | 1.5–2.5% for blotchy patches or hip weight leaking onto legs; 0.3–0.6% when tight clothing does not follow the body closely enough. 0 = raw nearest-surface weights. |
| **Collision strength** (Advanced) | 1.5 | How hard refinement pushes clipping clothing out versus keeping weights smooth. | 2–3 for stubborn clipping; 1 if refinement makes weights noisy. |
| **Search range** (Advanced) | 8% | Distance within which body collisions are measured; farther vertices are reported as distant. | 12–20% for long skirts, capes or very loose garments. |
| **Reverse pose facing** (Advanced) | Off | Flips generic forward/back pose directions. | On if test poses bend the wrong way (rig faces +Y). |
| **Preserve clothing-only bones** (Advanced) | On | Keeps existing weights on deform bones the body does not use (skirt, cape, strap chains). | Off to let the body's weights replace them. |

### Improving a result

| Problem | Try |
| --- | --- |
| Clothing already clips in the rest pose | Assisted Fit to move the area out; **Correct rest clipping** for millimeter-level clipping. |
| Clips in poses (armpits, groin, knees) | More **Refinement passes** (4–6), **Collision strength** 2–3, slightly higher **Clearance**; check with **Select Problems**. |
| Patchy or blotchy weights, hip weight on legs | Raise **Transition smoothing** to 1.5–2.5%. Make sure **Body** is the skin mesh. |
| Tight clothing slides over joints instead of following them | Lower **Transition smoothing** to 0.3–0.6%. |
| Skirt or coat tears between the legs | Raise **Transition smoothing**; give the skirt its own bones on the avatar and keep **Preserve clothing-only bones** on. |
| Belt or strap kinks | **L**, **Mark Belt / Strap**, then raise **Strap smoothing**. |
| Buttons, buckles or rings bend | **L**, **Mark Hardware / Accessory**. |
| Breasts, belly or butt poke through when they jiggle | Keep **Probe secondary bones** on; the avatar's PhysBone bones must weight the body there. |
| Fit takes too long | Automatic validation, 2–3 passes; split very dense garments. |
| Poses bend the wrong way | **Reverse pose facing**. |

After any change, run **Fit & Refine Weights** again; it always starts from the
current shape and the original weights in a fresh transaction.

### Belts, buttons and rings

For a flexible belt or strap, enter Edit Mode, hover it, press **L** and click
**Mark Belt / Strap**. Its weights are blended along the strap and through its
thickness while keeping joint motion and protected strap weights.

For buttons, buckles, rings and other detached pieces, click
**Mark Hardware / Accessory**. Each piece receives one shared, normalized weight
blend from its closest fabric contact (not just the dominant bone), so it
follows the fabric without bending, and Assisted Fit moves it rigidly.
**Detect metal hardware** finds pieces with metal material names or a constant
metallic shader automatically, ignoring materials the main piece also uses. Inspect detected
parts before fitting; mark hardware explicitly if needed.
**Clear Marks** removes marks.
**Apply Belt & Hardware Weights** repairs just these parts on the current
weights (Object Mode); rerun Fit & Refine for a fresh collision report.

Locked weights always win. Pieces with varying protected weights, or without
any fabric contact, are skipped and counted rather than pinned to a guessed bone.

### Clothing armature for separate export (VRCFury)

**Create Clothing Armature** duplicates the avatar armature (or the one already
deforming the clothing), parents the clothing to it, points its Armature
modifier at it and deletes every bone the clothing does not need — the usual
manual workflow for exporting clothing separately and attaching it with e.g.
VRCFury **Armature Link**. Bone names, rest positions and hierarchy match the
avatar; the avatar is never changed.

Before a fit, needed bones come from the body weights under the clothing;
after a fit, from the clothing's own weights. A bone stays when it carries at
least **Min bone weight** on any vertex; its ancestors always stay. A lighter
bone's weight is merged into its nearest kept parent first, so the weights remain normalized. Pruning can change deformation;
inspect the result in poses before export. Locked groups keep their bones. Other selected meshes on the same
armature share the new armature. **Restore Backup** undoes it and deletes the
armature once nothing uses it. The choice is stored as JSON on the armature
(`acf_bone_report`) and in a **Cloth Fit Armature – …** text.

### Reports

The full JSON report of each fit is the **Cloth Fit Report – …** text in the
Text Editor: poses, mapped bones, `before`/`after` clearance and deformation
measurements, screened and promoted library poses, warnings and timings.
Save the `.blend` to keep weights, keys, backups and reports.

## Practical limits

* This is a pose-guided weight fitter and a manual shaping tool, not a cloth
  simulation. Existing rest-pose self-intersections, complex layered dresses
  and freely swinging skirts can still need mesh edits or dedicated bones.
* Blender does not run Unity PhysBones; secondary bones are swung by fixed
  angles. Test the final avatar in Unity/VRChat.
* The body should be a clean, outward-facing surface; signed clearance uses the
  nearest triangle's normal.
* Screened library poses are evidence, not a certificate; use **Thorough** to
  validate them all.
* Non-armature modifiers are ignored during analysis and kept. Segmented
  B-Bones, masked/envelope/multi-armature deformation, linked/library override
  objects and absolute shape keys are rejected rather than handled incorrectly.
* Each fit considers only the chosen body and clothing; other garments in the
  scene are neither colliders nor modified.

