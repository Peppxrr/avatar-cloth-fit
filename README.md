# Avatar Cloth Fit

Fit clothing to a weighted avatar in Blender, then transfer and refine its skin weights.
Made for avatar workflows such as VRChat and VRM.

## What it does

- **Shape clothing by hand** with Smart Grab, Fit to Body, Shape Relax,
  Region Resize, and an Edit Mode Adjust Region tool.
- **Transfer and refine weights** from the avatar body, checking clothing
  in a range of poses with Blender's armature deformation.
- **Work with separate clothing pieces**, belts, straps, and rigid hardware.
- **Keep edits recoverable** with a separate shape key for Assisted Fit,
  backups, and restore controls.
- **Create a clothing armature** for exporting clothing separately.

Runs locally inside Blender. No external services or extra Python packages.

## Download and install

Requires **Blender 5.0 or newer**. Release checks use Blender 5.0.1.

1. Download **`avatar_cloth_fit-1.8.4.zip`** from the
   [latest release](https://github.com/Peppxrr/avatar-cloth-fit/releases/latest).
2. In Blender, open **Edit → Preferences → Get Extensions**, open the menu
   at the top right, and choose **Install from Disk**. Select the ZIP as-is.
3. In the 3D Viewport, press **N** and open the **Cloth Fit** tab.

Use the named extension ZIP in the release assets. GitHub's generated
“Source code” ZIP is a repository snapshot and is not the Blender installer.

## Documentation

- [User guide](avatar_cloth_fit/USER_GUIDE.md) — setup, tools, settings, troubleshooting, and export.
- [Changelog](CHANGELOG.md) — changes by version.
- [Report a problem](https://github.com/Peppxrr/avatar-cloth-fit/issues) — include your
  Blender version, extension version, and steps to reproduce it.

Clothing needs a reasonable starting shape. This is a manual shaping and
pose-guided weighting tool; complex garments can still need mesh edits or
dedicated bones. Check the result in your target application before export.

## Source and license

Extension source is in [`avatar_cloth_fit/`](avatar_cloth_fit/).
The install ZIP includes the source and user guide.

Licensed under [GPL-3.0-or-later](LICENSE).
