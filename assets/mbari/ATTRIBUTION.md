# MBARI tank assets

All three models are Sketchfab CC-BY (attribution required, commercial use allowed). Downloaded
2026-10-18 via the Sketchfab API; each model's own `license.txt` from the download is kept beside it.

| file | model | author | licence | notes |
|---|---|---|---|---|
| `mola/` | (Unofficial) MBARI MOLA AUV | see `mola/license.txt` | CC-BY | 84,619 faces, bbox 0.806 x 0.477 x 0.561 in file units |
| `dock/` | (Unofficial) MBARI Docking MOLA AUV | see `dock/license.txt` | CC-BY | 460 faces, bbox 2.478 x 2.149 x 1.824 — already real metres. Ships **AprilTag 36h11 textures**, the real docking fiducials |
| `diver/` | Scuba Diver | turelljc | CC-BY | 56,454 faces, bbox 1.79 m tall — already real metres |

The diver originally requested (`skfb.ly/puCPB`, "Character Scuba Diver V1") reports
`isDownloadable: false` and the API returns 403, so this is a substitute: the lightest full-body
downloadable diver on Sketchfab. It is an OOD object, so what matters is that it is a diver-shaped thing
absent from training, not that it is that specific diver.

The great white shark was CGTrader, which has no public download API. The OOD fish is procedural unless
a file is supplied.

`*.obj` are MuJoCo-ready conversions produced by `src/quickdraw/sim/mbari/convert_assets.py` (glTF is
not a MuJoCo mesh format). Each is recentred on its own geometric centre, so MJCF scale and pose are
set deliberately rather than inherited from whatever the artist used.
