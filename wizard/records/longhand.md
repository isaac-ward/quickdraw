# longhand — Swoosh right-arm world-model dataset

Branch `longhand`, forked from `main` @ `934ea19`. Lego is finished and pushed to the
`lego` branch; nothing from it is merged here and nothing here depends on it.

Goal: turn the Swoosh right-arm collection (Google Drive, campaigns 1–9) into a
quickdraw-compatible HuggingFace dataset via a new `longhand` processor.

---

## 1. What is actually in the Drive folder

`1NtfUR5kgCK8tMXdpiEoP0iJ-Q1DBEf81`, read via the Drive MCP connector on 2026-09-10.

| campaign | runs seen | disposition |
|---|---|---|
| `campaign1-tests` | 0 | **ignore** (user) |
| `campaign2-play` | 3 | **ignore** (user) |
| `campaign3-play` | 15 | train/val pool |
| `campaign4-rgb` | 11 | train/val pool |
| `campaign5-play-long` | 12 | train/val pool |
| `campaign6-combos` | ≥3 | train/val pool |
| `campaign7-precision` | 6 | train/val pool |
| `campaign8-purple-play` | ? | **eval only** |
| `campaign9-purple-stack` | ? | **eval only** |

**Also present, and NOT data:** `_rt`, `_rt2`, `_rt3`, `_rt4`, `shakedown`, `audit`.

`_rt*` are **my own test artefacts** — `tests/test_roundtrip.py` in the collection repo
wrote scratch runs into `campaigns/`, and they were swept up by the Drive sync. That is
a bug in the test (it should use a temp dir outside `campaigns/`), and they must be
excluded here regardless. `shakedown` and `audit` are bring-up runs, not dataset.

The listing hit the page limit, so run counts for 6/8/9 are lower bounds. `rclone ls`
will give the exact tree and total size.

## 2. Getting the data down — BLOCKED ON A BROWSER

`utils/gdrive_pull.sh` (already on main) is the right tool: resumable, parallel,
handles Google's large-file interstitial. rclone v1.75.1 is now installed at
`~/.local/bin/rclone` on this box, but the remote is **not configured** — that needs a
one-time OAuth round trip through a browser, which I cannot do.

**Why not the MCP connector, which IS authorised:** `download_file_content` returns the
file as base64 *in the tool result*, i.e. into the model's context. One camera mp4 is
40–53 MB (~70 MB as base64) and the full corpus is tens of GB. Fine for `run.json` and
listings; impossible for video. Not a permissions problem, a context-size one.

The exact sequence is documented at the top of `utils/gdrive_pull.sh`. Short form:

```bash
# 1. on the laptop, leave running:
ssh -N -L 53682:127.0.0.1:53682 <host>
# 2. on this box:
rclone authorize "drive" --drive-scope=drive.readonly     # paste the printed URL into the laptop browser
# 3. on this box, paste the returned token:
./utils/gdrive_pull.sh setup
# 4. look before leaping, then pull:
./utils/gdrive_pull.sh ls   https://drive.google.com/drive/folders/1NtfUR5kgCK8tMXdpiEoP0iJ-Q1DBEf81
nohup ./utils/gdrive_pull.sh pull <that-url> scratch/longhand > scratch/longhand.pull.log 2>&1 &
```

Nothing was previously downloaded to this machine — the campaign1/2 inspection was
metadata-only over MCP — so there is nothing local to delete. `scratch/` is empty.

## 3. Split design

User's spec: 1–2 ignored; **3–7** sliced and shuffled into train/val, **reserving the
longest trajectories for val**, at an overall **80/20**; **8–9** held out purely for
evaluation, structured identically for easy evals.

`starling_bags` already establishes the campaign-glob pattern (`excl_globs`,
`eval_globs`, `_split_suffix` → `eval_<suffix>` splits written verbatim). `longhand`
mirrors it, so the two processors read the same way.

**One thing the existing machinery cannot do.** `build_recorded_dataset` splits
train/val itself: `VAL_FRAC = 0.1`, random by episode, seed 0 (`processors.py:133`).
That is 90/10 and random — not 80/20, and not longest-first. So `longhand` needs an
explicit split rather than the default one.

**Longest-first val is the right call and worth stating why.** On lego the evaluable
rollout horizon was capped by the *shortest validation episode*, which cost 5× the
horizon. Putting the long trajectories in val directly buys long-horizon eval.

**But the two halves of the spec pull against each other**, and this is worth a
decision rather than a silent resolution:

- strict longest-first → val is a handful of very long episodes, possibly all from
  `campaign5-play-long`, so val stops being representative of the other campaigns
- strict 80/20 by frames → satisfied by many combinations, most of them short

Proposed resolution, to confirm: **stratify by campaign, then take longest-first within
each.** For each of campaigns 3–7, sort its episodes by length descending and take
episodes into val until that campaign contributes ~20% of *its own* frames. Result:
80/20 overall by frame count, val holds the longest trajectory from every campaign, and
no campaign is missing from val. Shuffling then happens at window level during training,
as usual.

## 4. Open questions

- [ ] Confirm the stratified longest-first split above, or state a different rule.
- [ ] 80/20 by **frames** or by **episode count**? Frames is the meaningful one for a
      world model and is what is assumed above.
- [ ] Cameras: all four, or scene-only to start? lego's lesson was that the codec is the
      binding constraint early on; one scene camera is the cheaper first target.
- [ ] `campaign4-rgb`, `campaign6-combos`, `campaign7-precision` are clearly different
      task distributions. Do they belong in one pooled train set, or should the
      processor tag `task_index` per campaign so a model can condition on it?
- [ ] Are `shakedown` and `audit` definitely excludable?

## 5. Decisions carried over from the collection side

These are properties of the recorded data and constrain the processor:

- **One monotonic clock** per run; camera frames stamped from the **V4L2 kernel buffer**
  at capture, so the ~21 ms read lag is already removed. No shifting needed.
- **`action` is the Xbox controller input** (5 dims). Commanded pose and the literal SDK
  arguments are separate columns and must stay separate.
- **Use `joints_real_deg`**, not `joints_deg` — the latter is the SDK's *planned* angles
  and was measured diverging by up to 5.02°.
- **`target_yaw_world_deg` is not absolute** — it resets on re-home, clear-errors and
  servo recovery. Absolute orientation is in `target_rpy_deg`.
- **Gripper polarity is inverted between streams**: `action` gripper 1 = closed;
  `gripper_pos` 850 = open.
- **"EE position" is the flange**, not the fingertips (`tcp_offset` is null).
- Feed orientation as **6D**, not euler — lego wrapped 702/802 times per arm.

---

## Log

**2026-09-10** — Branch cut from `main` @ `934ea19`. Lego work pushed to `lego` and
closed out. rclone installed; remote not yet authorised. Drive tree enumerated over
MCP: 9 campaigns plus 6 non-data folders, 4 of which are my own test leftovers.
Split design drafted; blocked on the download and on the questions in §4.
