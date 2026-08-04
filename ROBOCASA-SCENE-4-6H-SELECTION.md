# RoboCasa Scene `(4, 4)` Six-Hour Source Selection

This is a concrete source-episode selection for QuickDraw training-data generation in the exact L-shaped Island / Farmhouse scene `(layout_id=4, style_id=4)`.

Status: **superseded source-selection proposal, not the current package**. The full candidate pool contains 416 episodes and 6:09:07.40 of immutable native demonstrations. This document preserves the earlier six-task source-selection calculation, but its former action-replay acceptance language has been replaced by the protocol's direct swept-occupancy certificate. The current completed deliverable is the separate eight-task Scene-4 package recorded in `DATA-GENERATION-PROTOCOL.md`: 261 full episodes and 288,593 frames (4:00:29.65).

## Selection objective

For this selection, an object counts as interacted with when it is directly manipulated or participates in the demonstrated task as a source container, target receptacle, or object named by the success condition. A generic or reference-only distractor does not count.

The objective is lexicographic:

1. use only tasks with no explicit distractor object slots;
2. cover every selected task-owned object slot in aggregate;
3. separate the tasks' own placement regions across kitchen fixtures as much as the scene permits;
4. retain all scene-4 episodes from category-variable tasks; and
5. select exactly six accepted hours.

The resulting set covers all 19 of its task-owned object slots. Thirteen are directly manipulated and six are source or target containers that participate in task contact and success. Across the selected episode holdings, the set contains 32 sampled semantic categories and 128 distinct object asset paths.

## Full candidate pool

| Task | Scene-4 episodes | Frames | Native duration | Task-owned objects | Aggregate interaction coverage | Placement anchor |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `GetToastedBread` | 129 | 164,690 | 2:17:14.50 | 2 | 2 / 2 | toaster plus an island plate referenced to one stool |
| `DeliverStraw` | 85 | 79,992 | 1:06:39.60 | 2 | 2 / 2 | resolved drawer plus an island glass referenced to one stool |
| `ArrangeTea` | 34 | 34,345 | 0:28:37.25 | 3 | 3 / 3 | `cab_main_main_group` and `counter_main_main_group` |
| `LoadDishwasher` | 67 | 48,111 | 0:40:05.55 | 2 | 2 / 2 | `dishwasher_left_group` and its adjacent counter region |
| `PackIdenticalLunches` | 42 | 67,403 | 0:56:10.15 | 6 | 6 / 6 | `fridge_main_group` and right counter referenced to the fridge |
| `StirVegetables` | 59 | 48,407 | 0:40:20.35 | 4 | 4 / 4 | `stovetop_main_group` and right counter referenced to the stove |
| **Total** | **416** | **442,948** | **6:09:07.40** | **19** | **19 / 19** | — |

`PackIdenticalLunches` and `StirVegetables` are the category-variable tasks in this set. All of their scene-4 episodes are retained in the precomputed exact-six-hour subset.

## Precomputed exact-six-hour subset

If every candidate composite passes initialization compatibility and the direct swept-occupancy certificate, include every candidate-pool episode except these seven dataset episode indices:

| Task | Excluded episode index | Frames | Duration |
| --- | ---: | ---: | ---: |
| `GetToastedBread` | 45 | 1,585 | 0:01:19.25 |
| `GetToastedBread` | 161 | 1,669 | 0:01:23.45 |
| `GetToastedBread` | 211 | 1,553 | 0:01:17.65 |
| `GetToastedBread` | 336 | 1,490 | 0:01:14.50 |
| `GetToastedBread` | 460 | 1,449 | 0:01:12.45 |
| `GetToastedBread` | 462 | 1,679 | 0:01:23.95 |
| `DeliverStraw` | 371 | 1,523 | 0:01:16.15 |
| **Excluded** | **7 episodes** | **10,948** | **0:09:07.40** |

The resulting accounting is:

| Task | Selected episodes | Frames | Selected duration |
| --- | ---: | ---: | ---: |
| `GetToastedBread` | 123 | 155,265 | 2:09:23.25 |
| `DeliverStraw` | 84 | 78,469 | 1:05:23.45 |
| `ArrangeTea` | 34 | 34,345 | 0:28:37.25 |
| `LoadDishwasher` | 67 | 48,111 | 0:40:05.55 |
| `PackIdenticalLunches` | 42 | 67,403 | 0:56:10.15 |
| `StirVegetables` | 59 | 48,407 | 0:40:20.35 |
| **Total** | **409** | **432,000** | **6:00:00.00** |

This episode list is not allowed to override the acceptance rule. Test all 416 candidate composites for initialization compatibility and the direct swept-occupancy certificate first. If any precomputed member fails either check, recompute the 432,000-frame subset from the candidates that passed both, using the seven initially excluded episodes as the first reserve. If fewer than 432,000 frames pass, this task set does not satisfy the protocol.

## Task-owned placement and alias checks

No object may be moved into another task's placement pool to make this selection work.

- `GetToastedBread` and `DeliverStraw` both use the island, but each task independently selects a stool-referenced placement region. Their task-owned samplers must resolve non-aliasing stool regions and object poses for the combined initialization; otherwise reject it.
- `PackIdenticalLunches` and `StirVegetables` both use `counter_1_right_main_group`, but their task definitions reference different fixture neighborhoods: the fridge and the stove. Their own sampled positions must still be collision-checked; a collision is a rejection, not permission to relocate an object.
- `ArrangeTea` is confined to the main cabinet and main counter region.
- `LoadDishwasher` is confined to the dishwasher and its adjacent left-counter region.
- The active source episode's objects, fixture state, robot initialization, and actions remain immutable. Inactive tasks contribute freshly sampled object instances and poses from their own initializers only; they contribute no robot state, articulated fixture state, or actions.

This set deliberately omits otherwise useful tasks with explicit generic distractors or strong placement-region congestion. It maximizes the fraction of selected object slots that participate in demonstrated interactions; it is not a claim that 19 is the largest possible absolute object count under every feasible task combination.

## Fast acceptance execution

The 416 candidate source episodes are independent jobs. Parallel execution may change their scheduling, but it must not change any job's deterministic clutter seed, task-owned placement procedure, recorded model, simulator-state trace, or predicate identity.

Use a staged runner:

1. Run cached metadata, namespace, fixture-reservation, and coarse placement-region checks without constructing MuJoCo models.
2. Load one native source canary per task and verify its stored model/state geometry without added clutter. These canaries do not count as augmented acceptance; they catch asset, model, state-layout, and dataset-version mismatches before the expensive run.
3. Run one augmented canary per task. Construct the active episode plus the other five task inventories, using each inactive task's own deterministic placement procedure. Stop that task cohort on a systematic construction or namespace failure.
4. Run the remaining candidates in isolated headless processes. A worker constructs one composite, rejects incompatible initialization, maps the immutable recorded states into that same live model, and checks direct geometry at recorded and interpolated poses. Camera observations, offscreen rendering, videos, action application, and dynamics stepping remain disabled. Only candidates with predicate identity isolation and no meaningful observed intersection are accepted.

Balance the work queue by recorded frame count, longest first, rather than by episode count. Keep each simulator worker single-threaded (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and equivalent library settings), use processes rather than threads, and persist workers long enough to amortize Python imports and controller setup. Write one atomic result record per episode and reduce those records only after workers exit; no shared simulator state is allowed.

On the current host, 64 physical CPU cores and 251 GiB of RAM are available. Start with 32 headless workers and increase only if aggregate checked states per second rises while composite-model resident memory leaves a safe reserve. The RTX PRO 6000s are not used for authoritative acceptance; they render accepted frozen models afterward.

The runner can terminate the current six-task proposal early if rejected frames exceed the pool's 10,948-frame reserve, because fewer than 432,000 frames can then remain. Conversely, after accepted results make an exact 432,000-frame subset reachable under the selection constraints, remaining reserve jobs are unnecessary. Neither optimization permits accepting a metadata-only or initialization-only episode; the direct swept-occupancy certificate remains the final authority.
