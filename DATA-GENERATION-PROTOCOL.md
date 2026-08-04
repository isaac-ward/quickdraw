# RoboCasa Data-Generation Protocol

This protocol defines how built-in RoboCasa demonstrations are selected, augmented, and validated to produce the data used to train QuickDraw's world model. Recorded demonstration actions are never retargeted.

## 1. Select the tasks and scene

> Select a semantically diverse set of tasks whose built-in demonstrations share an exact RoboCasa scene and whose initializations can coexist without one task's objects occupying another task's required receptacles or conflicting with its fixture states.

Task and scene selection are a joint decision. Do not choose a layout/style pair first and then assume that suitable demonstrations exist for it.

For each candidate task, the selection process must establish:

- a built-in demonstration dataset is available;
- enough demonstrations share the same exact `(layout_id, style_id)` as the other selected tasks;
- its task semantics add a meaningfully different goal or interaction, rather than merely renaming another task;
- its required source objects, destination receptacles, and articulated-fixture state are compatible with the other selected tasks; and
- the selected demonstrations provide at least six hours of source trajectory data in total.

Different object inventories and different articulated fixtures are allowed. Compatibility means that the initializations can coexist without changing the recorded actions or making a demonstrated task infeasible.

### Select conservative task coverage

Prefer the largest task set whose compatibility follows directly from the task definitions. Do not add a task when compatibility depends on finding favorable episode-specific placements or fixture choices.

For each exact `(layout_id, style_id)`, the selection process must:

1. identify every task with built-in demonstrations in that scene;
2. remove task pairs with incompatible object, receptacle, or fixture-state initializations;
3. find a semantically diverse compatible task set without conditional compatibility assumptions; and
4. confirm that its selected source episodes provide at least six hours in total.

A task counts as selected only if it contributes source episodes to the corpus. Report both the eligibility upper bound and the selected task set certified by the episode metadata; do not present a source-code-only candidate as a certified result.

### Rejected preliminary route: seven pretraining tasks

An initial, narrowly scoped search considered 18 implemented environments. Three did not have registered built-in demonstration datasets, leaving 15 data-backed tasks in that preliminary universe.

The registered demonstrations impose a further split constraint:

- 14 tasks are available in the pretraining split, including `LoadDishwasher`;
- two tasks are available in the target split: `LoadDishwasher` and `WashFruitColander`; and
- pretraining and target scenes are different scene sets and cannot be combined to satisfy the exact-scene requirement.

The eligibility upper bound for that preliminary route was therefore 14 tasks in a pretraining scene. It was not an upper bound for RoboCasa's denser target-human data.

A conservative seven-task set to advance to that metadata audit is:

- `RestockCannedFood`
- `LoadFridgeByType`
- `MicrowaveCorrectMeal`
- `MicrowaveDefrostMeat`
- `ScrubBowl`
- `LoadDishwasher`
- `SimmeringSauce`

At the task-definition level, these tasks use compatible fixture states and do not initialize objects in one another's required receptacles. `AdjustHeat` is omitted because including it with `SimmeringSauce` would require selecting episodes with different burners and non-overlapping stove-area placements. This is candidate screening, not certification.

### Seven-task episode-metadata audit

The seven-task set was audited on 2026-07-31 against the official RoboCasa v1.0 pretraining-human archives registered at RoboCasa commit `b4684e6ee37d377cc392e98302a6b916d588b415`. For every source episode, the audit joined:

- `meta/info.json` for the dataset frame rate and aggregate counts;
- `meta/episodes.jsonl` for episode index and trajectory length; and
- `extras/episode_<id>/ep_meta.json` for exact layout, style, sampled objects, resolved fixture references, and robot initialization.

All seven datasets use 20 Hz trajectories. Their source holdings are:

| Task | Source episodes | Distinct `(layout_id, style_id)` | Frames | Source duration |
| --- | ---: | ---: | ---: | ---: |
| `RestockCannedFood` | 108 | 105 | 46,341 | 0:38:37.05 |
| `LoadFridgeByType` | 104 | 102 | 70,827 | 0:59:01.35 |
| `MicrowaveCorrectMeal` | 108 | 105 | 82,690 | 1:08:54.50 |
| `MicrowaveDefrostMeat` | 105 | 97 | 86,126 | 1:11:46.30 |
| `ScrubBowl` | 100 | 94 | 61,728 | 0:51:26.40 |
| `LoadDishwasher` | 100 | 99 | 84,383 | 1:10:19.15 |
| `SimmeringSauce` | 105 | 101 | 148,642 | 2:03:52.10 |
| **Total across different scenes** | **730** | — | **580,737** | **8:03:56.85** |

The intersection of the seven tasks' exact scene sets is empty. Consequently:

- the seven tasks do not share an exact `(layout_id, style_id)`;
- there is no shared scene to rank by usable seven-task demonstration count;
- the exact-scene-eligible source duration is zero, despite the 8.066 hours available when scenes are incorrectly pooled; and
- no trajectory from this seven-task set can yet count as an accepted augmented trajectory. Augmented-trajectory acceptance was not attempted because scene eligibility already failed.

The widest near misses are diagnostic only. Scene `(15, 43)` covers three tasks with four source episodes: two `MicrowaveCorrectMeal` episodes and one each from `RestockCannedFood` and `SimmeringSauce`, totaling 162.45 seconds. Scene `(47, 16)` covers `LoadDishwasher`, `MicrowaveCorrectMeal`, and `ScrubBowl` with one episode each, totaling 115.55 seconds. No scene covers four or more of the seven tasks.

Episode metadata also exposes an initialization hazard that task-definition screening alone does not resolve. In scene `(47, 16)`, all three represented tasks resolve their root clutter objects to `counter_03_right_group_1`. In scene `(15, 43)`, `RestockCannedFood` and `SimmeringSauce` both resolve their root clutter objects to `counter_1_left_group_1`. Naively combining independently recorded placement regions would therefore risk object overlap or obstruction even though the tasks use different goal fixtures. Added clutter must be placed and collision-checked independently of the source episode; inactive episode placements must not be copied as though they were jointly valid.

No cross-task `object_cfgs[*].name` collision or additional unconditional receptacle conflict was found in the episode JSON. In particular, `LoadFridgeByType` targets shelves in the refrigerator compartment while `MicrowaveDefrostMeat` initializes its tupperware in the freezer compartment. However, `ep_meta.json` identifies fixtures but does not record their articulated qpos, so episode JSON cannot establish fixture-state compatibility. That still requires the immutable source state and direct swept-occupancy certification.

The result is a failed data-availability audit for this sparse seven-task pretraining route, not a reversal of the structural task-definition audit and not a limit on RoboCasa generally. The corrected search below uses the target-human demonstrations, which are training data concentrated in ten held-out kitchens. Evaluation must use fresh simulator episodes rather than these target demonstrations.

### Correct target-task candidate universe

The corrected candidate universe contains all 32 target composite tasks. "Unseen" means unseen during broad pretraining; these tasks still have target demonstrations for target training or fine-tuning.

Composite-Seen:

- `DeliverStraw`
- `GetToastedBread`
- `KettleBoiling`
- `LoadDishwasher`
- `PackIdenticalLunches`
- `PreSoakPan`
- `PrepareCoffee`
- `RinseSinkBasin`
- `ScrubCuttingBoard`
- `SearingMeat`
- `SetUpCuttingStation`
- `StackBowlsCabinet`
- `SteamInMicrowave`
- `StirVegetables`
- `StoreLeftoversInBowl`
- `WashLettuce`

Composite-Unseen:

- `ArrangeBreadBasket`
- `ArrangeTea`
- `BreadSelection`
- `CategorizeCondiments`
- `CuttingToolSelection`
- `GarnishPancake`
- `GatherTableware`
- `HeatKebabSandwich`
- `MakeIceLemonade`
- `PanTransfer`
- `PortionHotDogs`
- `RecycleBottlesByType`
- `SeparateFreezerRack`
- `WaffleReheat`
- `WashFruitColander`
- `WeighIngredients`

### Target exact-scene metadata audit

All 32 official target-human composite archives were audited on 2026-07-31 using the same RoboCasa revision and the same `info.json` / `episodes.jsonl` / `ep_meta.json` join described above. The complete index contains 16,181 episodes, 12,726,552 frames, and 176:45:27.60 at 20 Hz.

The maximum exact-scene coverage is 31 tasks. Three scenes attain it:

| Scene | Description | Supported tasks | Missing task | Source episodes | Frames | Source duration |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `(2, 2)` | One Wall Island / Scandinavian | 31 | `LoadDishwasher` | 2,415 | 2,000,226 | 27:46:51.30 |
| `(10, 10)` | Wraparound / Transitional | 31 | `SeparateFreezerRack` | 1,856 | 1,527,501 | 21:12:55.05 |
| `(4, 4)` | L-shaped Island / Farmhouse | 31 | `SeparateFreezerRack` | 1,853 | 1,527,386 | 21:12:49.30 |

Scene `(2, 2)` has the most usable native source demonstrations among the 31-task scenes. Scene `(4, 4)` is nevertheless the leading protocol candidate because it retains `LoadDishwasher`, `PrepareCoffee`, `BreadSelection`, and `PackIdenticalLunches`. Scene `(10, 10)` supports the same 31-task set and is the alternative.

The 31-task upper bound is exact. `LoadDishwasher` has 501 target episodes only in scenes `(1, 1)`, `(3, 3)`, `(4, 4)`, `(5, 5)`, `(6, 6)`, and `(10, 10)`. `SeparateFreezerRack` has 501 target episodes only in `(2, 2)`, `(7, 7)`, and `(9, 9)`. Their scene sets are disjoint, so no target kitchen can support all 32 tasks.

This is a metadata-eligible upper bound, not a protocol-certified maximum. Scene `(4, 4)` has more than six hours of eligible source trajectories, but accepted duration is still zero until augmented initialization and trajectory acceptance are run. The scene-4 inventory, fixture, and swept-region audit is recorded in `ROBOCASA-SCENE-4-MANIFEST.md`.

A concrete interaction-focused source selection is recorded in `ROBOCASA-SCENE-4-6H-SELECTION.md`. Its six tasks have no explicit distractor slots and cover all 19 task-owned object slots in aggregate. The full candidate pool provides 6:09:07.40; the final six-hour subset must be chosen only from candidates that pass initialization compatibility and the direct swept-occupancy certificate below.

### Current conservative four-hour packaging set

A later interaction-focused audit tested a nine-task Scene-4 candidate:

- `DeliverStraw`
- `GatherTableware`
- `GetToastedBread`
- `HeatKebabSandwich`
- `LoadDishwasher`
- `MakeIceLemonade`
- `PrepareCoffee`
- `StackBowlsCabinet`
- `WashFruitColander`

That set is not conservatively compatible. `GatherTableware` initializes three mugs and a bowl across two resolved cabinets and, in Scene 4, always uses `cab_main_main_group` plus one other cabinet. `StackBowlsCabinet` may use `cab_main_main_group` as the demonstrated destination. Direct traces repeatedly intersected the inactive Gather inventory when the active Stack bowls entered that cabinet. Stack episodes using other cabinets can avoid the conflict only when Gather also resolves a favorable second-cabinet pair. Because selected-task membership may not depend on favorable episode-specific fixture choices, `StackBowlsCabinet` is excluded.

The current conservative set is therefore:

- `DeliverStraw`
- `GatherTableware`
- `GetToastedBread`
- `HeatKebabSandwich`
- `LoadDishwasher`
- `MakeIceLemonade`
- `PrepareCoffee`
- `WashFruitColander`

The native-validated source pool for these eight tasks contains 321,055 frames, or 4:27:32.75 at 20 Hz. Exact eight-inventory certification accepted 266 full source episodes totaling 291,637 frames (4:03:01.85); every accepted episode has an atomically frozen same-process model and source-to-composite joint mapping.

The finalized package selects 261 of those accepted full episodes totaling 288,593 frames (4:00:29.65). Independent validation passed all 261 certificate and artifact bindings, exact source-state and source-action equality, global annotation-code isolation, Parquet-schema equality, and all 783 video streams. This four-hour package is an interim training deliverable; it does not satisfy or weaken the protocol's general six-hour corpus requirement.

For rendering reproducibility, the accepting process must atomically save the exact compiled augmented MuJoCo model and its source-to-composite joint mapping. Re-running the initializer later is not an acceptable substitute: RoboCasa initialization uses process-global random state in addition to the explicit per-task seeds. Rendering loads this frozen accepted model, applies stored source states, and never reruns task initialization, actions, or dynamics.

## 2. Source episodes and the corpus

### Source episode

A source episode is one immutable built-in RoboCasa demonstration: its exact scene, task objects, object placements, fixture states, robot initialization, and recorded action sequence.

Each source episode is augmented and validated independently. Episodes do not need to be paired across tasks or organized into one-episode-per-task bundles.

### Corpus

The corpus contains any number of source episodes from each selected task, all sharing the selected exact `(layout_id, style_id)`. It must contain at least six hours of accepted augmented trajectories in total.

For each source episode:

- preserve its recorded task objects, placements, fixture states, and robot initialization;
- instantiate new objects from the other selected tasks' inventory definitions and add them as clutter without changing the source episode's task initialization; and
- use the recorded source-state trace and model geometry directly to certify that the demonstrated occupied and swept regions do not intersect the added clutter.

### Added clutter generation

Added clutter is sampled anew; it is not copied from another recorded demonstration. An inactive demonstration supplies neither exact object coordinates nor dynamic fixture state. Its task definition does supply the placement procedure for its objects. For each active source episode:

1. sample fresh object instances from each inactive task's inventory definition, preserving its instance counts, semantic object groups, scales, and object-in-object relationships;
2. resolve that inactive task's own placement fixtures and sample its objects only from the fixture regions, relative placements, and nested placements specified by that task's initializer;
3. do not import the inactive task's robot initialization, articulated fixture state, exact recorded object poses, or action sequence;
4. never place an inactive task's objects using the active task's placement pool or an invented neutral placement pool;
5. reject the combined initialization if independently sampled task placements overlap, occupy another task's required receptacle, or obstruct the active source; and
6. seed all inactive-task fixture and object sampling deterministically from the active source episode identifier, inactive task name, and protocol version; because RoboCasa also consumes process-global random state, atomically save the exact sampled initialization, compiled augmented model, and source-to-composite joint mapping in the accepting process rather than assuming the explicit seed alone can reconstruct it under a different process setup.

Exact clutter coordinates are not required to match across source episodes. RoboCasa's own task initializers sample object instances and poses within placement regions, so demonstrations of one task do not generally begin with identical object coordinates. Dataset consistency means using the same inventory rules, placement procedure, exclusions, and deterministic seeding scheme—not placing every object at one universal coordinate.

Duration is counted from augmented trajectories that pass the acceptance rules below, not merely from nominal dataset availability.

## 3. Initialization compatibility

Before trajectory validation, an augmented initialization must be rejected if adding the other task inventories would cause any of the following:

- an inactive task object occupies an active task's required receptacle or destination;
- an inactive task object obstructs a door, drawer, rack, knob, or appliance state required by another task;
- the added initialization assigns a state to an articulated fixture that conflicts with the source episode;
- task objects overlap physically in the augmented initial state;
- an added object's identity could be selected by, substituted into, or otherwise change the meaning of the active task's original success predicate; or
- MuJoCo body, joint, or site identities cannot be made unambiguous without changing the demonstration's meaning.

This is an initialization-compatibility audit. It does not pair the active episode with inactive demonstrations or try alternative actions; it independently runs each inactive task's own object-placement procedure while preserving the active episode's dynamic state.

Passing initialization compatibility is necessary but is not by itself trajectory acceptance.

## 4. Trajectory acceptance invariant

The recorded action sequence is never modified. Every augmented episode must pass the following certificate.

### Direct swept-occupancy certificate

Load the source episode's stored MuJoCo model and full simulator-state trace. At each recorded state, use MuJoCo's body and geometry transforms directly to recover the occupied geometry of the active robot, gripper, all source objects, and every articulated fixture part. Superimpose the added clutter at its proposed initial poses; do not import or force trajectories from inactive demonstrations.

The certificate passes only if all of the following are established:

1. every added object is stably supported at its proposed pose, with no unresolved settling or sliding;
2. the added objects are mutually collision-free except for their intended stable support or containment contacts;
3. the direct occupied geometry is checked at every recorded source state;
4. consecutive recorded poses are interpolated in generalized coordinates, including quaternion interpolation for free joints, so the pose-to-pose swept region is checked rather than only the 20 Hz endpoints;
5. no added-object collision geometry intersects the directly reconstructed occupied or swept geometry of the robot, gripper, source objects, or moving door, drawer, rack, knob, appliance part, or other articulated fixture part at any point in the trace; and
6. fixture-state compatibility and success-predicate identity isolation from the initialization audit remain valid for the full episode.

Give the recorded demonstration the benefit of the doubt: reject it for an observed geometry intersection deeper than a documented numerical contact tolerance, initialization incompatibility, instability, or predicate alias—not merely because two geometries are close, touch within numerical tolerance, or because a generic conservative bound overlaps. The interpolation method, resolution, and penetration tolerance must be recorded, but no additional positive clearance margin is required. A certified episode is accepted because the directly reconstructed source trajectory does not meaningfully intersect the added clutter.

### Acceptance record

For every accepted episode, record the active source episode identifier, exact clutter seed and sampled initialization, hashes of the frozen compiled augmented model and source-to-composite joint mapping, geometry/model and protocol versions, the interpolation method and resolution, and the minimum observed separation or absence of collision. Only certified episodes count toward the six-hour requirement.
