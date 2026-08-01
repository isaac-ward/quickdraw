# RoboCasa Data-Generation Protocol

This protocol defines how built-in RoboCasa demonstrations are selected and replayed to produce the data used to train QuickDraw's world model. Recorded demonstration actions are not retargeted.

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

### Current candidate accounting

The task list currently under consideration contains 18 implemented environments. Three do not have registered built-in demonstration datasets, leaving 15 data-backed tasks.

The registered demonstrations impose a further split constraint:

- 14 tasks are available in the pretraining split, including `LoadDishwasher`;
- two tasks are available in the target split: `LoadDishwasher` and `WashFruitColander`; and
- pretraining and target scenes are different scene sets and cannot be combined to satisfy the exact-scene requirement.

The current eligibility upper bound is therefore 14 tasks in a pretraining scene. The selected task set remains subject to exact-scene and duration confirmation from episode metadata.

A conservative seven-task set to advance to that metadata audit is:

- `RestockCannedFood`
- `LoadFridgeByType`
- `MicrowaveCorrectMeal`
- `MicrowaveDefrostMeat`
- `ScrubBowl`
- `LoadDishwasher`
- `SimmeringSauce`

These tasks use compatible fixture states and do not initialize objects in one another's required receptacles. `AdjustHeat` is omitted because including it with `SimmeringSauce` would require selecting episodes with different burners and non-overlapping stove-area placements.

## 2. Source episodes and the corpus

### Source episode

A source episode is one immutable built-in RoboCasa demonstration: its exact scene, task objects, object placements, fixture states, robot initialization, and recorded action sequence.

Each source episode is augmented and replayed independently. Episodes do not need to be paired across tasks or organized into one-episode-per-task bundles.

### Corpus

The corpus contains any number of source episodes from each selected task, all sharing the selected exact `(layout_id, style_id)`. It must contain at least six hours of accepted augmented trajectories in total.

For each source episode:

- preserve its recorded task objects, placements, fixture states, and robot initialization;
- add the selected objects from the other task inventories as clutter without changing the source episode's task initialization; and
- replay its recorded actions unchanged.

The added clutter initialization may be shared across episodes or derived deterministically for each episode. That choice is deliberately left open until the simplest replay-compatible option is established. Duration is counted from augmented trajectories that pass replay validation, not merely from nominal dataset availability.

## 3. Initialization compatibility

Before replay, an augmented initialization must be rejected if adding the other task inventories would cause any of the following:

- an inactive task object occupies an active task's required receptacle or destination;
- an inactive task object obstructs a door, drawer, rack, knob, or appliance state required by another task;
- the added initialization assigns a state to an articulated fixture that conflicts with the source episode;
- task objects overlap physically in the augmented initial state; or
- MuJoCo body, joint, or site identities cannot be made unambiguous without changing the demonstration's meaning.

This is an initialization-compatibility audit. It does not require pairing source episodes, resampling environments, or trying alternative actions.

## 4. Replay invariant

The recorded action sequence is never modified. An augmented trajectory is accepted as training data only if unchanged-action replay in the merged scene satisfies the original task's success predicate.
