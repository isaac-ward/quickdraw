# RoboCasa Scene `(4, 4)` Manifest

This manifest audits the 31 target composite tasks with native demonstrations in the exact L-shaped Island / Farmhouse scene `(layout_id=4, style_id=4)`. It separates immutable source-episode state, inactive-task inventory used as added clutter, initialization compatibility, and direct swept-occupancy acceptance.

Status: **metadata audit complete; conservative eight-task subset protocol-certified**. The official target-human holdings provide 1,853 source episodes, 1,527,386 frames, and 21:12:49.30 at 20 Hz. The 31-task union remains a metadata-only eligibility bound. The validated eight-task interim package contains 261 accepted full episodes, 288,593 frames, and 4:00:29.65 after task-owned clutter placement, predicate-identity isolation, and direct swept-occupancy certification.

Provenance: official RoboCasa v1.0 target-human LeRobot archives registered at RoboCasa commit `b4684e6ee37d377cc392e98302a6b916d588b415`, audited on 2026-07-31 by joining `meta/info.json`, `meta/episodes.jsonl`, and every `extras/episode_<id>/ep_meta.json`. `SeparateFreezerRack` is the sole target composite task without a scene-4 source episode.

## Demonstration holdings and object inventories

Object counts include generated receptacles such as the plate created by `try_to_place_in="plate"`. "Root / nested" distinguishes independently placed root objects from objects initialized in or on another object. Generic distractors remain generic; their exact category must be sampled and recorded for each augmented source episode.

| Task | Split | Episodes | Frames | Duration | Instances | Root / nested | Inventory categories |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `ArrangeBreadBasket` | unseen | 65 | 47,645 | 0:39:42.25 | 4 | 4 / 0 | bread, basket, 2 generic distractors |
| `ArrangeTea` | unseen | 34 | 34,345 | 0:28:37.25 | 3 | 3 / 0 | mug, non-electric kettle, tray |
| `BreadSelection` | unseen | 8 | 4,920 | 0:04:06.00 | 7 | 5 / 2 | cutting board, 2 plates, pastry, croissant, jam, generic distractor |
| `CategorizeCondiments` | unseen | 59 | 48,776 | 0:40:38.80 | 5 | 5 / 0 | 2 bottle condiments, 2 shakers, generic non-condiment distractor |
| `CuttingToolSelection` | unseen | 51 | 16,459 | 0:13:42.95 | 5 | 4 / 1 | peeler, knife, cutting board, food, generic non-food distractor |
| `DeliverStraw` | seen | 85 | 79,992 | 1:06:39.60 | 2 | 2 / 0 | straw, glass cup |
| `GarnishPancake` | unseen | 79 | 77,623 | 1:04:41.15 | 4 | 3 / 1 | strawberry, food distractor, plate, pancake |
| `GatherTableware` | unseen | 26 | 19,249 | 0:16:02.45 | 4 | 4 / 0 | 3 mugs, bowl |
| `GetToastedBread` | seen | 129 | 164,690 | 2:17:14.50 | 2 | 2 / 0 | sandwich bread, plate |
| `HeatKebabSandwich` | unseen | 42 | 54,006 | 0:45:00.30 | 3 | 1 / 2 | plate, kebab skewer, baguette |
| `KettleBoiling` | seen | 61 | 29,143 | 0:24:17.15 | 2 | 2 / 0 | non-electric kettle, pan-or-pot distractor |
| `LoadDishwasher` | seen | 67 | 48,111 | 0:40:05.55 | 2 | 2 / 0 | cup, bowl |
| `MakeIceLemonade` | unseen | 53 | 78,342 | 1:05:17.10 | 6 | 4 / 2 | lemon wedge, fridge distractor, bowl, 2 ice cubes, glass cup |
| `PackIdenticalLunches` | seen | 42 | 67,403 | 0:56:10.15 | 6 | 6 / 0 | 2 matched vegetables, 2 matched meats, 2 tupperware containers |
| `PanTransfer` | unseen | 48 | 18,920 | 0:15:46.00 | 4 | 3 / 1 | pan, vegetable, plate, generic non-pan/plate/vegetable distractor |
| `PortionHotDogs` | unseen | 79 | 67,867 | 0:56:33.35 | 7 | 3 / 4 | 2 plates, bowl, 2 buns, 2 sausages |
| `PreSoakPan` | seen | 45 | 37,502 | 0:31:15.10 | 2 | 2 / 0 | pan, sponge |
| `PrepareCoffee` | seen | 51 | 29,047 | 0:24:12.35 | 2 | 2 / 0 | mug, generic cabinet distractor |
| `RecycleBottlesByType` | unseen | 113 | 98,987 | 1:22:29.35 | 7 | 7 / 0 | 3 plastic bottles, 3 glass bottles, 1 sampled-type bottle |
| `RinseSinkBasin` | seen | 73 | 31,389 | 0:26:09.45 | 1 | 1 / 0 | plate distractor |
| `ScrubCuttingBoard` | seen | 54 | 25,068 | 0:20:53.40 | 2 | 2 / 0 | cutting board, sponge |
| `SearingMeat` | seen | 61 | 51,497 | 0:42:54.85 | 3 | 2 / 1 | pan, plate, meat |
| `SetUpCuttingStation` | seen | 62 | 44,988 | 0:37:29.40 | 4 | 3 / 1 | cutting board, knife, plate, meat |
| `StackBowlsCabinet` | seen | 44 | 15,103 | 0:12:35.15 | 2 | 2 / 0 | 2 differently scaled bowls |
| `SteamInMicrowave` | seen | 57 | 57,678 | 0:48:03.90 | 4 | 4 / 0 | bowl, vegetable, 2 generic counter distractors |
| `StirVegetables` | seen | 59 | 48,407 | 0:40:20.35 | 4 | 4 / 0 | spatula, pot, 2 vegetables |
| `StoreLeftoversInBowl` | seen | 115 | 85,442 | 1:11:12.10 | 7 | 5 / 2 | 2 plates, chicken drumstick, vegetable, bowl, 2 fridge distractors |
| `WaffleReheat` | unseen | 24 | 21,534 | 0:17:56.70 | 2 | 1 / 1 | bowl, waffle |
| `WashFruitColander` | unseen | 59 | 67,712 | 0:56:25.60 | 2–4 | 2–4 / 0 | 1–3 fruit, colander |
| `WashLettuce` | seen | 61 | 29,711 | 0:24:45.55 | 2 | 1 / 1 | colander, lettuce |
| `WeighIngredients` | unseen | 47 | 25,830 | 0:21:31.50 | 3 | 3 / 0 | 2 packaged foods, digital scale |
| **Total** | — | **1,853** | **1,527,386** | **21:12:49.30** | **113–115** | **94–96 / 19** | — |

## Fixture interactions and mandatory reservations

The table records logical fixture roles. Where an episode can bind several physical cabinets, drawers, stools, or counters, its resolved `fixture_refs` value is authoritative. Reservations are minimum semantic reservations; replay must additionally reserve the measured robot, gripper, source-object, and contact swept volumes for that exact episode.

| Task | Source → required destination | Required fixture state or interaction | Minimum reserved region |
| --- | --- | --- | --- |
| `ArrangeBreadBasket` | resolved cabinet + adjacent counter → `island_island_group` | cabinet starts closed; retrieve bread and move filled basket | cabinet interior/door sweep, basket, adjacent counter, island goal patch |
| `ArrangeTea` | `cab_main_main_group` + `counter_main_main_group` → tray on main counter | cabinet starts open and must finish closed | cabinet interior/door sweep, tray and counter work area |
| `BreadSelection` | main cabinet + main-counter plates → cutting board on main counter | cabinet starts open; retrieve jam | cabinet interior/door sweep, cutting board and both plate regions |
| `CategorizeCondiments` | resolved counter → resolved cabinet | cabinet starts open; place objects by matching cabinet examples | cabinet interior/door sweep, both category anchors, counter pickup region |
| `CuttingToolSelection` | resolved open drawer + resolved counter → food's cutting board | drawer starts open; choose the food-dependent tool | drawer volume/front sweep, both tools, board and food |
| `DeliverStraw` | resolved drawer → glass on island | preserve source drawer state; access drawer | drawer volume/front sweep, straw path, glass mouth and island patch |
| `GarnishPancake` | fridge → pancake/plate on island | preserve source fridge state; access fridge | fridge door/rack sweep, strawberry path, pancake/plate patch |
| `GatherTableware` | two resolved cabinets → predicate-defined mug cluster | both cabinets start open; predicate does not constrain a destination fixture | both cabinet interiors/door sweeps and demonstrated placement/sweep region |
| `GetToastedBread` | `toaster_main_group` → plate on island | bread must be detected in an energized toaster before removal | toaster slots/controls, bread path, plate and island patch |
| `HeatKebabSandwich` | plate on main counter → toaster-oven rack | toaster oven starts open and must finish closed with elapsed heat time | door sweep, both racks, controls, counter pickup region |
| `KettleBoiling` | right counter → selected stove burner | all stove knobs start off; target burner must be on | all burner/knob approach regions, kettle route, existing pan/pot distractor |
| `LoadDishwasher` | left counter → dishwasher rack | door starts open and rack extended; dishes on rack and door closed | full door/rack swept volumes, rack interior, adjacent pickup counter |
| `MakeIceLemonade` | fridge + right counter → glass on right counter | preserve source fridge state; access fridge | fridge door/rack sweep, ice bowl, cup mouth, counter work area |
| `PackIdenticalLunches` | open fridge → two tupperwares on right counter | fridge starts open | fridge door/racks, two containers and packing work area |
| `PanTransfer` | pan on stove → plate on right counter | no required articulated transition; robot must not touch food | pan interior, stove footprint, transfer arc, plate and counter patch |
| `PortionHotDogs` | bowl on island → two island plates | no articulated fixture state | bowl, both plates, distribution arcs across island |
| `PreSoakPan` | left counter → sink basin | sink starts off and must be on with pan and sponge fully inside | basin, faucet/handle sweep, adjacent counter pickup area |
| `PrepareCoffee` | open left cabinet → coffee machine | cabinet starts open; mug positioned and coffee machine turned on | cabinet door/interior, machine receptacle/button, mug route |
| `RecycleBottlesByType` | island middle → type clusters at island ends | no articulated fixture state; end assignment varies by episode | three stool-referenced island patches and all bottle routes |
| `RinseSinkBasin` | sink; plate remains a distractor on left counter | sink starts off; water must run with spout observed left, center, and right | basin, handle, full spout sweep, nearby robot approach |
| `ScrubCuttingBoard` | left-counter sponge → board on same counter | no articulated fixture state; at least 0.1 m contact sweep | board footprint plus scrub/gripper swept volume |
| `SearingMeat` | open main cabinet + right counter → pan on selected burner | cabinet starts open and selected knob starts off; actual predicate requires selected-burner placement and meat in pan | cabinet door/interior, selected burner/knob, pan, meat route |
| `SetUpCuttingStation` | resolved open drawer + counter → cutting board on same counter | drawer starts open | drawer/front sweep, board, meat/plate, knife routes |
| `StackBowlsCabinet` | resolved counter → resolved open cabinet | cabinet starts open; both bowls inside and nested | cabinet door/interior, both bowls, pickup counter |
| `SteamInMicrowave` | sink + left counter → microwave | sink starts off; microwave starts open and must finish closed/on | sink/faucet, bowl, microwave interior/door/button, counter work area |
| `StirVegetables` | right counter + pot on stove → pot | selected stove knob starts on; vegetables must be stirred in pot | selected burner/knob, pot interior and stirring sweep, pickup counter |
| `StoreLeftoversInBowl` | island + open fridge → fridge rack | fridge starts open; filled bowl must contact rack | island plates/bowl, packing routes, fridge door/rack sweep |
| `WaffleReheat` | bowl on left counter → microwave | microwave must be on; predicate does not explicitly require door closure | bowl, microwave interior/door/button, pickup counter |
| `WashFruitColander` | left counter → sink | sink starts off; colander must be under water with all 1–3 fruit | fruit/colander pickup region, basin, faucet/handle sweep |
| `WashLettuce` | colander on left counter → sink water stream | lettuce must remain under water for 25 state updates | colander/lettuce, basin, faucet and handle approach |
| `WeighIngredients` | resolved open cabinet → scale on resolved counter | cabinet starts open and must finish closed | cabinet interior/door sweep, scale top, packaged-food route |

## Episode-level compatibility findings

- Exact scene equality does not imply fixed task bindings. Scene-4 `DeliverStraw` episodes use 15 drawer/stool combinations; `CuttingToolSelection` and `SetUpCuttingStation` each use five fixture combinations; several cabinet tasks use three or four. `SearingMeat` and `StirVegetables` each use five possible stove knobs. `WashFruitColander` samples one, two, or three fruit.
- Original placement regions are heavily shared: 14 tasks place 27 task-scoped root slots on `counter_1_left_left_group`; 11 tasks place 21 on `counter_1_right_main_group`; eight tasks place 20 on the island; and eight tasks place 16 on `counter_main_main_group`. Added objects must still be sampled from their own tasks' configured fixture regions; they may not be moved into the active task's placement pool or an invented neutral region. A combined initialization that aliases must be rejected.
- RoboCasa task definitions specify sampling regions rather than one fixed coordinate set. Even demonstrations of the same task may sample different object assets, poses, fixture references, and robot initializations. Consistency therefore applies to the sampling rule and seed derivation, not to identical object coordinates across the corpus.
- There are 115 task-scoped object slots but only 93 raw slot names. Thirteen raw names collide across tasks, including `obj` in five tasks and `bowl` and `plate` in four tasks each. Prefix every added MuJoCo body, joint, site, and object name with a task namespace; do not rename or replace the active source objects.
- The 31 inventories span 152 sampled object categories across the scene-4 holdings. Every possible independently sampled 31-task combination contains at least 36 distinct categories. Exact category deduplication has no single count until a sampling and deduplication policy is chosen, so the 113–115 separate-instance count remains the conservative manifest.
- `ep_meta.json` records object configs, resolved fixtures, and robot base initialization, but not articulated fixture qpos or full trajectory swept volumes. The active source's serialized model/state remains authoritative.

## Union placement feasibility

The metadata-only capacity test is inconclusive; it neither certifies nor rules out the 31-task union.

- The full separate-instance inventory has 113–115 objects. Ninety-four to 96 are root objects requiring independent placement; 19 are nested in or on another object.
- Scene 4 has 8.34 m² of gross counter top in its layout YAML: 3.92 m² on the island and 4.42 m² on wall counters. Usable area is smaller because this gross value includes the sink/stove cutouts, fixed accessories, unreachable margins, the active source objects, required receptacles, and demonstration swept regions.
- Metadata exposes an advertised minimum footprint for only 61 of the 96 possible root slots. The sum of the per-slot maximum observed minimum footprints for those 61 slots is 2.665 m²; the other 35 root slots have no comparable footprint field. Rectangular footprint sums also do not prove collision-free packability.
- The worst pressure cases differ by active episode: island tasks reserve broad island paths, while sink, dishwasher, microwave, stove, and cabinet tasks reserve different wall-counter and articulated-fixture volumes. A single global arrangement is therefore not justified. The appropriate next test is deterministic rejection sampling that draws every added object from its own task's configured placement regions and rejects the complete initialization when those draws alias.

Consequently, the 31-task metadata upper bound is not the protocol-certified maximum. Direct state-trace testing subsequently exposed a conditional incompatibility between StackBowlsCabinet and GatherTableware: Gather always occupies cab_main_main_group plus a second cabinet in Scene 4, while Stack may use that main cabinet as its demonstrated destination. Avoiding the conflict requires favorable episode-specific fixture choices, which the conservative selection rule forbids. The validated four-hour package therefore contains the eight-task set documented in `DATA-GENERATION-PROTOCOL.md`: 261 full episodes and 288,593 frames (4:00:29.65). It does not claim that all 31 metadata-eligible tasks are compatible or that the original six-hour requirement is complete.
