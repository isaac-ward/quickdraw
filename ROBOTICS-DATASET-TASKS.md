# Robotics Dataset and Environment Roadmap

**Research target:** turn Quickdraw's torus demonstration into a reproducible study of semantic
steering through action-conditioned world models and learned rewards.  
**Research snapshot:** 2026-07-25  
**Depends on:** `DATASET-TEMPLATE-1140bbcfd1435e6e82bc8193646343d261706244.md`

## Recommendation

Use **RoboCasa365 as the flagship environment**, **LIBERO as the first external benchmark**, and
**ManiSkill3 as the adapter/conformance testbed**.

This is a deliberate three-level choice:

1. **ManiSkill3 proves the infrastructure.** It is pip-installable, Gymnasium-shaped, batched on GPU,
   has downloadable demonstrations, privileged state, RGB/depth/segmentation, success predicates,
   and task cards. It is the fastest way to prove that the new collector is not torus-specific.
2. **LIBERO proves comparability.** It supplies a LeRobot-ready dataset, standardized language
   suites, a maintained LeRobot evaluation path, and a recognized 400-episode evaluation protocol.
   The 10 long-horizon tasks are a compact first paper-facing benchmark.
3. **RoboCasa365 supplies the paper's semantic depth.** It has 365 tasks across 2,500 kitchens,
   2,200+ hours of demonstrations, 12-DoF mobile manipulation, held-out task/scene structure, and
   current per-timestep annotations for subtask, atomic skill, stage, and language. Its datasets are
   already distributed in LeRobot format.

If only one environment can be carried to the final paper, choose RoboCasa365. If only one can be
integrated immediately, choose LIBERO. Do not begin by converting Open X-Embodiment or full DROID:
they are useful training corpora, but they do not supply the closed-loop simulator needed for the
core Quickdraw evaluation.

## Why RoboCasa365 fits the scientific question

The current torus experiment asks whether an action-conditioned learned world can remain coherent
and whether a learned semantic reward can steer sampled action sequences. RoboCasa365 preserves
those primitives while making them consequential:

- observations contain three camera views and 16-dimensional proprioception;
- actions are continuous 12-dimensional mobile-manipulator controls;
- atomic tasks expose clean success predicates;
- composite tasks require ordered sequences of meaningful actions;
- kitchens contain articulated and stateful appliances;
- scenes, objects, tasks, and compositions create natural OOD axes;
- per-frame subtask, skill, stage, and instruction labels provide reward-model supervision without
  commissioning a new annotation effort.

The project's most defensible new claim would not be "we trained another VLA." It would be:

> An action-conditioned world model can propose and evaluate counterfactual futures, while a
> language-conditioned semantic reward selects futures that complete ordered manipulation
> subgoals; this improves closed-loop success and semantic compositional generalization over policy-
> only and non-semantic planning baselines.

RoboCasa's current release is unusually well matched to that claim. It supports atomic skills such
as `CloseFridge`, `OpenDrawer`, `TurnOnMicrowave`, and navigation, plus roughly 300 composite tasks
in cooking, cleaning, organizing, and related categories. The official dataset release contains
human and MimicGen demonstrations; current composite-task data includes a per-frame subtask index,
atomic-skill name, stage, and natural-language instruction.

Official sources:

- [RoboCasa project and RoboCasa365 release](https://robocasa.ai/)
- [RoboCasa repository](https://github.com/robocasa/robocasa)
- [Dataset overview](https://robocasa.ai/docs/build/html/datasets/datasets_overview.html)
- [LeRobot-format dataset structure and downloader](https://robocasa.ai/docs/build/html/datasets/using_datasets.html)
- [LeRobot RoboCasa365 integration and evaluation](https://huggingface.co/docs/lerobot/main/robocasa)

Licensing is also usable for open research: the repository identifies the code as MIT and the assets
and datasets as CC BY 4.0.

## Candidate matrix

Ratings are relative to Quickdraw's needs, not general judgments of benchmark quality.

| Candidate | Data + live sim | Semantic sequences | Standard evaluation | Start-up cost | Recommended role |
|---|---|---|---|---|---|
| **RoboCasa365** | Yes | Excellent | Current and growing | Medium | Flagship |
| **LIBERO** | Yes | Very good | Excellent | Low–medium | First external benchmark |
| **ManiSkill3** | Yes | Moderate; extensible | Good | Low | Adapter smoke test and scalable collection |
| **CALVIN** | Yes | Excellent | Canonical long-horizon | Medium–high due older stack | Reproduction/secondary comparison |
| **Language Table** | Yes | Excellent language, simpler mechanics | Established | Medium due TensorFlow-era stack | Semantic steering ablation |
| **RLBench** | Yes | Good | Established | High due CoppeliaSim/PyRep stack | Do not prioritize |
| **SIMPLER** | Evaluation sim plus real datasets | Task language, limited breadth | Strong real-to-sim framing | Medium | Later external-validity evaluation |
| **Open X / DROID / BridgeData V2** | Data only | Good language; no resettable source world | Dataset standards, not closed-loop sim | Low for samples, high at full scale | Pretraining or offline transfer |
| **RoboNet / BAIR pushing** | Data only | Weak | Historic world-model baselines | Low–medium | Action-adherence regression test |

## Best downloadable full-stack options

### 1. RoboCasa365 — flagship

What is available now:

- 365 everyday kitchen tasks;
- 2,500+ kitchens and 3,200+ object assets;
- 600+ hours of human demonstrations and 1,600+ hours of synthetic demonstrations;
- LeRobot datasets with Parquet low-dimensional streams and MP4 camera streams;
- a task-specific downloader, so an integration does not require the whole corpus;
- atomic/composite seen and composite-unseen target groups;
- three camera views, 16D state, and 12D bounded continuous actions in the LeRobot environment
  interface;
- current LeRobot commands for single-task, multi-task, and benchmark-group evaluation.

Lowest-risk entry:

1. Load the ready-to-use LeRobot `CloseFridge` dataset referenced by the official LeRobot docs.
2. Run the simulator and the released SmolVLA checkpoint as an environment sanity check.
3. Add `OpenDrawer`, `TurnOnMicrowave`, `PickPlaceCounterToStove`, and `NavigateKitchen`.
4. Move to one annotated composite family only after action alignment and closed-loop replay pass.

Important integration caveat:

RoboCasa and `robosuite` are installed from source, and RoboCasa's current setup pins an older
LeRobot. The official LeRobot integration explicitly works around that pin with editable installs
and `--no-deps`. Quickdraw should isolate the environment in an optional dependency group or
container and test one selected LeRobot version rather than mutating the existing unbounded
dependency in place.

### 2. LIBERO — first paper-facing checkpoint

LIBERO provides 130 language-conditioned tasks:

- 10 spatial-reasoning tasks;
- 10 object-transfer tasks;
- 10 goal-transfer tasks;
- 90 short-horizon tasks;
- 10 long-horizon tasks.

The official project includes teleoperated demonstrations, PDDL scene descriptions, workspace and
wrist RGB, proprioception, and language. The current LeRobot integration supplies a preprocessed
dataset (`HuggingFaceVLA/libero`), an environment wrapper, and the established evaluation of 10
episodes per task over Spatial, Object, Goal, and Long: 400 episodes total.

Why start here:

- It directly exercises language-conditioned action sequences.
- Its standard suites create clean claims about spatial, object, and goal transfer.
- The current LeRobot API exposes `observation.state`, two cameras, and a 7D continuous end-effector
  delta plus gripper action.
- It is much smaller and conceptually narrower than RoboCasa365.

Constraint: current LeRobot documentation supports the environment on Linux and MuJoCo. The original
LIBERO repository's own pinned stack is old; use the maintained LeRobot integration rather than
building the original Python 3.8/Torch 1.11 environment as the primary path.

Official sources:

- [LIBERO repository and dataset downloader](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [LIBERO dataset contents](https://libero-project.github.io/datasets)
- [Current LeRobot LIBERO integration](https://huggingface.co/docs/lerobot/main/libero)
- [LIBERO paper](https://arxiv.org/abs/2306.03310)

### 3. ManiSkill3 — infrastructure testbed

ManiSkill3 is the best environment for implementing and testing the generic simulator adapter:

- installable with `pip install mani_skill`;
- created through `gymnasium.make`;
- standard single and vector Gymnasium wrappers;
- GPU-parallel simulation and rendering;
- task-specific demonstration and asset download commands;
- task cards declaring robots, randomizations, rewards, success/failure, and demo availability;
- configurable state, RGB, depth, segmentation, point cloud, camera intrinsic/extrinsic, and
  privileged-state observations;
- raw demonstrations that preserve actions, initial states, and seeds for deterministic replay;
- replay/conversion tools that add desired observations, rewards, or control modes.

The raw demonstration format is HDF5 plus JSON rather than LeRobot, so it is ideal for testing the
conversion layer. Start with `PickCube-v1` and `PegInsertionSide-v1`: one forgiving pick/place task
and one contact-sensitive precision task.

Use ManiSkill to prove throughput, replay, segmentation, camera calibration, and Gym compatibility.
Do not make its simplest single-skill tasks the final semantic headline.

Official sources:

- [ManiSkill repository](https://github.com/haosulab/ManiSkill)
- [Tasks and demo download](https://maniskill.readthedocs.io/en/latest/tasks/)
- [Gymnasium and vector wrappers](https://maniskill.readthedocs.io/en/latest/user_guide/reinforcement_learning/setup.html)
- [Observation and semantic segmentation modes](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html)
- [Demonstration replay/conversion](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/replay.html)

The code is Apache-2.0; the repository notes that some assets are CC BY-NC 4.0, which should be
recorded per selected task.

### 4. CALVIN — canonical but operationally older

CALVIN remains a canonical long-horizon language benchmark. Its central evaluation asks a robot to
complete five consecutive language-specified tasks, and its dataset can be downloaded in a 1.3 GB
debug form before taking the full release.

It is scientifically attractive because it has a long history of VLA and compositional-policy
comparisons. It is operationally less attractive because its official install path uses Python 3.8
and an older simulation/training stack, and its repository documents historical corrections to
language and scene metadata. Use it after LIBERO/RoboCasa only if direct comparison to its long-
horizon leaderboard materially strengthens the paper.

Official sources:

- [CALVIN repository, download, and challenge](https://github.com/mees/calvin)
- [CALVIN paper](https://arxiv.org/abs/2112.03227)

### 5. Language Table — excellent semantic ablation

Google Research's Language Table is unusually relevant to semantic steering:

- 442,226 real-robot language-labeled episodes;
- 181,020 human-controlled simulated episodes;
- multiple 200,000-episode oracle simulation datasets;
- an open simulated continuous-control environment;
- free-form language and long-horizon rearrangement goals.

Its mechanics are simpler than full mobile manipulation, which is a virtue for a controlled
semantic-reward ablation and a limitation for the main robotics claim. It is also a TensorFlow
Datasets/GCS-era stack rather than LeRobot. Consider it if a clean demonstration of language reward
generalization is needed between Torus and RoboCasa.

Official sources:

- [Language Table repository, environment, datasets, and GCS locations](https://github.com/google-research/language-table)
- [Google Research description](https://research.google/blog/talking-to-robots-in-real-time/)

### 6. RLBench — broad, but not low-friction

RLBench has 100 hand-designed tasks, RGB/depth/segmentation, language descriptions, and effectively
unbounded motion-planner demonstrations. It is a respected benchmark, but its official setup is
tied to CoppeliaSim 4.1.0 and PyRep, including nontrivial headless rendering configuration.

It does not satisfy the "without immense setup work" priority as well as the three recommended
choices. Its value is primarily comparison to established PerAct-style results.

Official sources:

- [RLBench repository](https://github.com/stepjam/RLBench)
- [RLBench paper](https://arxiv.org/abs/1909.12271)

## Downloadable offline datasets

These are valuable, but none alone can implement protocol D because the original world cannot be
reset and stepped under counterfactual actions.

### Open X-Embodiment

Open X-Embodiment standardized data from 22 robot embodiments across 21 institutions and 527 skills.
It is the strongest cross-embodiment VLA pretraining choice. Its heterogeneity is also exactly what
makes it a poor first integration: coordinate frames, action spaces, camera sets, and task
vocabularies must be normalized across source datasets.

Use it only after Quickdraw's modality registry and action-spec metadata are truly generic.

Source: [Open X-Embodiment paper and project](https://arxiv.org/abs/2310.08864).

### DROID

DROID provides about 76,000 Franka demonstrations, 350 hours, 564 scenes, and 86 tasks. The current
release has three language annotations for 95% of successful episodes and improved calibration for
a large subset.

The complete RLDS release is roughly 1.7 TB. LeRobot's official porting guide documents a 100-episode
2 GB sample, which is the correct way to test a converter before any full ingestion.

Use DROID for real-world pretraining or reward-model robustness after the simulator experiment is
working. Do not use its full download as the first milestone.

Official sources:

- [DROID project](https://droid-dataset.github.io/)
- [DROID repository](https://github.com/droid-dataset/droid)
- [LeRobot large-dataset porting guide](https://github.com/huggingface/lerobot/blob/main/docs/source/porting_datasets_v3.mdx)

### BridgeData V2

BridgeData V2 contains 60,096 trajectories across 24 environments and 13 broad skills, with language
labels, multiple camera/depth subsets, and 7D end-effector-plus-gripper actions. It is an accessible
real-robot corpus and matches the WidowX setup in SIMPLER.

This pairing is useful later: train or evaluate a world/action model on BridgeData V2, then test
policy behavior under a related resettable SIMPLER environment. It is not a perfect replayable
digital twin, so claims must be about correlated policy evaluation rather than reconstruction of the
original episodes.

Official sources:

- [BridgeData V2 paper and project link](https://arxiv.org/abs/2308.12952)
- [SIMPLER repository](https://github.com/simpler-env/SimplerEnv)

### RoboNet

RoboNet provides 15 million frames from seven robot platforms and was explicitly studied with
forward video prediction ("visual foresight") and inverse models. It is a better historical
action-conditioned world-model baseline than a modern semantic benchmark.

Use a small slice to verify that Quickdraw can ingest a heterogeneous action-video corpus and measure
counterfactual action adherence. Do not make it the main semantic result.

Source: [RoboNet project and dataset](https://www.robonet.wiki/).

### BAIR robot pushing

The small BAIR pushing dataset is the easiest conventional action-conditioned video-prediction
regression test: 43,264 training examples, 256 test examples, two 64×64 cameras, 4D actions, and 3D
end-effector positions. TensorFlow Datasets reports a 30.06 GiB download.

It has almost no useful task semantics. A strong result on it verifies video prediction plumbing,
not semantic steering or robotics planning.

Source: [TensorFlow Datasets BAIR robot pushing](https://www.tensorflow.org/datasets/catalog/bair_robot_pushing_small).

## The red/blue collaboration environments

The strongest match to the description is **Google DeepMind's 2019 Capture the Flag work**. Its
public artwork shows small red and blue spherical/cartoon-like agents in blocky maze arenas. The
research used aesthetically modified Quake III Arena: agents had to cooperate with arbitrary
teammates and compete with the opposing team.

- [DeepMind: Capture the Flag, the emergence of complex cooperative agents](https://deepmind.google/blog/capture-the-flag-the-emergence-of-complex-cooperative-agents/)
- [Open-source DeepMind Lab platform](https://github.com/google-deepmind/lab)

Three nearby projects are often conflated with it:

- If the characters had arms/legs and moved yellow boxes or ramps, it was almost certainly
  **OpenAI's 2019 Hide-and-Seek**, with blue hiders and red seekers:
  [official project](https://openai.com/index/emergent-tool-use/).
- If the agents were simple colored circles in a flat 2D world, it was likely **OpenAI's
  Multi-Agent Particle Environment**:
  [official repository](https://github.com/openai/multiagent-particle-envs).
- If it was a top-down 2D suite of social dilemmas such as cleanup/resource sharing, it was likely
  **DeepMind's Melting Pot**:
  [official repository](https://github.com/google-deepmind/meltingpot).

For a downloadable collaboration/competition extension today, choose Melting Pot, not the old
Capture-the-Flag research setup. Melting Pot is installable from PyPI and provides more than 50
multi-agent substrates and more than 256 test scenarios covering cooperation, competition,
deception, reciprocity, trust, and other social interactions.

It should be a separate follow-on chapter, not the first robotics environment. A multi-agent world
model introduces joint actions, partial observations, nonstationary partner policies, agent-specific
rewards, and social generalization all at once. The generic dataset contract should reserve
`agent_id`, per-agent observations/actions/rewards, team, and partner-policy identity now, but the
robotics single-agent benchmark should land first.

## Proposed semantic task ladder

The experiment should grow in controlled stages rather than jumping directly to all 365 tasks.

### Tier 1: atomic state change

Candidate tasks:

- close/open fridge;
- open/close drawer;
- turn microwave on/off;
- turn stove burner on/off.

Why: success is a clear articulated-state predicate, action sequences are short, and reward labels
are unambiguous.

### Tier 2: object transport

Candidate tasks:

- pick an object from a counter;
- place an object in a cabinet;
- move an object from counter to stove;
- put an object in a drawer.

Why: introduces approach, grasp, transport, place, release, object identity, and spatial relations.

### Tier 3: navigation plus manipulation

Candidate tasks:

- navigate to an appliance, open it, and retrieve an object;
- carry an object across kitchen regions and place it at a named destination.

Why: tests long horizon, camera change, mobile-base action, and compositional reward.

### Tier 4: composite household task

Candidate sequences:

- open fridge → retrieve item → place on counter → close fridge;
- clear table → sort objects by destination → close storage;
- prepare tea → manipulate appliance states in the correct order;
- load dishwasher → preserve object/category and containment relations.

Why: these sequences create meaningful semantic forks. A plausible-looking future with the wrong
object, wrong order, or unclosed appliance must receive lower reward than a semantically correct
future.

## Planner design implication

Torus MPPI samples raw two-dimensional actions. Directly extending that planner to 12D mobile-
manipulator actions over hundreds of steps will be inefficient and is unlikely to be a competitive
robotics result.

The robotics planner should sample **action chunks from a learned proposal policy**, not isotropic
raw controls:

```text
language goal
    -> candidate semantic skill / subgoal
    -> VLA or diffusion-policy action-chunk proposals
    -> action-conditioned world-model rollouts
    -> learned semantic reward + safety/value scoring
    -> execute a short receding-horizon prefix
```

Two planning levels should be compared:

1. low-level action-chunk planning only;
2. hierarchical planning over skill/stage tokens followed by action chunks.

RoboCasa's per-frame skill and stage annotations make the second comparison possible without new
manual labels. This is the connection between "interesting semantic action sequences" and a
tractable sampling-based planner.

## Required experimental axes

### World-model quality

- one-step state and image prediction;
- long-horizon open-loop state/image prediction;
- object-state and articulated-state accuracy;
- contact/event timing;
- semantic skill/stage prediction;
- calibration or sample diversity for stochastic futures.

Pixel metrics alone are not sufficient. A visually plausible rollout that ignores the action is a
failed world-action model.

### Action adherence

From the same initial simulator snapshot:

1. sample deliberately different action chunks;
2. roll each chunk in the true simulator and learned world model;
3. compare end-effector, object, appliance, and semantic outcomes;
4. report whether world-model differences have the correct sign and ranking.

This counterfactual test should be a headline diagnostic.

### Semantic reward

Measure:

- frame/clip retrieval for task, skill, stage, object, and relation;
- progress ranking within a task;
- preference accuracy between correct and plausible-but-wrong futures;
- robustness to instruction paraphrase;
- robustness to unseen task composition;
- false-positive reward under stalled, unsafe, or visually deceptive behavior.

Compare:

- simulator oracle success/progress;
- learned reward from privileged semantic labels;
- learned reward from pixels and language only;
- a frozen VLM-as-reward baseline;
- the present caption-distillation approach.

### Closed-loop planning

Report:

- task success;
- ordered subgoal completion;
- time/actions to success;
- collision, drop, and unsafe-state rates;
- replans and sampled rollouts per executed action;
- wall-clock planning latency;
- performance under held-out scene/object/task/composition splits.

Compare:

- released or trained behavior policy/VLA without planning;
- world-model planning with oracle reward;
- world-model planning with learned reward;
- learned reward without a world model;
- deterministic versus stochastic world model;
- low-level versus hierarchical proposals;
- true-simulator MPC as an upper bound where feasible.

## Work breakdown

Tasks are ordered; each exit criterion is meant to prevent expensive downstream work on an invalid
foundation.

### QD-R0 — repair the current Torus benchmark

- [ ] Make long-horizon evaluation load `eval_ood_horizon`.
- [ ] Port visual/geometric/dynamics OOD evaluation to current model layouts.
- [ ] Add `transition.valid`; remove the sampled unused final action.
- [ ] Correct transition and duration accounting.
- [ ] Pin the generation Python/LeRobot version.
- [ ] Explicitly finalize each LeRobot dataset writer.
- [ ] Add dataset manifest and integrity validator.

**Exit:** all six splits are consumed by at least one current model in automated smoke tests, with
no silent empty evaluation.

### QD-R1 — implement the environment-independent contract

- [ ] Add `SimulatorAdapter`, `ActionSource`, `ObservationRenderer`, `TaskSemantics`, and
  `MetricSuite`.
- [ ] Add environment and feature registries.
- [ ] Generalize normalization over named numeric features.
- [ ] Generalize loaders to multiple cameras and optional semantic streams.
- [ ] Implement Gymnasium single/vector wrappers around Torus.
- [ ] Preserve Torus numerical behavior apart from the explicit final-transition correction.

**Exit:** the generic collector and loader contain no import from `quickdraw.environments.torus`.

### QD-R2 — ManiSkill conformance adapter

- [ ] Add `PickCube-v1` with state and RGB.
- [ ] Add `PegInsertionSide-v1` with RGB, depth, segmentation, camera calibration, and contacts if
  exposed.
- [ ] Download and replay official demonstrations.
- [ ] Convert HDF5/JSON trajectories to the Quickdraw LeRobot contract.
- [ ] Verify seeds, action replay, terminal observations, rewards, and success.
- [ ] Benchmark CPU single-env and GPU vector collection.

**Exit:** the same collection, validation, windowing, and open-loop evaluator used by Torus operates
on both tasks without environment-specific branches.

### QD-R3 — LIBERO dataset and environment

- [ ] Load `HuggingFaceVLA/libero` through the generic LeRobot adapter.
- [ ] Map 8D state, two cameras, 7D action, task language, and success.
- [ ] Run one LIBERO-Long task end to end.
- [ ] Run the official 10-episode-per-task evaluation wrapper.
- [ ] Add Spatial/Object/Goal/Long split metadata to the experiment manifest.
- [ ] Train a small action-conditioned world model and demonstrate counterfactual action adherence.

**Exit:** Quickdraw reproduces a valid environment success rate for a released policy and can score
its own world-model rollout on the same episodes.

### QD-R4 — RoboCasa365 single-task wedge

- [ ] Create an isolated RoboCasa/robosuite environment with one tested LeRobot revision.
- [ ] Download the `CloseFridge` dataset only.
- [ ] Map three cameras, 16D state, 12D action, task instruction, termination, and success.
- [ ] Run released SmolVLA policy evaluation as an integration baseline.
- [ ] Validate dataset action alignment by replay or simulator-state restoration.
- [ ] Store simulator state/snapshot handles separately from policy observations.

**Exit:** dataset playback, learned-policy rollout, true-simulator rollout, and Quickdraw world-model
rollout can be compared from a common initial state.

### QD-R5 — semantic labels and reward data

- [ ] Import per-frame subtask, atomic-skill, stage, and instruction annotations.
- [ ] Version vocabularies in the dataset manifest.
- [ ] Add object, appliance, relation, grasp/contact, and success event features available from the
  simulator.
- [ ] Generate hard negative pairs: wrong object, wrong destination, wrong order, incomplete
  close/open state, dropped object, and visually similar no-progress behavior.
- [ ] Split by episode, scene, object, task, and language paraphrase without leakage.

**Exit:** a reviewed sample and automated predicate checks show semantic labels are temporally and
logically consistent.

### QD-R6 — robotics world-action model

- [ ] Support three cameras plus proprio/object state.
- [ ] Condition on action chunks and instruction/task tokens.
- [ ] Predict state, image latents, object/appliance state, success, skill, and stage.
- [ ] Add stochastic rollouts or multiple future samples where behavior is multimodal.
- [ ] Implement snapshot-based counterfactual action tests.
- [ ] Compare data-space, latent-space, and diffusion variants under equal compute.

**Exit:** held-out action sequences produce measurably different and correctly ranked predicted
outcomes, not merely plausible unconditional video.

### QD-R7 — semantic reward model

- [ ] Train an oracle-semantic reward from simulator labels.
- [ ] Train a pixels-plus-language reward with no privileged test-time features.
- [ ] Port the caption-distillation reward as a baseline.
- [ ] Add VLM-as-reward and binary-success baselines.
- [ ] Measure temporal progress, paraphrase robustness, composition generalization, and hard-negative
  false positives.

**Exit:** the learned reward ranks correct counterfactual futures above the defined semantic hard
negatives on held-out tasks/scenes.

### QD-R8 — sampling planner

- [ ] Train or adopt a VLA/diffusion proposal distribution for action chunks.
- [ ] Implement receding-horizon candidate rollout and scoring.
- [ ] Add semantic skill/stage proposal planning.
- [ ] Enforce action bounds and safety constraints before execution.
- [ ] Compare raw-action MPPI, policy-proposal sampling, and hierarchical proposal sampling.
- [ ] Profile samples, latency, GPU memory, and model calls.

**Exit:** world-model plus learned-reward planning improves at least one closed-loop semantic task
over the identical proposal policy without planning, under matched execution budget.

### QD-R9 — benchmark suite

- [ ] Atomic seen tasks.
- [ ] Composite seen tasks.
- [ ] Composite unseen tasks.
- [ ] Held-out kitchens/scenes.
- [ ] Held-out objects and language paraphrases.
- [ ] Dynamics/camera/action-latency shifts.
- [ ] Official LIBERO Spatial/Object/Goal/Long evaluation.
- [ ] Fixed seeds, episode counts, checkpoint selection, and compute reporting.

**Exit:** one command produces a versioned experiment manifest, model/reward checkpoints, all
rollouts, aggregate tables, and failure videos.

### QD-R10 — paper-critical ablations

- [ ] No world model.
- [ ] No learned reward.
- [ ] Oracle reward.
- [ ] Static VLM reward.
- [ ] No semantic auxiliary prediction.
- [ ] No action conditioning.
- [ ] No counterfactual/hard-negative reward data.
- [ ] Low-level versus hierarchical sampling.
- [ ] Deterministic versus stochastic world model.
- [ ] Dataset scale and semantic-label fraction.
- [ ] Torus-to-LIBERO-to-RoboCasa transfer of the same protocol.

**Exit:** every claimed mechanism has a controlled ablation and every headline result is a
closed-loop simulator outcome, not only an offline reconstruction metric.

## What not to do first

- Do not download full DROID or Open X-Embodiment before a 100-episode converter passes.
- Do not treat LeRobot storage compatibility as environment compatibility.
- Do not call a dataset-only evaluation "closed loop."
- Do not report only PSNR/SSIM for an action-conditioned world model.
- Do not scale raw-action MPPI directly from 2D Torus control to long-horizon 12D manipulation.
- Do not mix task, scene, object, dynamics, and horizon shifts in one undiagnosable OOD split.
- Do not allow a learned reward to be evaluated only on successful demonstrations.
- Do not use the collaborator's "Gym-style" label as an interface specification; implement and test
  the actual Gymnasium contract.

## Near-term milestone

The first credible vertical slice is:

```text
repair Torus
    -> generic adapter/transition contract
    -> ManiSkill PickCube conformance
    -> LIBERO-Long dataset + live environment
    -> action-conditioned state/image world model
    -> simulator-success and semantic-progress reward
    -> proposal-based receding-horizon planning
    -> official LIBERO closed-loop evaluation
```

Once that works, move the same interfaces—not a forked pipeline—to RoboCasa365 `CloseFridge`, then
to one annotated composite task family. That sequence minimizes integration risk while keeping the
end state aimed at a current, semantically rich benchmark.
