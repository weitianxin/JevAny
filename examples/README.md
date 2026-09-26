# Games, robotics, and application examples

## Open the playground

From the repository root, after creating the Python environment in the
[installation guide](../README.md#installation):

```bash
python -m pip install -e .
jevany demo
```

Open `http://127.0.0.1:8090` if your browser does not open automatically.
Replays work offline and need no GPU, model weights, Docker, or cloud account.
Use `--port 8091` to change the port or `--no-open` on a remote machine. For a
remote host, forward the chosen port to your computer before opening the URL.

The playground has three modes:

| Mode | What runs | Requirements |
|---|---|---|
| **Replay** | Packaged frames and action records, with pause, step and seek controls | Base installation |
| **Play yourself** | Your button presses execute native environment actions on CPU | The environment's optional extra |
| **Run model** | JevAny receives the camera image, measured state and recent actions, chooses an action, and displays its probabilities | Optional extra and a running JevAny server |

All three replays show successful JevAny-27B-SFT runs and retain the original
option probabilities. Crafter uses model-selected objectives before each native
action; the robot uses primitive Cartesian controls and a measured subgoal
harness. These are demonstrations, not success-rate benchmarks. Fresh runs keep
the model's actual choices, including failures.

## Play locally or connect your model

For both games:

```bash
python -m pip install -e '.[demo]'
jevany demo
```

Choose **Play yourself** and click the action controls. No inference server is
needed. Change the seed and start a new run to change the initial world.

For multimodal decisions, start the model server with a shared image directory:

```bash
mkdir -p /tmp/jevany-media
JEVANY_MEDIA_ROOT=/tmp/jevany-media jevany serve \
  --checkpoint tianxinwei/JevAny-27B-SFT --device cuda --dtype bf16
```

In a second terminal on the same host:

```bash
jevany demo --base-url http://127.0.0.1:8008 --media-root /tmp/jevany-media
```

Choose **Run model**, then **One decision** or **Run automatically**. Use
**Pause after this step** to stop after the current request finishes. The
environment pauses while inference runs; Doom does not keep advancing while
waiting for the model. Each action then executes a bounded amount of simulation.
The browser shows the environment's result and the returned probability for
every candidate, without replacing a failed choice with a scripted action.

The demo writes each camera image under `--media-root` while inference runs,
then removes the file. The model server must see that directory at the same
path. A compatible server accepting inline images can omit `--media-root`.
Use `--text-only` to send measurements without images; images remain visible
in the browser. See the [server setup](../README.md#http-server) for prerequisites.
Use `--model MODEL_ID` to set the request's model identity and `--timeout 300`
if your server needs longer than the default 120 seconds per request.
Hardware requirements belong to the model server; the playground itself runs
on CPU. **Download trace** saves the states, actions and distributions from the
current run. Live trace exports do not include image frames.

### Platform requirements

The base replay viewer uses only the ordinary JevAny client installation.
For live games, ViZDoom's prebuilt wheels support recent Linux, Apple Silicon
macOS and x86-64 Windows. Older Linux distributions may fall back to a source
build; use a recent Linux environment to avoid that step. On Intel macOS,
install `vizdoom==1.2.4` together with the game extra.

The optional robot environment is installed separately:

```bash
python -m pip install -e '.[robotics]'
```

PyBullet 3.2.7 has no Python 3.12 wheels on PyPI, so this step requires a C++
build toolchain and can take several minutes. This does not affect robot
replays or either game. The game and robot extras can coexist with the training
and serving extras.

## Doom corridor (3D)

[ViZDoom](https://github.com/Farama-Foundation/ViZDoom) runs its
`deadly_corridor` scenario with the included Freedoom assets. No commercial Doom
installation or separate WAD download is required. The goal is to reach the
green armor while preserving health and ammunition.

Eight controls cover forward/backward movement, strafing, turning, shooting and
waiting. A move or shot advances eight game ticks; a turn advances four.
Observations contain the current rendered view, health, ammunition and visible
object boxes. An episode is limited to 160 decisions. In the included model
replay, the player reaches the armor by moving through the corridor.

![Accelerated checkpoint replay of Doom corridor navigation](../docs/demos/playground-doom.gif)

## Crafter survival (2D)

[Crafter](https://github.com/danijar/crafter) is a pixel-art survival game with
17 native actions. This task asks for a complete crafting sequence: gather wood,
place a table, make a wood pickaxe, and mine stone while remaining alive.
Movement, crafting, food, health and resource use are handled by the game.

The adapter preserves all native actions, including unavailable crafting
attempts that consume a turn. The checkpoint sees the current screenshot,
inventory, recipe requirements, a 9 × 7 visible map, and remembered locations
from earlier views. It first chooses an immediate objective, then chooses one
of the 17 native actions using the same screenshot. Position history and actual
action effects provide feedback for the next turn. Runs stop at goal completion,
death or 200 decisions. The included model replay completes the crafting sequence.

![Accelerated checkpoint replay of Crafter gathering wood, crafting and mining stone](../docs/demos/playground-crafter.gif)

## Robot peg insertion

This task adapts JevAny's existing Franka Panda
[peg-insertion case](../docs/CASES.md#robotics-assembly-and-laboratory-automation).
The 14 controls move the gripper along ±X, ±Y and ±Z in 1 cm or 5 cm increments,
or close and open its fingers in place. PyBullet contact and gravity determine
whether the peg is grasped, carried and inserted.

A deterministic harness supplies the current pick-and-place subgoal, target
coordinates, signed position error and current grasp feedback. Jev receives
these measurements with the camera image and chooses every primitive action
from the full set. This example tests action selection with supplied subgoals;
the model does not have to discover the task plan.

Success requires a prior two-finger grasp, alignment with the cyan socket,
correct insertion depth, an upright peg, released fingers and a settled object.
An empty grasp or a peg left outside the socket fails these checks. Runs have
a 120-decision limit. See [the harness and recorded results](../docs/ROBOTICS.md)
for its stages, recovery behavior and evaluation conditions.

![Accelerated local browser replay of the Franka peg-insertion task with recorded JevAny-27B-SFT probabilities](../docs/demos/playground-arm.gif)

## Environment API

The implementations live in [`jevany/demos`](../jevany/demos). Optional engines
load only when a live environment is created:

```python
from jevany.demos import make_environment

env = make_environment("crafter", seed=17)
try:
    state = env.observe()
    actions = env.get_all_actions()
    state, reward, done, info = env.step("move_right")
    image = env.render()  # RGB array
finally:
    env.close()
```

Environments implement `reset`, `step`, `get_all_actions`, `render` and `close`,
and provide `ACTION_LOOKUP` descriptions. They can also be used with
[`jevany.agent.run_episode`](../jevany/agent.py), whose default request contains
structured state only. The browser adds camera images and game-specific context.
Crafter makes two checkpoint calls per turn; the arm's
`decision_request(model, history)` method supplies its measured subgoals.

The viewer serves packaged HTML, CSS, JavaScript and frames without a frontend
build or CDN. To record fresh game replays, connect the recorder to your running
model server:

```bash
python scripts/record_demo_previews.py \
  --base-url http://127.0.0.1:8008 --model tianxinwei/JevAny-27B-SFT \
  --media-root /tmp/jevany-media --out /tmp/jevany-replays
```

The recorder uses the same harness as the browser and preserves failed runs too.
Recorded game imagery and upstream notices are described in
[`recordings/LICENSES.txt`](../jevany/demos/recordings/LICENSES.txt).

## Command-line application examples

Run from the repository root after [starting a server](../docs/DEPLOYMENT.md).
These examples adapt three scenarios from the existing
[showcase](../docs/CASES.md) into small, text-only applications.

```bash
python -m examples.inbox
python -m examples.sql_repair
python -m examples.service_recovery
```

| Example | What happens | Local effect |
|---|---|---|
| [Inbox](inbox.py) | Three messages are classified with choice and binary questions | Prints decisions; does not send or move mail |
| [SQL repair](sql_repair.py) | The model picks a provided query; SQLite runs it; an independent calculation checks totals | In-memory database |
| [Service recovery](service_recovery.py) | A bounded agent selects, prepares, canaries and promotes a replica | Local simulator, at most 12 decisions |

The SQL and recovery programs exit with status 1 if their checks fail. They
report the model's actual choices; they do not replace mistakes with a scripted
solution. New runs can differ from the selected successful showcase recordings.

Use your own server or load a checkpoint directly, with the same application code:

```bash
python -m examples.sql_repair --base-url http://127.0.0.1:8008
python -m examples.sql_repair --checkpoint runs/my-jev --device cuda
python -m examples.service_recovery \
  --checkpoint tianxinwei/JevAny-27B-SFT --device cuda --dtype bf16
```

`--checkpoint` needs the `local` extra; the released vision-capable checkpoint
also needs `multimodal`. The full base model must fit on the chosen device.

For your own environment, reuse `jevany.agent.run_episode`. It takes a callable
client, a goal, and an environment implementing `reset`, `step`, and
`get_all_actions`. The environment supplies finite actions and owns execution.
The [recovery example](service_recovery.py) shows this protocol.

[`request.json`](request.json) is a ready-to-send request:

```bash
jevany decide examples/request.json
jevany decide examples/request.json --checkpoint runs/my-jev
```

The Bedrock [harness](bedrock_harness.py) and
[symbolic tree](bedrock_symbolic.py) examples use an LLM planner to construct
bounded decisions. See
[INTEGRATIONS.md](../docs/INTEGRATIONS.md) for their additional dependencies.
