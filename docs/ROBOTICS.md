# Cartesian peg insertion with Jev

The Franka Panda demo uses 14 motor commands: ±X, ±Y and ±Z translation by
1 cm or 5 cm, plus opening and closing the fingers in place. The gripper
orientation stays fixed. Contact, gravity and joint motors determine the result;
there are no object attachments or automatic moves to named objects.

The [playground guide](../examples/README.md#play-locally-or-connect-your-model)
shows how to run it with camera images and a local model server.

![Recorded Jev run with primitive controls and measured subgoals](demos/playground-arm.gif)

## What the harness supplies

[`ArmHarness`](../jevany/demos/arm_policy.py) tracks open, align, descend, grasp,
lift, transport, lower and release stages. It supplies a target pose and finger
state for the current stage. Jev chooses every action; all 14 choices remain
available, and the harness neither ranks the candidates nor replaces a choice.
This is a supplied task plan with model-selected primitive commands.

Each model request contains the current camera image and a compact numerical
observation: current and target XYZ in centimetres, signed target-minus-current
error, required and observed finger state, current two-finger contact, and the
last three actions with feedback.

The grasp target is 8.5 cm above the table; transport height is 31 cm and the
release target is 14 cm. Position transitions use a 0.55 cm tolerance per axis.
The harness advances from grasp only after measured contact confirms that the
peg is held. If contact disappears during lifting or transport, it returns to
opening and alignment. A historical contact flag is not treated as current
possession. The harness provides recovery goals; Jev must still select the
actions that execute them.

The simulator accepts commands within X=[0.20, 0.75], Y=[-0.40, 0.40] and
Z=[0.045, 0.50] metres. Commands beyond those bounds are clipped and reported.
An episode ends at success or 120 decisions. Success requires prior two-finger
contact, centering in the cyan socket, insertion depth, upright orientation,
open fingers and a settled peg.

## Recorded evaluation

All runs used the unchanged `tianxinwei/JevAny-27B-SFT` checkpoint at revision
`ad7b48b7056a9742f54aacc9b98b6b46dc2ce167`, camera images, and argmax selection
from the full action set. No motor commands were scripted or substituted in
model runs. Responses confirmed that nonempty image tensors reached the model.

Four harness designs were first compared on seeds 17 and 29:

| Information supplied | Completed | Decisions |
|---|---:|---|
| Stage and target pose | 2/2 | 58, 58 |
| Stage, target pose and signed error | 2/2 | 48, 50 |
| Above, plus each action's predicted position error | 2/2 | 48, 48 |
| Above, plus a specified coordinate to correct first | 2/2 | 45, 45 |

The published harness uses stage, target pose and signed error. It completed
all eight additional seeds on Python 3.12 and PyBullet 3.2.7:

| Seed | Decisions | Result |
|---:|---:|---|
| 41 | 58 | All six checks passed |
| 53 | 50 | All six checks passed |
| 67 | 47 | All six checks passed |
| 79 | 48 | All six checks passed |
| 101 | 50 | All six checks passed |
| 113 | 46 | All six checks passed |
| 127 | 46 | All six checks passed |
| 137 | 46 | All six checks passed |

The packaged replay shows seed 41, the first additional seed. The
[compressed decision archive](demos/arm-checkpoint-runs.json.gz) retains all
16 trials, their requests, probabilities and before/after measurements.
These seeds vary the peg's initial X coordinate by a few millimetres; this is
a small fixture evaluation, not a benchmark of arbitrary robot manipulation or
independent task planning.

## Using the harness directly

`PegInsertion.decision_request(model, history)` returns the model's structured
choice request. Attach an image from `env.render()` using the client's media
interface, submit the request, and pass the returned action to `env.step()`.
`reset()` resets both the physical environment and its harness.

The browser does this automatically. With `--media-root`, it writes a temporary
PNG beneath the model server's `JEVANY_MEDIA_ROOT`, keeps it available during
the request, and removes it on success or error. Both processes must share
that filesystem path. The server's existing file validation remains in effect.
