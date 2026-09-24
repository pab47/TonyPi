# TonyPi Pro — Sim-to-Real Action Playback & Custom Behaviors

A MuJoCo simulator for the [Hiwonder TonyPi Pro](https://www.hiwonder.com/products/tonypi) that mirrors the robot's real hardware interface. There are two different kinds of behavior in this repo — keep them separate in your head:

1. **Real to Sim.** `controller_action.py`, in sim or on hardware, replays the Hiwonder-authored `.d6a` Action Group files (bow, wave, sit-ups, walking, etc.) that ship with the robot. This is just playing back vendor-created motions, and it's what the sim-vs-hardware comparison video below shows.
2. **Custom IK-generated behaviors.** `controller_stand.py` and `controller_walk.py`, run in sim, are state-machine-based behaviors I designed myself, using inverse kinematics (via [mink](https://github.com/kevinzakka/mink) + MuJoCo) rather than hand choreography. They get baked into `.d6a` files and then played on hardware with the same `controller_action.py --hw` used for stock playback. See [Section 4](#4-custom-ik-generated-behaviors-controller_standpy--controller_walkpy) below. A new YouTube video demoing these is in progress — placeholder for now.

Stock Action Group files: https://github.com/Hiwonder/TonyPi/tree/main/ActionGroups
Stock hardware example scripts: https://github.com/Hiwonder/TonyPi/tree/main/Example

Sim-vs-hardware comparison video (stock playback, flavor 1):

[![Sim vs hardware comparison](https://img.youtube.com/vi/VVifUPAAEM0/maxresdefault.jpg)](https://youtu.be/VVifUPAAEM0)

Custom walking/standing behaviors (flavor 2):

[![Custom IK-generated walking/standing behaviors](https://img.youtube.com/vi/xbgYgt3ijhk/maxresdefault.jpg)](https://youtu.be/xbgYgt3ijhk)

## Folder layout

```
./                          <-- this file lives at the repo root
  action/                   .d6a Action Group files (stand.d6a, chest.d6a, sit_ups.d6a, wave.d6a,
                             tonypi_stand_left.d6a, tonypi_stand_right.d6a,
                             tonypi_walk_forward.d6a, tonypi_walk_backward.d6a, ...)
  model/                    MuJoCo model: scene.xml + robot.xml + assets/
  control/
    controller_action.py    play back .d6a files, in sim or on hardware (--hw)
    controller.py            write your own control code against the same interface
    controller_stand.py      IK-based one-leg-stand state machine (sim); bakes to .d6a with --record
    controller_walk.py       IK-based forward/backward walk state machine (sim); bakes to .d6a with --record
    servo_map.py             per-servo raw-pulse <-> radian calibration
    tonypi_demo.sh            11-move demo sequence, SIMULATION
    tonypi_demo_hw.sh         same 11-move sequence, HARDWARE
    TonyPi_sw/tonypi_sim.py      TonyPiSim  -- MuJoCo backend (do not edit)
    TonyPi_hw/tonypi_hw.py       TonyPiHW   -- real-robot backend (do not edit)
    TonyPi_hw/read_sensors.py    hardware-only: reads the IMU and joint angles
    TonyPi_hw/write_servos.py    hardware-only: writes to the servos
```

All commands below are run from inside `control/`:

```bash
cd control
```

## 1. Simulation (stock playback)

One-time setup:

```bash
pip3 install mujoco
```

Play a single Action Group (or a sequence of them) against the MuJoCo model. Files are looked up by bare name in `../action/` automatically:

```bash
# chest workout
python3 controller_action.py stand.d6a chest.d6a

# sit-ups
python3 controller_action.py stand.d6a sit_ups.d6a
```

Run the full demo sequence (wave, wing chun, uppercut, chest, bow, twist, march, sit-ups, lift/put-down, walk forward, fall/recover):

```bash
chmod +x tonypi_demo.sh
./tonypi_demo.sh
```

Useful flags:

```bash
python3 controller_action.py --list stand.d6a          # print the frames instead of moving anything
python3 controller_action.py --duration 8 stand.d6a wave.d6a   # stop the viewer after N seconds
```

Close the MuJoCo viewer window to end a run early.

## 2. Hardware (stock playback and custom behaviors)

Deploy: copy this whole `control/` folder onto the robot's Raspberry Pi, into `TonyPi/Example/` (see the [stock Example folder](https://github.com/Hiwonder/TonyPi/tree/main/Example) for reference on where that lives).

On the robot, run the same commands with `--hw`, which drives the real servos via `TonyPi_hw/tonypi_hw.py` instead of MuJoCo:

```bash
# chest workout
python3 controller_action.py --hw stand.d6a chest.d6a

# sit-ups
python3 controller_action.py --hw stand.d6a sit_ups.d6a
```

Run the full demo sequence on hardware:

```bash
chmod +x tonypi_demo_hw.sh
./tonypi_demo_hw.sh
```

`--hw` mode needs the `hiwonder.Board` / `hiwonder.Mpu6050` packages that only exist on the robot's own Raspberry Pi image — it will not import on a dev PC/Mac, which is expected.

`TonyPi_hw/` also has two standalone hardware-only scripts (not used by `controller_action.py`, and they will not run anywhere but on the robot):

* `TonyPi_hw/read_sensors.py` — reads the IMU and joint angles from the real hardware.
* `TonyPi_hw/write_servos.py` — writes commanded positions to the real servos.

**Custom behaviors on hardware:** `controller_stand.py` and `controller_walk.py` only run in sim (they need `mink`, which needs MuJoCo, which is not installed on the rpi4b). To run a custom behavior on the robot, record it to a `.d6a` file first (`--record`, see Section 4) and play it back with the same `controller_action.py --hw` command used above, e.g.:

```bash
python3 controller_action.py --hw tonypi_walk_forward.d6a
```

Four such files are already recorded and ready to run on hardware today — `tonypi_stand_left.d6a`, `tonypi_stand_right.d6a`, `tonypi_walk_forward.d6a`, and `tonypi_walk_backward.d6a`.

## 3. Comparing sim vs. hardware

Run the same Action Group(s) both ways (with and without `--hw`) and compare the resulting motion. This is the core sim-to-real check: since both backends share the identical `getAngles` / `setAngles` / `getIMU` interface below, any given `.d6a` file drives the exact same commanded trajectory in both places — differences you see are the sim-to-real gap (contact, friction, PD gains, calibration, etc.), not a difference in what was commanded. This applies equally to stock Action Groups and to `.d6a` files baked from the custom controllers in Section 4.

## 4. Custom IK-generated behaviors (`controller_stand.py` / `controller_walk.py`)

Instead of hand-choreographing each frame, these controllers modulate the center of mass and foot location using inverse kinematics and plan a static trajectory from that. Both `controller_stand.py` and `controller_walk.py` are state machines built on [mink](https://github.com/kevinzakka/mink) (an IK solver for MuJoCo).

Since mink needs MuJoCo, and MuJoCo isn't installed on the rpi4b, these controllers only run on a dev machine, in sim. To get the resulting motion onto the robot, you record it to a `.d6a` file (`--record`) and play that back on hardware with `controller_action.py --hw`, exactly as in Section 2.

**Installing mink:**

```bash
pip3 install mink
```

(This is in addition to the `mujoco` install from Section 1 — mink builds on it.)

### Walking controller

1. To play in MuJoCo:

```bash
python3 controller_walk.py                                  # live viewer, all defaults (3 steps, forward)
python3 controller_walk.py --forward                        # explicit forward (same as default)
python3 controller_walk.py --backward                       # walk backward instead, same 3 steps
python3 controller_walk.py --steps 4                        # walk 4 individual foot placements
python3 controller_walk.py --steps 4 --backward             # 4 steps, backward
python3 controller_walk.py --record                         # bake the 3-step forward walk to action/tonypi_walk.d6a
python3 controller_walk.py --record --backward              # bake the 3-step backward walk to action/tonypi_walk_backward.d6a
python3 controller_walk.py --record --steps 8               # bake an 8-step forward walk to action/tonypi_walk.d6a
python3 controller_walk.py --record --steps 8 --backward    # bake an 8-step backward walk to action/tonypi_walk_backward.d6a
```

2. To record to `action/tonypi_walk.d6a`:

```bash
python3 controller_walk.py --record
```

3. Move the file to the robot's `ActionGroups` folder, then on the robot navigate to `TonyPi/Example/tonypi` and run:

```bash
python3 controller_action.py --hw tonypi_walk.d6a
```

### Standing on one leg controller

1. To play in MuJoCo:

```bash
python3 controller_stand.py                                   # live viewer, all defaults (left leg)
python3 controller_stand.py --leg right                       # stand on the right leg instead
python3 controller_stand.py --leg left                        # stand on the left leg (same as the default)
python3 controller_stand.py --stand-time 6                    # hold the one-leg pose longer (default 3 s)
python3 controller_stand.py --shift-time 5 --powerup-time 6   # slower COM shift / slower initial crouch, if needed
python3 controller_stand.py --leg right --record              # bake the right-leg version to action/pranav_stand.d6a
python3 controller_stand.py --leg right --powerup-time 4 --shift-time 3 --stand-time 3 --settle-time 1.5 --record
```

2. To record to `action/tonypi_stand.d6a`:

```bash
python3 controller_stand.py --record
```

3. Move the file to the robot's `ActionGroups` folder, then on the robot navigate to `TonyPi/Example/tonypi` and run:

```bash
python3 controller_action.py --hw tonypi_stand.d6a
```

### Already-recorded files

These four `.d6a` files are already baked and ready to copy over and run on hardware without recording anything yourself:

* `tonypi_stand_left.d6a`
* `tonypi_stand_right.d6a`
* `tonypi_walk_forward.d6a`
* `tonypi_walk_backward.d6a`

## Writing your own controller from scratch

`controller.py` is the file to edit if you want to write custom motions/logic that aren't already covered by `controller_stand.py` / `controller_walk.py`, rather than replay a stock Action Group. It uses the same interface as `controller_action.py`.

Unlike `controller_action.py`, `controller.py` picks sim vs. hardware by which import line at the top of the file is uncommented, not by a command-line flag — comment/uncomment as needed, then run `python3 controller.py` the same way on either side:

```python
from TonyPi_sw.tonypi_sim import TonyPiSim, SERVO_IDS, NAME   # for simulation
```

```python
from TonyPi_hw.tonypi_hw import TonyPiHW as TonyPiSim, SERVO_IDS, NAME   # for hardware
```

The interface itself:

| call | returns / does |
|---|---|
| `sim.getAngles(servo_No)` | joint angle in **radians**, servo ids 1..16 |
| `sim.setAngles(servo_No, desired_angle, time_in_ms)` | non-blocking: PD set-point ramps linearly from the current measured angle to `desired_angle` over `time_in_ms` |
| `sim.getIMU()` | `(accel, gyro)` dicts with keys `x,y,z`; accel in g, gyro in deg/s, body frame (X forward, Y left, Z up) |
| `sim.sleep(seconds)` | wait on the simulation clock (== `time.sleep` in real-time mode) |
| `sim.run(controller, duration=None)` | run; `duration` = seconds of sim time before the window closes |
| `sim.stop()` | end the run from inside the controller |

Servo ids: 1 L_ANK_ROLL, 2 L_ANK_PITCH, 3 L_KNEE, 4 L_HIP_PITCH, 5 L_HIP_ROLL, 6 L_ELBOW, 7 L_SH_ROLL, 8 L_SH_PITCH; 9..16 the same for the right side. Head pan/tilt and the grippers are not controlled (held at 0 by their PD).

To move `controller.py` to the real robot, copy it along with `TonyPi_hw/` and `servo_map.py`, then switch to the hardware import line shown above.

## Notes

* Every servo is the `<position>` actuator from `robot.xml` (kp=12, dampratio=0.6, torque cap 1.667 N·m) — the "default PD controller".
* The sim window is MuJoCo's managed viewer: Space pauses, the right panel has a speed slider, Backspace resets.
* Physics timestep is 4 ms.
* `.d6a` files are SQLite databases (Hiwonder's Action Group Editor format); raw servo pulses (0-1000) are converted to radians via each servo's calibration in `servo_map.py`. This is true whether the file came from Hiwonder (flavor 1) or was baked with `--record` from `controller_stand.py` / `controller_walk.py` (flavor 2).
* Acknowledgement: Salvador Echeveste created the xml model from .stp file provided by HiWonder. This simulation is based on his work.
