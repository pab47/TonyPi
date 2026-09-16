# TonyPi Pro — Sim-to-Real Action Playback

A MuJoCo simulator for the [Hiwonder TonyPi Pro](https://www.hiwonder.com/products/tonypi) that mirrors the robot's real hardware interface. The same Hiwonder `.d6a` Action Group files that drive the physical robot's open-loop motions (bow, wave, sit-ups, walking, etc.) can be replayed here in simulation, so sim and hardware behavior can be compared side by side.

Stock Action Group files: https://github.com/Hiwonder/TonyPi/tree/main/ActionGroups
Stock hardware example scripts: https://github.com/Hiwonder/TonyPi/tree/main/Example

Sim-vs-hardware comparison video:

[![Sim vs hardware comparison](https://img.youtube.com/vi/VVifUPAAEM0/maxresdefault.jpg)](https://youtu.be/VVifUPAAEM0)

## Folder layout

```
./                          <-- this file lives at the repo root
  action/                   .d6a Action Group files (stand.d6a, chest.d6a, sit_ups.d6a, wave.d6a, ...)
  model/                    MuJoCo model: scene.xml + robot.xml + assets/
  control/
    controller_action.py    play back .d6a files, in sim or on hardware (--hw)
    controller.py            write your own control code against the same interface
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

## 1. Simulation

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

## 2. Hardware

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

## 3. Comparing sim vs. hardware

Run the same Action Group(s) both ways (with and without `--hw`) and compare the resulting motion. This is the core sim-to-real check: since both backends share the identical `getAngles` / `setAngles` / `getIMU` interface below, any given `.d6a` file drives the exact same commanded trajectory in both places — differences you see are the sim-to-real gap (contact, friction, PD gains, calibration, etc.), not a difference in what was commanded.

## Writing your own controller

`controller.py` is the file to edit if you want to write custom motions/logic rather than replay a stock Action Group. It uses the same interface as `controller_action.py`.

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
* `.d6a` files are SQLite databases (Hiwonder's Action Group Editor format); raw servo pulses (0-1000) are converted to radians via each servo's calibration in `servo_map.py`.
* Acknowledgement: Salvador Echeveste created the xml model from .stp file provided by HiWonder. This simulation is based on his work.
