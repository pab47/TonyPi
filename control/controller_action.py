#!/usr/bin/env python3
"""
controller_action.py -- play back Hiwonder ".d6a" action-group files on the
TonyPi (sim or hardware), using the same sim.getAngles / sim.setAngles /
sim.getIMU / sim.sleep interface as controller.py.

WHAT A .d6a FILE IS
--------------------
A ".d6a" action group (created with Hiwonder's "Action Group Editor" /
saved from the app) is just a SQLite database with a single table:

    ActionGroup(
        [Index]  INTEGER PRIMARY KEY,   -- frame number, 1..N
        Time     INT,                   -- ms to ramp from the previous
                                         -- frame into this one
        Servo1   INT,                   -- raw servo pulse, 0..1000
        Servo2   INT,
        ...
        Servo18  INT
    )

Each row is one "waypoint": move every listed servo to its raw pulse value,
taking `Time` milliseconds to get there, then hold before starting the next
row. That is exactly the (angle, time_in_ms) contract of sim.setAngles(), so
playback is: for every frame, call setAngles() for each servo, then sleep
for Time seconds before moving on to the next frame.

RAW UNITS -> RADIANS
---------------------
The raw values are 0..1000 across each servo's ~240 degree travel, but the
pulse that means "sim angle 0" is NOT the same 500 for every servo, and
some servos are mounted mirrored (increasing pulse = decreasing sim angle).
That per-servo calibration already lives in this project's servo_map.py
(the same module tonypi_hw.py uses to talk to the real robot):

    angle_rad = SIGN[servo_id] * (raw_value - ZERO[servo_id]) / PULSE_PER_RAD

so this script imports servo_map.pulse_to_sim() rather than assuming a
single symmetric (center=500, sign=+1) formula -- a uniform formula gets
the ~0.0042 rad/count scale right but silently drives half the joints
backwards and off zero, since it ignores each servo's ZERO/SIGN.

USAGE
-----
    python3 controller_action.py stand.d6a
    python3 controller_action.py stand_up_back.d6a stand_up_front.d6a
    python3 controller_action.py --list stand.d6a      # just print the frames, don't move the robot

With no arguments it looks for stand.d6a / stand_up_back.d6a /
stand_up_front.d6a next to this script and plays whichever it finds, in
that order, pausing between each.
"""
import argparse
import math
import os
import sqlite3
import sys

try:
    # servo_map.py must sit next to this script (it's the same per-servo
    # pulse<->radian calibration tonypi_hw.py uses for the real robot).
    from servo_map import pulse_to_sim, SERVO_IDS as CALIBRATED_SERVO_IDS
except ImportError:
    pulse_to_sim = None
    CALIBRATED_SERVO_IDS = []

# Servo-id -> friendly-name table. Duplicated (not imported) from
# TonyPi_sw/tonypi_sim.py and TonyPi_hw/tonypi_hw.py, which both build the
# exact same dict -- importing it from either one here would drag in that
# backend's own dependencies (mujoco for the sim side) even when this
# script is only using the other backend, which is what broke --hw on the
# robot (print_status() was pulling in TonyPi_sw.tonypi_sim -> mujoco).
_LEFT_NAMES = ['ANK_ROLL', 'ANK_PITCH', 'KNEE', 'HIP_PITCH',
               'HIP_ROLL', 'ELBOW', 'SH_ROLL', 'SH_PITCH']
NAME = {i + 1: 'L_' + n for i, n in enumerate(_LEFT_NAMES)}
NAME.update({i + 9: 'R_' + n for i, n in enumerate(_LEFT_NAMES)})


def raw_to_rad(raw, servo_id):
    """
    Convert a raw 0..1000 servo pulse value to sim/MuJoCo radians for one
    specific servo, using this project's per-servo calibration (zero-pulse
    offset + direction sign) in servo_map.py.
    """
    if pulse_to_sim is None:
        raise RuntimeError(
            "servo_map.py not found next to controller_action.py -- it's "
            "required for a correct pulse-to-radian conversion (each servo "
            "has its own zero offset and direction sign).")
    return float(pulse_to_sim(raw, servo_id=servo_id))


def load_action_group(path):
    """
    Read a .d6a action-group file and return it as a list of frames:

        [(time_ms, {servo_id: raw_value, ...}), ...]

    ordered by the file's [Index] column. Only the ServoN columns that
    actually exist in the file are included.
    """
    con = sqlite3.connect(path)
    try:
        cur = con.cursor()
        cur.execute("PRAGMA table_info(ActionGroup)")
        columns = [row[1] for row in cur.fetchall()]
        servo_cols = [c for c in columns if c.startswith('Servo')]

        cur.execute(
            'SELECT Time, {} FROM ActionGroup ORDER BY [Index]'.format(
                ', '.join(servo_cols))
        )
        frames = []
        for row in cur.fetchall():
            time_ms = row[0]
            servos = {}
            for col, value in zip(servo_cols, row[1:]):
                servo_id = int(col[len('Servo'):])
                servos[servo_id] = value
            frames.append((time_ms, servos))
        return frames
    finally:
        con.close()


def play_action_group(sim, frames, servo_ids=None, verbose=True):
    """
    Play back the frames returned by load_action_group() on `sim`.

    servo_ids restricts playback to servo numbers the sim/hardware actually
    has (defaults to SERVO_IDS from tonypi_sim); any other columns in the
    file are ignored (e.g. a head pan/tilt channel the sim doesn't model).
    """
    if pulse_to_sim is None:
        # Fail loudly. Without this check, the servo_ids &= ... line below
        # would quietly become an empty set and every servo would just be
        # skipped for the rest of this function -- no error, no movement,
        # which looks exactly like "the conversion isn't happening" with no
        # clue why.
        raise RuntimeError(
            "servo_map.py not found next to controller_action.py -- it's "
            "required for a correct pulse-to-radian conversion (each servo "
            "has its own zero offset and direction sign). Nothing was played.")

    if servo_ids is None:
        servo_ids = set(CALIBRATED_SERVO_IDS)
    else:
        servo_ids = set(servo_ids)
    # Only servos servo_map.py has a ZERO/SIGN calibration for can be
    # converted correctly; anything else (e.g. a head pan/tilt column some
    # .d6a files carry) is skipped rather than driven with a guessed offset.
    servo_ids &= set(CALIBRATED_SERVO_IDS)

    for frame_no, (time_ms, servos) in enumerate(frames, start=1):
        time_ms = max(time_ms, 1)   # guard against a 0 ms frame
        if verbose:
            print('  frame {:2d}/{:2d}  {:4d} ms'.format(
                frame_no, len(frames), time_ms))
        for servo_id, raw in servos.items():
            if servo_id not in servo_ids:
                continue
            sim.setAngles(servo_id, raw_to_rad(raw, servo_id), time_ms)
        sim.sleep(time_ms / 1000.0)
        if verbose:
            # TEMP DIAGNOSTIC: per-frame IMU pitch/roll/gyro, so a stalled
            # get-up (legs reach the target angle but the torso never
            # rotates off the floor) is visible frame-by-frame instead of
            # only before/after the whole file.
            a, w = sim.getIMU()
            pitch, roll = sim.pitch_roll_from_accel(a)
            gyro_mag = (w['x'] ** 2 + w['y'] ** 2 + w['z'] ** 2) ** 0.5
            print('    -> pitch {:+6.1f} deg  roll {:+6.1f} deg  '
                  '|gyro| {:6.1f} deg/s'.format(pitch, roll, gyro_mag))


def print_status(sim):
    for i in CALIBRATED_SERVO_IDS:
        q = sim.getAngles(i)
        print('{:2d} {:<11s} angle: {:+7.3f} rad  {:+7.1f} deg'
              .format(i, NAME[i], q, math.degrees(q)))
    a, w = sim.getIMU()
    pitch, roll = sim.pitch_roll_from_accel(a)
    print('IMU accel [g]  body: x {:+6.3f}  y {:+6.3f}  z {:+6.3f}'.format(a['x'], a['y'], a['z']))
    print('IMU gyro [deg/s] body: x {:+7.2f} y {:+7.2f} z {:+7.2f}'.format(w['x'], w['y'], w['z']))
    print('IMU pitch {:+6.1f} deg   roll {:+6.1f} deg\n'.format(pitch, roll))


def action_search_dirs(extra_dirs=()):
    """
    Directories to look for a bare action-group filename in, in priority
    order: any caller-supplied extra_dirs (e.g. --action-dir) first, then
    next to this script, then every folder depth this project has actually
    used so the same filename works unmodified on both layouts:
      - Mac dev checkout : .../tonypi/controller_action.py + .../action/*.d6a
                            (one level up from this script, into "action")
      - robot deployment : .../ActionGroups/*.d6a sitting two levels up
                            from wherever this script ends up on the Pi
    """
    here = os.path.dirname(os.path.abspath(__file__))
    return list(extra_dirs) + [
        here,
        os.path.join(here, '..', 'action'),
        os.path.join(here, '..', 'ActionGroups'),
        os.path.join(here, '..', '..', 'action'),
        os.path.join(here, '..', '..', 'ActionGroups'),
    ]


def resolve_action_path(name_or_path, extra_dirs=()):
    """
    Turn a filename or path typed on the command line into a real, absolute
    path. If it already exists as given (relative to the cwd, or absolute),
    it's used as-is -- so "../action/stand.d6a" style paths still work
    unchanged. Otherwise its basename is looked up in action_search_dirs()
    so a bare "stand.d6a" resolves correctly regardless of how deep the
    action folder sits on this particular machine. If it's not found
    anywhere, the best-guess absolute path is still returned so main()'s
    missing-file check can report a clear error instead of a confusing one.
    """
    if os.path.exists(name_or_path):
        return os.path.abspath(name_or_path)
    base = os.path.basename(name_or_path)
    for d in action_search_dirs(extra_dirs):
        candidate = os.path.join(d, base)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(name_or_path)


def default_action_files(extra_dirs=()):
    search_dirs = action_search_dirs(extra_dirs)
    #names = ['stand.d6a', 'wave.d6a'] #works
    #names = ['stand.d6a', 'right_kick.d6a'] #works
    #names = ['stand.d6a', 'left_kick.d6a'] #works
    #names = ['stand.d6a', 'stepping.d6a'] #works
    #names = ['stand.d6a', 'chest.d6a'] #works
    #names = ['stand.d6a', 'bow.d6a'] #works
    #names = ['stand.d6a', 'wing_chun.d6a'] #works
    #names = ['stand.d6a', 'sit_ups.d6a'] #works
    #names = ['stand.d6a', 'twist.d6a'] #works
    #names = ['stand.d6a', 'twist.d6a'] #works
    names = ['stand.d6a', '0.d6a'] #works

    #names = ['stand.d6a', 'squat_down.d6a','squat_up.d6a'] #works
    #names = ['stand.d6a', 'go_forward.d6a'] #works
    #names = ['stand.d6a', 'move_up.d6a'] #works 

    #names = ['fall_back.d6a','stand_up_back.d6a'] #does not work
    #names = ['fall_forward.d6a','stand_up_front.d6a'] #does not work
    #names = ['stand_slow.d6a'] #works 

    found = []
    for name in names:
        for d in search_dirs:
            candidate = os.path.join(d, name)
            if os.path.exists(candidate):
                found.append(candidate)
                break
        else:
            print(f"warning: default action '{name}' not found in any of "
                  f"{search_dirs} -- skipping it", file=sys.stderr)
    return found


def make_controller(action_paths):
    """Build a controller(sim) function that plays the given .d6a files in order."""

    def controller(sim):
        sim.sleep(0.5)   # let the robot settle on the floor
        print('--- start ---')
        print_status(sim)

        for path in action_paths:
            print('=== playing {} ==='.format(os.path.basename(path)))
            frames = load_action_group(path)
            play_action_group(sim, frames)
            print('--- after {} ---'.format(os.path.basename(path)))
            print_status(sim)

    return controller


def list_frames(path):
    calibrated = set(CALIBRATED_SERVO_IDS)
    frames = load_action_group(path)
    print('{}  ({} frame(s))'.format(path, len(frames)))
    for frame_no, (time_ms, servos) in enumerate(frames, start=1):
        parts = []
        for sid, raw in sorted(servos.items()):
            if sid in calibrated:
                parts.append('S{}={} ({:+.3f} rad)'.format(sid, raw, raw_to_rad(raw, sid)))
            else:
                parts.append('S{}={} (uncalibrated, not played)'.format(sid, raw))
        print('  [{:2d}] {:4d} ms : {}'.format(frame_no, time_ms, ', '.join(parts)))


def main():
    parser = argparse.ArgumentParser(
        description='Play back Hiwonder .d6a action-group files on the TonyPi.')
    parser.add_argument('files', nargs='*',
                         help='.d6a action-group file(s) to play, in order')
    parser.add_argument('--list', action='store_true',
                         help='print the frames in each file instead of moving the robot')
    parser.add_argument('--hw', action='store_true',
                         help="drive the REAL robot via TonyPi_hw/tonypi_hw.py instead of "
                              "the MuJoCo sim. Run this only on the robot's own Raspberry Pi "
                              "-- it needs hiwonder.Board / hiwonder.Mpu6050, which only exist "
                              "on that Pi's image, not on a dev PC/Mac.")
    parser.add_argument('--duration', type=float, default=None,
                         help='stop after this many seconds (sim default: 15s; '
                              'hardware default: run until the action(s) finish, then return)')
    parser.add_argument('--action-dir', action='append', default=[],
                         help='extra folder to search for a bare .d6a filename in '
                              '(repeatable). Use this if the action folder is somewhere '
                              "action_search_dirs() doesn't already guess -- e.g. a "
                              'nonstandard layout on the robot.')
    args = parser.parse_args()

    extra_dirs = args.action_dir
    action_paths = args.files if args.files else default_action_files(extra_dirs)
    if not action_paths:
        parser.error('no .d6a files given, and none found next to this script')

    # Resolve every filename to a real absolute path now, before TonyPiSim.run()
    # changes the working directory (sim, to resolve MuJoCo mesh files under
    # model/) or the controller callback otherwise runs from a different cwd.
    # A bare name like "stand.d6a" is looked up via action_search_dirs() /
    # --action-dir; a path that already exists (e.g. "../action/stand.d6a")
    # is used exactly as given.
    action_paths = [resolve_action_path(p, extra_dirs) for p in action_paths]
    missing = [p for p in action_paths if not os.path.exists(p)]
    if missing:
        parser.error('file(s) not found: {}'.format(', '.join(missing)))

    if args.list:
        for path in action_paths:
            list_frames(path)
        return

    if args.hw:
        # Real hardware. TonyPi_hw/ (the whole folder, alongside this script
        # and servo_map.py) must be copied onto the robot -- this import
        # will fail on a PC/Mac with no hiwonder package, which is expected;
        # it's only meant to succeed when run ON the robot.
        from TonyPi_hw.tonypi_hw import TonyPiHW as TonyPiSim
        sim = TonyPiSim()
        duration = args.duration   # None = play the full action(s), then return
    else:
        from TonyPi_sw.tonypi_sim import TonyPiSim
        sim = TonyPiSim(viewer=True)     # viewer=False, realtime=False -> fast headless run
        sim.set_camera(azimuth=120, elevation=-15, distance=0.9, lookat=[0.0, 0.0, 0.17])
        duration = args.duration if args.duration is not None else 20

    controller = make_controller(action_paths)
    sim.run(controller, duration=duration)   # sim: None = stay open until viewer closed; hw: None = run to completion


if __name__ == '__main__':
    main()
