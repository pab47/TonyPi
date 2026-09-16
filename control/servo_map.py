#!/usr/bin/env python3
"""
Sim <-> hardware joint mapping for TonyPi Pro bus servos 1-16.

    pulse = ZERO[id] + SIGN[id] * angle_rad * PULSE_PER_RAD
    angle = SIGN[id] * (pulse - ZERO[id]) / PULSE_PER_RAD

ZERO  = pulse read with the robot in the sim's zero (standing) pose
SIGN  = +1 if increasing pulse moves the joint in MuJoCo's positive direction
Scale = 1000 pulses per 240 deg (Hiwonder LX-series spec; verify later)

Usage:
    from servo_map import sim_to_pulse, pulse_to_sim, SERVO_IDS
    pulses = sim_to_pulse(q)          # q: 16 sim angles (rad), order = servo ids 1..16
    q      = pulse_to_sim(pulses)     # inverse
    sim_to_pulse(0.3, servo_id=8)     # single joint
"""
import math
import numpy as np

DEG_PER_PULSE = 240.0 / 1000.0
PULSE_PER_RAD = 1000.0 / math.radians(240.0)     # ~238.73

SERVO_IDS = list(range(1, 17))

# pulse at sim zero pose (measured, robot standing)
# ZERO = {
#      1: 498,  2: 479,  3: 247,  4: 413,
#      5: 508,  6: 500,  7: 870,  8: 664,
#      9: 495, 10: 515, 11: 762, 12: 584,
#     13: 500, 14: 516, 15: 129, 16: 337,
# }
ZERO = {
     1: 498,  2: 479,  3: 247,  4: 413,
     5: 508,  6: 500,  7: 870,  8: 664,
     9: 495, 10: 515, 11: 762, 12: 584,
    13: 500, 14: 516, 15: 129, 16: 337,
}

# +1: pulse up == sim angle up ; -1: opposite
# SIGN = {
#      1: -1,  2: +1,  3: +1,  4: -1,
#      5: +1,  6: -1,  7: -1,  8: +1,
#      9: -1, 10: -1, 11: -1, 12: +1,
#     13: +1, 14: -1, 15: -1, 16: -1,
# }
SIGN = {
     1: +1,  2: +1,  3: +1,  4: -1,
     5: -1,  6: -1,  7: -1,  8: +1,
     9: +1, 10: -1, 11: -1, 12: +1,
    13: -1, 14: -1, 15: -1, 16: -1,
}

_ZERO = np.array([ZERO[i] for i in SERVO_IDS], dtype=float)
_SIGN = np.array([SIGN[i] for i in SERVO_IDS], dtype=float)

# angle range each servo can physically reach given its zero (rad) — the
# servo itself stops at pulse 0 and 1000; the joint's link-on-link limit
# will be tighter and should be measured separately.
ANGLE_MIN = {i: min(SIGN[i] * (0 - ZERO[i]), SIGN[i] * (1000 - ZERO[i])) / PULSE_PER_RAD for i in SERVO_IDS}
ANGLE_MAX = {i: max(SIGN[i] * (0 - ZERO[i]), SIGN[i] * (1000 - ZERO[i])) / PULSE_PER_RAD for i in SERVO_IDS}


def sim_to_pulse(q, servo_id=None, clip=True):
    """q in radians. Vector of 16 (servo order 1..16) or scalar with servo_id."""
    if servo_id is not None:
        p = ZERO[servo_id] + SIGN[servo_id] * q * PULSE_PER_RAD
        p = int(round(p))
        return max(0, min(1000, p)) if clip else p
    q = np.asarray(q, dtype=float)
    p = _ZERO + _SIGN * q * PULSE_PER_RAD
    if clip:
        p = np.clip(p, 0, 1000)
    return np.rint(p).astype(int)


def pulse_to_sim(pulse, servo_id=None):
    """Inverse: pulse (int or vector of 16) -> radians."""
    if servo_id is not None:
        return SIGN[servo_id] * (pulse - ZERO[servo_id]) / PULSE_PER_RAD
    pulse = np.asarray(pulse, dtype=float)
    return _SIGN * (pulse - _ZERO) / PULSE_PER_RAD


if __name__ == '__main__':
    print(f"{'id':>2} {'zero':>5} {'sign':>4} | {'min deg':>8} {'max deg':>8}")
    for i in SERVO_IDS:
        print(f"{i:2d} {ZERO[i]:5d} {SIGN[i]:+4d} | "
              f"{math.degrees(ANGLE_MIN[i]):8.1f} {math.degrees(ANGLE_MAX[i]):8.1f}")
    # round-trip check
    q = np.zeros(16)
    assert np.array_equal(sim_to_pulse(q), _ZERO.astype(int))
    q = np.random.uniform(-0.5, 0.5, 16)
    assert np.allclose(pulse_to_sim(sim_to_pulse(q, clip=False)), q, atol=0.5 / PULSE_PER_RAD)
    print("round-trip OK")
