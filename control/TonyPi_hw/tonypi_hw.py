#!/usr/bin/env python3
"""
tonypi_hw.py -- the SAME interface as tonypi_sim.TonyPiSim, but talking to
the real TonyPi Pro (hiwonder Board + MPU-6050).  Copy this file together
with servo_map.py and your controller.py onto the robot, then change one
line in controller.py:

    from tonypi_hw import TonyPiHW as TonyPiSim

Everything else in controller.py stays as it is.  Untested on the robot as
written here; it is assembled from read_angles.py / write_angles.py.
"""
import math
import time

import hiwonder.Board as Board
import hiwonder.Mpu6050 as Mpu6050
from servo_map import pulse_to_sim, sim_to_pulse, SERVO_IDS   # noqa: F401

_LEFT_NAMES = ['ANK_ROLL', 'ANK_PITCH', 'KNEE', 'HIP_PITCH',
               'HIP_ROLL', 'ELBOW', 'SH_ROLL', 'SH_PITCH']
NAME = {i + 1: 'L_' + n for i, n in enumerate(_LEFT_NAMES)}
NAME.update({i + 9: 'R_' + n for i, n in enumerate(_LEFT_NAMES)})


def chip_to_body(v):
    """MPU-6050 chip axes -> body axes (X forward, Y left, Z up)."""
    return {'x': -v['z'], 'y': -v['x'], 'z': v['y']}


class _Stopped(Exception):
    pass


class TonyPiHW:
    def __init__(self, **kwargs):          # kwargs ignored (viewer=..., etc.)
        self._duration = None
        self._stop = False
        self.mpu = Mpu6050.mpu6050(0x68)
        self.mpu.set_accel_range(self.mpu.ACCEL_RANGE_2G)
        self.mpu.set_gyro_range(self.mpu.GYRO_RANGE_2000DEG)
        self._t0 = time.time()

    # --- same three calls as the sim -------------------------------------
    def getAngles(self, servo_No=None):
        if servo_No is None:
            return [self.getAngles(i) for i in SERVO_IDS]
        pulse = Board.getBusServoPulse(servo_No)
        return float(pulse_to_sim(pulse, servo_id=servo_No))

    def setAngles(self, servo_No, desired_angle, time_in_ms):
        if isinstance(servo_No, (list, tuple)):
            for s, q in zip(servo_No, desired_angle):
                self.setAngles(int(s), float(q), time_in_ms)
            return
        pulse = sim_to_pulse(desired_angle, servo_id=servo_No)
        Board.setBusServoPulse(servo_No, pulse, int(time_in_ms))

    def getIMU(self):
        accel = chip_to_body(self.mpu.get_accel_data(g=True))   # g
        gyro = chip_to_body(self.mpu.get_gyro_data())           # deg/s
        return accel, gyro

    # --- helpers matching the sim ----------------------------------------
    @staticmethod
    def pitch_roll_from_accel(accel):
        pitch = math.degrees(math.atan2(-accel['x'], math.hypot(accel['y'], accel['z'])))
        roll = math.degrees(math.atan2(accel['y'], accel['z']))
        return pitch, roll

    def time(self):
        return time.time() - self._t0

    def sleep(self, seconds):
        time.sleep(seconds)
        if self._stop or (self._duration is not None and self.time() >= self._duration):
            raise _Stopped()

    def set_camera(self, *args, **kwargs):     # no viewer on the robot
        pass

    def stop(self):
        self._stop = True

    def run(self, controller, duration=None):
        """duration: wall-clock seconds after which the controller is
        stopped (at its next sleep), like the sim's window closing."""
        self._duration = duration
        self._stop = False
        self._t0 = time.time()
        try:
            controller(self)
        except _Stopped:
            pass
