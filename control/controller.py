#!/usr/bin/env python3
"""
controller.py -- the ONLY file you edit.

Write your behaviour inside controller(sim) using the three hardware-like
calls:

    q          = sim.getAngles(servo_No)                 # rad
    sim.setAngles(servo_No, desired_angle, time_in_ms)   # non-blocking ramp
    accel, gyro = sim.getIMU()                           # g, deg/s (body frame)

plus sim.sleep(seconds) to wait (same as time.sleep when running real-time).

Run:   python3 controller.py        (close the MuJoCo window to exit)
"""
import math

#uncomment appropirate line below to use the simulator or the real robot
from TonyPi_sw.tonypi_sim import TonyPiSim, SERVO_IDS, NAME
#from TonyPi_hw.tonypi_hw import TonyPiHW as TonyPiSim, SERVO_IDS, NAME

# --- initial condition ------------------------------------------------------
# Free-floating base orientation (w, x, y, z) and the 16 servo joint angles
# [rad], applied once at startup on top of the 'home' keyframe -- edit these
# to start the robot tipped over / on its side / upside down / crouched
# instead of standing upright, e.g. to test getting up after a fall.
INIT_QUAT = (1.0, 0.0, 0.0, 0.0)      # identity = upright, no rotation
INIT_JOINT_ANGLES = [0.0] * 16        # SERVO_IDS order (1..16), all zero


def print_status(sim):
    for i in SERVO_IDS:
        q = sim.getAngles(i)
        print('{:2d} {:<11s} angle: {:+7.3f} rad  {:+7.1f} deg'
              .format(i, NAME[i], q, math.degrees(q)))
    a, w = sim.getIMU()
    pitch, roll = sim.pitch_roll_from_accel(a)
    print('IMU accel [g]  body: x {:+6.3f}  y {:+6.3f}  z {:+6.3f}'.format(a['x'], a['y'], a['z']))
    print('IMU gyro [deg/s] body: x {:+7.2f} y {:+7.2f} z {:+7.2f}'.format(w['x'], w['y'], w['z']))
    print('IMU pitch {:+6.1f} deg   roll {:+6.1f} deg\n'.format(pitch, roll))


def controller(sim):
    sim.sleep(0.5)                      # let the robot settle on the floor
    print('--- standing ---')
    print_status(sim)

    # The pose from write_angles.py (servo order 1..16), moved to in 500 ms (this is the hiwonder stand pose)
    joint_angles = [-0.0, -0.38117990863556156, 1.0639527120157433, -0.7791149780902686,-0.020943951023931952, 
                      -0.30159289474462014, 0.28064894372068816, 0.2513274122871834,
                 -0.016755160819145562, -0.39793506945470714, 1.1016518238588207, -0.7791149780902686,-0.0041887902047863905,
                         0.37699111843077515, -0.289026524130261, 0.2513274122871834]
    for i in SERVO_IDS:
        sim.setAngles(i, joint_angles[i - 1], 500)
    sim.sleep(1.0)
    print('--- crouch pose ---')
    print_status(sim)

    # Wave the left arm: shoulder pitch (servo 8) back and forth
    for _ in range(3):
        sim.setAngles(8, -1.5, 400)
        sim.sleep(0.5)
        sim.setAngles(8, 0.25, 400)
        sim.sleep(0.5)

    # Back to all angles 0.0 
    for i in SERVO_IDS:
        sim.setAngles(i, 0.0, 800)
    sim.sleep(1.5)
    print('--- back to standing ---')
    print_status(sim)


if __name__ == '__main__':
    sim = TonyPiSim(viewer=True,     # viewer=False, realtime=False -> fast headless run
                     init_quat=INIT_QUAT, init_joint_angles=INIT_JOINT_ANGLES)
    sim.set_camera(azimuth=120, elevation=-15, distance=0.9, lookat=[0.0, 0.0, 0.17])
    sim.run(controller, duration=None)   # seconds of sim time, then the window closes (None = stay open)
