import math
import time
import hiwonder.Board as Board
import hiwonder.Mpu6050 as Mpu6050

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from servo_map import pulse_to_sim, ZERO, SIGN,SERVO_IDS   # servo_map.py must be in the same folder


# servo id -> joint name (1-8 left, 9-16 right, same order)
_LEFT = ['ANK_ROLL', 'ANK_PITCH', 'KNEE', 'HIP_PITCH', 'HIP_ROLL', 'ELBOW', 'SH_ROLL', 'SH_PITCH']
NAME = {i + 1: 'L_' + n for i, n in enumerate(_LEFT)}
NAME.update({i + 9: 'R_' + n for i, n in enumerate(_LEFT)})

# IMU on the expansion board (MPU-6050, I2C address 0x68)
mpu = Mpu6050.mpu6050(0x68)
mpu.set_accel_range(mpu.ACCEL_RANGE_2G)
mpu.set_gyro_range(mpu.GYRO_RANGE_2000DEG)


def getBusServoStatus(servoNo):
    pulse = Board.getBusServoPulse(servoNo)
    q = pulse_to_sim(pulse, servo_id=servoNo)          # sim angle, rad
    print('{:2d} {:<11s} Pulse: {:4d} angle: {:+7.3f} rad  {:+7.1f} deg'
          .format(servoNo, NAME[servoNo], pulse, q, math.degrees(q)))
    time.sleep(0.05)


def chip_to_body(v):
    """MPU-6050 chip axes -> body axes (X forward, Y left, Z up).
    Found experimentally: upright -> +y_chip up; on face -> +z_chip up; on right side -> -x_chip up."""
    return {'x': -v['z'], 'y': -v['x'], 'z': v['y']}


def getImuStatus():
    a = chip_to_body(mpu.get_accel_data(g=True))   # accel in g,     body frame
    w = chip_to_body(mpu.get_gyro_data())          # gyro in deg/s,  body frame
    # ZYX Euler (yaw-pitch-roll) from the gravity direction; valid only when ~static.
    #   pitch: rotation about +Y, positive = leaning forward (nose down)
    #   roll : rotation about +X, positive = leaning right   (left side up)
    pitch = math.degrees(math.atan2(-a['x'], math.hypot(a['y'], a['z'])))
    roll  = math.degrees(math.atan2( a['y'], a['z']))
    print('IMU accel [g]  body: x {:+6.3f}  y {:+6.3f}  z {:+6.3f}'.format(a['x'], a['y'], a['z']))
    #print('IMU gyro [deg/s] body: x {:+7.2f} y {:+7.2f} z {:+7.2f}'.format(w['x'], w['y'], w['z']))
    print('IMU pitch {:+6.1f} deg   roll {:+6.1f} deg'.format(pitch, roll))


#1) unload all the servos (idea if you want to move them before reading the angles)
for i in SERVO_IDS:
    Board.unloadBusServo(i)
    time.sleep(0.01)

#2) Get the pulses of the servos (ideal if you want to read zero or the sign convention)
for i in SERVO_IDS:
    getBusServoStatus(i)
    
#3) write the joint angles in radians to the pulses for input to write_angles.py
pulses = [Board.getBusServoPulse(i) for i in SERVO_IDS]
joint_angles = pulse_to_sim(pulses)
print('\njoint angles (radians):')
print(joint_angles.tolist())

#4) IMU
print('\nIMU status:')
getImuStatus()
