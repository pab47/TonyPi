import hiwonder.Board as Board
import time

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from servo_map import sim_to_pulse, SERVO_IDS
 
#joint_angles = [0.0] * 16  # radians, order = servo ids 1..16
joint_angles = [-0.0, -0.38117990863556156, 1.0639527120157433, -0.7791149780902686, -0.020943951023931952, -0.30159289474462014, 0.28064894372068816, 0.2513274122871834, -0.016755160819145562, -0.39793506945470714, 1.1016518238588207, -0.7791149780902686, -0.0041887902047863905, 0.37699111843077515, -0.289026524130261, 0.2513274122871834]
for i in SERVO_IDS:
    pulse = sim_to_pulse(joint_angles[i - 1], servo_id=i)
    Board.setBusServoPulse(i, pulse, 500)

# servo 1 = pan (up/down), servo 2 = tilt (left/right) on TonyPi Pro
# setPWMServoPulse(id, pulse, time_ms)
Board.setPWMServoPulse(1, 1800, 500)    #1200 (down) to 1800 (up)
Board.setPWMServoPulse(2, 1500, 500)    #1000 (right) to 2000 (left)
time.sleep(0.5)