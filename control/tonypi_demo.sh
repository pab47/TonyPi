#!/bin/bash
# TonyPi Pro fidelity demo — run sequence
#
# Setup (one time):
#   chmod +x tonypi_demo.sh
# Run:
#   ./tonypi_demo.sh
#
# Requires fall_back.d6a to exist alongside the stock action groups
# (custom action created for demo #11).

set -e

echo "1. Wave hello"
python3 controller_action.py --duration 4 stand.d6a wave.d6a

echo "2. Wing Chun warm-up (hand rub gesture)"
python3 controller_action.py --duration 3 stand.d6a wing_chun.d6a

echo "3. Left hook/uppercut"
python3 controller_action.py --duration 2.5 stand.d6a left_uppercut.d6a

echo "4. Combo: both arms extend, then laugh (chest/arm motion)"
python3 controller_action.py --duration 10.5 stand.d6a 0.d6a chest.d6a

echo "5. Bow (greeting)"
python3 controller_action.py --duration 5 stand.d6a bow.d6a

echo "6. Waist twist (flexibility/balance demo)"
python3 controller_action.py --duration 4.5 stand.d6a twist.d6a

echo "7. March in place (gait demo, stays put)"
python3 controller_action.py --duration 3 stand.d6a stepping.d6a

echo "8. Sit-ups (dynamic core motion)"
python3 controller_action.py --duration 12.5 stand.d6a sit_ups.d6a

echo "9. Lift and present, then set down (falls on its face -- sim-to-real demo)"
python3 controller_action.py --duration 16.5 stand.d6a move_up.d6a put_down.d6a

echo "10. Walk forward step-by-step (discrete gait)"
python3 controller_action.py --duration 5.5 stand.d6a go_forward_start.d6a go_forward_one_step.d6a go_forward_one_step.d6a go_forward_end.d6a

echo "11. Fall backward, attempt recovery (sim-to-real gap demo, custom fall_back.d6a)"
python3 controller_action.py --duration 9.5 stand.d6a fall_back.d6a stand_up_back.d6a

echo "Demo sequence complete."
