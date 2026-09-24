#!/usr/bin/env python3
"""
controller_stand.py -- balance on one leg, choreographed with mink IK on top
of the hardware-like sim/hw interface. Built the same way as
controller_walk.py (same four pieces: FSM-ish phase schedule, Cartesian
targets, mink IKSolver, apply_control) -- see that file's header for the
fuller explanation of the split. Before any of that, there's a one-time
POWER-UP RAMP (controller()'s and record_d6a()'s own code, not a
stand_ticks() phase -- see DEFAULT_POWERUP_DURATION below): an explicit,
slow ramp from wherever the servos actually are (power-up default, or
wherever a previous action left them) into this file's own standing pose
(knees noticeably bent). THIS was the actual hardware fall point, not any
of the phases below -- a too-fast version of exactly this jump. Once that
ramp finishes, the one-leg-stand routine itself has five phases, run once
(not cycled):

    "stand_before" -- plain, both-feet standing (arms down) -- the SAME
                pose the power-up ramp just arrived at -- held for
                SETTLE_DURATION seconds (see --settle-time) as a visible
                pause before anything else moves.
    "shift"  -- from that standing pose: COM eases onto the stance foot,
                the free foot lifts and splays out to the side, and the
                arms raise out to the sides for balance. Takes
                SHIFT_DURATION seconds (see --shift-time).
    "hold"   -- everything stays put at the one-leg pose. Takes
                STAND_DURATION seconds (see --stand-time).
    "return" -- the mirror image of "shift": COM eases back to center, the
                free foot lowers and comes back in, arms come back down.
                Also takes SHIFT_DURATION seconds.
    "stand_after" -- plain, both-feet standing again, held for
                SETTLE_DURATION seconds, so the routine visibly ENDS in a
                stable stand too.

The sim/hw abstraction (TonyPi_sw.tonypi_sim / TonyPi_hw.tonypi_hw) and its
model/scene.xml are untouched by this file -- IKSolver loads its own model
from model/scene_mink.xml purely to do the kinematics; `sim` keeps using
whatever scene.xml TonyPi_sw already defaults to. Two independent
mujoco.MjModel/MjData pairs, each indexed from 0, no cross-talk.

Run:   python3 controller_stand.py                     (live viewer; close the window to exit)
       python3 controller_stand.py --leg right          (stand on the RIGHT leg instead of the default left)
       python3 controller_stand.py --powerup-time 6      (slower still into the initial standing
                                                          pose -- raise this first if it still
                                                          falls right at the start)
       python3 controller_stand.py --record             (bake the routine into action/pranav_stand.d6a
                                                          instead -- see record_d6a() below)
"""
import argparse
import functools
import math
import os
import sqlite3
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import mink

from servo_map import sim_to_pulse

# uncomment appropriate line below to use the simulator or the real robot
#
# NOTE: this file's __main__/controller() path needs mujoco+mink (for
# IKSolver) AND whichever of these two you import here -- it's a
# sim/viewer development tool, and stays that way even with TonyPi_hw
# selected, since the Pi doesn't have mujoco/mink to run IKSolver at all.
# To actually run the one-leg stand on real hardware, bake it once on this
# machine with --record into a .d6a file, then play that back on the Pi
# with its own native action-group player -- no mujoco, no mink, no Python
# controller needed there at all.
from TonyPi_sw.tonypi_sim import TonyPiSim, SERVO_IDS, NAME, SERVO_JOINT
#from TonyPi_hw.tonypi_hw import TonyPiHW as TonyPiSim, SERVO_IDS, NAME, SERVO_JOINT

# ---------------------------------------------------------------------------
# Paths -- ADJUST if your model/ folder isn't a sibling of this file's parent
# (this mirrors tonypi_sim.py's own `../../model/scene.xml` from one level
# deeper inside TonyPi_sw/). This is the ONE line to fix if the stand fails
# to find the file.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_MINK_XML = _HERE / ".." / "model" / "scene_mink.xml"

# ---------------------------------------------------------------------------
# Stand constants -- start here when tuning: nothing below this block needs
# to change for a first pass. The three "time" parameters (COM shift time,
# one-leg hold time, and how long to hold plain standing before/after) are
# CLI flags (--shift-time/--stand-time/--settle-time, see __main__ below)
# rather than constants, since they're meant to be varied per run; these
# are just their defaults.
#
# 2026-09-24/25 hardware runs, in order:
#   1. First cut (2 s shift, no explicit before/after standing hold) fell
#      over. Read at the time as the shift-into-single-support being too
#      fast; DEFAULT_SHIFT_DURATION was more than doubled and
#      DEFAULT_SETTLE_DURATION ("stand_before"/"stand_after", see
#      stand_ticks() below) was added in response.
#   2. Actual root cause, found on the next run: the fall was happening
#      EARLIER than that, in the very first transition -- from whatever
#      pose the servos power up/were last left in, into this file's own
#      standing pose (which has the knees noticeably bent, inherited
#      unchanged from controller_walk.py's 'stand' keyframe). That jump
#      was being commanded in a single CONTROL_HZ tick (40 ms) -- adding
#      DEFAULT_SETTLE_DURATION didn't fix it because "stand_before"'s
#      ticks all target the SAME already-converged crouched pose, so only
#      the very first of them actually had to move, and it still only got
#      40 ms to do it; the rest were no-op repeats of a pose already
#      reached (or not) by tick 1. DEFAULT_POWERUP_DURATION below is the
#      real fix: an explicit multi-second ramp, issued as ONE setAngles()
#      call per servo BEFORE stand_ticks() starts, from wherever the
#      servos actually are to the settled standing pose (see
#      controller()'s and record_d6a()'s use of _settled_stand_angles()).
#      Since this was identified as the actual failure point,
#      DEFAULT_SHIFT_DURATION/DEFAULT_SETTLE_DURATION have been eased back
#      down some -- the shift/hold/return motions themselves were fine at
#      a brisker pace, it was only ever this first power-up transition
#      that fell.
#   3. Still some quick motion right at the start after (2)'s fix -- traced
#      by comparing every tick's commanded joint angles to the previous
#      tick's (in sim: a script that just diffs stand_ticks()'s successive
#      "angles" dicts) to an ACTUAL BUG, not another hardware-speed issue:
#      stand_ticks() built its COM target as an interpolation starting
#      from the bare geometric MIDPOINT of the two feet, not from
#      ik.com0 (the real standing-COM target, which includes
#      COM_FORE_AFT_TRIM and isn't exactly that midpoint) -- so at frac=0
#      ("stand_before", right after the power-up ramp) it was silently
#      asking for a DIFFERENT COM position than the one the power-up ramp
#      had just settled into, and the very first "stand_before" tick
#      jumped to correct that in a single 40 ms control tick (one servo
#      moved ~12.6 deg in that one tick in sim, ~314 deg/s -- easily
#      enough to destabilize real hardware, and invisible in sim's own
#      viewer since MuJoCo's PD actuators just track it). Fixed in
#      stand_ticks() by interpolating the COM from ik.com0[:2] instead of
#      the foot-midpoint, which makes frac=0 an EXACT match for
#      _settled_stand_angles()'s target -- verified in sim: the
#      tick-to-tick jump at that boundary is now exactly 0.
#   4. '--leg right' fell specifically at touchdown (foot not flat landing
#      from the splayed/lifted pose). Originally "fixed" by leveling the
#      captured foot-orientation targets (a since-removed _level_rotation()
#      helper) -- but see item 6 below: that diagnosis was wrong, caught
#      only once this file started testing against full physics instead of
#      kinematics alone. Left here for the historical record; item 6 is
#      the real fix for this AND the 2026-09-26 report of the same thing.
#   5. Falls BENDING BACKWARD right after reaching the crouched stand pose,
#      before the free leg even lifts -- i.e. during/just after the
#      power-up ramp, not any later phase (confirmed by asking which phase
#      specifically). This turned out to be a STATIC target-placement bug,
#      not a speed/trajectory one -- see COM_FORE_AFT_TRIM below for the
#      diagnosis (in short: the trim, copied unmodified from
#      controller_walk.py, pushed the crouched standing target's COM about
#      2 cm past center toward the heels, on top of an already
#      torque-demanding knees-bent crouch -- and unlike the earlier
#      "too-fast transition" fixes above, no amount of slowing the ramp
#      down helps a target that's statically past the tipping point).
#      Zeroed for now; nudge it back up gradually if hardware testing later
#      shows a small residual forward lean, rather than restoring the old
#      value outright.
#   6. Falls on BOTH legs specifically as the swing foot is placed back on
#      the floor at the end of "return". Reported as "feet not flat, touch
#      at the back end" -- but this is where testing moved from kinematics
#      (IK/FK checks in a script) to a full PHYSICS sim (real contact +
#      PD dynamics, TonyPiSim's actual MuJoCo model), for the first time in
#      this file's whole debugging history, and root-caused as something
#      else entirely. TWO compounding bugs, both only visible under real
#      physics:
#        a) "return" drove the free foot AND the COM/arms off the SAME
#           `frac`, so the COM was shifting back toward center for most of
#           the phase while only the stance foot was actually down --
#           asking the robot to move its weight toward a foot that hadn't
#           landed yet. Torso roll was already visibly diverging by the
#           HALFWAY point of "return", free foot still ~2 cm off the
#           ground -- well before any actual touchdown contact. Fixed with
#           RETURN_FOOT_LEAD_FRACTION below (foot lowers and plants FIRST,
#           COM/arms recenter SECOND, once double support is real).
#        b) Item (4)'s _level_rotation() fix -- forcing both feet's
#           standing-orientation target to exact world-level -- turned out
#           to have been chasing the wrong problem AND made whole-body
#           balance worse: a full-physics static plain-STANDING test (both
#           feet down, no leg lift at all) settled at a persistent -12 deg
#           torso ROLL with both feet leveled, vs only -3 to -4 deg with
#           both left at their native keyframe tilt. (a) alone, tested with
#           leveling still in place, delayed the fall but didn't prevent
#           it; reverting (b) as well (see self._foot_rotation's own
#           comment in IKSolver.__init__) was needed too. Fixed together,
#           both legs now complete the full routine in a full-physics sim
#           without tipping over. What looked and felt like a foot-flatness
#           problem at the moment of falling was actually the tail end of a
#           whole-body roll that started well earlier in the phase, made
#           worse by a standing pose that was already carrying 3x its
#           natural balance bias.
# ---------------------------------------------------------------------------
CONTROL_HZ = 25.0            # sim.setAngles() ramp time = 1/CONTROL_HZ
DEFAULT_LEG = "left"         # which leg to stand on if --leg isn't given
DEFAULT_POWERUP_DURATION = 4.0  # s, default --powerup-time: how long to
                              # ramp from wherever the servos currently are
                              # (power-up default, or wherever a previous
                              # action left them) into this file's standing
                              # pose (knees bent), BEFORE anything else
                              # happens. This is the one-time "unknown pose
                              # -> known crouched stand" jump that fell on
                              # hardware when it was implicitly given only
                              # a single 40 ms control tick -- see the
                              # 2026-09-24/25 note above. Generous on
                              # purpose; this happens exactly once per run
                              # so there's little cost to taking it slow.
DEFAULT_SETTLE_DURATION = 1.5  # s, default --settle-time: how long to hold
                              # plain, both-feet standing (arms down) --
                              # AFTER the power-up ramp above has already
                              # gotten there -- before the shift into
                              # one-leg stance begins, and again after the
                              # return back down finishes. Shorter than the
                              # power-up ramp on purpose: by this point the
                              # robot is already safely standing, this is
                              # just a visible pause, not another risky
                              # transition.
DEFAULT_SHIFT_DURATION = 3.0  # s, default --shift-time (COM shift / free-leg
                              # lift-and-splay / arm-raise, and its mirror
                              # image on the way back down). This one turned
                              # out NOT to be the failure point (see the
                              # 2026-09-24/25 note above) -- eased back down
                              # from an earlier, overly conservative 5.0 s.
DEFAULT_STAND_DURATION = 3.0  # s, default --stand-time (how long to hold
                              # the one-leg pose)
STAND_SHIFT_FRACTION = 1.0   # 0-1 (or >1 to overshoot), how far the COM
                              # travels from center toward the stance foot.
                              # 1.0 = fully over the stance foot's own
                              # (x, y) -- true single-support stance. Unlike
                              # the walk gait's SHIFT_FRACTION (which stays
                              # below 1.0 because the robot immediately
                              # swings the other foot through and doesn't
                              # need to actually balance on one leg for
                              # long), this one is meant to be held, so it
                              # goes all the way by default.
FREE_LEG_SPLAY_LATERAL = 0.05  # m, how far outward (away from the midline,
                              # beyond its normal standing offset) the free
                              # foot swings when splayed. Same order of
                              # magnitude as the walk gait's STEP_LENGTH
                              # (0.04); first guess, nudge and re-check the
                              # viewer -- too much will run the hip out of
                              # its roll range or make the free leg's mass
                              # harder for the COM task to compensate for.
FREE_LEG_LIFT_HEIGHT = 0.06  # m, how high the free foot lifts off the
                              # ground when splayed -- same magnitude as the
                              # walk gait's swing-foot CLEARANCE (0.05).
STANCE_WIDTH_SCALE = 1.0     # 0-1 (or >1 to go wider than the keyframe),
                              # scales each foot's lateral (x) offset from
                              # the midline in ik.feet0 -- see
                              # controller_walk.py's own copy of this
                              # constant for the full rationale. 1.0 = the
                              # 'stand' keyframe's own foot separation,
                              # unmodified.
COM_HEIGHT_TRIM = -0.02      # m, added to the standing COM height captured
                              # from the 'stand' keyframe -- see
                              # controller_walk.py's own copy of this
                              # constant for the full rationale.
COM_FORE_AFT_TRIM = 0.0      # m, added to the standing COM's fore-aft (y)
                              # position captured from the 'stand' keyframe;
                              # +y = backward. controller_walk.py uses +0.02
                              # here (see its own copy of this constant) --
                              # that value was blindly copied into this file
                              # without being independently re-checked for
                              # THIS file's very different use case (a long,
                              # static double/single-support HOLD, not brief
                              # weight-shifts during continuous walking) and
                              # is the diagnosed cause of the 2026-09-26
                              # "falls bending backward right after it
                              # crouches, before the leg even lifts" hardware
                              # report: checked in sim by reconstructing the
                              # actual hardware starting pose (the shipped
                              # action/stand.d6a frame -- confirmed via
                              # controller_action.py that it's what plays
                              # before any other action group) and comparing
                              # its own (foot-anchored) COM fore-aft position
                              # to ik.com0. Untrimmed, the 'stand' keyframe's
                              # own COM is already close to neutral (y =
                              # +0.003, i.e. centered over the feet); with
                              # the inherited +0.02 trim, the crouched
                              # standing target asked for y = +0.023 --
                              # roughly 2x the fore-aft travel actually
                              # needed from the real starting pose (y =
                              # -0.014) and well past center, toward the
                              # heels. A knees-bent crouch already raises
                              # the ankle torque needed just to hold still;
                              # asking it to hold that torque while also
                              # leaning back past center is exactly a slow
                              # backward tip-and-fall, and it's a STATIC
                              # target-placement problem, not a speed/
                              # trajectory one -- which is why none of the
                              # earlier "slow everything down" fixes touched
                              # it. Zeroed here rather than re-signed/
                              # re-scaled since there's no hardware
                              # measurement yet to justify a different
                              # nonzero value; if hardware testing later
                              # shows a persistent (small, forward-only)
                              # lean once this is fixed, nudge this back up
                              # a little at a time rather than reintroducing
                              # the old value outright.
ARM_SHOULDER_ROLL_HORIZONTAL_DEG = 80.0  # deg, shoulder ABDUCTION magnitude
                              # (away from the torso) the arms raise to
                              # during "shift", so the hands end up roughly
                              # horizontal / out to the sides for balance
                              # (classic tightrope-walker pose) instead of
                              # hanging at the 'stand' keyframe's own ~16 deg
                              # default. Sign convention verified by FK in
                              # controller_walk.py (not assumed): despite
                              # both shoulder_roll axes being defined the
                              # same nominal way in robot_mink.xml, the two
                              # arms are mirrored, so moving the RIGHT hand
                              # away from the body needs *negative*
                              # r_shoulder_roll and the LEFT hand away needs
                              # *positive* l_shoulder_roll -- applied with
                              # those opposite signs in
                              # IKSolver._cache_arm_targets(). 80 deg (not a
                              # full 90) leaves a little headroom inside
                              # each side's joint range; first guess, nudge
                              # and re-check the viewer.
ARM_SHOULDER_PITCH_HORIZONTAL_DEG = 0.0  # deg, shoulder pitch target while
                              # the arms are raised -- 0 keeps them from
                              # swinging fore/aft, so with shoulder_roll
                              # near 80 deg and the elbow straight (below)
                              # the arm points straight out to the side.
                              # This axis isn't mirrored between sides (see
                              # robot_mink.xml: same range both sides), so
                              # no sign flip needed.
ARM_ELBOW_HORIZONTAL_DEG = 0.0  # deg, elbow target while the arms are
                              # raised -- 0 = straight, so the whole arm
                              # (not just the upper arm) reads as
                              # horizontal. Also unmirrored between sides.
RETURN_FOOT_LEAD_FRACTION = 0.4  # 0-1, fraction of "return"'s own duration
                              # spent ONLY lowering/retracting the free foot
                              # (COM and arms held at their full one-leg-
                              # stance values) before the COM/arms are
                              # allowed to start recentering. See the
                              # 2026-09-26 finding below stand_ticks()'s
                              # "return" branch: with a single shared `frac`
                              # driving the free foot AND the COM/arms in
                              # lockstep, the COM was being asked to shift
                              # back toward center throughout MOST of
                              # "return" while the free foot was still
                              # airborne (it only reaches the ground right
                              # at the very end) -- reproduced in a full
                              # PHYSICS sim (not just kinematics): torso roll
                              # visibly starts diverging around the HALFWAY
                              # point of "return", free foot still ~2 cm off
                              # the ground, and runs away to a full 90 deg
                              # tip-over well before touchdown, on BOTH legs.
                              # Splitting "return" into foot-lowers-first,
                              # COM/arms-recenter-second (mirroring "shift",
                              # which is inherently safe because it always
                              # moves the COM onto a foot that's ALREADY
                              # grounded) fixes the ordering: the free foot
                              # is flat on the ground -- genuine double
                              # support -- before any weight is asked to
                              # move back toward it. 0.4 is a first guess
                              # (lowering the foot isn't gravity-loaded, so
                              # it shouldn't need much of the phase; the
                              # recentering portion gets the rest) -- raise
                              # it if hardware testing shows the foot is
                              # STILL not fully down/settled by the time
                              # weight starts coming back onto it.
ARM_POSTURE_COST = 2.0       # posture_task's default cost (1e-1, shared by
                              # every joint it regularizes) is far too weak
                              # for the arms to actually reach
                              # ARM_SHOULDER_ROLL_HORIZONTAL_DEG: com_task
                              # (cost 10.0) and torso_task (cost 2.0) end up
                              # dominating the weighted QP, and the arms
                              # settle at a steady-state equilibrium well
                              # short of the target (verified in sim:
                              # cost=1e-1 plateaus around 56-62 deg of the
                              # requested 80, even after many seconds of
                              # "hold" -- not a transient lag, a real
                              # steady-state shortfall). Raising ONLY the
                              # six arm dofs' cost (via
                              # IKSolver._cache_arm_targets below, everything
                              # else stays at 1e-1) to this value gets them
                              # to within ~0.3 deg of the target with no
                              # measurable effect on foot/COM tracking error
                              # (checked in sim: foot/COM error changes by
                              # under half a millimeter). Cost=1.0 was
                              # already enough in that check; this keeps a
                              # bit of margin.


# ---------------------------------------------------------------------------
# Smooth profile helper -- zero VELOCITY (not just zero value) at both ends
# of [0, 1], so the "shift" and "return" phases start and end without a
# jerk, and hand off cleanly into/out of the static "hold" phase.
# ---------------------------------------------------------------------------
def smoothstep(t):
    """0->1 with zero velocity at both ends."""
    t = min(max(t, 0.0), 1.0)
    return 3 * t**2 - 2 * t**3


# NOTE: this file used to have a _level_rotation() helper here, which forced
# each foot's captured standing orientation to be exactly world-level before
# using it as the (fixed, whole-routine) foot-orientation target. Removed
# 2026-09-26 -- see IKSolver.__init__'s own comment where self._foot_rotation
# is captured, just below, for why: it fixed the wrong problem and cost more
# than it gave back, found only once this file started validating against
# full contact/PD physics instead of kinematics alone.


# ---------------------------------------------------------------------------
# joint -- IK. Owns a completely separate MjModel/MjData from the sim's; it
# is a pure kinematic reference generator, never stepped as physics.
# ---------------------------------------------------------------------------
class IKSolver:
    def __init__(self, xml_path, dt):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.configuration = mink.Configuration(self.model)
        self.data = self.configuration.data
        self.dt = dt
        self.solver = "daqp"  # swap for "quadprog"/"osqp" if daqp isn't installed

        self.posture_task = mink.PostureTask(self.model, cost=1e-1)
        self.com_task = mink.ComTask(cost=10.0)
        # orientation_cost raised from an original 1.0 to 8.0 -- tracks
        # whatever the foot-orientation target actually is (see
        # self._foot_rotation below -- a 2026-09-25 attempt to LEVEL that
        # target instead of raising this cost was reverted 2026-09-26, but
        # tracking it more tightly is independently still worth keeping:
        # checked in sim, no measurable cost to position tracking).
        self.foot_tasks = {
            "right": mink.FrameTask(frame_name="right_foot", frame_type="site",
                                     position_cost=10.0, orientation_cost=8.0, lm_damping=1.0),
            "left": mink.FrameTask(frame_name="left_foot", frame_type="site",
                                    position_cost=10.0, orientation_cost=8.0, lm_damping=1.0),
        }
        # Orientation-only: keeps the torso from tipping to make COM/feet
        # targets kinematically easier to hit; position_cost=0 leaves COM
        # translation entirely to com_task.
        self.torso_task = mink.FrameTask(frame_name="torso", frame_type="body",
                                          position_cost=0.0, orientation_cost=2.0,
                                          lm_damping=1.0)
        self.tasks = [self.posture_task, self.com_task, self.torso_task,
                      self.foot_tasks["right"], self.foot_tasks["left"]]
        self.limits = [mink.ConfigurationLimit(self.model)]

        self.configuration.update_from_keyframe("stand")

        # Baseline posture target: the plain 'stand' keyframe, untouched.
        # Captured BEFORE any arm-raising, so _apply_arm_posture() below has
        # a fixed starting point (arms down) to interpolate away from every
        # tick, and everything that ISN'T an arm joint (head, legs,
        # grippers) keeps pulling gently toward this same baseline the
        # whole routine, exactly like posture_task normally does.
        self._posture_base = self.data.qpos.copy()
        self.posture_task.set_target(self._posture_base)
        mujoco.mj_forward(self.model, self.data)

        # Torso target: dead level, straight from the 'stand' keyframe's own
        # orientation -- no lean bias (unlike controller_walk.py's
        # TORSO_PITCH_TRIM_DEG, which compensates for a walking-specific,
        # hardware-tuned forward tip). Balancing on one leg is a different
        # balance problem (a large LATERAL COM shift, not a fore/aft gait
        # bias); if hardware testing shows the robot leans a particular way
        # once it's on one foot, add a rotation here the same way
        # controller_walk.py's IKSolver.__init__ does for TORSO_PITCH_TRIM_DEG.
        torso_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.torso_task.set_target(
            mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(self.data.xmat[torso_bid].reshape(3, 3).copy()),
                self.data.xpos[torso_bid].copy(),
            )
        )

        # Cache the standing feet/COM Cartesian pose -- this is what
        # stand_ticks() starts its sequence from, and the fixed foot
        # ORIENTATION held throughout for BOTH feet (this routine never
        # rotates a foot, same as controller_walk.py's swing foot).
        #
        # Taken VERBATIM from the keyframe's own FK -- NOT leveled, despite
        # that FK being measurably several degrees off exact world-level
        # for both feet. An earlier version of this file (2026-09-25) DID
        # level it, via a _level_rotation() helper (since removed), after a
        # '--leg right' hardware fall that looked like a foot-flatness
        # problem at touchdown. That turned out to be the wrong fix, caught
        # only once this file started testing against FULL PHYSICS (real
        # contact + PD dynamics) instead of kinematics/FK alone -- see
        # RETURN_FOOT_LEAD_FRACTION above: the actual cause of THAT fall,
        # and of the near-identical one reported again on 2026-09-26, was a
        # "return"-phase timing bug, not foot flatness at all; leveling was
        # chasing the wrong symptom. Worse, leveling had its own hidden
        # cost: a full-physics static plain-STANDING test (both feet down,
        # no leg lift at all) settled at a persistent -12 deg torso ROLL
        # with both feet leveled, vs only -3 to -4 deg with both feet left
        # at their native keyframe tilt (matching the -2.8 deg the RAW,
        # non-IK keyframe settles at on its own) -- i.e. whatever these
        # per-foot tilts are (mesh/calibration/authoring offset), they
        # evidently pair with each other into a better-balanced stance
        # than forcing either one to abstract world-level independently;
        # leveling even just ONE foot (tested both ways) still left a
        # -7 to -10 deg bias. So: capture as-is.
        self.feet0 = {}
        self._foot_rotation = {}
        for side, site_name in (("right", "right_foot"), ("left", "left_foot")):
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            self.feet0[side] = self.data.site_xpos[sid].copy()
            self._foot_rotation[side] = mink.SO3.from_matrix(
                self.data.site_xmat[sid].reshape(3, 3).copy()
            )

        # Narrow the stance: pull each foot's lateral (x) offset from the
        # midline in by STANCE_WIDTH_SCALE -- see controller_walk.py's own
        # copy of this constant for the full rationale. Feeds every
        # downstream consumer of feet0 (stand_ticks() and record_d6a()'s
        # standing-pose settle), since this runs before any of them read it.
        mid_x = 0.5 * (self.feet0["left"][0] + self.feet0["right"][0])
        for side in self.feet0:
            self.feet0[side][0] = mid_x + STANCE_WIDTH_SCALE * (self.feet0[side][0] - mid_x)

        self.com0 = self.data.subtree_com[1].copy()
        self.com0[2] += COM_HEIGHT_TRIM
        self.com0[1] += COM_FORE_AFT_TRIM

        self._cache_arm_targets()

    def _cache_arm_targets(self):
        """Resolve qpos addresses for the six arm joints (shoulder_pitch,
        shoulder_roll, elbow x2 sides) and cache their START angle (the
        plain 'stand' keyframe's own value, from self._posture_base) and
        END angle (the horizontal/raised-for-balance target, from the
        ARM_*_HORIZONTAL_DEG constants above, mirrored per side). Called
        once, up front, so _apply_arm_posture() is just a cheap lerp per
        tick -- no mj_name2id lookups in the hot loop."""
        # sign: right needs the shoulder_roll target NEGATIVE to go
        # outward, left needs it POSITIVE -- see ARM_SHOULDER_ROLL_
        # HORIZONTAL_DEG above for why. Pitch and elbow aren't mirrored
        # (same joint range both sides), so no sign flip for those.
        targets_deg = {
            "left": {"shoulder_pitch": ARM_SHOULDER_PITCH_HORIZONTAL_DEG,
                     "shoulder_roll": ARM_SHOULDER_ROLL_HORIZONTAL_DEG,
                     "elbow": ARM_ELBOW_HORIZONTAL_DEG},
            "right": {"shoulder_pitch": ARM_SHOULDER_PITCH_HORIZONTAL_DEG,
                      "shoulder_roll": -ARM_SHOULDER_ROLL_HORIZONTAL_DEG,
                      "elbow": ARM_ELBOW_HORIZONTAL_DEG},
        }
        self._arm_qadr = {"left": {}, "right": {}}
        self._arm_start = {"left": {}, "right": {}}
        self._arm_end = {"left": {}, "right": {}}
        prefix = {"left": "l_", "right": "r_"}
        # Start from the posture task's own default cost (uniform 1e-1
        # across every dof) and raise just the six arm dofs -- see
        # ARM_POSTURE_COST above for why the default is too weak for them
        # to actually reach the horizontal target.
        cost = np.full(self.model.nv, 1e-1)
        for side in ("left", "right"):
            for joint in ("shoulder_pitch", "shoulder_roll", "elbow"):
                jname = prefix[side] + joint
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                qadr = self.model.jnt_qposadr[jid]
                self._arm_qadr[side][joint] = qadr
                self._arm_start[side][joint] = float(self._posture_base[qadr])
                self._arm_end[side][joint] = math.radians(targets_deg[side][joint])
                cost[self.model.jnt_dofadr[jid]] = ARM_POSTURE_COST
        self.posture_task.set_cost(cost)

    def _apply_arm_posture(self, frac):
        """Update the posture target's six arm entries to `frac` of the way
        (0 = normal standing arms, 1 = fully raised/horizontal) from START
        to END, leaving every other posture entry (head, legs, grippers,
        floating base) at the plain 'stand' baseline. frac is expected to
        already be eased (smoothstep'd) by the caller -- this is a plain
        lerp."""
        frac = min(max(frac, 0.0), 1.0)
        target = self._posture_base.copy()
        for side in ("left", "right"):
            for joint, qadr in self._arm_qadr[side].items():
                q0 = self._arm_start[side][joint]
                q1 = self._arm_end[side][joint]
                target[qadr] = q0 + frac * (q1 - q0)
        self.posture_task.set_target(target)

    def free_foot_target(self, free_side, frac):
        """Cartesian target for the free (non-stance) foot at `frac` of the
        way (0 = normal standing position, 1 = fully splayed out and
        lifted) through the splay. frac is expected to already be eased by
        the caller. Only x (lateral) and z (height) move; y (fore-aft) stays
        put -- "splayed out to the side", not forward/back."""
        frac = min(max(frac, 0.0), 1.0)
        base = self.feet0[free_side]
        mid_x = 0.5 * (self.feet0["left"][0] + self.feet0["right"][0])
        outward = 1.0 if base[0] >= mid_x else -1.0
        x = base[0] + outward * FREE_LEG_SPLAY_LATERAL * frac
        z = base[2] + FREE_LEG_LIFT_HEIGHT * frac
        return np.array([x, base[1], z])

    def solve(self, com_xyz, foot_xyz, arm_frac=None):
        """One damped-IK step toward the given Cartesian/posture targets.
        Returns servo angles as {servo_id: radians}, true SERVO_IDS (1..16)
        order. arm_frac, if given, updates the posture target's arm entries
        first (see _apply_arm_posture)."""
        if arm_frac is not None:
            self._apply_arm_posture(arm_frac)
        self.com_task.set_target(np.asarray(com_xyz))
        for side, task in self.foot_tasks.items():
            target = mink.SE3.from_rotation_and_translation(
                self._foot_rotation[side], np.asarray(foot_xyz[side])
            )
            task.set_target(target)
        vel = mink.solve_ik(
            self.configuration, self.tasks, self.dt, self.solver,
            damping=1e-1, limits=self.limits,
        )
        self.configuration.integrate_inplace(vel, self.dt)
        return self._servo_angles()

    def _servo_angles(self):
        return {
            sid: float(self.data.qpos[self.model.joint(jname).qposadr[0]])
            for sid, jname in SERVO_JOINT.items()
        }


# ---------------------------------------------------------------------------
# control -- push a pose out to the real sim/hw interface.
# ---------------------------------------------------------------------------
def apply_control(sim, angles, time_in_ms):
    for sid in SERVO_IDS:
        sim.setAngles(sid, angles[sid], time_in_ms)


# ---------------------------------------------------------------------------
def _settled_stand_angles(ik, iterations=50):
    """Pre-converge the IK to the plain, both-feet standing pose (arms
    down): the raw 'stand' keyframe angles IKSolver.__init__ starts from
    aren't quite the same as the actual IK equilibrium (ARM_POSTURE_COST,
    the configuration limits, com/torso task residuals all nudge it
    slightly), so this settles onto the real fixed point by solving
    repeatedly at the fixed standing targets with everything else held
    still. Both controller() and record_d6a() use this SAME settled pose
    as the target of their (separate) power-up ramps, so the two can't
    drift apart."""
    angles = None
    for _ in range(iterations):
        angles = ik.solve(ik.com0, ik.feet0, arm_frac=0.0)
    return angles


# ---------------------------------------------------------------------------
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


def stand_ticks(ik, stance_leg=DEFAULT_LEG,
                 shift_duration=DEFAULT_SHIFT_DURATION,
                 stand_duration=DEFAULT_STAND_DURATION,
                 settle_duration=DEFAULT_SETTLE_DURATION):
    """Generate every IK tick of the one-leg-stand routine -- "stand_before"
    then "shift" then "hold" then "return" then "stand_after" -- as plain
    dicts, touching neither `sim` nor MuJoCo's viewer/physics at all:

        {"angles": {servo_id: rad}, "tick_ms": float,
         "phase": "stand_before"|"shift"|"hold"|"return"|"stand_after",
         "frac": float, "is_last_of_phase": bool}

    The "stand_before"/"stand_after" bookends hold the plain, both-feet
    standing pose (frac=0, same as normal standing) for settle_duration
    seconds before the shift into one-leg stance starts and after the
    return back down finishes -- so the routine always visibly starts AND
    ends in a stable stand, never straight into or out of motion.

    Both controller() (drives a live TonyPiSim/TonyPiHW tick by tick) and
    record_d6a() (bakes the same ticks into a .d6a file for the Pi's own
    action-group player) iterate this, so the two can never drift out of
    sync when the constants above get tuned again."""
    if stance_leg not in ("left", "right"):
        raise ValueError(f"stance_leg must be 'left' or 'right', got {stance_leg!r}")
    free_leg = "right" if stance_leg == "left" else "left"

    dt = ik.dt
    tick_ms = 1000.0 * dt
    # Interpolate the COM from its ACTUAL standing target (ik.com0 --
    # includes COM_FORE_AFT_TRIM and whatever asymmetry the real captured
    # COM has, same as _settled_stand_angles()/record_d6a()'s row 0 use)
    # toward the stance foot's own xy, NOT from the bare geometric
    # foot-midpoint. Using the foot-midpoint here used to be a real bug:
    # at frac=0 ("stand_before"/"stand_after") it silently targeted a
    # DIFFERENT com_xy than the powerup ramp had just settled the robot
    # into, so the very first "stand_before" tick jumped straight from the
    # true standing COM to the (unrelated) foot-midpoint in a single 40 ms
    # control tick -- found in sim by comparing tick-to-tick joint angles:
    # one servo moved ~12.6 deg in that one tick (~314 deg/s), right at the
    # start of the routine, exactly the "quick motion before it reaches its
    # one-leg pose" reported from hardware. Anchoring on ik.com0[:2]
    # instead makes frac=0 an exact match for the powerup ramp's target, so
    # there's nothing left to jump.
    com0_xy = ik.com0[:2]
    stance_xy = ik.feet0[stance_leg][:2]
    com_height = ik.com0[2]

    # "hold" runs at least one tick even if stand_duration is 0, so the
    # routine always visits the fully-splayed pose at least once (useful
    # for --stand-time 0 as a quick "touch and return" sanity check).
    # "stand_before"/"stand_after" are skipped (0 ticks) if settle_duration
    # is 0, for anyone who explicitly wants the old no-bookend behavior.
    phases = [("stand_before", settle_duration),
              ("shift", max(shift_duration, dt)),
              ("hold", max(stand_duration, dt) if stand_duration > 0 else dt),
              ("return", max(shift_duration, dt)),
              ("stand_after", settle_duration)]

    for phase, duration in phases:
        if duration <= 0:
            continue
        n_ticks = max(1, int(round(duration / dt)))
        for i in range(1, n_ticks + 1):
            t = i / n_ticks
            if phase == "shift":
                # COM/arms and the free foot move together -- always safe
                # here, since the COM is shifting ONTO the stance foot,
                # which has been planted since before this phase started.
                foot_frac = frac = smoothstep(t)
            elif phase == "hold":
                foot_frac = frac = 1.0
            elif phase == "return":
                # Foot-lowers-first, COM/arms-recenter-second -- see
                # RETURN_FOOT_LEAD_FRACTION above for why these can't share
                # a single `frac` the way "shift" does. foot_frac reaches 0
                # (fully lowered/retracted, flat on the ground) by
                # RETURN_FOOT_LEAD_FRACTION of the way through the phase and
                # then holds there; frac (COM/arms) stays at 1 until that
                # same point, then eases down to 0 over what's left.
                foot_t = min(t / RETURN_FOOT_LEAD_FRACTION, 1.0)
                foot_frac = 1.0 - smoothstep(foot_t)
                if RETURN_FOOT_LEAD_FRACTION < 1.0:
                    body_t = max((t - RETURN_FOOT_LEAD_FRACTION)
                                 / (1.0 - RETURN_FOOT_LEAD_FRACTION), 0.0)
                else:
                    body_t = 0.0
                frac = 1.0 - smoothstep(body_t)
            else:  # "stand_before" / "stand_after": plain standing, static
                foot_frac = frac = 0.0

            com_xy = com0_xy + STAND_SHIFT_FRACTION * frac * (stance_xy - com0_xy)
            com_xyz = np.array([com_xy[0], com_xy[1], com_height])
            foot_xyz = {
                stance_leg: ik.feet0[stance_leg].copy(),
                free_leg: ik.free_foot_target(free_leg, foot_frac),
            }
            angles = ik.solve(com_xyz, foot_xyz, arm_frac=frac)
            yield {
                "angles": angles, "tick_ms": tick_ms, "phase": phase,
                "frac": frac, "is_last_of_phase": i == n_ticks,
            }


def controller(sim, ik, stance_leg=DEFAULT_LEG,
               shift_duration=DEFAULT_SHIFT_DURATION,
               stand_duration=DEFAULT_STAND_DURATION,
               settle_duration=DEFAULT_SETTLE_DURATION,
               powerup_duration=DEFAULT_POWERUP_DURATION):
    sim.sleep(0.5)

    # Power-up ramp: `sim` starts wherever TonyPiSim/TonyPiHW's own
    # keyframe leaves it (NOT the same pose as this file's own knees-bent
    # standing target -- see the 2026-09-24/25 note by DEFAULT_POWERUP_
    # DURATION above). apply_control() below normally gets one CONTROL_HZ
    # tick (40 ms) per call, which is fine once consecutive IK targets are
    # already close together (the rest of this routine), but nowhere near
    # enough for this first, one-time jump -- so it's issued as its own
    # single ramp over powerup_duration seconds, and we wait for it to
    # actually finish (sim.sleep) before the per-tick loop below starts
    # sending fresh 40 ms commands on top of it.
    print('--- powering up: ramping to standing pose ---')
    stand_angles = _settled_stand_angles(ik)
    apply_control(sim, stand_angles, powerup_duration * 1000.0)
    sim.sleep(powerup_duration)
    print('--- standing ---')
    print_status(sim)

    for tick in stand_ticks(ik, stance_leg=stance_leg,
                             shift_duration=shift_duration,
                             stand_duration=stand_duration,
                             settle_duration=settle_duration):
        apply_control(sim, tick["angles"], tick["tick_ms"])
        sim.sleep(tick["tick_ms"] / 1000.0)
        if tick["is_last_of_phase"]:
            print(f'--- {tick["phase"]} done ---')

    print(f'--- one-leg stand on {stance_leg} complete ---')
    print_status(sim)


# ---------------------------------------------------------------------------
# record -- bake the same stand_ticks() trajectory into a Hiwonder .d6a
# action-group file, so it can be played on the real TonyPi Pro (Raspberry
# Pi 4B, no MuJoCo/mink available there) with the robot's own on-board
# action-group player. No TonyPiSim/TonyPiHW involved at all.
#
# .d6a format (reverse-engineered from action/convert.py, which reads and
# rewrites files in this exact shape, and confirmed by reading several
# shipped action/*.d6a files directly): a .d6a file is literally a SQLite
# database:
#
#     CREATE TABLE ActionGroup(
#         [Index] INTEGER PRIMARY KEY AUTOINCREMENT
#                  NOT NULL ON CONFLICT FAIL UNIQUE ON CONFLICT ABORT,
#         Time INT,               -- ramp duration INTO this row, ms
#         Servo1 INT, ..., Servo16 INT,  -- pulse 0-1000, bus servo ids 1..16
#         Servo17 INT, Servo18 INT       -- unused on TonyPi; every shipped
#                                         -- file has these at 500 (center)
#     );
#
# The Pi's action-group player walks the table in [Index] order and, for
# each row, ramps every servo from its current pulse to that row's pulse
# over `Time` ms -- exactly sim.setAngles(id, angle, time_in_ms) semantics,
# pre-baked instead of computed live.
# ---------------------------------------------------------------------------
D6A_OUT = _HERE / ".." / "action" / "tonypi_stand.d6a"
# First row: ramp from wherever the servos power up (or were left by a
# previous action) to the plain (both feet) standing pose, before the
# "shift" phase starts. Driven by --powerup-time (DEFAULT_POWERUP_DURATION
# above) -- this row IS the recorded-file counterpart of controller()'s own
# power-up ramp, and was too short (a flat 1000 ms) on the hardware run
# that identified this as the actual fall point; see the 2026-09-24/25
# note by DEFAULT_POWERUP_DURATION.
# The IK integrates at CONTROL_HZ (25 Hz -> 40 ms/tick) for a smooth
# reference trajectory, but the shipped action groups update roughly every
# ~150 ms (~6-7 Hz) -- that's what the bus-servo serial link and the
# on-board player are built to push through. Recording every 40 ms tick
# would very likely outrun that, so normally only every DECIMATE-th tick
# becomes a row (4 at 25 Hz -> one row every 160 ms, close to the shipped
# groups' own rate). An extra row is always forced at each phase boundary
# (every is_last_of_phase tick) -- see controller_walk.py's D6A_DECIMATE
# comment for why plain "every 4th tick" sampling alone can miss the exact
# hand-off instant.
D6A_DECIMATE = 4


def _write_d6a_rows(rows, path):
    """rows: list of (time_ms, [pulse_servo1..pulse_servo16])."""
    path = Path(path)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(str(path))
    c = conn.cursor()
    c.execute('''CREATE TABLE ActionGroup([Index] INTEGER PRIMARY KEY AUTOINCREMENT
    NOT NULL ON CONFLICT FAIL
    UNIQUE ON CONFLICT ABORT,
    Time INT,
    Servo1 INT, Servo2 INT, Servo3 INT, Servo4 INT, Servo5 INT, Servo6 INT,
    Servo7 INT, Servo8 INT, Servo9 INT, Servo10 INT, Servo11 INT, Servo12 INT,
    Servo13 INT, Servo14 INT, Servo15 INT, Servo16 INT, Servo17 INT, Servo18 INT);''')
    insert = ('INSERT INTO ActionGroup(Time, Servo1, Servo2, Servo3, Servo4, Servo5, '
              'Servo6, Servo7, Servo8, Servo9, Servo10, Servo11, Servo12, Servo13, '
              'Servo14, Servo15, Servo16, Servo17, Servo18) '
              'VALUES(' + ','.join(['?'] * 19) + ')')
    for time_ms, pulses in rows:
        c.execute(insert, (time_ms, *[int(p) for p in pulses], 500, 500))
    conn.commit()
    conn.close()


def _safe_copy(src, dst):
    """Plain read-all-then-write-all-then-fsync copy. shutil.copyfile()
    (and plain `cp`, sometimes) can use an accelerated sendfile/
    copy_file_range fast path that a cloud-sync folder (Dropbox, OneDrive,
    iCloud Drive) doesn't always handle correctly -- silently landing a
    0-byte file. Reading the whole (small) .d6a into memory and writing it
    back out with a normal write()+fsync() avoids that fast path
    entirely."""
    with open(src, 'rb') as f:
        data = f.read()
    with open(dst, 'wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def record_d6a(stance_leg=DEFAULT_LEG,
               shift_duration=DEFAULT_SHIFT_DURATION,
               stand_duration=DEFAULT_STAND_DURATION,
               settle_duration=DEFAULT_SETTLE_DURATION,
               powerup_duration=DEFAULT_POWERUP_DURATION,
               out_path=D6A_OUT):
    """Run the same one-leg-stand routine stand_ticks() drives live, but
    bake every (decimated) tick's servo pulses into a .d6a action-group
    file instead of a live sim/hw. IMPORTANT: SQLite needs real file
    locking, which a live-syncing Dropbox/OneDrive/iCloud folder often
    can't provide -- writing straight to such a path can leave a 0-byte
    .d6a plus a stray -journal file (exactly the failure this function
    works around). So the database is built in a genuinely local temp
    directory first, fully closed, and only the finished file is copied
    into place."""
    dt = 1.0 / CONTROL_HZ
    ik = IKSolver(_MINK_XML, dt)

    rows = []
    # Row 0: the power-up ramp, from wherever the real servos are when
    # playback starts to the settled standing pose -- see
    # DEFAULT_POWERUP_DURATION above for why this needs to be its own
    # generous ramp, not folded into the "stand_before" ticks below (those
    # all target the same already-converged pose and would add no motion).
    stand_angles = _settled_stand_angles(ik)
    stand_pulses = sim_to_pulse([stand_angles[s] for s in SERVO_IDS])
    rows.append((int(round(powerup_duration * 1000.0)), list(stand_pulses)))

    tick_ms = 1000.0 / CONTROL_HZ
    ticks_since_row = 0
    for step in stand_ticks(ik, stance_leg=stance_leg,
                             shift_duration=shift_duration,
                             stand_duration=stand_duration,
                             settle_duration=settle_duration):
        ticks_since_row += 1
        if ticks_since_row < D6A_DECIMATE and not step["is_last_of_phase"]:
            continue
        pulses = sim_to_pulse([step["angles"][s] for s in SERVO_IDS])
        rows.append((int(round(tick_ms * ticks_since_row)), list(pulses)))
        ticks_since_row = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "pranav_stand.d6a"
        _write_d6a_rows(rows, tmp_path)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _safe_copy(tmp_path, out_path)

    total_ms = sum(t for t, _ in rows)
    print(f'wrote {len(rows)} rows ({total_ms / 1000.0:.1f} s of playback) '
          f'standing on the {stance_leg} leg to {out_path.resolve()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--leg', choices=('left', 'right'), default=DEFAULT_LEG,
                         help=f"which leg to stand on (default: {DEFAULT_LEG})")
    parser.add_argument('--powerup-time', type=float, default=DEFAULT_POWERUP_DURATION,
                         metavar='SECONDS',
                         help='seconds to ramp from wherever the servos currently are '
                              'into this file\'s (knees-bent) standing pose, before '
                              'anything else happens -- the one-time transition that '
                              f'actually fell on hardware (default: {DEFAULT_POWERUP_DURATION})')
    parser.add_argument('--shift-time', type=float, default=DEFAULT_SHIFT_DURATION,
                         metavar='SECONDS',
                         help='seconds to shift the COM/lift the free leg/raise the '
                              f'arms into the one-leg stance, and its mirror image on '
                              f'the way back down (default: {DEFAULT_SHIFT_DURATION})')
    parser.add_argument('--stand-time', type=float, default=DEFAULT_STAND_DURATION,
                         metavar='SECONDS',
                         help='seconds to hold the one-leg stance before lowering back '
                              f'down (default: {DEFAULT_STAND_DURATION})')
    parser.add_argument('--settle-time', type=float, default=DEFAULT_SETTLE_DURATION,
                         metavar='SECONDS',
                         help='seconds to hold plain, both-feet standing before the '
                              'shift into one-leg stance begins, and again after the '
                              'return back down finishes -- so the routine visibly '
                              f'starts and ends in a stable stand (default: {DEFAULT_SETTLE_DURATION}; '
                              '0 skips the bookends entirely)')
    parser.add_argument('--record', action='store_true',
                         help='bake the routine into action/pranav_stand.d6a '
                              'instead of opening the live viewer')
    args = parser.parse_args()

    if args.record:
        record_d6a(stance_leg=args.leg, shift_duration=args.shift_time,
                   stand_duration=args.stand_time, settle_duration=args.settle_time,
                   powerup_duration=args.powerup_time)
    else:
        # Build the IK model/solver BEFORE the sim starts stepping physics.
        # mujoco.MjModel.from_xml_path() compiling a second model on the
        # controller thread while the main thread is concurrently calling
        # mj_step() on the sim's own model (sim.run()'s physics loop) is
        # not safe -- see controller_walk.py's own copy of this comment
        # for the full failure mode. Building the IK model here, before
        # sim.run() is even called, avoids the race entirely.
        dt = 1.0 / CONTROL_HZ
        ik = IKSolver(_MINK_XML, dt)

        sim = TonyPiSim(viewer=True)  # viewer=False, realtime=False -> fast headless run
        # sim.set_camera(azimuth=120, elevation=-15, distance=0.9, lookat=[0.0, 0.0, 0.17])
        sim.set_camera(azimuth=90, elevation=-5, distance=0.8, lookat=[0.0, 0.0, 0.2])
        sim.run(functools.partial(controller, ik=ik, stance_leg=args.leg,
                                   shift_duration=args.shift_time,
                                   stand_duration=args.stand_time,
                                   settle_duration=args.settle_time,
                                   powerup_duration=args.powerup_time),
                duration=None)   # seconds of sim time, then the window closes (None = stay open)
