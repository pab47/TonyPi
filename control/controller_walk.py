#!/usr/bin/env python3
"""
controller_walk.py -- quasi-static walk, choreographed with mink IK on top
of the hardware-like sim/hw interface.

Structure (four pieces, each doing one job):

    GaitFSM        -- scheduling: which phase are we in (shift weight onto a
                       foot, or swing the other foot forward), and how far
                       through that phase's duration are we (t in [0,1]).
    cartesian_targets -- given the FSM's phase/t, returns the Cartesian
                       (world-frame) targets for COM and both feet: where
                       they should be right now, as smooth curves in time
                       (not linear -- see the profile helpers below).
    IKSolver       -- joint: turns those Cartesian targets into servo
                       angles, by running one mink differential-IK step per
                       control tick on a *separate* MuJoCo model/data used
                       only for this kinematic computation. It never touches
                       the sim's own physics model.
    apply_control  -- control: pushes a dict of servo angles out through
                       sim.setAngles(), one call per servo, ramped over the
                       control period.

The sim/hw abstraction (TonyPi_sw.tonypi_sim / TonyPi_hw.tonypi_hw) and its
model/scene.xml are untouched by this file -- IKSolver loads its own model
from model/scene_mink.xml (the copy with the right_foot/left_foot/
right_palm/left_palm sites and the com/feet mocap targets) purely to do the
kinematics; `sim` keeps using whatever scene.xml TonyPi_sw already defaults
to. Two independent mujoco.MjModel/MjData pairs, each indexed from 0, no
cross-talk -- see the two-scenes discussion this came out of.

Run:   python3 controller_walk.py            (live viewer; close the window to exit)
       python3 controller_walk.py --record   (bake the walk into action/tonypi_walk.d6a
                                              instead -- see record_d6a() below;
                                              add --backward for
                                              action/tonypi_walk_backward.d6a)
"""
import argparse
import functools
import math
import os
import re
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
# To actually run a choreographed walk on real hardware, bake it once on
# this machine with record_walk_d6a.py (imports IKSolver/walk_ticks from
# this file, but never TonyPiSim/TonyPiHW) into a .d6a file, then play
# that back on the Pi with its own native action-group player -- no
# mujoco, no mink, no Python controller needed there at all.
from TonyPi_sw.tonypi_sim import TonyPiSim, SERVO_IDS, NAME, SERVO_JOINT
#from TonyPi_hw.tonypi_hw import TonyPiHW as TonyPiSim, SERVO_IDS, NAME, SERVO_JOINT

# ---------------------------------------------------------------------------
# Paths -- ADJUST if your model/ folder isn't a sibling of this file's parent
# (this mirrors tonypi_sim.py's own `../../model/scene.xml` from one level
# deeper inside TonyPi_sw/). This is the ONE line to fix if the walk fails
# to find the file.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_MINK_XML = _HERE / ".." / "model" / "scene_mink.xml"
_MODEL_DIR = _HERE / ".." / "model"
_STAND_SCENE_XML = _MODEL_DIR / "scene.xml"  # what TonyPiSim loads by default

# ---------------------------------------------------------------------------
# Gait constants -- the "details to work out" from before. Start here when
# tuning: nothing below this block needs to change for a first pass.
# ---------------------------------------------------------------------------
CONTROL_HZ = 25.0          # sim.setAngles() ramp time = 1/CONTROL_HZ
STEP_LENGTH = 0.04         # m, forward travel of the swing foot per step
CLEARANCE = 0.05          # m, peak swing-foot height off the ground
SHIFT_DURATION = 1.5       # s, weight-shift-onto-one-foot phase
SWING_DURATION = 2.0       # s, swing-the-other-foot phase
FORWARD = np.array([0.0, -1.0])  # world -y = forward (+y = back, per the IMU convention)
SHIFT_FRACTION = 0.8      # 0-1, how far the COM travels from center toward the
                           # stance foot during "shift" (1.0 = fully over the
                           # foot, like a real single-leg stance). Less than 1
                           # keeps the torso closer to upright / reduces lean,
                           # at the cost of a less complete weight transfer.
STANCE_WIDTH_SCALE = 1.0   # 0-1 (or >1 to go wider than the keyframe),
                           # scales each foot's lateral (x) offset from the
                           # midline in ik.feet0. 1.0 = the 'stand'
                           # keyframe's own foot separation (unmodified);
                           # less than 1 narrows it, more than 1 widens it
                           # beyond the keyframe. Was 0.85 (an earlier
                           # hardware run showed a wider-than-intended gap
                           # between the legs); raised back to 1.0 per
                           # request to increase the leg-to-leg distance --
                           # go above 1.0 for wider still, watch the viewer
                           # for the hips/ankles hitting their limits.
                           # Applied in IKSolver.__init__, right after
                           # feet0 is captured from the keyframe via FK.
COM_HEIGHT_TRIM = -0.02    # m, added to the standing COM height captured
                           # from the 'stand' keyframe -- raises the torso
                           # (straighter knees/hips) without touching
                           # scene.xml or the keyframe itself. This is a
                           # first modest step per request to increase COM
                           # height; nudge it up/down and re-check the
                           # viewer -- too much will run the legs out of
                           # extension range or make the stance/gait less
                           # stable. Applied in IKSolver.__init__, right
                           # after com0 is captured via subtree_com.
ARM_ROLL_TRIM_DEG = 30.0   # deg, rolls both shoulders outward (away from
                           # the torso) from the 'stand' keyframe's own arm
                           # pose, before that pose is captured as the
                           # posture target the arms hold throughout the
                           # whole gait (there's no dedicated arm/hand
                           # FrameTask -- posture_task's soft cost is all
                           # that's holding them). Requested ahead of
                           # lowering COM_HEIGHT_TRIM into negative territory
                           # (deeper crouch): at the keyframe's own arm
                           # angles, a lower torso brings the hands/forearms
                           # closer to the hips/thighs, so give them
                           # clearance first. Sign convention verified by FK
                           # (not assumed): despite both shoulder_roll axes
                           # being defined the same nominal way in robot_mink
                           # .xml, the two arms are mirrored, so moving the
                           # RIGHT hand away from the body needs *negative*
                           # r_shoulder_roll and the LEFT hand away needs
                           # *positive* l_shoulder_roll -- applied with those
                           # opposite signs in IKSolver.__init__, directly on
                           # qpos right after the keyframe loads and before
                           # the posture target / feet0 / com0 are captured,
                           # so the arm-mass shift is reflected everywhere
                           # downstream too. First guess; nudge and re-check
                           # the viewer for the elbow/wrist looking natural.
COM_FORE_AFT_TRIM = 0.01   # m, added to the standing COM's fore-aft (y)
                           # position captured from the 'stand' keyframe;
                           # +y = backward (see FORWARD above), so this
                           # shifts the COM TARGET the whole gait tracks
                           # further back than the model's own subtree_com
                           # says it is. Rationale: hardware persistently
                           # leans/falls forward relative to sim (TORSO_
                           # PITCH_TRIM_DEG already compensates for this on
                           # the torso's ORIENTATION; this is the same
                           # underlying real-vs-modeled mass-distribution
                           # gap, but attacking the COM's TRANSLATIONAL
                           # target instead) -- most likely the model's
                           # link masses/CoM (CAD-derived, no wiring/
                           # battery/connectors) put the REAL center of
                           # mass further forward than this model assumes,
                           # so commanding "COM centered" here actually
                           # ends up forward-of-center on the real robot.
                           # Confirmed on hardware (2026-09-24): fixed the
                           # forward-walk falling-forward problem once
                           # _shift_target_xy() started actually applying
                           # it mid-walk (see that method's own history).
                           # Applied in IKSolver.__init__ right after com0
                           # is captured, AND in CartesianPlanner for every
                           # forward-walk shift/swing/recenter target.
COM_FORE_AFT_TRIM_BACKWARD = 0.03  # m, same units/sign/rationale as
                           # COM_FORE_AFT_TRIM above, used for backward
                           # walking instead. Kept as its own constant, not
                           # derived from COM_FORE_AFT_TRIM, because a
                           # same-magnitude SIGN-FLIPPED trim was tried
                           # first for backward walking (reasoning: momentum
                           # reverses with direction, so the compensating
                           # lean should too) and made the robot fall
                           # FORWARD on hardware, COM visibly too far
                           # forward. That's the wrong model: COM_FORE_
                           # AFT_TRIM corrects a STATIC real-vs-modeled
                           # mass gap (see above), which doesn't reverse
                           # with gait direction, so backward walking needs
                           # the SAME-SIGN (backward) lean, just possibly a
                           # different magnitude.
                           #
                           # 0.01 (matching forward) still fell on hardware,
                           # specifically on the step where the right foot
                           # swings back past the left (the first swing of
                           # the walk -- see CartesianPlanner: right swings
                           # first, STEP_LENGTH, then left overtakes by
                           # 2*STEP_LENGTH, etc.). Bumped by +0.02 (2026-
                           # 09-24), the same increment used to dial in
                           # forward's COM_FORE_AFT_TRIM originally. Swept
                           # in sim first: stable (no tip, n_steps=6) up to
                           # 0.05, first tips at 0.06, so 0.03 sits with
                           # plenty of margin below the sim ceiling -- sim
                           # was already "stable" at 0.01 despite the real
                           # hardware fall, though, so that margin is a
                           # sanity check against gross instability, not
                           # confirmation this value is enough on hardware.
                           # Next move if it still falls: another +0.02;
                           # if it starts falling backward instead, back
                           # this off.
TORSO_PITCH_TRIM_DEG = -3.0  # deg. On 2026-09-23's hardware run the robot
                           # tipped FORWARD, worst on the very first step.
                           # NEGATIVE biases the torso's IK target to lean
                           # BACK (POSITIVE leans it forward) -- see
                           # IKSolver.__init__ for why this is a rotation
                           # about world +x, not the "pitch" slot of an rpy
                           # triple (TonyPi's world has +x lateral, not
                           # forward). This is a first guess to try on
                           # hardware, not a validated number -- nudge it a
                           # couple degrees at a time (more negative if it
                           # still tips forward, back toward 0 if it now
                           # tips backward) and re-record.


# ---------------------------------------------------------------------------
# Smooth profile helper -- zero VELOCITY (not just zero value) at both ends
# of [0, 1], so consecutive phases/sub-phases hand off without a jerk.
# ---------------------------------------------------------------------------
def smoothstep(t):
    """0->1 with zero velocity at both ends. Used for every leg of every
    phase's motion (COM shift, and the swing foot's horizontal travel) so
    each hand-off is jerk-free."""
    t = min(max(t, 0.0), 1.0)
    return 3 * t**2 - 2 * t**3


def clearance_bump(t, height):
    """Raised-cosine 0->height->0 bump over [0, 1], zero velocity at both
    ends (like smoothstep) -- used for the swing foot's height so it lifts
    and lands smoothly, peaking at the swing's midpoint."""
    t = min(max(t, 0.0), 1.0)
    return height * (1.0 - math.cos(2.0 * math.pi * t)) / 2.0


# ---------------------------------------------------------------------------
# fsm -- scheduling only. Knows nothing about Cartesian space or IK: just
# which phase we're in, how far through it, and which foot is which role.
# ---------------------------------------------------------------------------
class GaitFSM:
    # cycle: shift weight onto a foot, then swing the OTHER foot forward.
    _CYCLE = [
        ("shift", "left"),   # move COM over the left foot
        ("swing", "right"),  # swing the right foot forward
        ("shift", "right"),  # move COM over the right foot
        ("swing", "left"),   # swing the left foot forward
    ]
    _DURATIONS = {"shift": SHIFT_DURATION, "swing": SWING_DURATION}

    def __init__(self):
        self.phase_index = 0
        self.phase_time = 0.0

    @property
    def phase(self):
        return self._CYCLE[self.phase_index]

    def advance(self, dt):
        """Step the clock by dt. Returns (phase_type, active_foot, t,
        just_finished) -- just_finished is True on the single tick where a
        phase completes, so the caller can commit the new foot position."""
        phase_type, active_foot = self.phase
        duration = self._DURATIONS[phase_type]
        self.phase_time += dt
        t = min(self.phase_time / duration, 1.0)
        just_finished = t >= 1.0
        if just_finished:
            self.phase_index = (self.phase_index + 1) % len(self._CYCLE)
            self.phase_time = 0.0
        return phase_type, active_foot, t, just_finished


# ---------------------------------------------------------------------------
# cart -- Cartesian target sequence. Pure math: given the FSM's state and
# the walk's persistent foot/COM bookkeeping, return where COM/feet should
# be RIGHT NOW. No mink, no MuJoCo here.
# ---------------------------------------------------------------------------
class CartesianPlanner:
    def __init__(self, feet0, com0, total_swings=None, direction="forward"):
        """feet0: {"left": xyz, "right": xyz} at the standing pose.
        com0: xyz of the whole-body COM at the standing pose.
        total_swings: total number of swing phases the whole walk will run
        (2 per stride -- one per foot). When given, the LAST swing is
        treated as a closing step (STEP_LENGTH, landing level with the
        other foot) instead of an overtaking step (2*STEP_LENGTH), so the
        walk ends with feet together instead of mid-stride.
        direction="forward"|"backward" (added 2026-09-24): backward
        mirrors the SWING displacement (self._sign flips which way FORWARD
        points) so the robot actually travels the other way. A literal
        time-reversal of the recorded forward trajectory was tried first
        and tips the robot over in sim (85+ deg pitch, full roll) --
        walking dynamics aren't time-reversible the way this quasi-static
        PLANNING is direction-mirror-symmetric, so this re-solves IK for a
        genuinely backward gait instead of replaying the forward one
        backwards.

        COM_FORE_AFT_TRIM/COM_FORE_AFT_TRIM_BACKWARD are deliberately NOT
        just +-self._sign of one shared value, even though that was the
        first thing tried (2026-09-24) -- it made the backward walk fall
        FORWARD on hardware, COM visibly too far forward. That ruled out
        the "trim compensates for direction-dependent gait momentum"
        theory: COM_FORE_AFT_TRIM's own comment above already says why --
        it corrects a STATIC real-vs-modeled mass-distribution gap (the
        real robot's CoM sits further forward than the CAD-derived model
        assumes), which doesn't flip when the feet start stepping the
        other way. So both directions lean the SAME way (backward, +y);
        they just get independently-tunable magnitudes since hardware may
        not need identical amounts in both directions."""
        self.feet = {side: np.array(pos, dtype=float) for side, pos in feet0.items()}
        self.com_height = float(com0[2])       # held fixed throughout
        self.com_xy = np.array(com0[:2], dtype=float)
        self._shift_start_xy = self.com_xy.copy()
        self._swing_start_xy = None
        self.total_swings = total_swings
        if direction not in ("forward", "backward"):
            raise ValueError(f'direction must be "forward" or "backward", got {direction!r}')
        self.direction = direction
        # Only mirrors the SWING (travel direction); COM_FORE_AFT_TRIM is
        # NOT mirrored by this -- see class docstring for why.
        self._sign = 1.0 if direction == "forward" else -1.0
        self._com_trim = COM_FORE_AFT_TRIM if direction == "forward" else COM_FORE_AFT_TRIM_BACKWARD
        # Both feet start level (fore-aft) with each other. The FIRST swing
        # only has to cover STEP_LENGTH to get one foot-length ahead. Every
        # swing after that (other than the closing one) is the trailing
        # foot overtaking a foot that's already STEP_LENGTH ahead of it, so
        # it has to cover 2*STEP_LENGTH to end up STEP_LENGTH ahead in turn
        # -- otherwise the two feet just end up level again every other
        # step instead of alternating ahead by a full stride.
        self._swing_count = 0
        self._current_step_length = STEP_LENGTH

    def _shift_target_xy(self, stance_foot):
        """Where the COM should sit while weight is on `stance_foot`: not the
        foot's own (x, y) -- that's a full single-leg-stance lean -- but
        SHIFT_FRACTION of the way there from the midpoint of both feet, so
        the torso doesn't have to tip as far to keep the COM over the
        support foot. COM_FORE_AFT_TRIM applied here too (2026-09-24): this
        is the ONE function that sets the COM's fore-aft target during
        EVERY "shift" and "swing" phase of the whole walk -- before this,
        COM_FORE_AFT_TRIM only reached the very start of the first shift
        (as com0, blended away by the first smoothstep) and the final
        recenter, so it had essentially no effect on the actual walking
        gait despite being tuned as if it did. This is the fix for that:
        the same backward bias now applies throughout the walk, not just
        at the ends of it."""
        mid_xy = 0.5 * (self.feet["left"][:2] + self.feet["right"][:2])
        full_target_xy = self.feet[stance_foot][:2]
        target_xy = mid_xy + SHIFT_FRACTION * (full_target_xy - mid_xy)
        # +y = backward (world convention). NOT mirrored by direction --
        # see this class's docstring for why (static mass-gap correction,
        # not a momentum-direction one).
        return target_xy + np.array([0.0, self._com_trim])

    @staticmethod
    def _swing_xy_fraction(t):
        """0->1 fraction of horizontal travel, eased over the full swing
        duration. NOTE: a later version of this held x, y fixed for a
        trailing "descend" fraction of the swing (so z alone finished the
        touchdown) to try to fix hardware grazing/slipping at contact.
        Verified in sim (headless, 6 strides, fresh process per variant) to
        measurably destabilize the gait -- squeezing the same horizontal
        travel into a shorter window necessarily raises its peak velocity
        (any zero-velocity-at-both-ends curve covering a fixed distance in
        less time needs a higher peak speed), and every fraction tried
        (0.15, 0.25, 0.4) tipped the robot over by the 2nd-3rd stride, worse
        as the fraction (and so the compression) grew. Reverted to plain
        smoothstep over the full [0, 1], which is stable at 6 strides. The
        grazing fix instead lives in record_d6a()'s row sampling -- see
        D6A_DECIMATE below."""
        return smoothstep(t)

    @staticmethod
    def _swing_z(t):
        """Swing-foot height: a smooth 0->CLEARANCE->0 bump, zero velocity
        at both ends -- see _swing_xy_fraction's note for why this isn't a
        separate translate/descend split."""
        return clearance_bump(t, CLEARANCE)

    def targets(self, phase_type, active_foot, t, just_finished):
        """Returns (com_xyz, {"left": xyz, "right": xyz})."""
        if phase_type == "shift":
            if t == 0.0 or self._swing_start_xy is not None:
                self._shift_start_xy = self.com_xy.copy()
                self._swing_start_xy = None
            target_xy = self._shift_target_xy(active_foot)
            s = smoothstep(t)
            com_xy_now = self._shift_start_xy + s * (target_xy - self._shift_start_xy)
            self.com_xy = com_xy_now
            foot_targets = {side: pos.copy() for side, pos in self.feet.items()}
        else:  # "swing"
            stance_foot = "left" if active_foot == "right" else "right"
            if self._swing_start_xy is None:
                self._swing_start_xy = self.feet[active_foot][:2].copy()
                is_first = self._swing_count == 0
                is_last = (self.total_swings is not None
                           and self._swing_count == self.total_swings - 1)
                self._current_step_length = (
                    STEP_LENGTH if (is_first or is_last) else 2.0 * STEP_LENGTH
                )
                self._swing_count += 1
            xy_now = self._swing_start_xy + self._swing_xy_fraction(t) * (
                self._sign * self._current_step_length * FORWARD
            )
            z_now = self._swing_z(t)
            foot_targets = {side: pos.copy() for side, pos in self.feet.items()}
            foot_targets[active_foot] = np.array([xy_now[0], xy_now[1], z_now])
            # COM stays put, at the same (partially) shifted-to position the
            # preceding "shift" phase established over the stance foot
            self.com_xy = self._shift_target_xy(stance_foot)
            if just_finished:
                # commit the swing foot's new planted position for next cycle
                self.feet[active_foot] = np.array([xy_now[0], xy_now[1], 0.0])

        com_xyz = np.array([self.com_xy[0], self.com_xy[1], self.com_height])
        return com_xyz, foot_targets


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
        self.foot_tasks = {
            "right": mink.FrameTask(frame_name="right_foot", frame_type="site",
                                     position_cost=10.0, orientation_cost=1.0, lm_damping=1.0),
            "left": mink.FrameTask(frame_name="left_foot", frame_type="site",
                                    position_cost=10.0, orientation_cost=1.0, lm_damping=1.0),
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

        # Roll both shoulders outward from the keyframe's own arm pose --
        # see ARM_ROLL_TRIM_DEG above for why and for the sign convention.
        arm_trim = math.radians(ARM_ROLL_TRIM_DEG)
        for jname, sign in (("r_shoulder_roll", -1.0), ("l_shoulder_roll", 1.0)):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            qadr = self.model.jnt_qposadr[jid]
            lo, hi = self.model.jnt_range[jid]
            self.data.qpos[qadr] = min(max(self.data.qpos[qadr] + sign * arm_trim, lo), hi)

        self.posture_task.set_target_from_configuration(self.configuration)
        mujoco.mj_forward(self.model, self.data)

        # Torso pitch trim: bias torso_task's target to lean back (or
        # forward) by TORSO_PITCH_TRIM_DEG instead of dead level. Built by
        # hand with an explicit rotation matrix rather than
        # mink.SO3.from_rpy_radians(roll, pitch, yaw), because that
        # "pitch" slot means rotation about the Y axis -- and on TonyPi's
        # world axes (+x lateral, +y backward, -y forward, +z up, per
        # tonypi_sim.py's IMU convention) the rotation that actually tips
        # the torso fore/aft is about the WORLD X axis, not Y. Rotating
        # the "up" vector (0, 0, 1) by +theta about +x gives a Y-component
        # of -sin(theta): since -y is forward, positive theta tips the
        # torso forward and negative theta tips it back -- hence the sign
        # convention on TORSO_PITCH_TRIM_DEG above.
        torso_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        theta = math.radians(TORSO_PITCH_TRIM_DEG)
        c, s = math.cos(theta), math.sin(theta)
        trim_rot = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
        stand_rot = self.data.xmat[torso_bid].reshape(3, 3).copy()
        torso_target = mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(trim_rot @ stand_rot),
            self.data.xpos[torso_bid].copy(),
        )
        self.torso_task.set_target(torso_target)

        # Cache the standing feet/COM Cartesian pose -- this is what
        # CartesianPlanner starts its sequence from, and the fixed foot
        # ORIENTATION we hold throughout (this gait never rotates a foot).
        self.feet0 = {}
        self._foot_rotation = {}
        for side, site_name in (("right", "right_foot"), ("left", "left_foot")):
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            self.feet0[side] = self.data.site_xpos[sid].copy()
            self._foot_rotation[side] = mink.SO3.from_matrix(
                self.data.site_xmat[sid].reshape(3, 3).copy()
            )

        # Narrow the stance: pull each foot's lateral (x) offset from the
        # midline in by STANCE_WIDTH_SCALE. Doesn't touch fore-aft (y) or
        # height (z), and feeds every downstream consumer of feet0 --
        # CartesianPlanner's whole gait AND record_d6a()'s standing-pose
        # settle -- since this runs before any of them read it.
        mid_x = 0.5 * (self.feet0["left"][0] + self.feet0["right"][0])
        for side in self.feet0:
            self.feet0[side][0] = mid_x + STANCE_WIDTH_SCALE * (self.feet0[side][0] - mid_x)

        self.com0 = self.data.subtree_com[1].copy()
        self.com0[2] += COM_HEIGHT_TRIM
        self.com0[1] += COM_FORE_AFT_TRIM

    def solve(self, com_xyz, foot_xyz):
        """One damped-IK step toward the given Cartesian targets. Returns
        servo angles as {servo_id: radians}, true SERVO_IDS (1..16) order."""
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
# kp overrides -- let controller_walk.py run its own live-physics TonyPiSim
# with different actuator kp gains than whatever model/robot.xml currently
# has committed, WITHOUT editing that file. robot.xml is shared (via
# scene.xml) with controller_stand.py's own TonyPiSim, and a blanket kp
# change there previously fixed this file's walk-gait foot-flatness problem
# but broke controller_stand.py's one-leg balance tuning -- see robot.xml's
# "REVERTED PITCH-STIFFNESS FIX" comment for the full history. That comment
# suggested exactly this: scope the kp change per-controller with a runtime
# override instead of a blanket file edit. This is that.
# ---------------------------------------------------------------------------
WALK_KP_OVERRIDES = {
    # The exact 6 sagittal-leg actuators and value from the reverted fix
    # above -- swept there, dropped to <1.5 deg foot-pitch tilt at
    # touchdown (from ~12 deg at the class default kp=12), no tip-over.
    # Change this dict (or pass a different one to sim_with_kp_overrides())
    # to try other values -- nothing here touches robot.xml on disk.
    "r_hip_pitch": 64, "r_knee": 64, "r_ankle_pitch": 64,
    "l_hip_pitch": 64, "l_knee": 64, "l_ankle_pitch": 64,
}


def sim_with_kp_overrides(kp_overrides, scene_path=_STAND_SCENE_XML, **tonypisim_kwargs):
    """Build a TonyPiSim() with some actuator kp gains changed from
    model/robot.xml's own committed values, without writing to that file
    (or to scene.xml) at all -- so anything else that loads the real
    model/scene.xml (controller_stand.py, in particular) is completely
    unaffected by calling this.

    kp_overrides: {joint_name: kp}, e.g. WALK_KP_OVERRIDES above. Each
    joint_name must already have its own
        <position name="{j}" joint="{j}" class="body_servo" .../>
    line in robot.xml -- with or without an existing kp="..." attribute,
    either is replaced with the requested value.

    How: copies robot.xml's TEXT (small, cheap) into a throwaway temp
    directory, patches ONLY the requested actuators' kp in that copy,
    copies scene.xml alongside it unchanged, and symlinks model/assets/
    (the STL meshes robot.xml references, not worth copying) into the same
    temp directory so the include/mesh paths still resolve. Then points an
    ordinary TonyPiSim(model_path=...) at the patched scene.xml there.
    MuJoCo fully compiles the model into memory inside TonyPiSim.__init__
    -- by the time that call returns, nothing on disk is needed anymore --
    so the temp directory is deleted again before this function returns.
    model/robot.xml and model/scene.xml are opened read-only and never
    written to.

    Any keyword TonyPiSim() itself accepts (viewer, realtime, speed,
    keyframe, init_quat, init_joint_angles, ...) can be passed through."""
    scene_path = Path(scene_path)
    robot_xml = (scene_path.parent / "robot.xml").read_text()

    for joint, kp in kp_overrides.items():
        pattern = re.compile(
            rf'(<position name="{re.escape(joint)}" joint="{re.escape(joint)}"'
            rf' class="body_servo")(?: kp="[^"]*")?( ctrlrange=)'
        )
        robot_xml, n = pattern.subn(rf'\1 kp="{kp}"\2', robot_xml)
        if n != 1:
            raise ValueError(
                f'expected exactly 1 actuator named {joint!r} in '
                f'{scene_path.parent / "robot.xml"}, found {n} -- check the '
                f'joint name against robot.xml\'s <actuator> block'
            )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        (tmp_dir / "robot.xml").write_text(robot_xml)
        (tmp_dir / "scene.xml").write_text(scene_path.read_text())
        (tmp_dir / "assets").symlink_to((scene_path.parent / "assets").resolve())
        return TonyPiSim(model_path=str(tmp_dir / "scene.xml"), **tonypisim_kwargs)


# ---------------------------------------------------------------------------
# control -- push a pose out to the real sim/hw interface.
# ---------------------------------------------------------------------------
def apply_control(sim, angles, time_in_ms):
    for sid in SERVO_IDS:
        sim.setAngles(sid, angles[sid], time_in_ms)


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


def walk_ticks(ik, n_steps=3, direction="forward"):
    """Generate every IK tick of the full choreographed walk -- the gait
    phases, then the final recenter -- as plain dicts, touching neither
    `sim` nor MuJoCo's viewer/physics at all:

        {"angles": {servo_id: rad}, "tick_ms": float, "section": "gait"|"recenter",
         "phase_type": "shift"|"swing"|None, "active_foot": "left"|"right"|None,
         "just_finished": bool, "completed": int, "n_cycles": int}

    n_steps counts individual foot placements (one foot swinging forward
    [or backward -- see `direction`] once = one step), NOT strides -- a
    "stride" (both feet stepping once) is 2 steps. Was n_strides (a full
    shift+swing+shift+swing cycle) until 2026-09-24, which meant
    n_strides=4 produced 8 individual foot placements -- surprising if
    you're thinking of "steps" the ordinary way. GaitFSM._CYCLE alternates
    shift/swing/shift/swing regardless of where n_steps stops (odd values
    work fine -- the walk just ends on whichever foot the count lands on),
    so this only changes the counting, not the phase sequence itself.

    direction="forward" (default) or "backward" (added 2026-09-24): all
    the mirroring lives in CartesianPlanner (see its docstring) -- this
    function just passes direction through to it and to the recenter
    phase's own trim sign below. A literal time-reversal of the recorded
    forward tick sequence was tried first (playing the same poses back in
    reverse order) and reliably tips the robot over in sim (85+ deg pitch,
    full roll) within one walk -- the contact/damping dynamics that make
    forward walking stable are not time-symmetric, even though the
    quasi-static PLANNING is direction-mirror-symmetric. This re-solves
    IK for a genuinely backward gait every tick instead.

    Both controller() (drives a live TonyPiSim/TonyPiHW tick by tick) and
    record_d6a() (bakes the same ticks into a .d6a file for the Pi's own
    action-group player) iterate this, so the two can never drift out of
    sync when the gait constants above get tuned again."""
    dt = ik.dt
    tick_ms = 1000.0 * dt

    fsm = GaitFSM()
    n_cycles = n_steps * 2   # 2 FSM phases (one shift, one swing) per step
    total_swings = n_steps   # one swing per step, by definition
    planner = CartesianPlanner(ik.feet0, ik.com0, total_swings=total_swings, direction=direction)

    completed = 0
    while completed < n_cycles:
        phase_type, active_foot, t, just_finished = fsm.advance(dt)
        com_xyz, foot_xyz = planner.targets(phase_type, active_foot, t, just_finished)
        angles = ik.solve(com_xyz, foot_xyz)
        if just_finished:
            completed += 1
        yield {
            "angles": angles, "tick_ms": tick_ms, "section": "gait",
            "phase_type": phase_type, "active_foot": active_foot,
            "just_finished": just_finished, "completed": completed, "n_cycles": n_cycles,
        }

    # Recenter: the closing step just landed the swing foot level with the
    # other one, but the COM is still shifted over whichever foot the last
    # "shift" phase put it on. Ease it back to the midpoint between both
    # (now level) feet so the walk ends standing square, not leaning.
    # COM_FORE_AFT_TRIM applied here too (2026-09-24): this target used to
    # be purely feet-derived, untouched by COM_FORE_AFT_TRIM -- so every
    # bit of that trim tuned so far had NO effect on the final post-walk
    # standing pose, only on mid-walk COM targets. Hardware showed the
    # standing-after-walk weight line landing well forward of the ankle
    # (between the ankle and the front of the foot) -- clean evidence of a
    # static bias independent of anything walking-dynamics related, so the
    # same backward correction now applies here as well.
    recenter_steps = max(1, int(round(SHIFT_DURATION / dt)))
    start_xy = planner.com_xy.copy()
    mid_xy = 0.5 * (planner.feet["left"][:2] + planner.feet["right"][:2])
    mid_xy = mid_xy + np.array([0.0, planner._com_trim])  # +y = backward, not direction-mirrored
    for i in range(1, recenter_steps + 1):
        s = smoothstep(i / recenter_steps)
        com_xy_now = start_xy + s * (mid_xy - start_xy)
        com_xyz = np.array([com_xy_now[0], com_xy_now[1], planner.com_height])
        foot_targets = {side: pos.copy() for side, pos in planner.feet.items()}
        angles = ik.solve(com_xyz, foot_targets)
        yield {
            "angles": angles, "tick_ms": tick_ms, "section": "recenter",
            "phase_type": None, "active_foot": None,
            "just_finished": i == recenter_steps, "completed": n_cycles, "n_cycles": n_cycles,
        }
    planner.com_xy = mid_xy.copy()


def controller(sim, ik, n_steps=3, direction="forward"):
    sim.sleep(0.5)
    print(f'--- standing ({direction}) ---')
    print_status(sim)

    for tick in walk_ticks(ik, n_steps=n_steps, direction=direction):
        apply_control(sim, tick["angles"], tick["tick_ms"])
        sim.sleep(tick["tick_ms"] / 1000.0)
        if tick["section"] == "gait" and tick["just_finished"]:
            print(f'--- completed phase {tick["completed"]}/{tick["n_cycles"]} '
                  f'({tick["phase_type"]} {tick["active_foot"]}) ---')
        elif tick["section"] == "recenter" and tick["just_finished"]:
            print('--- recentered ---')

    print('--- walk sequence done ---')
    print_status(sim)


# ---------------------------------------------------------------------------
# record -- bake the same walk_ticks() trajectory into a Hiwonder .d6a
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
D6A_OUT = _HERE / ".." / "action" / "tonypi_walk.d6a"
# Backward walk gets its own file (literal time-reversal of the
# forward one -- see walk_ticks()'s "backward" direction) so
# recording one direction never clobbers the other. 2026-09-24.
D6A_OUT_BACKWARD = D6A_OUT.parent / (D6A_OUT.stem + "_backward" + D6A_OUT.suffix)
D6A_N_STEPS = 3           # individual foot placements, not strides --
                          # see walk_ticks()'s docstring. Must match what
                          # you want to play on hardware.
D6A_STAND_HOLD_MS = 1000  # first row: ramp from wherever the servos power
                          # up to the standing pose, before the gait starts
# The IK integrates at CONTROL_HZ (25 Hz -> 40 ms/tick) for a smooth
# reference trajectory, but the shipped action groups (e.g.
# action/go_forward_one_step.d6a) update roughly every ~150 ms (~6-7 Hz) --
# that's what the bus-servo serial link and the on-board player are built
# to push through. Recording every 40 ms tick would very likely outrun
# that, so normally only every DECIMATE-th tick becomes a row (4 at 25 Hz
# -> one row every 160 ms, close to the shipped groups' own rate).
#
# BUT 25 ticks/swing-phase isn't a multiple of DECIMATE=4, so plain
# "every 4th tick" sampling can land 1-3 ticks short of the exact touchdown
# tick, skip it, and only pick back up a couple of ticks INTO the next
# phase -- by which point the swing foot's z has already snapped to
# exactly 0 (planted) AND the next phase's COM-shift has already started,
# both blended into one ~160 ms linear ramp on the hardware player. That's
# almost certainly the "feet graze/slip at the end of the step" seen on
# real hardware: not the continuous IK trajectory, which already has zero
# velocity in x, y, and z right at touchdown (see _swing_xy_fraction
# above), but this file's own sampling missing that exact instant. The fix
# below (in record_d6a()) forces an extra row at every phase boundary
# (every `just_finished` tick), with that row's Time set to however many
# ticks actually elapsed since the previous row -- so the exact
# just-touched-down pose is always its own short row, never blended into
# the next phase's motion.
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


def record_d6a(n_steps=D6A_N_STEPS, out_path=None, direction="forward"):
    """Run the same gait walk_ticks() drives live, but bake every
    (decimated) tick's servo pulses into a .d6a action-group file instead
    of a live sim/hw. IMPORTANT: SQLite needs real file locking, which a
    live-syncing Dropbox/OneDrive/iCloud folder often can't provide --
    writing straight to such a path can leave a 0-byte .d6a plus a stray
    -journal file (exactly the failure this function works around). So the
    database is built in a genuinely local temp directory first, fully
    closed, and only the finished file is copied into place.

    out_path=None (default) picks D6A_OUT or D6A_OUT_BACKWARD from
    `direction`, so forward and backward recordings never overwrite each
    other unless you explicitly point them at the same path."""
    if out_path is None:
        out_path = D6A_OUT if direction == "forward" else D6A_OUT_BACKWARD
    dt = 1.0 / CONTROL_HZ
    ik = IKSolver(_MINK_XML, dt)

    rows = []
    # ik._servo_angles() right after IKSolver.__init__ would just be the
    # raw 'stand' keyframe angles -- IK hasn't actually solved anything
    # yet, so they wouldn't reflect TORSO_PITCH_TRIM_DEG at all. Settle a
    # few IK iterations at the nominal standing COM/feet targets first, so
    # row 0 (and everything after it) is consistent with the same
    # torso-biased solve the whole gait uses.
    for _ in range(50):
        settled_angles = ik.solve(ik.com0, ik.feet0)
    stand_pulses = sim_to_pulse([settled_angles[s] for s in SERVO_IDS])
    rows.append((D6A_STAND_HOLD_MS, list(stand_pulses)))

    tick_ms = 1000.0 / CONTROL_HZ
    ticks_since_row = 0
    for step in walk_ticks(ik, n_steps=n_steps, direction=direction):
        ticks_since_row += 1
        if ticks_since_row < D6A_DECIMATE and not step["just_finished"]:
            continue
        pulses = sim_to_pulse([step["angles"][s] for s in SERVO_IDS])
        rows.append((int(round(tick_ms * ticks_since_row)), list(pulses)))
        ticks_since_row = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "pranav_walk.d6a"
        _write_d6a_rows(rows, tmp_path)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _safe_copy(tmp_path, out_path)

    total_ms = sum(t for t, _ in rows)
    print(f'wrote {len(rows)} rows ({total_ms / 1000.0:.1f} s of playback) '
          f'to {out_path.resolve()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', action='store_true',
                         help='bake the walk into a .d6a action-group file '
                              '(action/tonypi_walk.d6a, or '
                              'action/tonypi_walk_backward.d6a with '
                              '--backward) instead of opening the live viewer')
    parser.add_argument('--steps', type=int, default=3,
                         help='number of individual foot placements to walk '
                              '(one foot swinging forward once = one step; '
                              'what the code calls n_steps throughout); '
                              'default 3')
    direction_group = parser.add_mutually_exclusive_group()
    direction_group.add_argument('--forward', action='store_true',
                                  help='walk forward (default)')
    direction_group.add_argument('--backward', action='store_true',
                                  help='walk backward -- a genuinely '
                                       're-solved mirror-image gait, not a '
                                       'time-reversed replay (see '
                                       'CartesianPlanner\'s docstring)')
    args = parser.parse_args()
    direction = "backward" if args.backward else "forward"

    if args.record:
        record_d6a(n_steps=args.steps, direction=direction)
    else:
        # Build the IK model/solver BEFORE the sim starts stepping physics.
        # mujoco.MjModel.from_xml_path() compiling a second model on the
        # controller thread while the main thread is concurrently calling
        # mj_step() on the sim's own model (sim.run()'s physics loop) is
        # not safe: it intermittently raises "engine error: Python
        # exception raised" out of from_xml_path. TonyPiSim's
        # _controller_wrapper catches that, stores it in
        # self._controller_error, and -- since viewer=True keeps
        # self._running True -- leaves the controller thread dead and the
        # robot frozen in its initial pose with the viewer window still
        # open. That's the "robot is stationary" symptom. Building the IK
        # model here, before sim.run() is even called, avoids the race
        # entirely.
        dt = 1.0 / CONTROL_HZ
        ik = IKSolver(_MINK_XML, dt)

        # WALK_KP_OVERRIDES bumps the sagittal-leg actuators' kp for THIS
        # sim only, via a throwaway patched copy of scene.xml/robot.xml --
        # model/robot.xml itself is left exactly as controller_stand.py
        # needs it. See sim_with_kp_overrides() above for why.
        sim = sim_with_kp_overrides(WALK_KP_OVERRIDES, viewer=True)  # viewer=False, realtime=False -> fast headless run
        sim.set_camera(azimuth=90, elevation=-5, distance=0.9, lookat=[0.0, 0.0, 0.2])
        sim.run(functools.partial(controller, ik=ik, n_steps=args.steps, direction=direction), duration=None)   # seconds of sim time, then the window closes (None = stay open)
