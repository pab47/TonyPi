#!/usr/bin/env python3
"""
tonypi_sim.py -- MuJoCo stand-in for the TonyPi Pro hardware.

Exposes the same three calls your hardware scripts use, so a controller
written against this file can be moved to the robot by swapping the sim
object for the Board/Mpu6050 calls (see tonypi_hw.py):

    sim.getAngles(servo_No)                       -> joint angle [rad]
    sim.setAngles(servo_No, desired_angle, time_in_ms)
    sim.getIMU()                                  -> (accel[g], gyro[deg/s])

Servo numbering is the TonyPi bus-servo numbering (1..16) in exactly the
order used by read_angles.py / write_angles.py / servo_map.py:

    1 L_ANK_ROLL   2 L_ANK_PITCH  3 L_KNEE   4 L_HIP_PITCH
    5 L_HIP_ROLL   6 L_ELBOW      7 L_SH_ROLL 8 L_SH_PITCH
    9..16 = same order, right side

Head pan/tilt and the two grippers are NOT controlled (they sit at zero on
their default PD controller, exactly like the other joints).

Servo model: setAngles() starts a linear ramp of the PD set-point from the
joint angle measured at the time of the call to desired_angle, lasting
time_in_ms.  The MuJoCo <position> actuator defined in robot.xml (kp,
dampratio, forcerange) does the rest -- that is the "default PD controller"
of every servo.  Just like Board.setBusServoPulse, setAngles() returns
immediately; the motion happens in the background.

Typical use (see controller.py):

    from tonypi_sim import TonyPiSim

    def controller(sim):
        sim.setAngles(3, 0.8, 500)     # left knee to 0.8 rad over 500 ms
        sim.sleep(0.5)
        print(sim.getAngles(3))
        a, w = sim.getIMU()

    TonyPiSim().run(controller)

Physics stepping, real-time pacing, and the viewer are all handled by
run(); you never call mj_step yourself.

Runs with plain   python3 controller.py   on macOS, Linux and Windows (the
viewer is MuJoCo's managed viewer, the one behind `python3 -m mujoco.viewer`).
TonyPiSim(viewer=False) gives a headless run.
"""
import math
import os
import sys
import threading
import time
import traceback

import mujoco
import mujoco.viewer
import numpy as np

# ---------------------------------------------------------------------------
# Servo id -> MuJoCo joint/actuator name (order = TonyPi bus servo ids 1..16)
# ---------------------------------------------------------------------------
_LEFT = ['ankle_roll', 'ankle_pitch', 'knee', 'hip_pitch',
         'hip_roll', 'elbow', 'shoulder_roll', 'shoulder_pitch']
SERVO_JOINT = {i + 1: 'l_' + n for i, n in enumerate(_LEFT)}
SERVO_JOINT.update({i + 9: 'r_' + n for i, n in enumerate(_LEFT)})
SERVO_IDS = list(range(1, 17))

# Human-readable names, identical to read_angles.py
_LEFT_NAMES = ['ANK_ROLL', 'ANK_PITCH', 'KNEE', 'HIP_PITCH',
               'HIP_ROLL', 'ELBOW', 'SH_ROLL', 'SH_PITCH']
NAME = {i + 1: 'L_' + n for i, n in enumerate(_LEFT_NAMES)}
NAME.update({i + 9: 'R_' + n for i, n in enumerate(_LEFT_NAMES)})

_G = 9.80665
_DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '../..', 'model', 'scene.xml')


class _SimStopped(Exception):
    """Raised inside sleep() when the simulation has ended."""


class TonyPiSim:
    """Hardware-like interface around a MuJoCo model of the TonyPi Pro."""

    def __init__(self, model_path=_DEFAULT_MODEL, viewer=True, realtime=True,
                 speed=1.0, keyframe='home',
                 init_quat=None, init_joint_angles=None):
        """
        model_path : scene.xml (includes robot.xml)
        viewer     : open the interactive MuJoCo window (always real time;
                     the window stays open after the controller returns)
        realtime   : headless only: pace the physics to the wall clock.
                     False = run as fast as possible (use sim.sleep, not
                     time.sleep, in the controller in that case).
        speed      : headless real-time factor (0.5 = half speed); in the
                     viewer use the speed slider in the right panel
        keyframe   : initial pose: 'home' (standing), 'prone', 'supine'
        init_quat  : (w, x, y, z) orientation to force on the free-floating
                     base after `keyframe` is loaded. None (default) leaves
                     whatever the keyframe gives -- for keyframe='home' that
                     is (1, 0, 0, 0), i.e. upright. Pass an explicit
                     quaternion to start the robot tipped over, on its
                     side, upside down, etc. (e.g. to test recovering from
                     a fall). Not normalized for you unless it already has
                     unit length.
        init_joint_angles : 16 angles [rad], one per servo id 1..16
                     (SERVO_IDS order), to force after `keyframe` is loaded.
                     None (default) leaves whatever the keyframe gives --
                     for keyframe='home' that is all zero. The base
                     position (x, y, z) always comes from the keyframe;
                     raise it yourself in the model/keyframe if a chosen
                     orientation would otherwise clip through the floor.
        """
        self.model = mujoco.MjModel.from_xml_path(os.path.abspath(model_path))
        self.data = mujoco.MjData(self.model)
        self.viewer_enabled = viewer
        self.realtime = realtime
        self.speed = float(speed)
        self.dt = self.model.opt.timestep

        # --- resolve ids ----------------------------------------------------
        self._qadr = {}     # servo id -> qpos address
        self._dadr = {}     # servo id -> qvel address
        self._act = {}      # servo id -> actuator index
        self._ctrlrange = {}
        for sid, jname in SERVO_JOINT.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
            if jid < 0 or aid < 0:
                raise RuntimeError(f'joint/actuator "{jname}" not found in model')
            self._qadr[sid] = self.model.jnt_qposadr[jid]
            self._dadr[sid] = self.model.jnt_dofadr[jid]
            self._act[sid] = aid
            self._ctrlrange[sid] = tuple(self.model.actuator_ctrlrange[aid])

        self._gyro_adr = self._sensor_adr('imu_gyro')
        self._acc_adr = self._sensor_adr('imu_acc')
        self._quat_adr = self._sensor_adr('imu_quat')

        # --- initial pose ---------------------------------------------------
        kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        if kid >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
        else:
            mujoco.mj_resetData(self.model, self.data)
        if init_quat is not None:
            q = np.asarray(init_quat, dtype=float)
            n = np.linalg.norm(q)
            if n > 0:
                q = q / n
            self.data.qpos[3:7] = q
        if init_joint_angles is not None:
            if len(init_joint_angles) != len(SERVO_IDS):
                raise ValueError(
                    f'init_joint_angles must have {len(SERVO_IDS)} values '
                    f'(one per servo id, SERVO_IDS order), got '
                    f'{len(init_joint_angles)}')
            for sid, ang in zip(SERVO_IDS, init_joint_angles):
                self.data.qpos[self._qadr[sid]] = float(ang)
        mujoco.mj_forward(self.model, self.data)

        # --- per-servo interpolation plan: (q_start, q_goal, t_start, T) ----
        self._plan = {sid: (self.data.qpos[self._qadr[sid]],
                            self.data.qpos[self._qadr[sid]], 0.0, 0.0)
                      for sid in SERVO_IDS}
        for sid in SERVO_IDS:
            self.data.ctrl[self._act[sid]] = self.data.qpos[self._qadr[sid]]

        self._lock = threading.Lock()
        self._running = False
        self._controller_error = None
        # fast mode: physics only advances while the controller is in sleep()
        self._run_until = math.inf if realtime else 0.0
        self.model_path = os.path.abspath(model_path)
        self._duration = None
        self._cam_req = {}          # set_camera() requests
        self._viewer_cam = None     # MjvCamera of the open window, if any
        self._simulate = None

    # ======================================================================
    # Hardware-equivalent API
    # ======================================================================
    def getAngles(self, servo_No=None):
        """Joint angle [rad] of bus servo servo_No (1..16).
        Called with no argument, returns a list of all 16 in servo order."""
        with self._lock:
            if servo_No is None:
                return [float(self.data.qpos[self._qadr[s]]) for s in SERVO_IDS]
            self._check_id(servo_No)
            return float(self.data.qpos[self._qadr[servo_No]])

    def getVelocities(self, servo_No=None):
        """Joint velocity [rad/s] (not available on the real servo bus;
        provided for debugging only)."""
        with self._lock:
            if servo_No is None:
                return [float(self.data.qvel[self._dadr[s]]) for s in SERVO_IDS]
            self._check_id(servo_No)
            return float(self.data.qvel[self._dadr[servo_No]])

    def setAngles(self, servo_No, desired_angle, time_in_ms):
        """Move servo servo_No to desired_angle [rad] in time_in_ms [ms].

        The PD set-point ramps linearly from the CURRENT measured joint
        angle to desired_angle over time_in_ms, then holds.  Returns at
        once (non-blocking), like Board.setBusServoPulse.  servo_No may
        also be a list of ids with a matching list of angles."""
        if isinstance(servo_No, (list, tuple, np.ndarray)):
            for s, q in zip(servo_No, desired_angle):
                self.setAngles(int(s), float(q), time_in_ms)
            return
        self._check_id(servo_No)
        lo, hi = self._ctrlrange[servo_No]
        goal = float(min(max(desired_angle, lo), hi))
        T = max(0.0, float(time_in_ms) / 1000.0)
        with self._lock:
            q_now = float(self.data.qpos[self._qadr[servo_No]])
            self._plan[servo_No] = (q_now, goal, self.data.time, T)
            if T == 0.0:
                self.data.ctrl[self._act[servo_No]] = goal

    def getIMU(self):
        """IMU reading in the BODY frame (X forward, Y left, Z up), same
        convention as chip_to_body() in read_angles.py:

            accel, gyro = sim.getIMU()
            accel = {'x','y','z'} in g       (+1 on z when standing still)
            gyro  = {'x','y','z'} in deg/s
        """
        with self._lock:
            acc = self.data.sensordata[self._acc_adr:self._acc_adr + 3].copy()
            gyr = self.data.sensordata[self._gyro_adr:self._gyro_adr + 3].copy()
        # MuJoCo imu site frame = torso frame: +x right, +y back, +z up.
        # Body frame (hardware): X forward = -y_site, Y left = -x_site, Z up.
        accel = {'x': float(-acc[1] / _G), 'y': float(-acc[0] / _G), 'z': float(acc[2] / _G)}
        gyro = {'x': math.degrees(-gyr[1]), 'y': math.degrees(-gyr[0]), 'z': math.degrees(gyr[2])}
        return accel, gyro

    # ----------------------------------------------------------------------
    # Convenience (not on the hardware)
    # ----------------------------------------------------------------------
    @staticmethod
    def pitch_roll_from_accel(accel):
        """Static pitch/roll [deg] from the accelerometer, exactly as in
        read_angles.py: pitch + = leaning forward, roll + = leaning right."""
        pitch = math.degrees(math.atan2(-accel['x'], math.hypot(accel['y'], accel['z'])))
        roll = math.degrees(math.atan2(accel['y'], accel['z']))
        return pitch, roll

    def time(self):
        """Simulation clock [s]."""
        with self._lock:
            return float(self.data.time)

    def sleep(self, seconds):
        """Wait until the simulation clock has advanced by `seconds`.
        Equivalent to time.sleep() when realtime=True, and the right thing
        to use when realtime=False.  Returns early if the sim stops."""
        t_end = self.time() + float(seconds)
        if not self.realtime:
            self._run_until = t_end      # let the physics thread advance to here
        while self._running and self.time() < t_end:
            time.sleep(0.0005)
        if not self._running:
            raise _SimStopped()        # unwinds the controller quietly

    def reset(self, keyframe='home'):
        """Reset the robot to a keyframe pose and cancel all ramps."""
        with self._lock:
            kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
            if kid >= 0:
                mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
            else:
                mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            for sid in SERVO_IDS:
                q = float(self.data.qpos[self._qadr[sid]])
                self._plan[sid] = (q, q, self.data.time, 0.0)
                self.data.ctrl[self._act[sid]] = q

    def set_camera(self, azimuth=None, elevation=None, distance=None, lookat=None):
        """Viewer camera.  Usable before run() (initial view) or from inside
        the controller while the window is open.

            sim.set_camera(azimuth=150, elevation=-15, distance=0.9,
                           lookat=[0, 0, 0.17])

        azimuth/elevation in degrees, distance in metres, lookat [x, y, z].
        Arguments left as None are unchanged."""
        if azimuth is not None:
            self._cam_req['azimuth'] = float(azimuth)
        if elevation is not None:
            self._cam_req['elevation'] = float(elevation)
        if distance is not None:
            self._cam_req['distance'] = float(distance)
        if lookat is not None:
            self._cam_req['lookat'] = [float(v) for v in lookat]
        if self._viewer_cam is not None:          # window already open
            self._apply_camera(self._viewer_cam)

    def stop(self):
        """End the simulation (closes the viewer).  Callable from the
        controller."""
        self._running = False
        if self._simulate is not None:
            self._simulate.exit()

    # ======================================================================
    # Simulation loop (hidden from the controller)
    # ======================================================================
    def run(self, controller, duration=None):
        """Run `controller(sim)` in a worker thread while the physics is
        stepped for it.  With viewer=True the MuJoCo managed viewer (the
        same one as `python3 -m mujoco.viewer`) owns the window and the
        real-time physics loop -- plain python3 on every OS, no mjpython.
        Servo ramps are applied from a MuJoCo control callback, so they run
        inside whatever loop is stepping.

        duration : seconds of SIMULATION time after which the run ends and
                   the window closes (None = stay open until closed by hand;
                   headless runs end when the controller returns)."""
        self._running = True
        self._duration = None if duration is None else float(duration)
        worker = threading.Thread(target=self._controller_wrapper,
                                  args=(controller,), daemon=True)
        mujoco.set_mjcb_control(self._control_cb)
        worker.start()
        try:
            if self.viewer_enabled:
                self._viewer_loop()
            else:
                self._headless_loop(worker)
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            self._viewer_cam = None
            self._simulate = None
            mujoco.set_mjcb_control(None)
        if self._controller_error is not None:
            raise self._controller_error

    # ----------------------------------------------------------------------
    def _viewer_loop(self):
        """Managed MuJoCo viewer (window + real-time physics thread), built
        from the same pieces mujoco.viewer.launch() uses so that we keep a
        handle on the camera and can close the window programmatically."""
        from mujoco import viewer as mv
        import glfw
        import atexit

        cam = mujoco.MjvCamera()
        opt = mujoco.MjvOption()
        pert = mujoco.MjvPerturb()
        simulate = mv._Simulate(cam, opt, pert, None, True, None)
        simulate.ui0_enable = True
        simulate.ui1_enable = True

        if mv._MJPYTHON is None:
            if not glfw.init():
                raise mujoco.FatalError('could not initialize GLFW')
            atexit.register(glfw.terminate)

        # Giving the loader a file name makes the viewer call
        # mjv_defaultFreeCamera on load; we then overwrite with the request.
        def loader():
            return self.model, self.data, self.model_path

        self._simulate = simulate
        physics = threading.Thread(target=mv._physics_loop, args=(simulate, loader))
        watchdog = threading.Thread(target=self._viewer_watchdog, args=(simulate, cam),
                                    daemon=True)
        physics.start()
        watchdog.start()
        simulate.render_loop()          # blocks until the window closes
        self._running = False
        physics.join()
        simulate.destroy()

    def _viewer_watchdog(self, simulate, cam):
        """Apply the requested camera once the model is loaded, then close
        the window when `duration` of sim time has elapsed."""
        while self._running and simulate.m is None and not simulate.exitrequest:
            time.sleep(0.01)
        time.sleep(0.05)                # let LoadOnRenderThread run first
        with simulate.lock():
            self._apply_camera(cam)
        self._viewer_cam = cam
        while self._running and not simulate.exitrequest:
            if self._duration is not None and self.data.time >= self._duration:
                simulate.exit()
                break
            time.sleep(0.02)

    def _apply_camera(self, cam):
        r = self._cam_req
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        if 'azimuth' in r:
            cam.azimuth = r['azimuth']
        if 'elevation' in r:
            cam.elevation = r['elevation']
        if 'distance' in r:
            cam.distance = r['distance']
        if 'lookat' in r:
            cam.lookat[:] = r['lookat']

    def _headless_loop(self, worker):
        t_wall0 = time.perf_counter()
        t_sim0 = self.data.time
        while self._running and worker.is_alive():
            if self._duration is not None and self.data.time >= self._duration:
                break
            if self.realtime:
                target = t_sim0 + (time.perf_counter() - t_wall0) * self.speed
                if self.data.time >= target:
                    time.sleep(0.0005)
                    continue
                n = int(min((target - self.data.time) / self.dt, 0.1 / self.dt)) + 1
            else:
                # fast mode: advance only up to the time the controller asked
                # for in sleep(), so results do not depend on scheduling
                if self.data.time >= self._run_until:
                    time.sleep(0.0002)
                    continue
                n = int(min((self._run_until - self.data.time) / self.dt, 200)) + 1
            with self._lock:
                for _ in range(n):
                    mujoco.mj_step(self.model, self.data)

    def _control_cb(self, model, data):
        """mjcb_control: called by MuJoCo inside every mj_step."""
        self._apply_ramps()

    # ----------------------------------------------------------------------
    def _apply_ramps(self):
        """Linear interpolation of the PD set-points."""
        t = self.data.time
        for sid in SERVO_IDS:
            q0, q1, t0, T = self._plan[sid]
            if T <= 0.0:
                self.data.ctrl[self._act[sid]] = q1
            else:
                a = (t - t0) / T
                a = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)
                self.data.ctrl[self._act[sid]] = q0 + a * (q1 - q0)

    def _controller_wrapper(self, controller):
        try:
            controller(self)
        except _SimStopped:
            pass
        except Exception as e:      # noqa: BLE001
            traceback.print_exc()
            self._controller_error = e
        finally:
            if not self.viewer_enabled:
                self._running = False

    def _check_id(self, sid):
        if sid not in SERVO_JOINT:
            raise ValueError(f'servo id must be 1..16 (got {sid}); head and '
                             f'grippers are not controlled in this sim')

    def _sensor_adr(self, name):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            raise RuntimeError(f'sensor "{name}" not in model')
        return self.model.sensor_adr[sid]
