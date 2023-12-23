import mujoco
import mujoco.viewer
import numpy as np
import time

# Controller parameters.
integration_dt: float = 1.0  # Integration timestep (seconds).
damping: float = 1e-5  # Damping term for the pseudoinverse.


def main() -> None:
    # Load the model and data.
    model = mujoco.MjModel.from_xml_path("universal_robots_ur5e/scene.xml")
    data = mujoco.MjData(model)

    # Set PD gains.
    Kp = np.asarray([2000.0, 2000.0, 2000.0, 500.0, 500.0, 500.0])
    Kd = np.asarray([100.0, 100.0, 100.0, 50.0, 50.0, 50.0])
    model.actuator_gainprm[:, 0] = Kp
    model.actuator_biasprm[:, 1] = -Kp
    model.actuator_biasprm[:, 2] = -Kd

    # Enable gravity compensation. Set to 0.0 to disable.
    model.body_gravcomp[:] = 1.0

    # Simulation and control timesteps. Feel free to change these.
    dt = 0.002
    model.opt.timestep = dt
    control_dt = 0.02  # Must be a multiple of dt.
    n_steps = int(round(control_dt / dt))

    # End-effector site we wish to control.
    site_name = "attachment_site"
    site_id = model.site(site_name).id

    # Get the dof and actuator ids for the joints we wish to control. These are copied
    # from the XML file. Feel free to comment out some joints to see the effect on
    # the controller.
    joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
    ]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    # Initial joint configuration saved as a keyframe in the XML file.
    key_name = "home"
    key_id = model.key(key_name).id

    # Mocap body we will control with our mouse.
    mocap_name = "target"
    mocap_id = model.body(mocap_name).mocapid[0]

    # Reset the simulation.
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    # Pre-allocate numpy arrays.
    jac = np.zeros((6, model.nv))
    diag = damping * np.eye(model.nv)
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

    def circle(t: float, r: float, h: float, k: float, f: float) -> np.ndarray:
        """Return the (x, y) coordinates of a circle with radius r centered at (h, k)
        as a function of time t and frequency f."""
        x = r * np.cos(2 * np.pi * f * t) + h
        y = r * np.sin(2 * np.pi * f * t) + k
        return np.array([x, y])

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        while viewer.is_running():
            step_start = time.time()

            data.mocap_pos[mocap_id, 0:2] = circle(data.time, 0.1, 0.5, 0.0, 0.5)

            # Spatial velocity (aka twist).
            twist[:3] = data.mocap_pos[mocap_id] - data.site(site_id).xpos
            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
            mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)

            # Jacobian.
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

            # Solve for joint velocities: J * dq = twist using damped least squares.
            dq = np.linalg.solve(jac.T @ jac + diag, jac.T @ twist)

            # Integrate joint velocities to obtain joint positions.
            q = data.qpos.copy()  # Note the copy here is important.
            mujoco.mj_integratePos(model, q, dq, integration_dt)
            np.clip(q, *model.jnt_range.T, out=q)

            # Set the control signal and step the simulation.
            data.ctrl[actuator_ids] = q[dof_ids]
            mujoco.mj_step(model, data, n_steps)

            viewer.sync()
            time_until_next_step = control_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
