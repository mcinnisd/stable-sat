#!/usr/bin/env python3
"""
Unified Simulation Driver for Satellite Attitude Control
"""

import argparse
import logging
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

import visualization
from visualization import plot_method_comparison
from dynamics import SatelliteEnv, SatelliteRLEnv
from controllers import PIDController, LQRController
from stable_baselines3 import PPO
import os
import re

def run_animation(env, output: str) -> None:
    """
    Runs the simulation in animation mode and saves the output.

    Parameters:
        env: The satellite environment.
        output (str): The filename for the output animation.
    """
    frames: int = int(env.sim_time / env.dt)

    def update(frame: int) -> list:
        obs, reward, done, _ = env.step()
        env.render()
        return []

    ani = animation.FuncAnimation(env.fig, update, frames=frames, interval=100, blit=False)
    ani.save(output, writer="pillow", fps=10)
    plt.close('all')
    logging.info("Animation saved to %s", output)

def run_interactive(env) -> None:
    """
    Runs the simulation in interactive mode.

    Parameters:
        env: The satellite environment.
    """
    done: bool = False
    while not done:
        _, _, done, _ = env.step()
        env.render()
    plt.close('all')
    logging.info("Interactive simulation ended.")

def run_static_simulation(env, title_prefix=None) -> None:
    """
    Runs the simulation in static mode and plots the simulation results.

    Parameters:
        env: The satellite environment.
        title_prefix: Optional string to prepend to plot titles.
    """
    times, euler_angles, omega_history, wheel_speeds, q_scalar_history, desired_q_scalar_history = collect_time_series(env)
    # Compute attitude error from quaternion scalar
    import numpy as _np
    error_history = _np.degrees(2 * _np.arccos(_np.clip(q_scalar_history, -1.0, 1.0)))
    visualization.plot_static_simulation(
        times, euler_angles, omega_history,
        error_history=error_history,
        desired_euler_points=None,
        title_prefix=title_prefix
    )
    visualization.plot_wheel_speeds(times, wheel_speeds)
    logging.info("Static simulation plots generated.")


def collect_time_series(env, model=None, desired_scalar=1.0):
    """
    Record state history from the environment.

    If *model* is supplied, the agent’s actions are used; otherwise the
    environment relies on its internal control policy.
    Returns six numpy arrays: times, euler_angles, omega_history,
    wheel_speeds, q_scalar_history, desired_q_scalar_history.
    """
    times, eulers, omegas, wheels, q_scalars, q_des_scalars = [], [], [], [], [], []
    obs = env._get_obs() if model else None
    done = False
    while env.current_time < env.sim_time and not done:
        times.append(env.current_time)
        eulers.append(R.from_quat(env.q).as_euler('xyz', degrees=True))
        omegas.append(env.omega.copy())
        wheels.append(env.omega_w.copy())
        q_scalars.append(env.q[3])
        q_des_scalars.append(desired_scalar)
        if model:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
        else:
            obs, _, done, _ = env.step()
    return (np.array(times), np.array(eulers), np.array(omegas),
            np.array(wheels), np.array(q_scalars), np.array(q_des_scalars))

def main() -> None:
    """
    Main function to run the satellite simulation with the chosen control mode.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description="Unified Satellite Attitude Simulation")
    parser.add_argument('--sim_time', type=float, default=50.0, help="Total simulation time (s)")
    parser.add_argument('--dt', type=float, default=0.05, help="Time step (s)")
    parser.add_argument('--inertia', type=float, nargs=3, default=[0.108, 0.083, 0.042], help="Satellite inertia (Ix Iy Iz)")
    parser.add_argument('--I_w', type=float, default=0.1, help="Reaction wheel inertia")
    parser.add_argument('--initial_euler', type=float, nargs=3, default=[20, 0, 0], help="Initial Euler angles (deg)")
    parser.add_argument('--control_mode', type=str, default='PID', choices=['PID', 'LQR', 'RL', 'None'], help="Control mode: PID, LQR, RL, or None")
    parser.add_argument('--model_path', type=str, default='best', help="Path to RL model (for RL mode)")
    parser.add_argument('--controlled_axes', type=str, default='0,1,2',
                        help="Comma-separated list of axis indices for RL control (e.g., '0,1,2')")
    parser.add_argument('--max_torque', type=float, default=0.007,
                        help="Maximum torque for RL environment")
    parser.add_argument('--interactive', action='store_true', help="Run interactive simulation")
    parser.add_argument('--save_animation', action='store_true', help="Save animation as GIF")
    parser.add_argument('--output', type=str, default='satellite_simulation.gif', help="Output filename for animation")
    parser.add_argument('--plot_static', action='store_true', help="Plot static simulation results")
    parser.add_argument('--evaluate', action='store_true',
                        help="Evaluate all controllers over multiple random trials")
    parser.add_argument('--num_trials', type=int, default=100,
                        help="Number of trials for evaluation")
    parser.add_argument('--eval_rl', action='store_true',
                        help="Evaluate RL models at different training steps using a model prefix")
    parser.add_argument('--model_prefix', type=str, default=None,
                        help="Prefix of the RL model files to evaluate (e.g., 'ppo_satellite_agent')")
    args = parser.parse_args()
    # Parse controlled_axes into a list of ints for RL use
    args.controlled_axes = [int(x.strip()) for x in args.controlled_axes.split(',')]
    # Preserve the full axes list for any RL evaluation slices
    full_axes = args.controlled_axes.copy()
    # RL models evaluation mode: evaluate multiple RL checkpoints
    if args.eval_rl:
        if not args.model_prefix:
            print("Please specify --model_prefix for RL evaluation.")
            return
        # Discover model files matching prefix_<steps>
        files = os.listdir('.')
        pattern = re.compile(rf'^{re.escape(args.model_prefix)}_(\d+)(?:\.\w+)?$')
        model_entries = []
        for f in files:
            m = pattern.match(f)
            if m:
                steps = int(m.group(1))
                model_entries.append((steps, f))
        if not model_entries:
            print(f"No model files found for prefix '{args.model_prefix}_<steps>'.")
            return
        model_entries.sort(key=lambda x: x[0])
        training_steps = []
        mean_times = []
        mean_torques = []
        for steps, fname in model_entries:
            times_list = []
            torques_list = []
            for _ in range(args.num_trials):
                # sample a random initial quaternion
                rand_q = np.random.randn(4)
                init_q = rand_q / np.linalg.norm(rand_q)
                model_eval = PPO.load(fname)
                n_act = model_eval.action_space.shape[0]
                axes = args.controlled_axes[:n_act]
                env_eval = SatelliteRLEnv(sim_time=args.sim_time, dt=args.dt,
                                              inertia=args.inertia, I_w=args.I_w,
                                              max_torque=args.max_torque,
                                              controlled_axes=axes)
                obs_eval = env_eval.reset(initial_q=init_q)
                done_eval = False
                torque_sum = 0.0
                while not done_eval:
                    action, _ = model_eval.predict(obs_eval, deterministic=True)
                    obs_eval, _, done_eval, info_eval = env_eval.step(action)
                    torque_sum += np.linalg.norm(info_eval['torque'])
                times_list.append(env_eval.current_time)
                torques_list.append(torque_sum)
            training_steps.append(steps)
            mean_times.append(np.mean(times_list))
            mean_torques.append(np.mean(torques_list))
        # Plot performance across training steps
        visualization.plot_rl_model_performance(training_steps,
                                                mean_times,
                                                mean_torques)
        return
    # Evaluation mode: compare PID, LQR, and RL over random seeds
    if args.evaluate:
        controllers = ['PID', 'LQR', 'RL']
        results = {c: {'times': [], 'torques': [], 'init_errors': []} for c in controllers}
        target_q = np.array([0., 0., 0., 1.])
        # convergence thresholds
        rl_error_thr = 0.5       # degrees
        rl_omega_thr = 0.25      # rad/s

        for _ in range(args.num_trials):
            # sample a random initial quaternion
            rand_q = np.random.randn(4)
            init_q = rand_q / np.linalg.norm(rand_q)
            # compute initial attitude error
            err_rot = R.from_quat(init_q) * R.from_quat(target_q).inv()
            init_error = np.degrees(np.linalg.norm(err_rot.as_rotvec()))

            for mode in controllers:
                if mode == 'PID':
                    cp = PIDController()
                    env_eval = SatelliteEnv(sim_time=args.sim_time, dt=args.dt,
                                            inertia=args.inertia, I_w=args.I_w,
                                            control_policy=cp)
                    obs_eval = env_eval.reset(initial_q=init_q)
                elif mode == 'LQR':
                    cp = LQRController()
                    env_eval = SatelliteEnv(sim_time=args.sim_time, dt=args.dt,
                                            inertia=args.inertia, I_w=args.I_w,
                                            control_policy=cp)
                    obs_eval = env_eval.reset(initial_q=init_q)
                else:  # RL
                    # Load the model and determine its expected action dimension
                    model_eval = PPO.load(args.model_path)
                    # Extract full list of controlled axes (parsed above)s
                    axes = full_axes
                    # Trim or pad axes to match model action dimension
                    n_act = model_eval.action_space.shape[0]
                    axes = axes[:n_act]
                    env_eval = SatelliteRLEnv(sim_time=args.sim_time, dt=args.dt,
                                              inertia=args.inertia, I_w=args.I_w,
                                              max_torque=args.max_torque,
                                              controlled_axes=axes)
                    obs_eval = env_eval.reset(initial_q=init_q)

                # record initial error
                results[mode]['init_errors'].append(init_error)

                # run until convergence or timeout
                torque_sum = 0.0
                convergence_time = args.sim_time
                while env_eval.current_time < args.sim_time:
                    if mode == 'RL':
                        action, _ = model_eval.predict(obs_eval, deterministic=True)
                        obs_eval, _, _, info_eval = env_eval.step(action)
                    else:
                        obs_eval, _, _, info_eval = env_eval.step()
                    torque_sum += np.linalg.norm(info_eval['torque'])
                    # check convergence thresholds
                    q_cur = env_eval.q
                    omega_cur = env_eval.omega
                    err_rot_cur = R.from_quat(q_cur) * R.from_quat(target_q).inv()
                    error_cur = np.degrees(np.linalg.norm(err_rot_cur.as_rotvec()))
                    if mode == 'RL':
                        if error_cur < rl_error_thr and np.linalg.norm(omega_cur) < rl_omega_thr:
                            convergence_time = env_eval.current_time
                            break
                    else:
                        # classical controllers: only check angle error
                        # use their internal orientation_error_threshold
                        thr = np.degrees(env_eval.control_policy.orientation_error_threshold)
                        if error_cur < thr:
                            convergence_time = env_eval.current_time
                            break

                results[mode]['times'].append(convergence_time)
                results[mode]['torques'].append(torque_sum)

        # Summarize results
        print(f"Evaluation over {args.num_trials} trials:")
        for mode in controllers:
            times = np.array(results[mode]['times'])
            torques = np.array(results[mode]['torques'])
            print(f"{mode}: time {times.mean():.2f} ± {times.std():.2f}s, "
                  f"torque {torques.mean():.2f} ± {torques.std():.2f}")

        # --- Scatter plots of convergence time vs initial error ---
        import visualization as _viz
        for mode in controllers:
            _viz.plot_performance_vs_initial_error(
                results[mode]['init_errors'],
                results[mode]['times'],
                results[mode]['torques'],
                label=mode
            )

        # After printing summary, plot method comparisons
        methods = controllers
        time_means, time_lows, time_highs = [], [], []
        torque_means, torque_lows, torque_highs = [], [], []
        for mode in controllers:
            t = np.array(results[mode]['times'])
            tau = np.array(results[mode]['torques'])
            time_means.append(t.mean())
            time_lows.append(t.min())
            # cap the maximum time bound at 49.9 seconds
            time_highs.append(min(t.mean() + t.std(), 49.9))
            torque_means.append(tau.mean())
            torque_lows.append(tau.min())
            torque_highs.append(tau.max())
        # Plot convergence time comparison
        plot_method_comparison(
            methods,
            time_means, time_lows, time_highs,
            ylabel='Convergence Time (s)',
            title='Method Convergence Times'
        )
        # Plot torque usage comparison
        plot_method_comparison(
            methods,
            torque_means, torque_lows, torque_highs,
            ylabel='Total Torque',
            title='Method Torque Usage'
        )
        return

    

    initial_q = R.from_euler('xyz', np.radians(args.initial_euler)).as_quat()

    if args.control_mode == 'PID':
        control_policy = PIDController(max_torque=args.max_torque)
        env = SatelliteEnv(sim_time=args.sim_time, dt=args.dt, inertia=args.inertia, I_w=args.I_w, control_policy=control_policy)
        env.reset(initial_q=initial_q)
        # When plotting static PID simulation, target only the zero orientation
        if args.plot_static or args.interactive:
            control_policy.desired_points = [np.array([0.0, 0.0, 0.0, 1.0])]
            control_policy.current_target_idx = 0
    elif args.control_mode == 'LQR':
        control_policy = LQRController(max_torque=args.max_torque)
        env = SatelliteEnv(sim_time=args.sim_time, dt=args.dt, inertia=args.inertia, I_w=args.I_w, control_policy=control_policy)
        env.reset(initial_q=initial_q)
        # When plotting static LQR simulation, target only the zero orientation
        if args.plot_static or args.interactive:
            control_policy.desired_points = [np.array([0.0, 0.0, 0.0, 1.0])]
            control_policy.current_target_idx = 0
        
    elif args.control_mode == 'RL':
        env = SatelliteRLEnv(sim_time=args.sim_time, dt=args.dt, inertia=args.inertia,
                             I_w=args.I_w, max_torque=args.max_torque,
                             controlled_axes=args.controlled_axes)
        model = PPO.load(args.model_path)
        print(f"Loaded RL model from {args.model_path}")
        obs = env.reset(initial_q=initial_q)
    else:
        env = SatelliteEnv(sim_time=args.sim_time, dt=args.dt, inertia=args.inertia, I_w=args.I_w, control_policy=None)
        env.reset(initial_q=initial_q)

    if args.save_animation:
        run_animation(env, args.output)
    elif args.interactive:
        if args.control_mode == 'RL':
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                env.render()
            env.close()
        else:
            run_interactive(env)
    elif args.plot_static:
        if args.control_mode == 'RL':
            times, euler_angles, omega_history, wheel_speeds, q_scalar_history, desired_q_scalar_history = \
                collect_time_series(env, model)
            import numpy as _np
            error_history = _np.degrees(2 * _np.arccos(_np.clip(q_scalar_history, -1.0, 1.0)))
            visualization.plot_static_simulation(
                times, euler_angles, omega_history,
                error_history=error_history,
                desired_euler_points=None,
                title_prefix=args.model_path
            )
            visualization.plot_wheel_speeds(times, wheel_speeds)
        else:
            run_static_simulation(env, title_prefix=args.control_mode)
    else:
        if args.control_mode == 'RL':
            times, euler_angles, omega_history, wheel_speeds, q_scalar_history, desired_q_scalar_history = \
                collect_time_series(env, model)
            import numpy as _np
            error_history = _np.degrees(2 * _np.arccos(_np.clip(q_scalar_history, -1.0, 1.0)))
            visualization.plot_static_simulation(
                times, euler_angles, omega_history,
                error_history=error_history,
                desired_euler_points=None,
                title_prefix=args.model_path
            )
            visualization.plot_wheel_speeds(times, wheel_speeds)
        else:
            run_static_simulation(env, title_prefix=args.control_mode)

if __name__ == '__main__':
    main()