#!/usr/bin/env python3
"""
Unified Simulation Driver for Satellite Attitude Control

This script simulates the satellite environment with different control modes (PID, LQR, etc.)
and provides options for interactive simulation, animation saving, or static plotting.
"""

import argparse
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

import visualization
from dynamics import SatelliteEnv, normalize_quaternion
from controllers import PIDController, LQRController

def main():
    parser = argparse.ArgumentParser(description="Unified Satellite Attitude Simulation")
    parser.add_argument('--sim_time', type=float, default=50.0, help="Total simulation time (s)")
    parser.add_argument('--dt', type=float, default=0.1, help="Time step (s)")
    parser.add_argument('--inertia', type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help="Satellite inertia values (Ix, Iy, Iz)")
    parser.add_argument('--I_w', type=float, default=0.05, help="Reaction wheel inertia")
    parser.add_argument('--initial_euler', type=float, nargs=3, default=[20, 0, 0],
                        help="Initial Euler angles (roll, pitch, yaw in degrees)")
    parser.add_argument('--control_mode', type=str, default='PID', choices=['PID', 'LQR', 'None'],
                        help="Control mode to use: PID, LQR, or None")
    parser.add_argument('--interactive', action='store_true', help="Run interactive simulation")
    parser.add_argument('--save_animation', action='store_true', help="Save animation as gif/video")
    parser.add_argument('--output', type=str, default='satellite_simulation.gif', help="Output filename for animation")
    args = parser.parse_args()
    
    # Convert initial Euler angles to quaternion.
    initial_euler = np.radians(args.initial_euler)
    initial_q = R.from_euler('xyz', initial_euler).as_quat()
    
    # Select control policy based on the control_mode.
    if args.control_mode == 'PID':
        control_policy = PIDController()
    elif args.control_mode == 'LQR':
        control_policy = LQRController()
    else:
        control_policy = None  # Environment will default to zero control.
    
    # Instantiate the environment with the selected control policy.
    env = SatelliteEnv(sim_time=args.sim_time, dt=args.dt, inertia=args.inertia, I_w=args.I_w,
                       control_policy=control_policy)
    env.reset(initial_q=initial_q)
    
    # Run simulation in one of three modes.
    if args.save_animation:
        frames = int(env.sim_time / env.dt)
        def update(frame):
            obs, reward, done, _ = env.step()
            env.render()
            return []
        ani = animation.FuncAnimation(env.fig, update, frames=frames, interval=100, blit=False)
        ani.save(args.output, writer="pillow", fps=10)
        plt.close('all')
    elif args.interactive:
        done = False
        while not done:
            _, _, done, _ = env.step()
            env.render()
        plt.close('all')
    else:
        times = []
        euler_angles = []
        omega_history = []
        wheel_speeds = []
        while env.current_time < env.sim_time:
            times.append(env.current_time)
            euler = R.from_quat(env.q).as_euler('xyz', degrees=True)
            euler_angles.append(euler)
            omega_history.append(env.omega.copy())
            wheel_speeds.append(env.omega_w.copy())
            env.step()
        visualization.plot_static_simulation(np.array(times), np.array(euler_angles), np.array(omega_history))
        visualization.plot_wheel_speeds(np.array(times), np.array(wheel_speeds))
    
if __name__ == '__main__':
    main()