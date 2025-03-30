#!/usr/bin/env python3
"""
Satellite Attitude Simulation (Step 1)

This script simulates the 3D satellite attitude dynamics with an external torque input.
The external torque input can be chosen among different modes.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import argparse

# Import visualization and torque functions as before
import visualization
from torqueFunctions import external_torque_zero, EXTERNAL_TORQUE_MODES
from satelliteEnv import SatelliteAttitudeEnv
from dynamics import update_state

def simulate_satellite(torque_mode='impulse', sim_time=10.0, dt=0.1, inertia=[1.0, 1.0, 1.0]):
    """
    Simulates the satellite attitude dynamics over time.
    
    Returns arrays for time, Euler angles (degrees), and angular velocities.
    """
    I_sat = np.diag(inertia)
    # Initial state: identity quaternion and zero angular velocity
    q = np.array([0, 0, 0, 1])
    omega = np.zeros(3)
    
    times = []
    euler_angles = []
    omega_history = []
    
    torque_func = EXTERNAL_TORQUE_MODES.get(torque_mode, external_torque_zero)
    
    t = 0.0
    while t <= sim_time:
        times.append(t)
        euler = R.from_quat(q).as_euler('xyz', degrees=True)
        euler_angles.append(euler)
        omega_history.append(omega.copy())
        
        tau_ext = torque_func(t)
        q, omega = update_state(q, omega, dt, tau_ext, I_sat)
        
        t += dt
        
    return np.array(times), np.array(euler_angles), np.array(omega_history)

def main():
    parser = argparse.ArgumentParser(description="Satellite Attitude Simulation")
    parser.add_argument('--torque_mode', type=str, default='impulse', choices=list(EXTERNAL_TORQUE_MODES.keys()),
                        help="Mode for external torque: impulse, sinusoid, step, zero")
    parser.add_argument('--sim_time', type=float, default=10.0, help="Total simulation time in seconds")
    parser.add_argument('--dt', type=float, default=0.1, help="Time step for simulation")
    parser.add_argument('--inertia', type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help="Satellite inertia values (Ix, Iy, Iz)")
    parser.add_argument('--interactive', action='store_true', help="Run interactive simulation using Gym environment stub")
    parser.add_argument('--save_animation', action='store_true',
                        help="Save the simulation as a gif or video instead of running interactively")
    parser.add_argument('--output', type=str, default='satellite_simulation.gif',
                        help="Output filename for saved animation")
    args = parser.parse_args()
    
    if args.save_animation:
        import matplotlib.animation as animation
        env = SatelliteAttitudeEnv(torque_mode=args.torque_mode, sim_time=args.sim_time, dt=args.dt, inertia=args.inertia)
        obs = env.reset()
        frames = int(env.sim_time / env.dt)
        
        def update(frame):
            obs, reward, done, _ = env.step()
            env.render()
            return []
        
        ani = animation.FuncAnimation(env.fig, update, frames=frames, interval=100, blit=False)
        ani.save(args.output, writer="pillow", fps=10)
        plt.close('all')
    elif args.interactive:
        env = SatelliteAttitudeEnv(torque_mode=args.torque_mode, sim_time=args.sim_time, dt=args.dt, inertia=args.inertia)
        obs = env.reset()
        done = False
        while not done:
            obs, reward, done, _ = env.step()
            env.render()
        plt.close('all')
    else:
        times, euler_angles, omega_history = simulate_satellite(
            torque_mode=args.torque_mode,
            sim_time=args.sim_time,
            dt=args.dt,
            inertia=args.inertia
        )
        visualization.plot_static_simulation(times, euler_angles, omega_history)

if __name__ == '__main__':
    main()