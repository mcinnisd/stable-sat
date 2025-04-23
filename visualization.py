#!/usr/bin/env python3
"""
Visualization Module

This module contains functions for drawing the satellite in a 3D matplotlib axis,
plotting static simulation data, and additional plotting functions such as for reaction wheel speeds.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def draw_satellite(ax, q, inertia, current_time, q_desired=None):
    """
    Draws the satellite on the provided 3D axis.
    
    Parameters:
        ax: matplotlib 3D axis.
        q: Current quaternion (array-like) for the satellite orientation.
        inertia: Inertia values [Ix, Iy, Iz] used to scale the satellite's shape.
        current_time: Simulation time (displayed in the title).
        q_desired: Desired quaternion for target orientation.
    """
    ax.cla()  # Clear dynamic content
    origin = np.array([0, 0, 0])
    # Draw inertial frame axes
    ax.quiver(*origin, 1, 0, 0, color='r', length=1, normalize=True)
    ax.quiver(*origin, 0, 1, 0, color='g', length=1, normalize=True)
    ax.quiver(*origin, 0, 0, 1, color='b', length=1, normalize=True)
    
    # Draw satellite as a rectangular prism scaled by inertia
    avg_inertia = np.mean(inertia)
    l_x = 0.5 * (inertia[0] / avg_inertia)
    l_y = 0.5 * (inertia[1] / avg_inertia)
    l_z = 0.5 * (inertia[2] / avg_inertia)
    
    cube_points = np.array([
        [-l_x, -l_y, -l_z],
        [-l_x, -l_y,  l_z],
        [-l_x,  l_y,  l_z],
        [-l_x,  l_y, -l_z],
        [ l_x, -l_y, -l_z],
        [ l_x, -l_y,  l_z],
        [ l_x,  l_y,  l_z],
        [ l_x,  l_y, -l_z],
    ])
    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    ]
    R_body = R.from_quat(q).as_matrix()
    cube_points_world = (R_body @ cube_points.T).T
    for edge in edges:
        p1, p2 = cube_points_world[edge[0]], cube_points_world[edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='k')
    
    # Draw satellite body axes
    body_x = R_body @ np.array([1, 0, 0])
    body_y = R_body @ np.array([0, 1, 0])
    body_z = R_body @ np.array([0, 0, 1])
    ax.quiver(*origin, *body_x, color='m', length=0.8, normalize=True)
    ax.quiver(*origin, *body_y, color='c', length=0.8, normalize=True)
    ax.quiver(*origin, *body_z, color='y', length=0.8, normalize=True)

    # If target orientation is provided, draw its coordinate axes as dashed lines
    if q_desired is not None:
        R_target = R.from_quat(q_desired).as_matrix()
        target_x = R_target @ np.array([1, 0, 0])
        target_y = R_target @ np.array([0, 1, 0])
        target_z = R_target @ np.array([0, 0, 1])
        # Draw dashed lines for the target axes
        ax.plot([0, target_x[0]*0.8], [0, target_x[1]*0.8], [0, target_x[2]*0.8], 'r--', label='Target X')
        ax.plot([0, target_y[0]*0.8], [0, target_y[1]*0.8], [0, target_y[2]*0.8], 'g--', label='Target Y')
        ax.plot([0, target_z[0]*0.8], [0, target_z[1]*0.8], [0, target_z[2]*0.8], 'b--', label='Target Z')

    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    # ax.set_title(f"Satellite Orientation at t = {current_time:.2f}s")

def plot_static_simulation(times, euler_angles, omega_history, error_history=None, desired_euler_points=None, title_prefix=None):
    """
    Plots static graphs for the satellite simulation.
    
    Plots:
      - Euler angles (roll, pitch, yaw) over time.
      - Angular velocity components over time.
      - Attitude error over time if error_history is provided.
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    prefix = f"{title_prefix + ' - ' if title_prefix else ''}"
    
    # Plot Euler angles
    axs[0].plot(times, euler_angles[:, 0], label="Roll")
    axs[0].plot(times, euler_angles[:, 1], label="Pitch")
    axs[0].plot(times, euler_angles[:, 2], label="Yaw")
    axs[0].set_xlabel("Time, s")
    axs[0].set_ylabel("Angle, deg")
    # axs[0].set_title(prefix + "Satellite Euler Angles Over Time")
    axs[0].legend()
    
    if desired_euler_points is not None:
        text_lines = ["Desired Set Points:"]
        for idx, point in enumerate(desired_euler_points):
            text_lines.append(f"P{idx}: {np.round(point, 2)}")
        axs[0].text(0.05, 0.65, "\n".join(text_lines), transform=axs[0].transAxes, 
                    bbox=dict(facecolor='white', alpha=0.7))
    
    # Plot Angular Velocity
    ax_av = axs[1]
    omega_history = np.array(omega_history)
    ax_av.plot(times, omega_history[:, 0], label="$\omega_x$")
    ax_av.plot(times, omega_history[:, 1], label="$\omega_y$")
    ax_av.plot(times, omega_history[:, 2], label="$\omega_z$")
    ax_av.set_xlabel("Time, s")
    ax_av.set_ylabel("Angular velocity, rad/s")
    # ax_av.set_title(prefix + "Satellite Angular Velocity Over Time")
    ax_av.legend()
    
    # Plot Attitude Error
    ax_err = axs[2]
    if error_history is not None:
        ax_err.plot(times, error_history, label="Attitude Error (deg)")
    ax_err.set_xlabel("Time, s")
    ax_err.set_ylabel("Error, deg")
    # ax_err.set_title(prefix + "Attitude Error Over Time")
    ax_err.legend()
    
    plt.tight_layout()
    plt.show()
    
def annotate_wheel_speeds(ax, wheel_speeds, error=None):
    """Annotates the reaction wheel speeds and error value as text on the given axis."""
    text = f"Wheel speeds: {np.round(wheel_speeds, 3)}"
    if error is not None:
        error_norm = np.linalg.norm(error)
        text += f" | Error: {error_norm:.2f} rad"
    ax.text2D(0.05, 0.95, text, transform=ax.transAxes)

def plot_wheel_speeds(times, wheel_speeds):
    """
    Plots the reaction wheel speeds over time.
    """
    plt.figure()
    wheel_speeds = np.array(wheel_speeds)
    plt.plot(times, wheel_speeds[:, 0], label="Wheel x")
    plt.plot(times, wheel_speeds[:, 1], label="Wheel y")
    plt.plot(times, wheel_speeds[:, 2], label="Wheel z")
    plt.xlabel("Time, s")
    plt.ylabel("Wheel speed, rad/s")
    # plt.title("Reaction Wheel Speeds Over Time")
    plt.legend()
    plt.show()

def plot_torque_over_time(times, torque_history):
    """
    Plots the control torque magnitude over time.
    
    Parameters:
        times (array-like): Time stamps for each step.
        torque_history (array-like): Torque magnitude applied at each time step.
    """
    plt.figure()
    plt.plot(times, torque_history, marker='o')
    plt.xlabel("Time, s")
    plt.ylabel("Torque magnitude, arb. u.")
    # plt.title("Control Torque Over Time")
    plt.show()

def plot_rl_model_performance(training_steps, convergence_times, total_torques):
    """
    Plots RL model performance versus training steps on a log-scale x-axis.
    
    Parameters:
        training_steps (array-like): Number of timesteps each RL model was trained for.
        convergence_times (array-like): Convergence times for each model.
        total_torques (array-like): Total torque usage for each model.
    """
    training_steps = np.array(training_steps)
    convergence_times = np.array(convergence_times)
    total_torques = np.array(total_torques)
    
    fig, ax1 = plt.subplots()
    # Plot convergence time
    line1, = ax1.plot(training_steps, convergence_times, marker='o', color='tab:blue', label='Convergence Time')
    ax1.set_xscale('log')
    ax1.set_xlabel("RL Training Steps (log scale)")
    ax1.set_ylabel("Convergence time, s")
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    # ax1.set_title("RL Model Performance vs Training Steps")
    
    ax2 = ax1.twinx()
    # Plot total torque
    line2, = ax2.plot(training_steps, total_torques, marker='x', color='tab:red', label='Total Torque')
    ax2.set_ylabel("Total torque, arb. u.")
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # Combined legend
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower left')
    
    # Fit quadratic curve to convergence times
    coeffs = np.polyfit(training_steps, convergence_times, deg=2)
    fit_vals = np.polyval(coeffs, training_steps)
    # Plot fit curve (no legend entry)
    ax1.plot(training_steps, fit_vals, linestyle='--')
    # Print fit equation for external use
    eq = f"Fit eqn: convergence_time = {coeffs[0]:.6g}*steps^2 + {coeffs[1]:.6g}*steps + {coeffs[2]:.6g}"
    print(eq)
    
    fig.tight_layout()
    plt.show()

def plot_performance_vs_initial_error(init_errors, convergence_times, total_torques, label):
    fig, ax1 = plt.subplots()
    ax1.scatter(init_errors, convergence_times, marker='o', label=f"{label} time")
    ax1.set_xlabel("Initial attitude error, deg")
    ax1.set_ylabel("Convergence time, s")
    # ax1.set_title("Performance vs Initial Error")
    
    ax2 = ax1.twinx()
    ax2.scatter(init_errors, total_torques, marker='x', label=f"{label} torque")
    ax2.set_ylabel("Total torque, arb. u.")
    
    fig.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


# === Method Comparison Plot ===
def plot_method_comparison(methods, means, lows, highs, ylabel, title):
    """
    Bar plot with error bars showing mean, low, and high values for different methods.
    """
    methods = list(methods)
    x = np.arange(len(methods))
    means = np.array(means)
    lows = np.array(lows)
    highs = np.array(highs)
    err_low = means - lows
    err_high = highs - means

    plt.figure(figsize=(8, 6))
    plt.bar(x, means, yerr=[err_low, err_high], capsize=5, color='skyblue', edgecolor='k')
    plt.xticks(x, methods)
    plt.ylabel(ylabel)
    # plt.title(title)
    plt.tight_layout()
    plt.show()