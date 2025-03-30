#!/usr/bin/env python3
"""
Single-Axis LQR Test
This script simulates the closed-loop dynamics of a single rotational axis
using the LQR gain computed for a double integrator system.
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

def compute_lqr_gain(I_i, Q, R_mat):
    """
    Compute the LQR gain for a single axis double integrator model.
    """
    A = np.array([[0, 1],
                  [0, 0]])
    B = np.array([[0],
                  [1.0 / I_i]])
    P = la.solve_continuous_are(A, B, Q, R_mat)
    K = np.linalg.inv(R_mat) @ (B.T @ P)
    return K

def simulate_single_axis_lqr(I_i=1.0, Q=np.diag([10, 1]), R_mat=np.array([[10]]),
                             initial_angle=0.5, initial_omega=0.0,
                             sim_time=5.0, dt=0.01):
    """
    Simulates the single axis dynamics using a double integrator model with LQR control.
    
    State:
        x = [angle, angular_velocity]
    Dynamics:
        d(angle)/dt = angular_velocity
        d(angular_velocity)/dt = torque (as computed by the LQR controller)
    
    The control law is:
        torque = -K * x
    """
    K = compute_lqr_gain(I_i, Q, R_mat)
    
    t = 0.0
    times = []
    angle_history = []
    omega_history = []
    
    angle = initial_angle  # initial error in angle (radians)
    omega = initial_omega  # initial angular velocity
    
    while t <= sim_time:
        times.append(t)
        angle_history.append(angle)
        omega_history.append(omega)
        
        # State vector
        x = np.array([angle, omega])
        # LQR control law
        torque = -K.dot(x)[0]
        
        # Dynamics: x_dot = [omega, torque]
        angle_dot = omega
        omega_dot = torque
        
        # Euler integration update
        angle = angle + angle_dot * dt
        omega = omega + omega_dot * dt
        
        t += dt
        
    return np.array(times), np.array(angle_history), np.array(omega_history)

if __name__ == '__main__':
    times, angle, omega = simulate_single_axis_lqr()
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(times, angle, label="Angle Error (rad)")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle Error")
    plt.title("Angle Response")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(times, omega, label="Angular Velocity (rad/s)", color='orange')
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Velocity")
    plt.title("Angular Velocity Response")
    plt.legend()

    plt.tight_layout()
    plt.show()