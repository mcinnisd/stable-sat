import numpy as np
import scipy.linalg as la
from scipy.spatial.transform import Rotation as R

def update_state(q, omega, dt, torque, I_sat):
    """
    Updates the satellite state given the current state, time step, applied torque, and inertia matrix.
    
    Parameters:
      q (np.array): Current quaternion [x, y, z, w].
      omega (np.array): Current angular velocity (rad/s).
      dt (float): Time step.
      torque (np.array): Applied external torque.
      I_sat (np.array): Satellite inertia matrix.
    
    Returns:
      q_new (np.array): Updated quaternion.
      omega_new (np.array): Updated angular velocity.
    """
    I_inv = np.linalg.inv(I_sat)
    # Calculate angular acceleration: ω̇ = I⁻¹ (τ - ω × (I * ω))
    omega_dot = I_inv @ (torque - np.cross(omega, I_sat @ omega))
    
    # Update angular velocity using Euler integration
    omega_new = omega + omega_dot * dt
    
    # Update quaternion: use the new angular velocity to compute the rotation vector
    delta_rot = R.from_rotvec(omega_new * dt)
    q_new = (R.from_quat(q) * delta_rot).as_quat()
    
    return q_new, omega_new


def compute_lqr_gain(I_i, Q, R_mat):
    """Compute LQR gain for a double integrator model for a given axis."""
    A = np.array([[0, 1],
                  [0, 0]])
    B = np.array([[0],
                  [1.0 / I_i]])
    P = la.solve_continuous_are(A, B, Q, R_mat)
    K = np.linalg.inv(R_mat) @ (B.T @ P)
    return K