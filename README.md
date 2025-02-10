The goal of this project is to compare standard methods for stabilizing an uncontrolled spinning satellite with an approach using reinforcement learning techniques to stabilize the satellite. 
General outline of the current plan:
1. Set up a visualization of a simple simulation of a satellite and its attitude
2. Implement dynamics to recover an uncontrolled spin using standard methods (probably, in 1 axis)
3. Implement reinforcement learning policty to correct an uncontrolled spin (in 1 axis)
4. Evaluate methods recovery metrics, such as speed, precision, oscillations?
5. Repeat methods from above but include 2 or all 3 axis for the uncontrolled spin to recover to
6. Investigate the recover abilities of reinforcement learning policy for satellties with different/ varying moments of inertia (such as a docking satellite or same control for different satellites)
