'''
3.2 Introduction
Comp_Sci_Navier_Stokes_Equations.src.Introduction

We aim to solve the quite involved equations by splitting the terms into sub problems of easier type. These are then
approximated by finite-difference methods on a cartesian grid.
'''

import numpy as np
import matplotlib.pyplot as plt

'''
Implement the upwind finite difference method for the linear advection equation with variable advection field in two 
spatial dimensions using operator splitting for each of the spatial derivatives. 
(u & a below are vectors)
(δ_t * u) + a * ∇u = 0
u(0, y, z) = u(y, z)
As boundary conditions consider the ones given in the previous section. Check which restrictions occur due to the 
explicit treatment of this term. 
'''
def upwind_step(field, a, delta, dt, axis, periodic=True):
    '''
    Upwind Finite Difference Update (Lie Splitting)
    
    Inputs:
        field: 2D array to update (e.g., v, w, or T)
        a: advection velocity in this direction
        delta: grid spacing in this direction (dy or dz)
        dt: time step
        axis: 0 → y-direction, 1 → z-direction
        periodic: True if field is periodic along this axis
    Output: 
        u_new (2D array): updated field after one explicit upwind time step along the given axis

    numpy functions:
        roll: shifts the array along a given axis
        where: like a ternary
    '''
    
    if a > 0:
        # Backward finite difference 
        if periodic:
            deriv = (field - np.roll(field, 1, axis=axis)) / delta
        else:
            deriv = np.zeros_like(field)
            if axis == 0: deriv[1:, :] = (field[1:, :] - field[:-1, :]) / delta
            else:         deriv[:, 1:] = (field[:, 1:] - field[:, :-1]) / delta
    else:
        # Forward finite difference 
        if periodic:
            deriv = (np.roll(field, -1, axis=axis) - field) / delta
        else:
            deriv = np.zeros_like(field)
            if axis == 0: deriv[:-1, :] = (field[1:, :] - field[:-1, :]) / delta
            else:         deriv[:, :-1] = (field[:, 1:] - field[:, :-1]) / delta
                
    return field - dt * a * deriv

'''
OPTIONAL: Implement a semi-lagrangian scheme for the two dimensional linear advection equation with variable advection
field. Such a scheme tracks the characteristic solutions back in time and subsequently interpolates the field at the old
time point to obtain the values transported along the charateristics.
'''

'''
Find suitable testcases, for which you can find analytical solutions and compare them with your results. Therefore
visualize the analytical/ approximated solutions and explain what you observe. Numerically test the consistency order 
of you implementation(s).
'''

# Grid and parameters
Ny, Nz = 100, 100
y = np.linspace(0, 1, Ny)
z = np.linspace(0, 1, Nz)
dy = 1.0 / Ny
dz = 1.0 / (Nz - 1) 
Y, Z = np.meshgrid(y, z, indexing='ij')

# Constant advection field
ay_val = 0.5
az_val = 0.3
ay = ay_val * np.ones_like(Y)
az = az_val * np.ones_like(Z)

dt = 0.5 * min(dy / abs(ay_val), dz / abs(az_val)) 
t_final = 0.5
n_steps = int(t_final / dt)

# Initial Conditions
v = np.sin(2 * np.pi * Y) * np.cos(2 * np.pi * Z)
w = np.sin(2 * np.pi * Y) * np.sin(np.pi * Z)
w[:, 0] = w[:, -1] = 0.0

# Time Stepping Loop
for _ in range(n_steps):
    # Operator splitting: Update y then update z sequentially
    v = upwind_step(v, ay_val, dy, dt, axis=0, periodic=True)
    w = upwind_step(w, ay_val, dy, dt, axis=0, periodic=True)
    
    v = upwind_step(v, az_val, dz, dt, axis=1, periodic=False)
    w = upwind_step(w, az_val, dz, dt, axis=1, periodic=False)
    
    # Enforce Dirichlet BCs for w 
    w[:, 0] = 0.0
    w[:, -1] = 0.0

# Analytical solutions
t_reached = n_steps * dt
v_exact = np.sin(2 * np.pi * (Y - ay_val * t_reached)) * np.cos(2 * np.pi * (Z - az_val * t_reached))
w_exact = np.sin(2 * np.pi * (Y - ay_val * t_reached)) * np.sin(np.pi * (Z - az_val * t_reached))
# enforce BCs on analytical w
w_exact[:, 0] = w_exact[:, -1] = 0.0 

# Visualization
plt.figure(figsize=(12, 8))

# --- v component ---
plt.subplot(2, 2, 1)
plt.imshow(v.T, origin='lower', extent=[0.0, 1.0, 0.0, 1.0]) 
plt.xlabel('y')
plt.ylabel('z')
plt.colorbar()
plt.title("Numerical Velocity v")

plt.subplot(2, 2, 2)
plt.imshow(v_exact.T, origin="lower", extent=[0.0, 1.0, 0.0, 1.0])
plt.xlabel('y')
plt.ylabel('z')
plt.colorbar()
plt.title("Analytical Velocity v")

# --- w component ---
plt.subplot(2, 2, 3)
plt.imshow(w.T, origin="lower", extent=[0.0, 1.0, 0.0, 1.0])
plt.xlabel('y')
plt.ylabel('z')
plt.colorbar()
plt.title("Numerical Velocity w")

plt.subplot(2, 2, 4)
plt.imshow(w_exact.T, origin="lower", extent=[0.0, 1.0, 0.0, 1.0])
plt.xlabel('y')
plt.ylabel('z')
plt.colorbar()
plt.title("Analytical Velocity w")

plt.tight_layout()
plt.show()

# --- Error ---
error_v = np.max(np.abs(v - v_exact))
error_w = np.max(np.abs(w - w_exact))

print("Max error in v:", error_v) # 0.38752306307575746
print("Max error in w:", error_w) # 0.42549213970769667