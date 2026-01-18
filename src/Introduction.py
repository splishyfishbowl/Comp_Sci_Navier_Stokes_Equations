'''
3.2 Introduction
Comp_Sci_Navier_Stokes_Equations.src.Introduction

We aim to solve the quite involved equations by splitting the terms into sub problems of easier type. These are then
approximated by finite-difference methods on a cartesian grid.
'''

'''
Explain the phrase 'operator splitting'. Why / how does it work?
// TODO: Refer to Ceren's math -- do we need to do anything specifically in the code?
'''

'''
Implement the upwind finite difference method for the linear advection equation with variable advection field in two 
spatial dimensions using operator splitting for each of the spatial derivatives. 
(u & a below are vectors)
(δ_t * u) + a * ∇u = 0
u(0, y, z) = u(y, z)
As boundary conditions consider the ones given in the previous section. Check which restrictions occur due to the 
explicit treatment of this term. 
// TODO: boundary conditions????
'''
import numpy as np
def upwind_finite_difference(u, ay, az, dy, dz, dt):
    '''
    Solves the 2-D linear advection equation with variable velocity using upwind finite differences
    
    Inputs:
        u  (2D array): Scalar field at time n
        ay (2D array): y–component of advection velocity
        az (2D array): z–component of advection velocity
        dy, dz (float): Grid spacing
        dt (float): Time step, must satisfy the CFL stability condition: dt ≤ min(dy / max|ay|, dz / max|az|)
    Output: 
        u_new (2D array): field at time n+1

    numpy functions:
        roll: shifts the array along a given axis
        where: like a ternary
    '''

    '''
    At every grid point, compute both the forward and backward finite differences in the y-direction, then select 
    the one consistent with the local flow direction.
    '''
    def y_advection(u, ay, dy):
        backward = (u - np.roll(u, 1, axis=0)) / dy
        forward  = (np.roll(u, -1, axis=0) - u) / dy
        return np.where(ay > 0, backward, forward)
    
    def z_advection(u, az, dz):
        backward = (u - np.roll(u, 1, axis=1)) / dz
        forward  = (np.roll(u, -1, axis=1) - u) / dz
        return np.where(az > 0, backward, forward)
    
    dudy = y_advection(u, ay, dy)
    dudz = z_advection(u, az, dz)
    return u - dt * (ay * dudy + az * dudz)

'''
Find suitable testcases, for which you can find analytical solutions and compare them with your results. Therefore
visualize the analytical/ approximated solutions and explain what you observe. Numerically test the consistency order 
of you implementation(s).
'''
import matplotlib.pyplot as plt

# --- Grid and initial condition ---
Ny, Nz = 100, 100
y = np.linspace(0, 1, Ny, endpoint=False)
z = np.linspace(0, 1, Nz, endpoint=False)
dy = y[1] - y[0]
dz = z[1] - z[0]
Y, Z = np.meshgrid(y, z, indexing='ij')

# Constant velocity
ay_val = 0.5
az_val = 0.3
ay = ay_val * np.ones_like(Y)
az = az_val * np.ones_like(Z)

# Initial condition
u0 = np.sin(2 * np.pi * Y) * np.sin(2 * np.pi * Z)

# Time stepping
dt = 0.5 * min(dy / ay_val, dz / az_val)  # CFL condition
t_final = 0.5
n_steps = int(t_final / dt)
u = u0.copy()

for n in range(n_steps):
    u = upwind_finite_difference(u, ay, az, dy, dz, dt)

# --- Analytical solution ---
def analytical_solution(Y, Z, t, ay_val, az_val):
    return np.sin(2 * np.pi * (Y - ay_val*t)) * np.sin(2 * np.pi * (Z - az_val*t))

u_exact = analytical_solution(Y, Z, dt*n_steps, ay_val, az_val)

# --- Visualization ---
plt.figure(figsize=(12,5))
plt.subplot(1,3,1)
plt.imshow(u0, origin='lower', extent=[0,1,0,1])
plt.title('Initial condition')

plt.subplot(1,3,2)
plt.imshow(u, origin='lower', extent=[0,1,0,1])
plt.title('Numerical solution')

plt.subplot(1,3,3)
plt.imshow(u_exact, origin='lower', extent=[0,1,0,1])
plt.title('Analytical solution')

plt.show()

# --- Error ---
error = np.max(np.abs(u - u_exact))
print("Max error:", error)