'''
3.2 Introduction
Comp_Sci_Navier_Stokes_Equations.src.Introduction

We aim to solve the quite involved equations by splitting the terms into sub problems of easier type. These are then
approximated by finite-difference methods on a cartesian grid.
'''

import numpy as np
import matplotlib.pyplot as plt

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
'''
def upwind_velocity_step(v, w, ay, az, dy, dz, dt):
    '''
    One explicit upwind step for velocity (v, w)
    
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

    def y_advection(u, ay, dy):
        # Periodic upwind derivative in y-direction
        backward = (u - np.roll(u, 1, axis=0)) / dy
        forward  = (np.roll(u, -1, axis=0) - u) / dy
        return np.where(ay > 0, backward, forward)
    
    def z_advection_periodic(u, az, dz):    
        # Periodic upwind derivative in z-direction
        backward = (u - np.roll(u, 1, axis=1)) / dz
        forward  = (np.roll(u, -1, axis=1) - u) / dz
        return np.where(az > 0, backward, forward)
    
    
    def z_advection_dirichlet(w, az, dz):
        '''
        Upwind derivative in z-direction for w with
        Dirichlet BC: w = 0 at z = 0, 1
        '''
        dwdz = np.zeros_like(w)
        backward = (w[:, 1:-1] - w[:, :-2]) / dz
        forward  = (w[:, 2:]   - w[:, 1:-1]) / dz
        dwdz[:, 1:-1] = np.where(az[:, 1:-1] > 0, backward, forward)

        # boundaries (values fixed by BC)
        dwdz[:, 0]  = 0.0
        dwdz[:, -1] = 0.0
        '''Setting the derivative also to zero is a numerical safety choice that prevents the time update from
        changing boundary values.
        '''

        return dwdz
    
    # y-advection (periodic)
    dvdy = y_advection(v, ay, dy)
    dwdy = y_advection(w, ay, dy)

    # z-advection
    dvdz = z_advection_periodic(v, az, dz)
    dwdz = z_advection_dirichlet(w, az, dz)

    # time update
    v_new = v - dt * (ay * dvdy + az * dvdz)
    w_new = w - dt * (ay * dwdy + az * dwdz)

    # enforce BCs for w
    w_new[:, 0]  = 0.0
    w_new[:, -1] = 0.0

    return v_new, w_new

'''
Find suitable testcases, for which you can find analytical solutions and compare them with your results. Therefore
visualize the analytical/ approximated solutions and explain what you observe. Numerically test the consistency order 
of you implementation(s).
'''

# Grid and parameters
Ny, Nz = 100, 100
y = np.linspace(0, 1, Ny, endpoint=False)
z = np.linspace(0, 1, Nz, endpoint=False)
dy = y[1] - y[0]
dz = z[1] - z[0]
Y, Z = np.meshgrid(y, z, indexing='ij')

# Constant velocity a
ay_val = 0.5
az_val = 0.3
ay = ay_val * np.ones_like(Y)
az = az_val * np.ones_like(Z)

# Initial conditions
# v: periodic in y and z
v0 = np.sin(2 * np.pi * Y) * np.cos(2 * np.pi * Z)

# w: zero at z = 0 and z = 1
w0 = np.sin(2 * np.pi * Y) * np.sin(np.pi * Z)

w0[:, 0]  = 0.0
w0[:, -1] = 0.0

v = v0.copy()
w = w0.copy()

# Time stepping
dt = 0.5 * min(dy / ay_val, dz / az_val)  # stability condition
t_final = 0.5
n_steps = int(t_final / dt)

for _ in range(n_steps):
    v, w = upwind_velocity_step(v, w, ay, az, dy, dz, dt)

t = n_steps * dt

# Analytical solutions
def analytical_v(Y, Z, t, ay, az):
    return np.sin(2 * np.pi * (Y - ay * t)) * np.cos(2 * np.pi * (Z - az * t))


def analytical_w(Y, Z, t, ay, az):
    return np.sin(2 * np.pi * (Y - ay * t)) * np.sin(np.pi * (Z - az * t))


v_exact = analytical_v(Y, Z, t, ay_val, az_val)
w_exact = analytical_w(Y, Z, t, ay_val, az_val)

# enforce BCs on analytical w
w_exact[:, 0]  = 0.0
w_exact[:, -1] = 0.0

# Visualization
plt.figure(figsize=(12, 8))

# --- v component ---
plt.subplot(2, 2, 1)
plt.imshow(v, origin="lower", extent=[0, 1, 0, 1])
plt.title("Numerical v")

plt.subplot(2, 2, 2)
plt.imshow(v_exact, origin="lower", extent=[0, 1, 0, 1])
plt.title("Analytical v")

# --- w component ---
plt.subplot(2, 2, 3)
plt.imshow(w, origin="lower", extent=[0, 1, 0, 1])
plt.title("Numerical w")

plt.subplot(2, 2, 4)
plt.imshow(w_exact, origin="lower", extent=[0, 1, 0, 1])
plt.title("Analytical w")

plt.tight_layout()
plt.show()

'''
The horizontal velocity component v satisfies periodic boundary conditions in both spatial directions, while the vertical component w
is subject to homogeneous Dirichlet conditions at the top and bottom boundaries. Therefore, both components are visualized separately
in order to assess the influence of the boundary treatment on the numerical solution.
'''

# --- Error ---
error_v = np.max(np.abs(v - v_exact))
error_w = np.max(np.abs(w - w_exact))

print("Max error in v:", error_v) # 0.04397972126737182
print("Max error in w:", error_w) # 0.42577929199786807