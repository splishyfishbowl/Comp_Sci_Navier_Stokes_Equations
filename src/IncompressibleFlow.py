'''
3.3 Incompressible Flow
Comp_Sci_Navier_Stokes_Equations.src.IncompressibleFlow
'''

'''
Familiarize yourself with the concept of Chorin’s projection method. Explain the necessity for a projection method.
'''

'''
Write a Poisson solver in two spatial dimensions, which discretizes the solution of (6) - (7) by finite differences 
on a cartesian grid.
(6)     Δp = f
(7)     ∂p / ∂n = 0
For this purpose define the interface and implement unit tests as well tests to check the consistency first.
'''

'''
Do the implementation of the solver.
'''

'''
Implement Chorin's projection method [1] for the Navier Stokes equations i.e. (2) - (3) with β = 0 using the numerical
methods built so far. Implement initial conditions to test your code, check the consistency order and visualize the data.
(2)     ∂_t~u + ~u · ∇~u + ∇p/ρ = ν∆~u + T βe_z
(3)     ∇ · ~u = 0
'''

import numpy as np
import matplotlib.pyplot as plt

# TODO: put this all in a main method for organization and import the functions from separate files for better modularity and readability

# ============================================================
# DEFINE DOMAIN AND NUMERICAL PARAMETERS
# ============================================================

Ny, Nz = 50, 50          # grid points in y and z directions
Ly, Lz = 1.0, 1.0        # domain size in y and z directions

# grid spacing (uniform grid)
dy = Ly / (Ny - 1)
dz = Lz / (Nz - 1)

# time discretization
dt = 0.001               # time step
nt = 100                 # number of time steps

# physical parameters
rho = 1.0                # density
nu = 0.01                # kinematic viscosity

# ============================================================
# INITIAL CONDITIONS
# ============================================================

y = np.linspace(0, Ly, Ny)
z = np.linspace(0, Lz, Nz)
Y, Z = np.meshgrid(y, z, indexing="ij")

v = np.sin(np.pi * Y) * np.cos(np.pi * Z)   # velocity in y-direction
w = -np.cos(np.pi * Y) * np.sin(np.pi * Z)  # velocity in z-direction
p = np.zeros((Ny, Nz))   # pressure field

# ============================================================
# FINITE DIFFERENCE OPERATORS
# ============================================================

def divergence(v, w):
    '''
    ∇ ⋅ u = ∂v/∂y + ∂w/∂z
    '''
    div = np.zeros_like(v)
    div[1:-1, 1:-1] = (
        (v[2:, 1:-1] - v[:-2, 1:-1]) / (2*dy) +
        (w[1:-1, 2:] - w[1:-1, :-2]) / (2*dz)
    )
    return div


def gradient(p):
    '''
    ∇p = (∂p/∂y, ∂p/∂z)
    '''
    dpdy = np.zeros_like(p)
    dpdz = np.zeros_like(p)

    dpdy[1:-1, 1:-1] = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2*dy)
    dpdz[1:-1, 1:-1] = (p[1:-1, 2:] - p[1:-1, :-2]) / (2*dz)

    return dpdy, dpdz


def laplacian(f):
    '''
    Δf = ∂²f/∂y² + ∂²f/∂z²
    '''
    lap = np.zeros_like(f)
    lap[1:-1, 1:-1] = (
        (f[2:, 1:-1] - 2*f[1:-1, 1:-1] + f[:-2, 1:-1]) / dy**2 +
        (f[1:-1, 2:] - 2*f[1:-1, 1:-1] + f[1:-1, :-2]) / dz**2
    )
    return lap

# ============================================================
# PREDICTOR STEP
# ============================================================

def predictor_step(v, w):
    '''
    computes a temporary velocity u* = (v*, w*) by solving the momentum equation without the pressure term,
    producing as velocity which isn't divergence free.
    This solves for the u* in the predictor equation in Chorin's projection method: (u* - u^n)/dt = -(u^n · ∇)u^n + nu(Δu^n)

    Componentwise:
        (v* - v^n)/dt = -((v^n)(∂v^n/∂y) + (w^n)(∂v^n/∂z)) + nu(Δv^n)
        (w* - w^n)/dt = -((v^n)(∂w^n/∂y) + (w^n)(∂w^n/∂z)) + nu(Δw^n)
    '''

    dv_dy = np.zeros_like(v)
    dv_dz = np.zeros_like(v)
    dw_dy = np.zeros_like(w)
    dw_dz = np.zeros_like(w)

    # approximate spatial derivatives using central difference on interior points. Time integration uses explicit forward Euler
    dv_dy[1:-1, 1:-1] = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2*dy) # ∂v/∂y ≈ (v[i+1, j] - v[i-1, j]) / (2*dy)
    dv_dz[1:-1, 1:-1] = (v[1:-1, 2:] - v[1:-1, :-2]) / (2*dz) # ∂v/∂z ≈ (v[i, j+1] - v[i, j-1]) / (2*dz)
    dw_dy[1:-1, 1:-1] = (w[2:, 1:-1] - w[:-2, 1:-1]) / (2*dy) # ∂w/∂y ≈ (w[i+1, j] - w[i-1, j]) / (2*dy)
    dw_dz[1:-1, 1:-1] = (w[1:-1, 2:] - w[1:-1, :-2]) / (2*dz) # ∂w/∂z ≈ (w[i, j+1] - w[i, j-1]) / (2*dz)

    # convection terms
    conv_v = v * dv_dy + w * dv_dz # (u ⋅ ∇)v = v(∂v/∂y) + w(∂v/∂z)
    conv_w = v * dw_dy + w * dw_dz # (u ⋅ ∇)w = v(∂w/∂y) + w(∂w/∂z)

    # diffusion terms 
    diff_v = nu * laplacian(v) # nu Δv
    diff_w = nu * laplacian(w) # nu Δw

    # explicit forward Euler time update (full predictor step)
    v_star = v + dt * (-conv_v + diff_v) # v* = v^n + dt * (-(u^n ⋅ ∇)v^n + nu Δv^n)
    w_star = w + dt * (-conv_w + diff_w) # w* = w^n + dt * (-(u^n ⋅ ∇)w^n + nu Δw^n)

    return v_star, w_star

# ============================================================
# PRESSURE POISSON SOLVER (JACOBI METHOD)
# ============================================================

def pressure_poisson(p, rhs, max_iter=500, tol=1e-5):
    '''
    computes the pressure field p by solving the Poisson equation that arises
    from enforcing incompressibility in Chorin's projection method.
    parameters:
        p: initial guess for pressure field (2D array)
        rhs: right-hand side of the Poisson equation (2D array); rhs = (rho/dt) * ∇ ⋅ u*
    returns:
        p: solution of the Poisson equation Δp = rhs with Neumann BCs

    '''
    p_new = np.zeros_like(p)

    for _ in range(max_iter):
        # discrete Lapacian
        p_new[1:-1, 1:-1] = (
            ((p[2:, 1:-1] + p[:-2, 1:-1]) * dz**2 +
             (p[1:-1, 2:] + p[1:-1, :-2]) * dy**2 -
             rhs[1:-1, 1:-1] * dy**2 * dz**2)
            / (2 * (dy**2 + dz**2))
        ) # p[i, j] = ((p[i+1, j] + p[i-1, j]) * dz^2 + (p[i, j+1] + p[i, j-1]) * dy^2 - rhs[i, j] * dy^2 * dz^2) / (2 * (dy^2 + dz^2))

        # Neumann BC ∂p/∂n = 0 → pressure doesn't change across the boundary, so we can just copy the adjacent interior value to the ghost cells
        p_new[0, :] = p_new[1, :]   # y = 0 boundary
        p_new[-1, :] = p_new[-2, :] # y = Ly boundary
        p_new[:, 0] = p_new[:, 1]   # z = 0 boundary
        p_new[:, -1] = p_new[:, -2] # z = Lz boundary

        # convergence check -- iterate until the maximum change in pressure is below the tolerance level
        if np.max(np.abs(p_new - p)) < tol:
            break

        p[:] = p_new[:]

    return p

# ============================================================
# PROJECTION STEP (VELOCITY CORRECTION)
# ============================================================

def projection_step(v_star, w_star, p):
    '''
    computes the divergence-free velocity field at the new time step by correcting the intermediate velocity u* with the pressure gradient.
    This implements the projection step in Chorin's method: u^(n+1) = u* - (dt/rho) ∇p
    Componentwise:
        v^(n+1) = v* - (dt/rho) ∂p/∂y
        w^(n+1) = w* - (dt/rho) ∂p/∂z
    '''
    dpdy, dpdz = gradient(p)
    v_new = v_star - (dt / rho) * dpdy
    w_new = w_star - (dt / rho) * dpdz
    return v_new, w_new

# ============================================================
# BOUNDARY CONDITIONS
# ============================================================

def apply_velocity_bc(v, w):
    # periodic in y
    v[0, :] = v[-2, :] # v[0, j] = v[Ny-2, j]
    v[-1, :] = v[1, :] # v[Ny-1, j] = v[1, j]
    w[0, :] = w[-2, :] # w[0, j] = w[Ny-2, j]
    w[-1, :] = w[1, :] # w[Ny-1, j] = w[1, j]

    # walls in z
    w[:, 0] = 0.0   # w[i, 0] = 0.0
    w[:, -1] = 0.0  # w[i, Nz-1] = 0.0

    return v, w

# ============================================================
# TIME LOOP
# ============================================================

for n in range(nt):

    # predictor step: compute intermediate velocity u* by solving the momentum equation without the pressure term
    v_star, w_star = predictor_step(v, w)

    # compute the right-hand side of the Poisson equation for pressure correction
    rhs = (rho / dt) * divergence(v_star, w_star)

    # solve the Poisson equation for pressure
    p = pressure_poisson(p, rhs)

    # projection step: correct the intermediate velocity u* with the pressure gradient to get the divergence-free velocity at the new time step
    v, w = projection_step(v_star, w_star, p)

    # apply boundary conditions to the velocity field
    v, w = apply_velocity_bc(v, w)

    # monitor the maximum divergence to check if the velocity field is approximately divergence-free after the projection step
    if n % 10 == 0:
        div_norm = np.max(np.abs(divergence(v, w)))
        print(f"Step {n:4d} | max divergence = {div_norm:.3e}")

# ============================================================
# PLOTTING
# ============================================================

speed = np.sqrt(v**2 + w**2)

plt.figure(figsize=(6,5))
plt.contourf(y, z, speed.T, levels=20)
plt.colorbar(label="|u|")
plt.quiver(y, z, v.T, w.T, color="white", scale=30)
plt.xlabel("y")
plt.ylabel("z")
plt.title("Velocity Magnitude + Vector Field")
plt.show()