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

# ============================================================
# GLOBAL PARAMETERS
# ============================================================

Ny, Nz = 50, 50          # grid points in y and z directions
Ly, Lz = 1.0, 1.0        # domain size in y and z directions

# grid spacing (uniform grid)
dy = Ly / Ny            # periodic in y
dz = Lz / (Nz - 1)      # walls included in z

# time discretization
dt = 0.001               # time step
nt = 100                 # number of time steps

# physical parameters
rho = 1.0                # density
nu = 0.01                # kinematic viscosity

# ============================================================
# GRID / INITIAL CONDITIONS
# ============================================================

def create_grid():
    y_endpoint = False  # periodic in y
    z_endpoint = True   # walls in z
    y = np.linspace(0, Ly, Ny, y_endpoint)  # periodic
    z = np.linspace(0, Lz, Nz, z_endpoint)  # walls included
    Y, Z = np.meshgrid(y, z, indexing="ij")
    return y, z, Y, Z

def initial_conditions(Y, Z):
    ky = 2*np.pi / Ly
    kz = np.pi / Lz

    v = np.sin(ky*Y) * np.cos(kz*Z)             # velocity in y-direction
    w = -(ky/kz) * np.cos(ky*Y) * np.sin(kz*Z)  # velocity in z-direction
    p = np.zeros((Ny, Nz))                      # pressure field
    return v, w, p

# ============================================================
# BOUNDARY CONDITIONS
# ============================================================

def apply_velocity_bc(v, w):
    v = v.copy()
    w = w.copy()

    # walls in z: w = 0
    w[:, 0] = 0.0   # w[i, 0] = 0.0
    w[:, -1] = 0.0  # w[i, Nz-1] = 0.0

    # free-slip for v at walls (∂v/∂n = 0)
    v[:, 0] = v[:, 1]   # v[i, 0] = v[i, 1]
    v[:, -1] = v[:, -2] # v[i, Nz-1] = v[i, Nz-2]

    return v, w

# ============================================================
# FINITE DIFFERENCE OPERATORS
# ============================================================

def divergence(v, w):
    '''
    ∇ ⋅ u = ∂v/∂y + ∂w/∂z
    periodic in y via roll, walls in z via central interior
    '''
    v, w = apply_velocity_bc(v.copy(), w.copy())

    dv_dy = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2*dy)

    dw_dz = np.zeros_like(w)
    dw_dz[:, 1:-1] = (w[:, 2:] - w[:, :-2]) / (2*dz)
    # one-sided at walls 
    dw_dz[:, 0]  = (w[:, 1] - w[:, 0]) / dz
    dw_dz[:, -1] = (w[:, -1] - w[:, -2]) / dz

    return dv_dy + dw_dz

def gradient(p):
    '''
    ∇p = (∂p/∂y, ∂p/∂z)
    periodic in y via roll
    Neumann in z handled by copying adjacent values before derivative
    '''
    p = p.copy()
    p[:, 0]  = p[:, 1]
    p[:, -1] = p[:, -2]

    dpdy = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / (2*dy)

    dpdz = np.zeros_like(p)
    dpdz[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2*dz)
    dpdz[:, 0]  = (p[:, 1] - p[:, 0]) / dz
    dpdz[:, -1] = (p[:, -1] - p[:, -2]) / dz

    return dpdy, dpdz

def laplacian(f):
    '''
    Δf = ∂²f/∂y² + ∂²f/∂z²
    periodic in y via roll
    in z: central interior + reflection at boundaries (Neumann-type)
    '''
    f = f.copy()
    # reflect in z for a stable 2nd derivative near walls
    f[:, 0]  = f[:, 1]
    f[:, -1] = f[:, -2]

    d2y = (np.roll(f, -1, axis=0) - 2*f + np.roll(f, 1, axis=0)) / (dy**2)

    d2z = np.zeros_like(f)
    d2z[:, 1:-1] = (f[:, 2:] - 2*f[:, 1:-1] + f[:, :-2]) / (dz**2)
    d2z[:, 0]  = (f[:, 1] - 2*f[:, 0] + f[:, 1]) / (dz**2)
    d2z[:, -1] = (f[:, -2] - 2*f[:, -1] + f[:, -2]) / (dz**2)

    return d2y + d2z

# ============================================================
# ADVECTION
# ============================================================

def upwind_step(field, a, delta, dt, axis, periodic=True):
    '''
    First-order upwind update for u_t + a * u_x = 0 along a single axis.
    Supports scalar a or array a(y,z).

    periodic=True: uses np.roll for neighbors
    periodic=False: uses one-sided interior differences; boundaries get zero-derivative by default
                    (you can overwrite boundaries via BCs after the step).
    '''
    field = field.copy()
    a = np.asarray(a)

    if periodic:
        f_plus  = np.roll(field, -1, axis=axis)
        f_minus = np.roll(field,  1, axis=axis)

        # backward (a>0): (f - f_minus)/delta
        db = (field - f_minus) / delta
        # forward  (a<0): (f_plus - f)/delta
        df = (f_plus - field) / delta

        deriv = np.where(a > 0, db, np.where(a < 0, df, 0.0))
        return field - dt * a * deriv

    # non-periodic: compute interior only
    deriv = np.zeros_like(field)
    db = np.zeros_like(field)
    df = np.zeros_like(field)

    if axis == 0:  # y-direction
        # backward difference for i=1..end
        db = np.zeros_like(field); db[1:, :]  = (field[1:, :] - field[:-1, :]) / delta
        # forward difference for i=0..end-1
        df = np.zeros_like(field); df[:-1, :] = (field[1:, :] - field[:-1, :]) / delta
    else:          # z-direction
        db = np.zeros_like(field); db[:, 1:]  = (field[:, 1:] - field[:, :-1]) / delta
        df = np.zeros_like(field); df[:, :-1] = (field[:, 1:] - field[:, :-1]) / delta

    deriv = np.where(a > 0, db, np.where(a < 0, df, 0.0))
    return field - dt * a * deriv

def advect_split_upwind(q, a_y, a_z):
    """
    Operator-split advection:
    first y-direction, then z-direction
    """
    q1 = upwind_step(q, a_y, dy, dt, axis=0, periodic=True)
    q2 = upwind_step(q1, a_z, dz, dt, axis=1, periodic=False)
    return q2

# ============================================================
# PREDICTOR STEP
# ============================================================

def predictor_step(v, w):
    '''
    computes a temporary velocity u* = (v*, w*) by solving the momentum equation without the pressure term,
    producing as velocity which isn't divergence free.
    This solves for the u* in the predictor equation in Chorin's projection method: 
        (u* - u^n)/dt = -(u^n · ∇)u^n + nu(Δu^n)
        solving for u* gives: u* = u^n + dt[-(u^n · ∇)u^n + nu Δu^n]

    Componentwise:
        (v* - v^n)/dt = -((v^n)(∂v^n/∂y) + (w^n)(∂v^n/∂z)) + nu(Δv^n)
        (w* - w^n)/dt = -((v^n)(∂w^n/∂y) + (w^n)(∂w^n/∂z)) + nu(Δw^n)
    '''

    # enforce BCs in z 
    v, w = apply_velocity_bc(v, w)

    # advect each component by the current velocity field
    v_adv = advect_split_upwind(v, v, w)
    w_adv = advect_split_upwind(w, v, w)

    # diffusion (still explicit)
    v_star = v_adv + dt * nu * laplacian(v)
    w_star = w_adv + dt * nu * laplacian(w)

    v_star, w_star = apply_velocity_bc(v_star, w_star)

    return v_star, w_star

# ============================================================
# PRESSURE POISSON SOLVER (JACOBI METHOD)
# ============================================================

def pressure_poisson(p, rhs, max_iter=20000, tol_rel=1e-5):
    '''
    computes the pressure field p by solving the Poisson equation that arises
    from enforcing incompressibility in Chorin's projection method. Solve the Δp = rhs with:
        - periodic in y
        - Neumann BCs in z (∂p/∂n = 0)
        - mean(p) = 0 to fix the gauge freedom 
    parameters:
        p: initial guess for pressure field (2D array)
        rhs: right-hand side of the Poisson equation (2D array); rhs = (rho/dt) * ∇ ⋅ u*
        max_iter: maximum number of Jacobi iterations
        tol_update: tolerance for maximum change in p between iterations for convergence (pressure change)
        tol_res: tolerance for maximum residual of the Poisson equation for convergence (Poisson residual)
    returns:
        p: solution of the Poisson equation Δp = rhs with Neumann BCs

    '''

    p = p.copy()
    rhs = rhs - rhs.mean()   # compatibility for Neumann Poisson
    rhs_scale = np.max(np.abs(rhs)) + 1e-14

    for _ in range(max_iter): # p[i, j] = ((p[i+1, j] + p[i-1, j]) * dz^2 + (p[i, j+1] + p[i, j-1]) * dy^2 - rhs[i, j] * dy^2 * dz^2) / (2 * (dy^2 + dz^2))
        
        p_old = p.copy()

        # enforce Neumann in z before update
        p_old[:, 0]  = p_old[:, 1]
        p_old[:, -1] = p_old[:, -2]

        p_y_plus  = np.roll(p_old, -1, axis=0)
        p_y_minus = np.roll(p_old,  1, axis=0)

        p_z_plus  = np.empty_like(p_old)
        p_z_minus = np.empty_like(p_old)
        p_z_plus[:, :-1] = p_old[:, 1:]
        p_z_plus[:, -1]  = p_old[:, -2]
        p_z_minus[:, 1:] = p_old[:, :-1]
        p_z_minus[:, 0]  = p_old[:, 1]

        p = ((p_y_plus + p_y_minus) * dz**2 +
             (p_z_plus + p_z_minus) * dy**2 -
             rhs * dy**2 * dz**2) / (2*(dy**2 + dz**2))

        # Neumann + gauge
        p[:, 0]  = p[:, 1]
        p[:, -1] = p[:, -2]
        p -= p.mean()

        # relative residual
        rel_res = np.max(np.abs(laplacian(p) - rhs)) / rhs_scale
        if rel_res < tol_rel:
            print(f"[Poisson] converged: iters={_}, rel_res={rel_res:.2e}")
            break

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
# SIMULATION
# ============================================================

def incompressible_flow_simulation(nt, v, w, p):
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

    return v, w, p

# ============================================================
# PLOTTING
# ============================================================

def plot_velocity_field(y, z, v, w):
    speed = np.sqrt(v**2 + w**2)
    plt.figure(figsize=(6,5))
    plt.contourf(y, z, speed.T, levels=20)
    plt.colorbar(label="|u|")
    plt.quiver(y, z, v.T, w.T, color="white", scale=30)
    plt.xlabel("y")
    plt.ylabel("z")
    plt.title("Velocity Magnitude + Vector Field")
    plt.show()

# ============================================================
# MAIN
# ============================================================

def main():
    y, z, Y, Z = create_grid()
    v, w, p = initial_conditions(Y, Z)

    print("Initial max divergence:", np.max(np.abs(divergence(v, w))))

    v, w, p = incompressible_flow_simulation(nt, v, w, p)

    print("Final max divergence:", np.max(np.abs(divergence(v, w))))

    plot_velocity_field(y, z, v, w)


if __name__ == "__main__":
    main()