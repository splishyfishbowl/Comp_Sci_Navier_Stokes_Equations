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
dy = Ly / Ny            # periodic in y --> no endpoint
dz = Lz / (Nz - 1)      # walls included in z --> endpoint included

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
    y = np.linspace(0, Ly, Ny, endpoint=False)  # periodic: don't repeat y=Ly
    z = np.linspace(0, Lz, Nz, endpoint=True)   # walls at z=0 and z=Lz
    Y, Z = np.meshgrid(y, z, indexing="ij")
    return y, z, Y, Z

def initial_conditions(Y, Z):
    '''
    Divergence-free initial condition (exact for the continuous operators):
        v =  sin(ky·y) cos(kz·z)
        w = -(ky/kz) cos(ky·y) sin(kz·z)
    ∇·u = ky cos(ky·y)cos(kz·z) - (ky/kz)·kz cos(ky·y)cos(kz·z) = 0  ✓
    w satisfies w=0 at z=0 and z=Lz because sin(0)=sin(π)=0.
    '''
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
    v, w = v.copy(), w.copy()
    w[:, 0]  = 0.0      # no-penetration at z = 0
    w[:, -1] = 0.0      # no-penetration at z = Lz
    v[:, 0]  = v[:, 1]  # free-slip: dv/dz = 0 at z = 0
    v[:, -1] = v[:, -2] # free-slip: dv/dz = 0 at z = Lz
    return v, w

# ============================================================
# FINITE DIFFERENCE OPERATORS
# ============================================================

def divergence(v, w):
    '''
    ∇ ⋅ u = ∂v/∂y + ∂w/∂z
    periodic in y via roll, walls in z via central interior
    '''
    v, w = v.copy(), w.copy()
    w[:, 0]  = 0.0   # no-penetration only; never touch v
    w[:, -1] = 0.0

    dv_dy = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2*dy)

    dw_dz = np.zeros_like(w)
    dw_dz[:, 1:-1] = (w[:, 2:] - w[:, :-2]) / (2*dz)
    # one-sided at walls (w=0 there, so these are 0 anyway, but kept for generality)
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

    # Neumann ghost cells
    p[:, 0]  = p[:, 1]
    p[:, -1] = p[:, -2]

    dpdy = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / (2*dy)

    dpdz = np.zeros_like(p)
    dpdz[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2*dz)
    dpdz[:, 0]  = (p[:, 1] - p[:, 0]) / dz      # = 0 by Neumann ghost
    dpdz[:, -1] = (p[:, -1] - p[:, -2]) / dz    # = 0 by Neumann ghost

    return dpdy, dpdz

def laplacian(f):
    '''
    Δf = ∂²f/∂y² + ∂²f/∂z²
    periodic in y via roll
    in z: central interior + reflection at boundaries (Neumann-type)
    '''
    f = f.copy()

    d2y = (np.roll(f, -1, axis=0) - 2*f + np.roll(f, 1, axis=0)) / (dy**2)
    d2z = np.zeros_like(f)
    d2z[:, 1:-1] = (f[:, 2:] - 2*f[:, 1:-1] + f[:, :-2]) / (dz**2)
    # Neumann reflection: f at ghost node = f at first interior node
    d2z[:, 0]  = (f[:, 1] - 2*f[:, 0] + f[:, 1]) / (dz**2)
    d2z[:, -1] = (f[:, -2] - 2*f[:, -1] + f[:, -2]) / (dz**2)

    return d2y + d2z

# ============================================================
# ADVECTION (first-order upwind, operator split)
# ============================================================

def upwind_step(field, a, delta, dt, axis, periodic=True):
    '''
    First-order upwind update for u_t + a * u_x = 0 along a single axis.
    Supports scalar a or array a(y,z). u_t + a·u_x = 0, one axis at a time.

    periodic=True: uses np.roll for neighbors
    periodic=False: uses one-sided interior differences; boundaries get zero-derivative by default
                    (you can overwrite boundaries via BCs after the step).
    '''
    field = field.copy()
    a = np.asarray(a)

    if periodic:
        f_plus  = np.roll(field, -1, axis=axis)
        f_minus = np.roll(field,  1, axis=axis)
        db = (field - f_minus) / delta  # backward diff (upwind if a>0)
        df = (f_plus - field) / delta   # forward  diff (upwind if a<0)
        return field - dt * a * np.where(a > 0, db, np.where(a < 0, df, 0.))

    # non-periodic: compute interior only
    db = np.zeros_like(field)
    df = np.zeros_like(field)

    if axis == 0:  # y-direction
        db[1:,  :] = (field[1:,  :] - field[:-1, :]) / delta # backward difference for i=1..end
        df[:-1, :] = (field[1:,  :] - field[:-1, :]) / delta # forward difference for i=0..end-1
    else:          # z-direction
        db[:, 1:]  = (field[:, 1:]  - field[:, :-1]) / delta
        df[:, :-1] = (field[:, 1:]  - field[:, :-1]) / delta

    return field - dt * a * np.where(a > 0, db, np.where(a < 0, df, 0.))

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

    # advective increment: -(u^n·∇)u^n · dt  (upwind gives u^n - dt·(u·∇)u)
    v_adv = advect_split_upwind(v, v, w)    # = v^n + dt·(advection of v)
    w_adv = advect_split_upwind(w, v, w)    # = w^n + dt·(advection of w)

    # diffusion (still explicit)
    v_star = v_adv + dt * nu * laplacian(v)
    w_star = w_adv + dt * nu * laplacian(w)

    v_star, w_star = apply_velocity_bc(v_star, w_star)

    return v_star, w_star

# ============================================================
# PRESSURE POISSON SOLVER
# ============================================================

def pressure_poisson(p, rhs, max_iter=10000, tol_rel=1e-5):
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
    rhs = rhs - rhs.mean()   # compatibility condition for Neumann Poisson
    rhs_scale = max(np.max(np.abs(rhs)), 1e-14)

    for iteration in range(max_iter):
        p_old = p.copy()
        p_old = p.copy()

        # y-neighbours: periodic
        p_yp = np.roll(p_old, -1, axis=0)
        p_ym = np.roll(p_old,  1, axis=0)

        # z-neighbours: match laplacian() stencil
        p_zp = np.empty_like(p_old)
        p_zm = np.empty_like(p_old)

        p_zp[:, 1:-1] = p_old[:, 2:]    # interior z+1
        p_zm[:, 1:-1] = p_old[:, :-2]   # interior z-1
        p_zp[:, 0]    = p_old[:, 1]     # wall j=0:   reflection ghost
        p_zm[:, 0]    = p_old[:, 1]     # wall j=0:   reflection ghost
        p_zp[:, -1]   = p_old[:, -2]    # wall j=N-1: reflection ghost
        p_zm[:, -1]   = p_old[:, -2]    # wall j=N-1: reflection ghost

        p = ((p_yp + p_ym) * dz**2 +
             (p_zp + p_zm) * dy**2 -
             rhs * dy**2 * dz**2) / (2*(dy**2 + dz**2))

        p -= p.mean()   # pin gauge freedom

        if iteration % 100 == 0:
            res = np.max(np.abs(laplacian(p) - rhs)) / rhs_scale
            if res < tol_rel:
                print(f"  [Poisson] converged at iter {iteration}, rel_res={res:.2e}")
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
# CFL CHECK
# ============================================================

def check_cfl(v, w):
    '''
    For upwind advection the stability requirement is:
        max|v|·dt/dy ≤ 1   and   max|w|·dt/dz ≤ 1
    A violated CFL will cause the simulation to blow up silently.
    '''
    cfl_y = np.max(np.abs(v)) * dt / dy
    cfl_z = np.max(np.abs(w)) * dt / dz
    if max(cfl_y, cfl_z) > 1.0:
        print(f"  [WARNING] CFL violated: CFL_y={cfl_y:.3f}, CFL_z={cfl_z:.3f}")
    return cfl_y, cfl_z

# ============================================================
# TESTS
# ============================================================

def run_tests():
    print("=" * 55)
    print("UNIT TESTS")
    print("=" * 55)
    y, z, Y, Z = create_grid()
    ky = 2*np.pi / Ly
    kz =   np.pi / Lz

    # ----------------------------------------------------------
    # Test 1: discrete divergence of IC is O(h^2), NOT machine zero
    # ----------------------------------------------------------
    v0, w0, _ = initial_conditions(Y, Z)
    div0 = np.max(np.abs(divergence(v0, w0)))
    tol1 = 50.0 * max(dy, dz)**2
    print(f"[Test 1] max|div(u_0)|  = {div0:.3e}  (expect < 50*h^2 = {tol1:.3e})")
    assert div0 < tol1, f"FAIL: {div0:.3e} >= {tol1:.3e}"

    # ----------------------------------------------------------
    # Test 2: Laplacian consistency — Δ(sin·cos) ~ known value
    # ----------------------------------------------------------
    f  = np.sin(ky*Y) * np.cos(kz*Z)
    Lf_exact = -(ky**2 + kz**2) * f
    Lf_num   = laplacian(f)
    # compare only interior (walls have one-sided stencil)
    err = np.max(np.abs(Lf_num[:, 1:-1] - Lf_exact[:, 1:-1]))
    print(f"[Test 2] Laplacian error (interior) = {err:.3e}  (expect < 0.1)")
    assert err < 0.1, f"Laplacian inconsistency: {err}"

    # ----------------------------------------------------------
    # Test 3: Poisson solver — manufactured solution
    # ----------------------------------------------------------
    # p_exact = cos(ky·y)·cos(kz·z)  satisfies Neumann BCs in z
    # Δp_exact = -(ky²+kz²)·p_exact
    p_exact = np.cos(ky*Y) * np.cos(kz*Z)
    p_exact -= p_exact.mean()
    rhs_test = laplacian(p_exact)          # compute exact rhs from exact p
    p_init   = np.zeros_like(p_exact)
    p_solved = pressure_poisson(p_init, rhs_test, max_iter=10000, tol_rel=1e-8)
    err_p = np.max(np.abs(p_solved - p_exact))
    print(f"[Test 3] Poisson solver error       = {err_p:.3e}  (expect < 1e-3)")
    assert err_p < 1e-3, f"Poisson solver inaccurate: {err_p}"

    # ----------------------------------------------------------
    # Test 4: Projection removes divergence
    # ----------------------------------------------------------
    # Perturb a divergence-free field by adding a gradient (pure curl-free part)
    A   = 0.1   # small amplitude keeps the divergence manageable
    phi = A * np.cos(ky*Y) * np.cos(2*kz*Z)
    dpdy_phi, dpdz_phi = gradient(phi)
    v_pert = v0 + dpdy_phi
    w_pert = w0 + dpdz_phi
    div_before = np.max(np.abs(divergence(v_pert, w_pert)))
    rhs4       = (rho / dt) * divergence(v_pert, w_pert)
    p_corr     = pressure_poisson(np.zeros_like(phi), rhs4,
                                  max_iter=15000, tol_rel=1e-7)
    v_c, w_c   = projection_step(v_pert, w_pert, p_corr)
    v_c, w_c   = apply_velocity_bc(v_c, w_c)
    div_after  = np.max(np.abs(divergence(v_c, w_c)))
    reduction  = (1 - div_after / div_before) * 100
    print(f"[Test 4] Projection: {div_before:.3e} -> {div_after:.3e}  "
          f"({reduction:.1f}% reduction, expect >= 95%)")
    assert div_after < 0.05 * div_before, \
        f"FAIL: {div_after:.3e} should be < 5% of {div_before:.3e}"

    print("All tests passed.\n")

# ============================================================
# SIMULATION
# ============================================================

def incompressible_flow_simulation(nt, v, w, p):
    for n in range(nt):
        check_cfl(v, w)

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
            speed    = np.max(np.sqrt(v**2 + w**2))
            print(f"Step {n:4d} | max|∇·u| = {div_norm:.3e} | max|u| = {speed:.4f}")

    return v, w, p

# ============================================================
# PLOTTING
# ============================================================

def plot_results(y, z, v, w, p):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    speed = np.sqrt(v**2 + w**2)
    im0 = axes[0].contourf(y, z, speed.T, levels=20, cmap='viridis')
    axes[0].quiver(y[::4], z[::4], v[::4, ::4].T, w[::4, ::4].T,
                   color='white', scale=10)
    plt.colorbar(im0, ax=axes[0], label="|u|")
    axes[0].set(xlabel="y", ylabel="z", title="Speed + velocity field")

    im1 = axes[1].contourf(y, z, p.T, levels=20, cmap='RdBu_r')
    plt.colorbar(im1, ax=axes[1], label="p")
    axes[1].set(xlabel="y", ylabel="z", title="Pressure")

    div = divergence(v, w)
    im2 = axes[2].contourf(y, z, div.T, levels=20, cmap='RdBu_r')
    plt.colorbar(im2, ax=axes[2], label="∇·u")
    axes[2].set(xlabel="y", ylabel="z", title=f"Divergence (max={np.max(np.abs(div)):.2e})")

    plt.tight_layout()
    plt.savefig("./imgs/incompressible_flow.png", dpi=150)
    plt.show()

# ============================================================
# MAIN
# ============================================================

def main():
    run_tests()

    y, z, Y, Z = create_grid()
    v, w, p = initial_conditions(Y, Z)

    print(f"Initial max|∇·u| = {np.max(np.abs(divergence(v, w))):.3e}")
    print(f"Grid: dy={dy:.4f}, dz={dz:.4f}, dt={dt:.4f}")
    print(f"CFL(y): {np.max(np.abs(v))*dt/dy:.3f}, CFL(z): {np.max(np.abs(w))*dt/dz:.3f}")
    print()

    v, w, p = incompressible_flow_simulation(nt, v, w, p)

    print(f"\nFinal max|∇·u| = {np.max(np.abs(divergence(v, w))):.3e}")

    plot_results(y, z, v, w, p)


if __name__ == "__main__":
    main()