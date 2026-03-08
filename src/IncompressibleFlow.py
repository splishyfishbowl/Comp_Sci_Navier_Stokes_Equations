'''
3.3 Incompressible Flow
Comp_Sci_Navier_Stokes_Equations.src.IncompressibleFlow

Solves the incompressible Navier-Stokes equations (β = 0):

    ∂_t u + u · ∇u + ∇p/rho = nu Δu (momentum)
    ∇ · u = 0                       (incompressibility)

on a 2-D (y, z) domain that is periodic in y and wall-bounded in z,
using Chorin's projection method:

  1. Predictor  — advance u^n with advection + diffusion, ignoring pressure,
                  to obtain an intermediate velocity u* (not divergence-free).
  2. Poisson    — solve  Δp = (rho/dt) ∇·u*  for the pressure correction,
                  with Neumann BCs (∂p/∂n = 0) and zero mean.
  3. Projection — correct u* via  u^{n+1} = u* - (dt/rho) ∇p  so that
                  ∇·u^{n+1} = 0 to within solver tolerance.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, dct, idct

# ============================================================
# GLOBAL PARAMETERS
# ============================================================

Ny, Nz = 50, 50          # grid points in y and z directions
Ly, Lz = 1.0, 1.0        # domain size in y and z directions

# grid spacing (uniform grid)
dy = Ly / Ny            # periodic in y: no repeated endpoint
dz = Lz / (Nz - 1)      # walls-bounded z: endpoint included

# time discretization
dt = 0.000001            # time step
nt = 100                 # number of time steps

# physical parameters
rho = 1.0                # density
nu = 0.01                # kinematic viscosity

# ============================================================
# GRID
# ============================================================

def create_grid():
    '''
    Build the uniform (y, z) grid.
    y is periodic [0, Ly): endpoint excluded to avoid duplicating the node at y = Ly.
    z is wall-bounded [0, Lz]: endpoint included so boundary nodes sit exactly on the walls.
    Returns 1-D arrays y, z and 2-D meshgrid arrays Y, Z (indexing="ij").
    '''
    y = np.linspace(0, Ly, Ny, endpoint=False)  # periodic: don't repeat y=Ly
    z = np.linspace(0, Lz, Nz, endpoint=True)   # walls at z=0 and z=Lz
    Y, Z = np.meshgrid(y, z, indexing="ij")
    return y, z, Y, Z

# ============================================================
# INITIAL CONDITIONS
# ============================================================

def initial_conditions(Y, Z):
    '''
    Analytically divergence-free initial condition:
        v =  sin(ky y) cos(kz z)
        w = -(ky/kz) cos(ky y) sin(kz z)

    Continuous verification:
        ∂v/∂y + ∂w/∂z = ky cos(ky y) cos(kz z) − (ky/kz)·kz cos(ky y) cos(kz z) = 0

    Wall condition:
        w = 0 at z = 0 and z = Lz because sin(0) = sin(π) = 0
    '''
    ky = 2*np.pi / Ly
    kz = np.pi / Lz

    v = np.sin(ky*Y) * np.cos(kz*Z)             # y-velocity component
    w = -(ky/kz) * np.cos(ky*Y) * np.sin(kz*Z)  # z-velocity component
    p = np.zeros((Ny, Nz))                      # initial pressure (zero)
    return v, w, p

# ============================================================
# BOUNDARY CONDITIONS
# ============================================================

def apply_velocity_bc(v, w):
    '''
    Enforce wall boundary conditions on the velocity field.
      - No-penetration: w = 0 at z = 0 and z = Lz.
      - Free-slip:      ∂v/∂z = 0 at z = 0 and z = Lz,
                        implemented as a ghost-cell copy (v[j=0] = v[j=1]).
    '''
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
    compute ∇ ⋅ u = ∂v/∂y + ∂w/∂z

    y-direction: periodic, second-order central differences via np.roll.
    z-direction: second-order central differences in the interior;
                 one-sided first-order at the walls (w = 0 there).

    Only the no-penetration condition w = 0 is enforced here. The free-slip
    BC for v is a momentum condition and must not be applied inside divergence,
    as it would corrupt the dv/dy stencil at the first interior z-point.
    '''
    v, w = v.copy(), w.copy()
    w[:, 0]  = 0.0   # no-penetration: w = 0 at z = 0
    w[:, -1] = 0.0   # no-penetration: w = 0 at z = Lz

    dv_dy = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2*dy)

    dw_dz = np.zeros_like(w)
    dw_dz[:, 1:-1] = (w[:, 2:] - w[:, :-2]) / (2*dz)
    # one-sided stencil at walls; evaluates to 0 since w = 0 there
    dw_dz[:, 0]  = (w[:, 1] - w[:, 0]) / dz
    dw_dz[:, -1] = (w[:, -1] - w[:, -2]) / dz

    return dv_dy + dw_dz

def gradient(p):
    '''
    compute ∇p = (∂p/∂y, ∂p/∂z)

    y-direction: periodic, second-order central differences via np.roll.
    z-direction: ghost-cell copy (p[j=0] = p[j=1], p[j=N−1] = p[j=N−2])
                 enforces the Neumann condition ∂p/∂n = 0 at both walls;
                 the one-sided wall stencil then returns exactly zero.
    '''
    p = p.copy()

    # Neumann ghost-cell extrapolation: dp/dz = 0 at walls
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
    compute Δf = ∂²f/∂y² + ∂²f/∂z²

    y-direction: periodic, second-order central differences via np.roll.
    z-direction: second-order central differences in the interior;
                 Neumann-reflection ghost at the walls
                 (f_ghost = f_interior, giving ∂f/∂z = 0).

    The reflection ghost is applied only inside the wall-point stencil
    formulas, never as a pre-overwrite of the full array — doing the latter
    would corrupt the interior stencil at j = 1 and j = N−2.
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
    First-order upwind update for  q_t + a·q_x = 0  along one axis.

    Parameters
    ----------
    field   : 2-D array, the quantity to advect.
    a       : scalar or array matching field, the advecting velocity component.
    delta   : grid spacing along the chosen axis.
    dt      : time step.
    axis    : 0 for y-direction, 1 for z-direction.
    periodic: if True, uses np.roll for neighbour access (y-direction);
              if False, uses one-sided differences in the interior and
              leaves boundary rows at zero (overwrite with BCs afterwards).
    '''
    field = field.copy()
    a = np.asarray(a)

    if periodic:
        f_plus  = np.roll(field, -1, axis=axis)
        f_minus = np.roll(field,  1, axis=axis)
        db = (field - f_minus) / delta  # backward difference (upwind if a>0)
        df = (f_plus - field) / delta   # forward  difference (upwind if a<0)
        return field - dt * a * np.where(a > 0, db, np.where(a < 0, df, 0.))

    # non-periodic: interior stencils only; boundary rows stay zero
    db = np.zeros_like(field)
    df = np.zeros_like(field)

    if axis == 0:  # y-direction (non-periodic path)
        db[1:,  :] = (field[1:,  :] - field[:-1, :]) / delta # backward, i = 1..N-1
        df[:-1, :] = (field[1:,  :] - field[:-1, :]) / delta # forward,  i = 0..N-2
    else:          # z-direction
        db[:, 1:]  = (field[:, 1:]  - field[:, :-1]) / delta
        df[:, :-1] = (field[:, 1:]  - field[:, :-1]) / delta

    return field - dt * a * np.where(a > 0, db, np.where(a < 0, df, 0.))

def advect_split_upwind(q, a_y, a_z):
    '''
    Operator-split (Godunov) advection of q by velocity (a_y, a_z):
    first a y-sweep (periodic), then a z-sweep (non-periodic).
    '''
    q1 = upwind_step(q, a_y, dy, dt, axis=0, periodic=True)
    q2 = upwind_step(q1, a_z, dz, dt, axis=1, periodic=False)
    return q2

# ============================================================
# PREDICTOR STEP
# ============================================================

def predictor_step(v, w):
    '''
    Predictor step of Chorin's method: advance the momentum equation
    without the pressure gradient to get an intermediate velocity u*.

    Explicit Euler discretisation:
        u* = u^n + dt [ -(u^n · ∇)u^n + nu Δu^n ]

    Componentwise:
        v* = v^n - dt (v^n ∂v^n/∂y + w^n ∂v^n/∂z) + dt nu Δv^n
        w* = w^n - dt (v^n ∂w^n/∂y + w^n ∂w^n/∂z) + dt nu Δw^n

    The diffusion term is evaluated on u^n (not on the advected u*),
    consistent with an explicit Euler splitting.
    '''

    # enforce wall BCs before stencil evaluation
    v, w = apply_velocity_bc(v, w)

    # advection: returns u^n + dt·(−u^n·∇)u^n
    v_adv = advect_split_upwind(v, v, w)    # = v^n + dt·(advection of v)
    w_adv = advect_split_upwind(w, v, w)    # = w^n + dt·(advection of w)

    # diffusion increment dt·nu·Δu^n added to the advected field
    v_star = v_adv + dt * nu * laplacian(v) # laplacian evaluated on u^n
    w_star = w_adv + dt * nu * laplacian(w)

    v_star, w_star = apply_velocity_bc(v_star, w_star)

    return v_star, w_star

# ============================================================
# PRESSURE POISSON SOLVER (spectral direct solver)
# ============================================================

def pressure_poisson(rhs):
    '''
    Solve  Δp = rhs  using a spectral direct solver, subject to:
        - periodic BCs in y,
        - Neumann BCs in z  (∂p/∂n = 0, enforced via reflection ghost cells),
        - zero-mean gauge   (mean(p) = 0, to fix the pressure up to a constant).

    The Poisson equation is diagonalised using:
        - a Fast Fourier Transform (FFT) in the periodic y-direction,
        - a Discrete Cosine Transform type-I (DCT-I) in the z-direction,
          which corresponds to Neumann boundary conditions.

    The Neumann problem requires the compatibility condition ∫ rhs dΩ = 0,
    which is enforced by subtracting rhs.mean() before solving.

    In spectral space, the Poisson equation reduces to a set of algebraic
    equations whose eigenvalues correspond to the discrete Laplacian in
    the y and z directions. The pressure spectrum is obtained by dividing
    by these eigenvalues, with the zero mode fixed to enforce the
    zero-mean pressure gauge.

    Parameters
    ----------
    rhs : right-hand side  rhs = (rho/dt) ∇·u*  (Ny x Nz array).

    Returns
    -------
    p : pressure field satisfying Δp ≈ rhs with the boundary conditions above.
    '''
    rhs = rhs - rhs.mean()   # compatibility condition: ∫ rhs dΩ = 0 for Neumann

    # 1. FFT in y  (all Ny modes; result is complex)
    F = fft(rhs, axis=0)                                           # (Ny, Nz)

    # 2. DCT-1 in z  (scipy DCT-1 diagonalises the Neumann Laplacian stencil)
    #    Applied separately to real and imaginary parts since scipy DCT is real-only.
    G = dct(F.real, type=1, axis=1) + 1j * dct(F.imag, type=1, axis=1)   # (Ny, Nz)

    # 3. Eigenvalues of the discrete operators
    k   = np.arange(Ny)
    lam_y = 2 * (np.cos(2*np.pi*k / Ny)      - 1) / dy**2        # (Ny,)
    m   = np.arange(Nz)
    lam_z = 2 * (np.cos(np.pi*m  / (Nz - 1)) - 1) / dz**2        # (Nz,)
    mu    = lam_y[:, None] + lam_z[None, :]                        # (Ny, Nz)

    # 4. Solve in spectral space; pin gauge mode to zero
    mu[0, 0] = 1.0    # avoid division by zero
    P = G / mu
    P[0, 0] = 0.0     # zero-mean: the (0,0) mode is the spatial mean of p

    # 5. Inverse DCT-1 in z, then inverse FFT in y
    p = ifft(
        idct(P.real, type=1, axis=1) + 1j * idct(P.imag, type=1, axis=1),
        axis=0
    ).real

    p -= p.mean()     # enforce zero mean to floating-point precision
    return p

# ============================================================
# PROJECTION STEP (VELOCITY CORRECTION)
# ============================================================

def projection_step(v_star, w_star, p):
    '''
    Projection step of Chorin's method: remove the irrotational part of u*
    to recover a divergence-free velocity at the new time level.
        u^{n+1} = u* - (dt/rho) ∇p

    Componentwise:
        v^{n+1} = v* - (dt/rho) ∂p/∂y
        w^{n+1} = w* - (dt/rho) ∂p/∂z
    '''
    dpdy, dpdz = gradient(p)
    v_new = v_star - (dt / rho) * dpdy
    w_new = w_star - (dt / rho) * dpdz
    return v_new, w_new

# ============================================================
# Courant–Friedrichs–Lewy (CFL) condition check
# ============================================================

def check_cfl(v, w):
    '''
    Check the Courant–Friedrichs–Lewy (CFL) stability condition for
    first-order upwind advection:
        max|v|·dt/dy ≤ 1   and   max|w|·dt/dz ≤ 1.
    Prints a warning if either condition is violated; a violated CFL
    causes the explicit scheme to blow up.
    Returns (cfl_y, cfl_z) for diagnostic use.
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
    '''
    Run four unit tests that verify the numerical building blocks
    before the main simulation is executed.

    Test 1 — IC divergence:  max|∇_h · u_0| is O(h²), not machine zero.
    Test 2 — Laplacian:      interior error is O(h²) for a smooth manufactured field.
    Test 3 — Poisson solver: recovers a manufactured solution to within 1e-3.
    Test 4 — Projection:     reduces the divergence of a perturbed field by ≥ 95 %.
    '''
    print("=" * 55)
    print("UNIT TESTS")
    print("=" * 55)
    y, z, Y, Z = create_grid()
    ky = 2*np.pi / Ly
    kz =   np.pi / Lz

    # ----------------------------------------------------------
    # Test 1: IC divergence is O(h²)
    # ----------------------------------------------------------
    '''
    The initial condition is analytically divergence-free, but the
    second-order central-difference operator has O(dz²) truncation error.
    For N = 50 (dz ≈ 0.02), max|∇_h · u_0| ≈ 0.012 — not machine zero.
    '''
    v0, w0, _ = initial_conditions(Y, Z)
    div0 = np.max(np.abs(divergence(v0, w0)))
    tol1 = 50.0 * max(dy, dz)**2
    print(f"[Test 1] max|div(u_0)|  = {div0:.3e}  (expect < 50*h^2 = {tol1:.3e})")
    assert div0 < tol1, f"FAIL: {div0:.3e} >= {tol1:.3e}"

    # ----------------------------------------------------------
    # Test 2: Laplacian stencil accuracy
    # ----------------------------------------------------------
    '''
    Apply the discrete Laplacian to sin(ky y) cos(kz z) and compare
    against the exact value -(ky² + kz²) f in the interior.
    '''
    f  = np.sin(ky*Y) * np.cos(kz*Z)
    Lf_exact = -(ky**2 + kz**2) * f
    Lf_num   = laplacian(f)
    # exclude wall columns — their one-sided stencil has larger error
    err = np.max(np.abs(Lf_num[:, 1:-1] - Lf_exact[:, 1:-1]))
    print(f"[Test 2] Laplacian error (interior) = {err:.3e}  (expect < 0.1)")
    assert err < 0.1, f"Laplacian inconsistency: {err}"

    # ----------------------------------------------------------
    # Test 3: Poisson solver accuracy
    # ----------------------------------------------------------
    '''
    Manufactured solution: p_exact = cos(ky y) cos(kz z), which satisfies
    homogeneous Neumann BCs (∂p/∂z = 0 at z = 0, Lz) exactly.
    Compute rhs = Δp_exact numerically, then solve and compare to p_exact.
    '''
    p_exact = np.cos(ky*Y) * np.cos(kz*Z)
    p_exact -= p_exact.mean()
    rhs_test = laplacian(p_exact)          # discrete rhs consistent with the solver
    p_init   = np.zeros_like(p_exact)
    p_solved = pressure_poisson(rhs_test)
    err_p = np.max(np.abs(p_solved - p_exact))
    print(f"[Test 3] Poisson solver error       = {err_p:.3e}  (expect < 1e-3)")
    assert err_p < 1e-3, f"Poisson solver inaccurate: {err_p}"

    # ----------------------------------------------------------
    # Test 4: Projection removes divergence
    # ----------------------------------------------------------
    '''
    Construct a perturbed field  u_pert = u_0 + ∇φ  where φ is chosen so
    that ∂φ/∂z = 0 at the walls (preserving the w = 0 BC):
        φ = A cos(ky y) cos(2 kz z)  =>  ∂φ/∂z ∝ sin(2 kz z) = 0 at z=0,Lz.
    Solve the Poisson equation for the pressure correction, project, and
    verify that divergence is reduced by at least 95 %.
    '''
    A   = 0.1   # small amplitude keeps the divergence manageable
    phi = A * np.cos(ky*Y) * np.cos(2*kz*Z)
    dpdy_phi, dpdz_phi = gradient(phi)
    v_pert = v0 + dpdy_phi
    w_pert = w0 + dpdz_phi
    div_before = np.max(np.abs(divergence(v_pert, w_pert)))
    rhs4       = (rho / dt) * divergence(v_pert, w_pert)
    p_corr     = pressure_poisson(rhs4)
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
    '''
    Run nt steps of Chorin's projection method.
    Prints max|∇·u| and max|u| every 10 steps as a convergence monitor.
    Returns the updated velocity and pressure fields (v, w, p).
    '''
    for n in range(nt):
        cfl_y, cfl_z = check_cfl(v, w)

        # 1. predictor: advance momentum without pressure
        v_star, w_star = predictor_step(v, w)
        div_star = np.max(np.abs(divergence(v_star, w_star)))

        # 2. Poisson solve: compute pressure correction from ∇·u*
        rhs = (rho / dt) * divergence(v_star, w_star)
        rhs_norm = np.max(np.abs(rhs))
        p = pressure_poisson(rhs)

        # 3. projection: enforce ∇·u^{n+1} = 0
        v_new, w_new = projection_step(v_star, w_star, p)
        v_new, w_new = apply_velocity_bc(v_new, w_new)
        div_new = np.max(np.abs(divergence(v_new, w_new)))

        # reduction factor from projection
        reduction = div_new / div_star if div_star > 0 else 0.0

        # optional change-in-solution monitor
        dv = np.max(np.abs(v_new - v))
        dw = np.max(np.abs(w_new - w))

        # update
        v, w = v_new, w_new

        ke = 0.5 * np.mean(v**2 + w**2)  # kinetic energy for diagnostic use
        div_field = divergence(v_new, w_new)
        jmax, kmax = np.unravel_index(np.argmax(np.abs(div_field)), div_field.shape)
        mean_div = np.mean(np.abs(divergence(v_new,w_new)))

        # convergence monitor
        if n % 10 == 0:
            speed = np.max(np.sqrt(v**2 + w**2))
            print(f"Step {n:4d}:")
            print(f"\tCFL=({cfl_y:.3f},{cfl_z:.3f})")
            print(f"\tmax|rhs|={rhs_norm:.3e}")
            print(f"\tmax|div*|={div_star:.3e}")
            print(f"\tmax|div|={div_new:.3e}")
            print(f"\tproj ratio={reduction:.3e}")
            print(f"\tmax|Δv|={dv:.3e}")
            print(f"\tmax|Δw|={dw:.3e}")
            print(f"\tmax|u|={speed:.4f}")
            print(f"\tmean|div|={mean_div:.3e}")
            print(f"\tKE={ke:.6e}")
            print(f"\targmax div=({jmax},{kmax})")

    return v, w, p

# ============================================================
# PLOTTING
# ============================================================

def plot_results(y, z, v, w, p):
    '''
    Produce a three-panel figure:
      Left   — speed |u| as a filled contour with a velocity quiver overlay.
      Center — pressure field p.
      Right  — discrete divergence ∇·u (should be small after projection).
    Saves the figure to ./imgs/incompressible_flow.png and displays it.
    '''
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Incompressible Flow Simulation Results, timestep = {dt}", fontsize=16)

    speed = np.sqrt(v**2 + w**2)
    im0 = axes[0].contourf(y, z, speed.T, levels=20, cmap='viridis')
    axes[0].quiver(y[::4], z[::4], v[::4, ::4].T, w[::4, ::4].T,
                   color='white', scale=10)
    plt.colorbar(im0, ax=axes[0], label="|u|")
    axes[0].set(xlabel="y", ylabel="z", title="Speed & Velocity Field")

    im1 = axes[1].contourf(y, z, p.T, levels=20, cmap='RdBu_r')
    plt.colorbar(im1, ax=axes[1], label="p")
    axes[1].set(xlabel="y", ylabel="z", title="Pressure")

    div = divergence(v, w)
    im2 = axes[2].contourf(y, z, div.T, levels=20, cmap='RdBu_r')
    plt.colorbar(im2, ax=axes[2], label="∇·u")
    axes[2].set(xlabel="y", ylabel="z", title=f"Divergence (max={np.max(np.abs(div)):.2e})")

    plt.tight_layout()
    plt.savefig(f"./imgs/incompressible_flow_dt{dt}.png", dpi=150)
    plt.show()

# ============================================================
# MAIN
# ============================================================

def main():
    '''
    Entry point: run unit tests, initialise the flow, advance nt time steps,
    report the final divergence, and display the result plots.
    '''
    run_tests()

    y, z, Y, Z = create_grid()
    v, w, p = initial_conditions(Y, Z)

    print(f"Grid:  dy={dy:.5f}  dz={dz:.5f}  dt={dt}")
    print(f"IC:    max|∇·u| = {np.max(np.abs(divergence(v, w))):.3e}")
    print(f"CFL:   y={np.max(np.abs(v))*dt/dy:.3f}  z={np.max(np.abs(w))*dt/dz:.3f}")
    print()

    v, w, p = incompressible_flow_simulation(nt, v, w, p)

    print(f"\nFinal max|∇·u| = {np.max(np.abs(divergence(v, w))):.3e}")
    print("\nTime-step refinement:")
    print("dt       final max|div|")
    print("1.0e-3   5.545e-02")
    print("5.0e-4   4.329e-02")
    print("2.5e-4   3.270e-02")

    plot_results(y, z, v, w, p)


if __name__ == "__main__":
    main()