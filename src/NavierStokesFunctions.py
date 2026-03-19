'''
Contains common functions for Navier-Stokes equations.
'''

import numpy as np
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
dt = None                # chosen automatically from CFL + diffusion bound
nt = 100                 # number of time steps

# physical parameters
rho = 1.0                # density
nu = 0.01                # kinematic viscosity

ky = 2*np.pi / Ly
kz =   np.pi / Lz

# ============================================================
# PROBLEM SETUP
# ============================================================

'''
Registry of built-in IC modes.
Each entry maps a mode name to:
    "v_fn"  : callable(Y, Z, ky, kz) -> v-field
    "w_fn"  : callable(Y, Z, ky, kz) -> w-field
    "label" : human-readable formula string for plot annotations
'''
IC_REGISTRY = {
    "mode1": {
        "v_fn":  lambda Y, Z, ky, kz:  np.sin(ky*Y) * np.cos(kz*Z),
        "w_fn":  lambda Y, Z, ky, kz: -(ky/kz) * np.cos(ky*Y) * np.sin(kz*Z),
        "label": "v = sin(ky·y)cos(kz·z),  w = -(ky/kz)cos(ky·y)sin(kz·z)",
    },
    "mode2": {
        "v_fn":  lambda Y, Z, ky, kz:  np.sin(2*ky*Y) * np.cos(kz*Z),
        "w_fn":  lambda Y, Z, ky, kz: -(2*ky/kz) * np.cos(2*ky*Y) * np.sin(kz*Z),
        "label": "v = sin(2ky·y)cos(kz·z),  w = -(2ky/kz)cos(2ky·y)sin(kz·z)",
    },
    "mode3": {
        "v_fn":  lambda Y, Z, ky, kz:  np.sin(ky*Y) * np.cos(2*kz*Z),
        "w_fn":  lambda Y, Z, ky, kz: -(ky/(2*kz)) * np.cos(ky*Y) * np.sin(2*kz*Z),
        "label": "v = sin(ky·y)cos(2kz·z),  w = -(ky/2kz)cos(ky·y)sin(2kz·z)",
    },
    "double": { # superposition of mode1 and mode2 (linearly combined)
        "v_fn":  lambda Y, Z, ky, kz: (
                     np.sin(ky*Y) * np.cos(kz*Z)
                   + np.sin(2*ky*Y) * np.cos(kz*Z)
                 ),
        "w_fn":  lambda Y, Z, ky, kz: (
                   -(ky/kz) * np.cos(ky*Y) * np.sin(kz*Z)
                   -(2*ky/kz) * np.cos(2*ky*Y) * np.sin(kz*Z)
                 ),
        "label": "v = sin(ky·y)cos(kz·z) + sin(2ky·y)cos(kz·z)  [mode1+mode2]",
    },
}

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

def initial_conditions(Y, Z, ic_mode="mode1", ic_custom_v=None, ic_custom_w=None):
    '''
    Return an analytically divergence-free initial velocity (v, w) and
    zero pressure, based on the chosen ic_mode.

    Parameters
    ----------
    Y, Z         : 2-D meshgrid arrays from create_grid().
    ic_mode      : str — one of "mode1", "mode2", "mode3", "double", "custom".
    ic_custom_v  : callable(Y, Z, ky, kz) -> array, required when ic_mode="custom".
    ic_custom_w  : callable(Y, Z, ky, kz) -> array, required when ic_mode="custom".

    Returns
    -------
    v, w  : velocity components (Ny x Nz arrays)
    p     : pressure            (Ny x Nz, initialised to zero)
    label : str describing the velocity formula (used in plot titles)

    Wall condition check (all built-in modes):
        w = 0 at z = 0 and z = Lz  because sin(·) = 0 there.
    '''

    if ic_mode == "custom":
        if ic_custom_v is None or ic_custom_w is None:
            raise ValueError(
                "ic_mode='custom' requires ic_custom_v and ic_custom_w callables."
            )
        v = ic_custom_v(Y, Z, ky, kz)
        w = ic_custom_w(Y, Z, ky, kz)
        label = "v = custom_v(Y,Z,ky,kz), w = custom_w(Y,Z,ky,kz)"

    elif ic_mode in IC_REGISTRY:
        entry = IC_REGISTRY[ic_mode]
        v     = entry["v_fn"](Y, Z, ky, kz)
        w     = entry["w_fn"](Y, Z, ky, kz)
        label = entry["label"]

    else:
        raise ValueError(
            f"Unknown ic_mode='{ic_mode}'. "
            f"Choose from {list(IC_REGISTRY.keys())} or 'custom'."
        )
    
    p = np.zeros((Ny, Nz)) # initial pressure (zero)
    return v, w, p, label

def check_cfl(v, w):
    '''
    Check the Courant-Friedrichs-Lewy (CFL) stability condition for
    first-order upwind advection:
        max|v|·dt/dy ≤ 1   and   max|w|·dt/dz ≤ 1.
    Prints a warning if either condition is violated; a violated CFL
    causes the explicit scheme to blow up.
    Returns (cfl_y, cfl_z) for diagnostic use.
    '''
    cfl_y = np.max(np.abs(v)) * dt / dy
    cfl_z = np.max(np.abs(w)) * dt / dz
    if max(cfl_y, cfl_z) > 1.0:
        print(f"[WARNING] CFL violated: CFL_y={cfl_y:.3f}, CFL_z={cfl_z:.3f}")
    return cfl_y, cfl_z

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

def advect_split_upwind(q, a_y, a_z):
    '''
    Operator-split (Godunov) advection of q by velocity (a_y, a_z):
    first a y-sweep (periodic), then a z-sweep (non-periodic).
    '''
    q1 = upwind_step(q,  a_y, dy, dt, axis=0, periodic=True)
    q2 = upwind_step(q1, a_z, dz, dt, axis=1, periodic=False)
    return q2

def upwind_step(field, a, delta, dt, axis, periodic=True):
    '''
    First-order upwind update for  q_t + a·q_x = 0  along one axis.

    Parameters
    ----------
    field   : 2D array, the quantity to advect.
    a       : scalar or 2D array matching field, the advecting velocity.
    delta   : grid spacing along the chosen axis.
    dt      : time step.
    axis    : 0 for y-direction, 1 for z-direction.
    periodic: if True, uses np.roll for neighbour access (y-direction);
              if False, uses one-sided differences in the interior and
              leaves boundary rows at zero (overwrite with BCs afterwards).

    Returns
    -------
    field_new : 2D array, updated field after one explicit upwind step.

    '''
    field = field.copy()
    a = np.asarray(a)

    if periodic:
        f_plus  = np.roll(field, -1, axis=axis)
        f_minus = np.roll(field,  1, axis=axis)
        db = (field - f_minus) / delta  # backward difference (upwind if a>0)
        df = (f_plus - field) / delta   # forward  difference (upwind if a<0)
    else:
        # non-periodic: interior stencils only; boundary rows stay zero
        db = np.zeros_like(field)
        df = np.zeros_like(field)
        if axis == 0:  # y-direction (non-periodic path)
            db[1:,  :] = (field[1:,  :] - field[:-1, :]) / delta # backward, i = 1..N-1
            df[:-1, :] = (field[1:,  :] - field[:-1, :]) / delta # forward,  i = 0..N-2
        else:          # z-direction
            db[:, 1:]  = (field[:, 1:]  - field[:, :-1]) / delta
            df[:, :-1] = (field[:, 1:]  - field[:, :-1]) / delta

    deriv = np.where(a > 0, db, np.where(a < 0, df, 0.))
    return field - dt * a * deriv

# ============================================================
# CHORIN PROJECTION METHOD STEPS
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
    rhs = rhs - rhs.mean() # compatibility condition: ∫ rhs dΩ = 0 for Neumann

    # 1. FFT in y  (all Ny modes; result is complex)
    F = fft(rhs, axis=0) # (Ny, Nz)

    # 2. DCT-1 in z (scipy DCT-1 diagonalises the Neumann Laplacian stencil)
    #    Applied separately to real and imaginary parts since scipy DCT is real-only.
    G = dct(F.real, type=1, axis=1) + 1j * dct(F.imag, type=1, axis=1) # (Ny, Nz)

    # 3. Eigenvalues of the discrete operators
    k   = np.arange(Ny)
    lam_y = 2 * (np.cos(2*np.pi*k / Ny) - 1) / dy**2 # (Ny,)
    m   = np.arange(Nz)
    lam_z = 2 * (np.cos(np.pi*m  / (Nz - 1)) - 1) / dz**2 # (Nz,)
    mu    = lam_y[:, None] + lam_z[None, :] # (Ny, Nz)

    # 4. Solve in spectral space; pin gauge mode to zero
    mu[0, 0] = 1.0 # avoid division by zero
    P = G / mu
    P[0, 0] = 0.0 # zero-mean: the (0,0) mode is the spatial mean of p

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
# SIMULATION
# ============================================================

def incompressible_flow_simulation(v, w, p):
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

        # update
        v, w = v_new, w_new

        # # diagnostics UNCOMMENT IF NEEDED FOR DEBUGGING PURPOSES
        # dv = np.max(np.abs(v_new - v))
        # dw = np.max(np.abs(w_new - w))
        # ke = 0.5 * np.mean(v**2 + w**2)  # kinetic energy for diagnostic use
        # div_field = divergence(v_new, w_new)
        # jmax, kmax = np.unravel_index(np.argmax(np.abs(div_field)), div_field.shape)
        # mean_div = np.mean(np.abs(divergence(v_new,w_new)))

        # # convergence monitor
        # if n % 10 == 0:
        #     speed = np.max(np.sqrt(v**2 + w**2))
        #     print(f"Step {n:4d}:")
        #     print(f"\tCFL=({cfl_y:.3f},{cfl_z:.3f})")
        #     print(f"\tmax|rhs|={rhs_norm:.3e}")
        #     print(f"\tmax|div*|={div_star:.3e}")
        #     print(f"\tmax|div|={div_new:.3e}")
        #     print(f"\tproj ratio={reduction:.3e}")
        #     print(f"\tmax|Δv|={dv:.3e}")
        #     print(f"\tmax|Δw|={dw:.3e}")
        #     print(f"\tmax|u|={speed:.4f}")
        #     print(f"\tmean|div|={mean_div:.3e}")
        #     print(f"\tKE={ke:.6e}")
        #     print(f"\targmax div=({jmax},{kmax})")

    return v, w, p