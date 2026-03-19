import NavierStokesFunctions as nsf
import RayleighBenardConvection as rb

import numpy as np

# ============================================================
# TESTS
# ============================================================

def intro_tests(ic_mode="mode1", ic_custom_v=None, ic_custom_w=None):
    '''
    Run four unit tests that verify the upwind advection scheme
    from section 3.2 before the main simulation is executed.
 
    Test 1 — Zero velocity:     upwind_step is the identity when a = 0.
    Test 2 — Constant field:    a uniform field is unchanged for any velocity.
    Test 3 — CFL restriction:   CFL > 1 causes growth; CFL ≤ 1 stays bounded.
    Test 4 — Convergence order: the global error halves as the grid is refined
                                 (first-order consistency in space and time).
    '''
    print(f"INTRODUCTION UPWIND UNIT TESTS: {ic_mode}")
    y, z, Y, Z = nsf.create_grid()
 
    ay_val = 0.5
    az_val = 0.3
 
    v0, w0, _, _ = nsf.initial_conditions(Y, Z, ic_mode, ic_custom_v, ic_custom_w)
    w0[:, 0] = w0[:, -1] = 0.0
    dt = 0.5 * min(nsf.dy / abs(ay_val), nsf.dz / abs(az_val))
 
    # ----------------------------------------------------------
    # Test 1: Zero velocity → identity
    # ----------------------------------------------------------
    '''
    With a = 0 the advection equation reduces to ∂_t u = 0, so the field
    must be unchanged after any number of steps.  We check both axes and
    both periodic / non-periodic paths.
    '''
    for axis, periodic in [(0, True), (1, False)]:
        delta = nsf.dy if axis == 0 else nsf.dz
        v_out = nsf.upwind_step(v0, 0.0, delta, dt, axis=axis, periodic=periodic)
        err = np.max(np.abs(v_out - v0))
        print(f"[Test 1] Zero-velocity identity  axis={axis}  error = {err:.3e}  (expect 0)")
        assert err == 0.0, f"FAIL axis={axis}: field changed with a=0, err={err:.3e}"
 
    # ----------------------------------------------------------
    # Test 2: Constant field → unchanged for any velocity
    # ----------------------------------------------------------
    '''
    A spatially uniform field has zero gradient everywhere, so the upwind
    derivative is zero and the field must not change regardless of velocity,
    time step, or boundary mode.
    '''
    field_const = 3.7 * np.ones((nsf.Ny, nsf.Nz))
    for axis, periodic in [(0, True), (1, False)]:
        delta = nsf.dy if axis == 0 else nsf.dz
        for a in (ay_val, -ay_val):
            out = nsf.upwind_step(field_const, a, delta, dt, axis=axis, periodic=periodic)
            err = np.max(np.abs(out - field_const))
            print(f"[Test 2] Constant-field invariance  axis={axis}  a={a:+.1f}  "
                  f"error = {err:.3e}  (expect 0)")
            assert err == 0.0, \
                f"FAIL axis={axis}, a={a}: constant field changed, err={err:.3e}"
 
    # ----------------------------------------------------------
    # Test 3: CFL restriction
    # ----------------------------------------------------------
    '''
    The explicit upwind scheme is stable iff  |a|·dt/δ ≤ 1 (CFL ≤ 1).
    We verify both sides of the boundary:
      - CFL = 0.4 (safe):   the L∞ norm must not grow after one step.
      - CFL = 1.5 (unsafe): the L∞ norm must grow (scheme blows up).
    The test uses the y-direction (periodic) with a smooth sinusoidal field
    so numerical diffusion does not mask instability.
    '''
    f_sin = np.sin(nsf.ky * Y) * np.ones_like(Z)
    norm0 = np.max(np.abs(f_sin))
 
    # stable step
    dt_stable   = 0.4 * nsf.dy / abs(ay_val)
    out_stable  = nsf.upwind_step(f_sin, ay_val, nsf.dy, dt_stable,  axis=0, periodic=True)
    norm_stable  = np.max(np.abs(out_stable))
    print(f"[Test 3] CFL=0.4 (stable):   norm {norm0:.4f} → {norm_stable:.4f}  (expect ≤ norm0)")
    assert norm_stable <= norm0 + 1e-12, \
        f"FAIL: stable step grew: {norm_stable:.4f} > {norm0:.4f}"
 
    # unstable step
    dt_unstable = 1.5 * nsf.dy / abs(ay_val)
    out_unstable = nsf.upwind_step(f_sin, ay_val, nsf.dy, dt_unstable, axis=0, periodic=True)
    norm_unstable = np.max(np.abs(out_unstable))
    print(f"[Test 3] CFL=1.5 (unstable): norm {norm0:.4f} → {norm_unstable:.4f}  (expect > norm0)")
    assert norm_unstable > norm0, \
        f"FAIL: expected growth for CFL=1.5 but norm did not increase"
 
    # ----------------------------------------------------------
    # Test 4: First-order convergence in space and time
    # ----------------------------------------------------------
    '''
    The exact solution of  u_t + a·∇u = 0  with periodic ICs is a pure
    translation: u(t,y,z) = u_0(y - ay·t, z - az·t).
    We run the split upwind scheme to t_final on three successively refined
    grids (N = Ny of the module, 2·Ny, 4·Ny) and measure the L∞ error against
    the exact translated field.  Halving the mesh size should halve the error
    (rate ≈ 1).  We accept any rate > 0.7 to allow for low-order constant factors.
 
    Only the y-direction (periodic, axis=0) is tested here because the
    z-direction has Dirichlet walls that break the pure-translation exact solution.
    '''
    t_final = 0.1
    errors  = []
 
    for N in (nsf.Ny, 2 * nsf.Ny, 4 * nsf.Ny):
        # build a refined grid for this N
        y_r   = np.linspace(0, nsf.Ly, N, endpoint=False)
        z_r   = np.linspace(0, nsf.Lz, nsf.Nz)
        dy_r  = y_r[1] - y_r[0]
        Y_r, Z_r = np.meshgrid(y_r, z_r, indexing="ij")
 
        dt_r    = 0.4 * dy_r / abs(ay_val)   # CFL = 0.4
        n_steps = int(t_final / dt_r)
        t_reached = n_steps * dt_r
 
        # IC from registry evaluated on the refined grid
        entry   = nsf.IC_REGISTRY.get(ic_mode, nsf.IC_REGISTRY["mode1"])
        field_r = entry["v_fn"](Y_r, Z_r, nsf.ky, nsf.kz)
 
        for _ in range(n_steps):
            field_r = nsf.upwind_step(field_r, ay_val, dy_r, dt_r,
                                      axis=0, periodic=True)
 
        exact_r = entry["v_fn"](Y_r - ay_val * t_reached, Z_r, nsf.ky, nsf.kz)
        errors.append(np.max(np.abs(field_r - exact_r)))
 
    rates = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    print(f"[Test 4] Convergence errors : {[f'{e:.3e}' for e in errors]}")
    print(f"[Test 4] Convergence rates  : {[f'{r:.2f}' for r in rates]}  (expect ≈ 1.0)")
    for r in rates:
        assert r > 0.7, \
            f"FAIL: convergence rate {r:.2f} < 0.7; scheme may not be first-order"
 
    print(f"{ic_mode}: All tests passed.\n")


def incomp_flow_tests(ic_mode="mode1", ic_custom_v=None, ic_custom_w=None, custom_components=None):
    '''
    Run four unit tests that verify the numerical building blocks
    before the main simulation is executed.

    Test 1 — IC divergence:  max|∇_h · u_0| is O(h²), not machine zero.
    Test 2 — Laplacian:      interior error is O(h²) for a smooth manufactured field.
    Test 3 — Poisson solver: recovers a manufactured solution to within 1e-3.
    Test 4 — Projection:     reduces the divergence of a perturbed field by ≥ 95 %.
    '''
    print(f"INCOMPRESSIBLE FLOW UNIT TESTS: {ic_mode}")
    y, z, Y, Z = nsf.create_grid()

    # ----------------------------------------------------------
    # Test 1: IC divergence is O(h²)
    # ----------------------------------------------------------
    '''
    The central-difference truncation error on ∇·u scales as
      (n·ky)²·dy²  +  (m·kz)²·dz²
    where n, m are the y- and z-wavenumber multipliers of the chosen mode.
    A fixed tolerance calibrated for mode1 (n=m=1) therefore under-estimates
    the expected error for higher modes such as mode2 (n=2) or mode3 (m=2).
    We derive the multipliers from the registry so the tolerance is always tight.
    '''
    _mode_components  = {
        "mode1":  [(1, 1)],         # sin(1·ky·y) cos(1·kz·z)
        "mode2":  [(2, 1)],         # sin(2·ky·y) cos(1·kz·z)
        "mode3":  [(1, 2)],         # sin(1·ky·y) cos(2·kz·z)
        "double": [(1, 1), (2, 1)], # mode1 + mode2: errors add
    }

    if ic_mode == "custom":
        components = custom_components if custom_components is not None else [(3, 1)]
    else:
        components = _mode_components.get(ic_mode, [(2, 2)])

    tol1 = 3.0 * sum((abs(ny) * nsf.ky)**3 * nsf.dy**2 + (abs(nz) * nsf.kz)**3 * nsf.dz**2
                 for ny, nz in components)

    v0, w0, _, _ = nsf.initial_conditions(Y, Z, ic_mode, ic_custom_v, ic_custom_w)
    div0 = np.max(np.abs(nsf.divergence(v0, w0)))
    print(f"[Test 1] max|div(u_0)| = {div0:.3e}  (expect < {tol1:.3e})")
    assert div0 < tol1, f"FAIL: {div0:.3e} >= {tol1:.3e}"

    # ----------------------------------------------------------
    # Test 2: Laplacian stencil accuracy
    # ----------------------------------------------------------
    '''
    Apply the discrete Laplacian to sin(ky y) cos(kz z) and compare
    against the exact value -(ky² + kz²) f in the interior.
    '''
    f  = np.sin(nsf.ky*Y) * np.cos(nsf.kz*Z)
    Lf_exact = -(nsf.ky**2 + nsf.kz**2) * f
    Lf_num   = nsf.laplacian(f)
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
    p_exact = np.cos(nsf.ky*Y) * np.cos(nsf.kz*Z)
    p_exact -= p_exact.mean()
    rhs_test = nsf.laplacian(p_exact) # discrete rhs consistent with the solver
    p_solved = nsf.pressure_poisson(rhs_test)
    err_p = np.max(np.abs(p_solved - p_exact))
    print(f"[Test 3] Poisson solver error = {err_p:.3e} (expect < 1e-3)")
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
    nsf.dt = 0.01
    A   = 0.1   # small amplitude keeps the divergence manageable
    phi = A * np.cos(nsf.ky*Y) * np.cos(2*nsf.kz*Z)
    dpdy_phi, dpdz_phi = nsf.gradient(phi)
    v_pert = v0 + dpdy_phi
    w_pert = w0 + dpdz_phi
    div_before = np.max(np.abs(nsf.divergence(v_pert, w_pert)))
    rhs4       = (nsf.rho / nsf.dt) * nsf.divergence(v_pert, w_pert)
    p_corr     = nsf.pressure_poisson(rhs4)
    v_c, w_c   = nsf.projection_step(v_pert, w_pert, p_corr)
    v_c, w_c   = nsf.apply_velocity_bc(v_c, w_c)
    div_after  = np.max(np.abs(nsf.divergence(v_c, w_c)))
    reduction  = (1 - div_after / div_before) * 100
    print(f"[Test 4] Projection: {div_before:.3e} -> {div_after:.3e}  "
          f"({reduction:.1f}% reduction, expect >= 95%)")
    assert div_after < 0.05 * div_before, \
        f"FAIL: {div_after:.3e} should be < 5% of {div_before:.3e}"

    print(f"{ic_mode}: All tests passed.\n")

def rayleigh_benard_tests(ic_mode="mode1", ic_custom_v=None, ic_custom_w=None):
    '''
    Run four unit tests that verify the Rayleigh-Benard building blocks
    before the main simulation is executed.
 
    Test 1 — Temperature BCs:       T = 1 at z = 0, T = 0 at z = Lz, always.
    Test 2 — Temperature Laplacian: Δ(1-z) = 0 in the interior (conduction profile).
    Test 3 — Buoyancy source term:  zero fluctuation → no buoyancy force on w.
    Test 4 — Coupled step:          one RB step preserves BCs and stays finite.
    '''
    print(f"RAYLEIGH-BENARD UNIT TESTS: {ic_mode}")
    y, z, Y, Z = nsf.create_grid()
 
    v0, w0, p0, _ = nsf.initial_conditions(Y, Z, ic_mode, ic_custom_v, ic_custom_w)
    w0[:, 0] = w0[:, -1] = 0.0
    nsf.dt = rb.choose_dt(v0, w0, safety=0.4)
 
    # ----------------------------------------------------------
    # Test 1: Temperature BCs are enforced by apply_temperature_bc
    # ----------------------------------------------------------
    '''
    apply_temperature_bc must pin T = 1 at z = 0 and T = 0 at z = Lz
    regardless of the interior values passed in.  We verify this by
    constructing a field that violates the BCs and checking that the
    function corrects both walls exactly.
    '''
    T_bad   = np.random.rand(nsf.Ny, nsf.Nz)   # deliberately wrong at walls
    T_fixed = rb.apply_temperature_bc(T_bad)
    err_bot = np.max(np.abs(T_fixed[:, 0]  - 1.0))
    err_top = np.max(np.abs(T_fixed[:, -1] - 0.0))
    print(f"[Test 1] T BC bottom wall error = {err_bot:.3e}  (expect 0)")
    print(f"[Test 1] T BC top    wall error = {err_top:.3e}  (expect 0)")
    assert err_bot == 0.0, f"FAIL: bottom wall T != 1, err={err_bot:.3e}"
    assert err_top == 0.0, f"FAIL: top wall T != 0, err={err_top:.3e}"
 
    # ----------------------------------------------------------
    # Test 2: Temperature Laplacian accuracy on conduction profile
    # ----------------------------------------------------------
    '''
    The steady conduction profile T_c(z) = 1 - z satisfies Δ T_c = 0 exactly
    (it is linear in z, so ∂²T/∂z² = 0, and uniform in y, so ∂²T/∂y² = 0).
    The boundary rows are excluded because laplacian_temperature sets them to
    zero by design — BCs are re-applied externally after each step.
    '''
    T_cond  = 1.0 - Z
    LT_num  = rb.laplacian_temperature(T_cond)
    err     = np.max(np.abs(LT_num[:, 1:-1]))   # interior only
    print(f"[Test 2] Δ(1-z) interior error = {err:.3e}  (expect ≈ 0)")
    assert err < 1e-10, \
        f"FAIL: Δ(1-z) should be 0 in the interior, got {err:.3e}"
 
    # ----------------------------------------------------------
    # Test 3: Zero temperature fluctuation → no buoyancy force on w
    # ----------------------------------------------------------
    '''
    The Boussinesq buoyancy term is  β·(T - T_c)·e_z  where T_c = 1 - z is
    the conductive background.  When T = T_c exactly the fluctuation is zero
    and predictor_step_boussinesq must return the same w as the standard
    predictor (i.e. the buoyancy increment on w is zero).
    The test compares the w-component of the Boussinesq predictor against
    that of the standard predictor; they must agree to machine precision.
    '''
    T_cond       = 1.0 - Z
    v_star_std,  w_star_std  = nsf.predictor_step(v0, w0)
    v_star_bous, w_star_bous = rb.predictor_step_boussinesq(v0, w0, T_cond, Z)
    err_w = np.max(np.abs(w_star_bous - w_star_std))
    print(f"[Test 3] Buoyancy increment at T=T_c: max|Δw| = {err_w:.3e}  (expect ≈ 0)")
    assert err_w < 1e-12, \
        f"FAIL: buoyancy should vanish for T=T_c, got max|Δw|={err_w:.3e}"
 
    # ----------------------------------------------------------
    # Test 4: One coupled RB step — finite values and BCs preserved
    # ----------------------------------------------------------
    '''
    A single call to rayleigh_benard_step must:
      (a) return arrays that are everywhere finite (no NaN / Inf),
      (b) preserve w = 0 at both walls (no-penetration BC),
      (c) preserve T = 1 at z = 0 and T = 0 at z = Lz (Dirichlet BCs).
    We start from the conduction profile with a small perturbation so that
    the buoyancy force is non-trivial and genuinely exercises the coupling.
    '''
    T0 = rb.initial_temperature_rb(Y, Z, eps=0.01)
    v_new, w_new, p_new, T_new = rb.rayleigh_benard_step(v0, w0, T0, Z)
 
    fin_v = np.isfinite(v_new).all()
    fin_w = np.isfinite(w_new).all()
    fin_T = np.isfinite(T_new).all()
    fin_p = np.isfinite(p_new).all()
    print(f"[Test 4] Finite values: v={fin_v}  w={fin_w}  T={fin_T}  p={fin_p}  (all expect True)")
    assert fin_v and fin_w and fin_T and fin_p, \
        "FAIL: non-finite values detected after one RB step"
 
    err_w_bot = np.max(np.abs(w_new[:, 0]))
    err_w_top = np.max(np.abs(w_new[:, -1]))
    print(f"[Test 4] w BC: bottom={err_w_bot:.3e}  top={err_w_top:.3e}  (expect 0)")
    assert err_w_bot == 0.0, f"FAIL: w != 0 at bottom wall, err={err_w_bot:.3e}"
    assert err_w_top == 0.0, f"FAIL: w != 0 at top wall, err={err_w_top:.3e}"
 
    err_T_bot = np.max(np.abs(T_new[:, 0]  - 1.0))
    err_T_top = np.max(np.abs(T_new[:, -1] - 0.0))
    print(f"[Test 4] T BC: bottom={err_T_bot:.3e}  top={err_T_top:.3e}  (expect 0)")
    assert err_T_bot == 0.0, f"FAIL: T != 1 at bottom wall, err={err_T_bot:.3e}"
    assert err_T_top == 0.0, f"FAIL: T != 0 at top wall, err={err_T_top:.3e}"
 
    print(f"{ic_mode}: All tests passed.\n")