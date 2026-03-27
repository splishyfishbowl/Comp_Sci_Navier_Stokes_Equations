'''
3.4 Rayleigh-Benard Convection
Comp_Sci_Navier_Stokes_Equations.src.Rayleigh-BenardConvection
'''

'''
Implement the scalar equation (1) using the same (explicit) methods as before.
(1)     ∂_tT + ~u · ∇T = ∆T
'''

'''
Implement the source term for β != 0 in (2) by an explicit time stepping method.
(2)     ∂_t~u + ~u · ∇~u + ∇p/ρ = ν∆~u + T βe_z
'''

'''
Find initial conditions, boundary conditions and parameters such that you can the classical Rayleigh-Benard 
convection cells.
'''

import NavierStokesFunctions as nsf
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

beta = 50.  # thermal expansion coefficient

def choose_dt(v, w, safety=0.1):
    """
    Stable dt for explicit RBC update.
    Includes:
      - advection CFL from current velocity
      - diffusion bound for momentum/temperature
    """
    vmax = max(np.max(np.abs(v)), 1e-8)
    wmax = max(np.max(np.abs(w)), 1e-8)

    dt_adv = safety * min(nsf.dy / vmax, nsf.dz / wmax)

    # diffusion bound (same viscosity / thermal diffusivity = 1 here for T)
    dt_diff_v = safety * 0.5 / (nsf.nu * (1.0/nsf.dy**2 + 1.0/nsf.dz**2))
    dt_diff_T = safety * 0.5 / (1.0 * (1.0/nsf.dy**2 + 1.0/nsf.dz**2))

    return min(dt_adv, dt_diff_v, dt_diff_T)

# ============================================================
# TEMPERATURE EQUATION FOR RAYLEIGH-BENARD CONVECTION
# ============================================================

def apply_temperature_bc(T):
    """
    Boundary conditions for temperature:
      - periodic in y (handled by np.roll in the operators)
      - T = 1 at z = 0   (hot bottom plate)
      - T = 0 at z = Lz  (cold top plate)
    """
    T = T.copy()
    T[:, 0]  = 1.0
    T[:, -1] = 0.0
    return T


def initial_temperature(Y, Z, perturbation=0.01):
    """
    Initial temperature profile:
      conduction state: T(z) = 1 - z
      plus a small periodic perturbation to trigger convection later.
    """
    T = 1.0 - Z

    # small perturbation, vanishes at top and bottom so BCs stay compatible
    T += perturbation * np.sin(2*np.pi*Y/nsf.Ly) * np.sin(np.pi*Z/nsf.Lz)

    return apply_temperature_bc(T)

def laplacian_temperature(T):
    """
    Laplacian for temperature with:
      - periodic in y
      - Dirichlet walls in z: T=1 at bottom, T=0 at top
    """
    T = T.copy()

    d2y = (np.roll(T, -1, axis=0) - 2*T + np.roll(T, 1, axis=0)) / (nsf.dy**2)

    d2z = np.zeros_like(T)
    d2z[:, 1:-1] = (T[:, 2:] - 2*T[:, 1:-1] + T[:, :-2]) / (nsf.dz**2)

    # boundary rows are fixed by Dirichlet BCs, so we do not evolve them
    d2z[:, 0]  = 0.0
    d2z[:, -1] = 0.0

    return d2y + d2z

def temperature_step(T, v, w):
    """
    One explicit time step for

        T_t + u · grad(T) = Delta T

    using:
      - split first-order upwind for advection
      - explicit Laplacian for diffusion
    """
    T = apply_temperature_bc(T)

    # advection with the same split upwind method as before
    T_adv = nsf.advect_split_upwind(T, v, w)

    # diffusion term treated explicitly
    T_new = T_adv + nsf.dt * laplacian_temperature(T)

    # re-apply Dirichlet BCs after the update
    T_new = apply_temperature_bc(T_new)

    return T_new


def temperature_simulation(nt_temp, v, w, Y, Z, T0=None, store_history=False):
    """
    Evolve the scalar temperature equation for nt_temp steps
    using a prescribed velocity field (v,w).

    Parameters
    ----------
    nt_temp : int
        number of time steps
    v, w : 2D arrays
        velocity field
    T0 : 2D array or None
        initial temperature field; if None, use default profile
    store_history : bool
        if True, store all temperature snapshots

    Returns
    -------
    T : final temperature
    hist : list of snapshots (optional)
    """

    if T0 is None:
        T = initial_temperature(Y, Z)
    else:
        T = apply_temperature_bc(T0)

    hist = [T.copy()] if store_history else None

    for n in range(nt_temp):
        T = temperature_step(T, v, w)

        if store_history:
            hist.append(T.copy())

    if store_history:
        return T, hist
    return T


def plot_temperature_comparison(Y, Z, T1, T2, ic_mode, title1, title2):
    """
    Plot two temperature fields side-by-side for comparison.
    """

    # Use same color scale for both plots
    vmin = min(T1.min(), T2.min())
    vmax = max(T1.max(), T2.max())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    # --- Left plot ---
    cf1 = axes[0].contourf(Y, Z, T1, levels=30, vmin=vmin, vmax=vmax)
    axes[0].set_title(title1)
    axes[0].set_xlabel("y")
    axes[0].set_ylabel("z")

    # --- Right plot ---
    cf2 = axes[1].contourf(Y, Z, T2, levels=30, vmin=vmin, vmax=vmax)
    axes[1].set_title(title2)
    axes[1].set_xlabel("y")
    axes[1].set_ylabel("z")

    # Shared colorbar
    fig.colorbar(cf2, ax=axes, label="T")
    plt.savefig(f"./imgs/rb_temp_compare_{ic_mode}.png", dpi=150)
    plt.show()

# ============================================================
# PREDICTOR STEP WITH BUOYANCY
# ============================================================

def predictor_step_boussinesq(v, w, T, Z):
    """
    Tentative velocity step for the Boussinesq system using
    temperature fluctuation relative to the conductive profile.
    """
    v, w = nsf.apply_velocity_bc(v, w)

    v_adv = nsf.advect_split_upwind(v, v, w)
    w_adv = nsf.advect_split_upwind(w, v, w)

    v_new = v_adv + nsf.dt * nsf.nu * nsf.laplacian(v)
    w_new = w_adv + nsf.dt * nsf.nu * nsf.laplacian(w)

    # buoyancy uses perturbation around conduction state
    T_fluct = T - (1.0 - Z)
    w_new = w_new + nsf.dt * beta * T_fluct

    v_new, w_new = nsf.apply_velocity_bc(v_new, w_new)
    return v_new, w_new

# ============================================================
# ONE RAYLEIGH-BENARD STEP
# ============================================================

def rayleigh_benard_step(v, w, T, Z):
    """
    One explicit Boussinesq / Rayleigh-Benard step:

      1) update temperature
      2) predictor for velocity with buoyancy
      3) pressure projection
    """
    # first update temperature using current velocity
    T_new = temperature_step(T, v, w)

    # tentative velocity with buoyancy
    v_star, w_star = predictor_step_boussinesq(v, w, T_new, Z)

    # standard Chorin projection
    rhs = (nsf.rho / nsf.dt) * nsf.divergence(v_star, w_star)
    p_new = nsf.pressure_poisson(rhs)

    v_new, w_new = nsf.projection_step(v_star, w_star, p_new)
    v_new, w_new = nsf.apply_velocity_bc(v_new, w_new)

    return v_new, w_new, p_new, T_new

# ============================================================
# FULL RAYLEIGH-BENARD SIMULATION
# ============================================================

def rayleigh_benard_simulation(nt_rb, v, w, p, T, Z, verbose_every=10, safety=0.2):
    """
    Run nt_rb steps of the coupled Rayleigh-Benard system.
    dt is updated adaptively from the current velocity field.
    """

    for n in range(nt_rb):
        # update dt using current velocity field
        nsf.dt = choose_dt(v, w, safety=safety)

        v, w, p, T = rayleigh_benard_step(v, w, T, Z)

        if not (np.isfinite(v).all() and np.isfinite(w).all() and np.isfinite(T).all() and np.isfinite(p).all()):
            print(f"Stopped at step {n}: non-finite values detected.")
            break

        # uncomment for debugging purposes
        # if n % verbose_every == 0:
        #     speed = np.sqrt(v**2 + w**2)
        #     ke = 0.5 * np.sum(v**2 + w**2) * nsf.dy * nsf.dz
        #     print(f"Step {n:4d}: "
        #           f"dt={nsf.dt:.3e}, "
        #           f"max|u|={np.max(speed):.4f}, "
        #           f"KE={ke:.6e}, "
        #           f"max|div|={np.max(np.abs(nsf.divergence(v, w))):.3e}, "
        #           f"T-range=[{np.min(T):.3f}, {np.max(T):.3f}]")

    return v, w, p, T

# ============================================================
# INITIAL TEMPERATURE FOR RBC
# ============================================================

def initial_temperature_rb(Y, Z, eps=0.05):
    """
    Conduction profile + small perturbation.
    """
    T = 1.0 - Z
    T += eps * np.cos(2*np.pi*Y/nsf.Ly) * np.sin(np.pi*Z/nsf.Lz)
    return apply_temperature_bc(T)

def plot_rb_state(y, z, v, w, T, ic_mode, title):
    Y, Z = np.meshgrid(y, z, indexing="ij")

    plt.figure(figsize=(6, 4))
    cf = plt.contourf(Y, Z, T, levels=30)
    plt.colorbar(cf, label="T")
    plt.quiver(Y[::3, ::3], Z[::3, ::3], v[::3, ::3], w[::3, ::3], color="white")
    plt.xlabel("y")
    plt.ylabel("z")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"./imgs/rb_state_{ic_mode}.png", dpi=150)
    plt.show()

# ============================================================
# RAYLEIGH-BENARD VISUALIZATION (updated)
# ============================================================

def plot_rb_diagnostics(y, z, Y, Z, v, w, T, ic_mode, title):
    """
    Show:
      1) full temperature T
      2) temperature fluctuation theta = T - (1-z)
      3) vertical velocity w
      4) streamlines of the velocity field
    """
    theta = T - (1.0 - Z)
    speed = np.sqrt(v**2 + w**2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    # --- full temperature ---
    cf1 = axes[0, 0].contourf(Y, Z, T, levels=30)
    fig.colorbar(cf1, ax=axes[0, 0])
    axes[0, 0].set_title("Full temperature T")
    axes[0, 0].set_xlabel("y")
    axes[0, 0].set_ylabel("z")

    # --- fluctuation temperature ---
    m = np.max(np.abs(theta))
    cf2 = axes[0, 1].contourf(Y, Z, theta, levels=30, vmin=-m, vmax=m, cmap="coolwarm")
    fig.colorbar(cf2, ax=axes[0, 1])
    axes[0, 1].set_title(r"Temperature fluctuation $\theta = T-(1-z)$")
    axes[0, 1].set_xlabel("y")
    axes[0, 1].set_ylabel("z")

    # --- vertical velocity ---
    m2 = np.max(np.abs(w))
    cf3 = axes[1, 0].contourf(Y, Z, w, levels=30, vmin=-m2, vmax=m2, cmap="coolwarm")
    fig.colorbar(cf3, ax=axes[1, 0])
    axes[1, 0].set_title("Vertical velocity w")
    axes[1, 0].set_xlabel("y")
    axes[1, 0].set_ylabel("z")

    # --- streamlines ---
    axes[1, 1].contourf(Y, Z, speed, levels=25, alpha=0.6)
    axes[1, 1].streamplot(y, z, v.T, w.T, density=1.4, color="k")
    axes[1, 1].set_title("Velocity streamlines")
    axes[1, 1].set_xlabel("y")
    axes[1, 1].set_ylabel("z")

    fig.suptitle(title, fontsize=14)
    plt.savefig(f"./imgs/rb_diagnostics_{ic_mode}.png", dpi=150)
    plt.show()

def animate_rb(ic_mode, nt_total=2000, save_every=50):
    # Project-standard grid
    y, z, Y, Z = nsf.create_grid()  # Ny, Nz defined globally in project

    # Initialize fields
    v = np.zeros((nsf.Ny, nsf.Nz))
    w = np.zeros((nsf.Ny, nsf.Nz))
    p = np.zeros((nsf.Ny, nsf.Nz))
    T = 1.0 - Z + 0.01 * np.random.randn(*Z.shape)  # small perturbation

    # Set up figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    def update(frame):
        nonlocal v, w, p, T

        # Clear previous plots
        for ax in axes.flat:
            ax.clear()

        # Advance simulation 
        for _ in range(save_every):
            nsf.dt = choose_dt(v, w, safety=0.4)  # adaptive timestep
            v, w, p, T = rayleigh_benard_step(v, w, T, Z)

        # Derived fields
        theta = T - (1.0 - Z)
        speed = np.sqrt(v**2 + w**2)

        #  Panel 1: Full Temperature
        axes[0,0].contourf(Y, Z, T, levels=30)
        axes[0,0].set_title("Full Temperature T")
        axes[0,0].set_xlabel("y")
        axes[0,0].set_ylabel("z")

        # Panel 2: Temperature Fluctuation
        m = np.max(np.abs(theta)) if np.max(np.abs(theta))>0 else 1e-5
        axes[0,1].contourf(Y, Z, theta, levels=30, vmin=-m, vmax=m, cmap="coolwarm")
        axes[0,1].set_title("Temperature Fluctuation θ")
        axes[0,1].set_xlabel("y")
        axes[0,1].set_ylabel("z")

        # Panel 3: Vertical Velocity
        m2 = np.max(np.abs(w)) if np.max(np.abs(w))>0 else 1e-5
        axes[1,0].contourf(Y, Z, w, levels=30, vmin=-m2, vmax=m2, cmap="coolwarm")
        axes[1,0].set_title("Vertical Velocity w")
        axes[1,0].set_xlabel("y")
        axes[1,0].set_ylabel("z")

        # Panel 4: Flow speed + streamlines
        axes[1,1].contourf(Y, Z, speed, levels=25, alpha=0.6)
        axes[1,1].streamplot(y, z, v.T, w.T, color='k', density=1.0)
        axes[1,1].set_title("Flow Speed & Streamlines")
        axes[1,1].set_xlabel("y")
        axes[1,1].set_ylabel("z")

    # Create animation 
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=nt_total // save_every,
        interval=100,
        blit=False
    )

    ani.save(f"./imgs/rb_animation_{ic_mode}.gif", writer="pillow", fps=10)

    plt.close()

# ============================================================
# MAIN
# ============================================================

def main(v, w, p, y, z, Y, Z, ic_mode="mode1", ic_custom_v=None, ic_custom_w=None):
    print("=" * 55)
    print("3.4 RAYLEIGH-BENARD CONVECTION")
    print("=" * 55)
    print()

    nsf.dt = choose_dt(v, w, safety=0.4)

    print(f"Timestep: dt={nsf.dt}")
    print(f"CFL     : y={np.max(np.abs(v))*nsf.dt/nsf.dy:.3f}  z={np.max(np.abs(w))*nsf.dt/nsf.dz:.3f}")
    print(f"max|v|={np.max(np.abs(v)):.4f}  max|w|={np.max(np.abs(w)):.4f}")
    print()

    # initial temperature
    # perturbation=0.3 is intentionally large here so the diffusive smoothing
    # is visible on the shared [0,1] colorscale — for physical runs use ~0.01
    T0 = initial_temperature(Y, Z, perturbation=0.3)
    T0copy = T0.copy()

    # evolve temperature with the current velocity field
    T_final = temperature_simulation(nt_temp=5000, v=v, w=w, Y=Y, Z=Z, T0=T0)

    # plot result
    plot_temperature_comparison(
        Y, Z,
        T0copy, T_final, ic_mode,
        title1="Initial temperature",
        title2="Temperature after advection-diffusion"
    )
    T = initial_temperature_rb(Y, Z, eps=1e-1)

    # run coupled simulation
    v, w, p, T = rayleigh_benard_simulation(nt_rb=10000, v=v, w=w, p=p, T=T, Z=Z, verbose_every=500, safety=0.4)

    # final divergence check
    div_final = nsf.divergence(v, w)

    max_div = np.max(np.abs(div_final))
    mean_div = np.mean(np.abs(div_final))

    print(f"\nFINAL DIVERGENCE CHECK:")
    print(f"Max divergence  = {max_div:.3e}")
    print(f"Mean divergence = {mean_div:.3e}")

    plt.figure()
    plt.contourf(div_final.T, levels=30)
    plt.title("Final Divergence Check")
    plt.colorbar()
    plt.savefig(f"./imgs/rb_divergence_{ic_mode}.png", dpi=150)
    plt.show()
    
    # plot final state
    plot_rb_state(y, z, v, w, T, ic_mode, title=f"Rayleigh-Benard state, beta={beta}")

    plot_rb_diagnostics(y, z, Y, Z, v, w, T, ic_mode, title=f"RB diagnostics, beta={beta}, nu={nsf.nu}")

    animate_rb(ic_mode, nt_total=2000, save_every=50)

if __name__ == "__main__":
    main()
