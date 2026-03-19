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

Initial condition modes (IC_MODE):
  "mode1"  — v =  sin(ky y) cos(kz z),   w = -(ky/kz) cos(ky y) sin(kz z)
  "mode2"  — v =  sin(2ky y) cos(kz z),  w = -(2ky/kz) cos(2ky y) sin(kz z)
  "mode3"  — v =  sin(ky y) cos(2kz z),  w = -(ky/2kz) cos(ky y) sin(2kz z)
  "double" — superposition of mode1 and mode2 (linearly combined)
  "custom" — user-supplied functions via IC_CUSTOM_V and IC_CUSTOM_W
             (see main() for how to provide them)
'''

import NavierStokesFunctions as nsf
import numpy as np
import matplotlib.pyplot as plt

def choose_dt(v, w, safety=0.4):
    '''
    Pick a stable explicit time step from:
      - advection CFL: max|v| dt/dy <= 1, max|w| dt/dz <= 1
      - diffusion bound: dt <= 1 / (2*nu*(1/dy^2 + 1/dz^2))
    safety < 1 adds margin.
    '''
    vmax = max(np.max(np.abs(v)), 1e-14)
    wmax = max(np.max(np.abs(w)), 1e-14)

    dt_adv = safety * min(nsf.dy / vmax, nsf.dz / wmax)
    dt_diff = safety * 0.5 / (nsf.nu * (1.0/nsf.dy**2 + 1.0/nsf.dz**2))

    return min(dt_adv, dt_diff)

# ============================================================
# PLOTTING
# ============================================================

def plot_results(y, z, v, w, p, ic_mode="mode1", ic_label=""):
    '''
    Produce a three-panel figure:
      Left   — speed |u| as a filled contour with a velocity quiver overlay.
      Center — pressure field p.
      Right  — discrete divergence ∇·u (should be small after projection).

    ic_label is printed as a subtitle so the velocity formula is visible on the plot.
    Saves the figure to ./imgs/incompressible_flow_dt<dt>.png and displays it.
    '''
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Main title: timestep info + IC formula as subtitle
    fig.suptitle(
        f"Incompressible Flow Simulation  |  dt = {nsf.dt:.4e}  |  timesteps = {nsf.nt}\n"
        f"{ic_label}",
        fontsize=11,
        y=0.98
    )

    speed = np.sqrt(v**2 + w**2)
    im0 = axes[0].contourf(y, z, speed.T, levels=20, cmap='viridis')
    axes[0].quiver(y[::4], z[::4], v[::4, ::4].T, w[::4, ::4].T,
                   color='white', scale=10)
    plt.colorbar(im0, ax=axes[0], label="|u|")
    axes[0].set(xlabel="y", ylabel="z", title="Speed & Velocity Field")

    im1 = axes[1].contourf(y, z, p.T, levels=20, cmap='RdBu_r')
    plt.colorbar(im1, ax=axes[1], label="p")
    axes[1].set(xlabel="y", ylabel="z", title="Pressure")

    div = nsf.divergence(v, w)
    im2 = axes[2].contourf(y, z, div.T, levels=20, cmap='RdBu_r')
    plt.colorbar(im2, ax=axes[2], label="∇·u")
    axes[2].set(xlabel="y", ylabel="z", title=f"Divergence (max={np.max(np.abs(div)):.2e})")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f"./imgs/incompressible_flow__{ic_mode}.png", dpi=150)
    plt.show()

# ============================================================
# MAIN
# ============================================================

def main(v, w, p, y, z, ic_mode="mode1", ic_custom_v=None, ic_custom_w=None):
    print("=" * 55)
    print(f"3.3 INCOMPRESSIBLE FLOW")
    print("=" * 55)
    print()

    nsf.dt = choose_dt(v, w)

    print(f"Timestep: dt={nsf.dt}")
    print(f"CFL     : y={np.max(np.abs(v))*nsf.dt/nsf.dy:.3f}  z={np.max(np.abs(w))*nsf.dt/nsf.dz:.3f}")
    print()

    v, w, p = nsf.incompressible_flow_simulation(v, w, p)

    print(f"\nFinal max|∇·u| = {np.max(np.abs(nsf.divergence(v, w))):.3e}")

    if ic_mode == "custom":
        label = "v = custom_v(Y,Z,ky,kz), w = custom_w(Y,Z,ky,kz)"
    else:
        label = nsf.IC_REGISTRY[ic_mode]["label"]
    plot_results(y, z, v, w, p, ic_mode, label)


if __name__ == "__main__":
    main()