'''
3.2 Introduction
Comp_Sci_Navier_Stokes_Equations.src.Introduction

We aim to solve the quite involved equations by splitting the terms into sub problems of easier type. These are then
approximated by finite-difference methods on a cartesian grid.
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

'''
OPTIONAL: Implement a semi-lagrangian scheme for the two dimensional linear advection equation with variable advection
field. Such a scheme tracks the characteristic solutions back in time and subsequently interpolates the field at the old
time point to obtain the values transported along the charateristics.
'''

'''
Find suitable testcases, for which you can find analytical solutions and compare them with your results. Therefore
visualize the analytical/ approximated solutions and explain what you observe. Numerically test the consistency order 
of you implementation(s).
'''

import NavierStokesFunctions as nsf
import numpy as np
import matplotlib.pyplot as plt

# Constant advection field
ay_val = 0.5
az_val = 0.3

t_final = 0.5

def upwind_simulation(v, w, dt, n_steps):
    # Time Stepping Loop
    for _ in range(n_steps):
        # Operator splitting: Update y then update z sequentially
        v = nsf.upwind_step(v, ay_val, nsf.dy, dt, axis=0, periodic=True)
        w = nsf.upwind_step(w, ay_val, nsf.dy, dt, axis=0, periodic=True)
        
        v = nsf.upwind_step(v, az_val, nsf.dz, dt, axis=1, periodic=False)
        w = nsf.upwind_step(w, az_val, nsf.dz, dt, axis=1, periodic=False)
        
        # Enforce Dirichlet BCs for w 
        w[:, 0] = 0.0
        w[:, -1] = 0.0
    return v, w

def analytical_solution(Y, Z, dt, n_steps, mode, ic_custom_v=None, ic_custom_w=None):
    t_reached = n_steps * dt
    Y_shift = Y - ay_val * t_reached
    Z_shift = Z - az_val * t_reached

    if mode == "custom":
        if ic_custom_v is None or ic_custom_w is None:
            raise ValueError(
                "mode='custom' requires ic_custom_v and ic_custom_w."
            )
        v_exact = ic_custom_v(Y_shift, Z_shift, nsf.ky, nsf.kz)
        w_exact = ic_custom_w(Y_shift, Z_shift, nsf.ky, nsf.kz)
        label = "v = custom_v(Y,Z,ky,kz), w = custom_w(Y,Z,ky,kz)"

    elif mode in nsf.IC_REGISTRY:
        mode_entry = nsf.IC_REGISTRY[mode]
        v_exact = mode_entry["v_fn"](Y_shift, Z_shift, nsf.ky, nsf.kz)
        w_exact = mode_entry["w_fn"](Y_shift, Z_shift, nsf.ky, nsf.kz)
        label = mode_entry["label"]

    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Available modes: {list(nsf.IC_REGISTRY.keys())} plus 'custom'"
        )

    # enforce BCs on analytical w
    w_exact[:, 0] = w_exact[:, -1] = 0.0 
    return v_exact, w_exact, label

def plot_results(v, w, v_exact, w_exact, mode, label, dt):
    fig = plt.figure(figsize=(12, 8))

    fig.suptitle(
        "2D Linear Advection (Upwind Scheme)\n"
        f"Numerical vs Analytical | nt={nsf.nt}, dt={dt:.2e}\n"
        f"Initial Conditions: {label}",
        fontsize=11,
        y=0.98
    )

    vmin_v = min(v.min(), v_exact.min())
    vmax_v = max(v.max(), v_exact.max())
    vmin_w = min(w.min(), w_exact.min())
    vmax_w = max(w.max(), w_exact.max())
    extent = [0.0, nsf.Ly, 0.0, nsf.Lz]

    # --- v component ---
    plt.subplot(2, 2, 1)
    plt.imshow(v.T, origin='lower', extent=extent, vmin=vmin_v, vmax=vmax_v) 
    plt.xlabel('y')
    plt.ylabel('z')
    plt.colorbar()
    plt.title("Numerical Velocity v")

    plt.subplot(2, 2, 2)
    plt.imshow(v_exact.T, origin="lower", extent=extent, vmin=vmin_v, vmax=vmax_v)
    plt.xlabel('y')
    plt.ylabel('z')
    plt.colorbar()
    plt.title("Analytical Velocity v")

    # --- w component ---
    plt.subplot(2, 2, 3)
    plt.imshow(w.T, origin="lower", extent=extent, vmin=vmin_w, vmax=vmax_w)
    plt.xlabel('y')
    plt.ylabel('z')
    plt.colorbar()
    plt.title("Numerical Velocity w")

    plt.subplot(2, 2, 4)
    plt.imshow(w_exact.T, origin="lower", extent=extent, vmin=vmin_w, vmax=vmax_w)
    plt.xlabel('y')
    plt.ylabel('z')
    plt.colorbar()
    plt.title("Analytical Velocity w")

    plt.tight_layout()
    plt.savefig(f"./imgs/upwind_{mode}.png", dpi=150)
    plt.show()

def main(v, w, Y, Z, ic_mode="mode1", ic_custom_v=None, ic_custom_w=None):
    print("=" * 55)
    print(f"3.2 INTRODUCTION")
    print("=" * 55)

    w[:, 0] = w[:, -1] = 0.0
    dt = 0.5 * min(nsf.dy / abs(ay_val), nsf.dz / abs(az_val))
    n_steps = int(t_final / dt)

    print(f"Timestep    : dt={dt}")
    print(f"Steps       : {n_steps}  (t_final={t_final})")
    print()
    
    v, w = upwind_simulation(v, w, dt, n_steps)
    v_exact, w_exact, label = analytical_solution(Y, Z, dt, n_steps, ic_mode, ic_custom_v, ic_custom_w)
    plot_results(v, w, v_exact, w_exact, ic_mode, label, dt)

    # --- Error ---
    error_v = np.max(np.abs(v - v_exact))
    error_w = np.max(np.abs(w - w_exact))

    print("Max error in v:", error_v)
    print("Max error in w:", error_w) 
    print()

if __name__ == "__main__":
    main()