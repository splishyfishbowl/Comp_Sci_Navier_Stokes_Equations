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
Do the implementation of the solver subsequently.
'''

'''
Implement Chorin's projection method [1] for the Navier Stokes equations i.e. (2) - (3) with β = 0 using the numerical
methods built so far. Implement initial conditions to test your code, check the consistency order and visualize the data.
(2)     ∂_t~u + ~u · ∇~u + ∇p/ρ = ν∆~u + T βe_z
(3)     ∇ · ~u = 0
'''