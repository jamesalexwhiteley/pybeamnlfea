from pybeamnlfea.model.frame import Frame
from pybeamnlfea.model.material import LinearElastic
from pybeamnlfea.model.section import Section 
from pybeamnlfea.model.element import ThinWalledBeamElement 
from pybeamnlfea.model.boundary import BoundaryCondition 
from pybeamnlfea.model.load import NodalLoad 
import numpy as np

# Author: James Whiteley (github.com/jamesalexwhiteley) 

# Create a beam structure 
n = 10
L = 1
beam = Frame() 
beam.add_nodes([[L*i/n, 0, 0] for i in range(n+1)])

# # Steel properties
# E = 210e9   # N/m2
# G = 80e9    # N/m2 
# rho = 7850  # kg/m3

# # UB127x76x13 section properties
# A = 1650e-6    # m2
# Iy = 4.73e-6   # m4 
# Iz = 0.557e-6  # m4
# J = 0.0285e-6  # m4 
# Iw = 0.002e-6  # m6   

E, G, rho = 1, 1, 1  
rho, E, G = 1, 1, 1
A, Iy, Iz, J, Iw = 1, 1, 1, 1, 1

beam.add_material("steel", LinearElastic(rho=rho, E=E, G=G))
beam.add_section("UB127x76x13", Section(A=A, Iy=Iy, Iz=Iz, J=J, Iw=Iw, y0=0, z0=0))

# Add element 
beam.add_elements([[i, i+1] for i in range(n)], "steel", "UB127x76x13", element_class=ThinWalledBeamElement) 

# Add boundary conditions and loads
beam.add_boundary_condition(0, [0, 0, 0, 0, 1, 1, 1], BoundaryCondition) 
beam.add_boundary_condition(n, [1, 0, 0, 0, 1, 1, 1], BoundaryCondition) 
[beam.add_boundary_condition(i+1, [1, 1, 0, 0, 1, 1, 0], BoundaryCondition) for i in range(n-1)]  # torsional and major axis modes constrained 
beam.add_nodal_load(n, [-1, 0, 0, 0, 0, 0, 0], NodalLoad) 

# Linear buckling analysis 
eigenvalues, eigenvectors = beam.solve_eigen(num_modes=3) 
for n in range(len(eigenvalues)):
    load_analytic = (n+1)**2 * np.pi**2 / L**2 * E * min(Iz,Iy)
    error = (np.abs(eigenvalues[n] - load_analytic)) / load_analytic * 100
    print(f"mode {n+1}: buckling analytic = {load_analytic:.4e} | buckling fea = {eigenvalues[n]:.4e} N | error {error:.4f} %")
    beam.show_mode_shape(eigenvectors[n], scale=3, show_local_axes=True, cross_section_scale=3/4)
