from pybeamnlfea.model.frame import Frame
from pybeamnlfea.model.material import LinearElastic
from pybeamnlfea.model.section import Section 
from pybeamnlfea.model.element import ThinWalledBeamElement 
from pybeamnlfea.model.boundary import BoundaryCondition 
from pybeamnlfea.model.load import UniformLoad, NodalLoad
import numpy as np

# Author: James Whiteley (github.com/jamesalexwhiteley) 

# Create a beam structure 
n = 20
L = 3/2 # m 
beam = Frame() 
beam.add_nodes([[i*L/n, 0, 0] for i in range(n+1)])
# beam.add_nodes([[0, i*L/n, 0] for i in range(n+1)])

# Steel properties
E = 210e9   # N/m2
G = 80e9    # N/m2 
rho = 7850  # kg/m3

# # Rectangular section (no warping) 
# b, d = 0.05, 0.5 # m 
# A = b*d          # m2
# Iy = b*d**3/12   # m4
# Iz = b**3*d/12   # m4
# J = d**3/3 * (b-0.63*d*(1-d**4/(12*b**4))) # m4
# Iw = 0           # m6  

# UB127x76x13 section (warping)
A = 1650e-6    # m2
Iy = 4.73e-6   # m4 
Iz = 0.557e-6  # m4
J = 0.0285e-6  # m4
Iw = 0.002e-6  # m6    

# E, G, rho = 1, 1, 1  
# rho, E, G = 1, 1, 1
# A, Iy, Iz, J, Iw = 1, 1, 1, 1, 1

beam.add_material("steel", LinearElastic(rho=rho, E=E, G=G))
beam.add_section("rectangular", Section(A=A, Iy=Iy, Iz=Iz, J=J, Iw=Iw, y0=0, z0=0))

# Add element 
beam.add_elements([[i, i+1] for i in range(n)], "steel", "rectangular", element_class=ThinWalledBeamElement) 

# Add boundary conditions 
beam.add_boundary_condition(0, [0, 0, 0, 0, 1, 1, 1], BoundaryCondition)
beam.add_boundary_condition(n, [0, 0, 0, 0, 1, 1, 1], BoundaryCondition)

# Add loads (pure bending)
beam.add_nodal_load(0, [0, 0, 0, 0, 1, 0, 0], NodalLoad)
beam.add_nodal_load(n, [0, 0, 0, 0, -1, 0, 0], NodalLoad)
# beam.add_nodal_load(0, [0, 0, 0, 0, 0, 1, 0], NodalLoad)
# beam.add_nodal_load(n, [0, 0, 0, 0, 0, -1, 0], NodalLoad)

# # Linear solver 
# results = beam.solve() 
# beam.show_deformed_shape(scale=1e-3, cross_section_scale=5) 

# Linear eigenvalue analysis 
eigenvalues, eigenvectors = beam.solve_eigen(num_modes=1) 
for n in range(len(eigenvalues)):
    load_analytic = (n+1) * (np.pi / L) * np.sqrt(E * Iz * G * J) * np.sqrt(1 + (np.pi / L)**2 * (E * Iw / (G * J)))  # analytic solution (warping) 
    # load_analytic = (n+1) * (np.pi / L) * np.sqrt(E * Iy * G * J) * np.sqrt(1 + (np.pi / L)**2 * (E * Iw / (G * J)))  # analytic solution (warping)  
    error = (np.abs(eigenvalues[n] - load_analytic)) / load_analytic * 100
    print(f"mode {n+1}: m0cr analytic = {load_analytic:.4e} | m0cr fea {eigenvalues[n]:.4e} | error = {error:.2f} %") 
    beam.show_mode_shape(eigenvectors[n], scale=5, cross_section_scale=3/4)