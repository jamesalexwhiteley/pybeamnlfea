import numpy as np
from pybeamnlfea.model.frame import Frame
from pybeamnlfea.model.material import LinearElastic
from pybeamnlfea.model.section import Section 
from pybeamnlfea.model.element import ThinWalledBeamElement 
from pybeamnlfea.model.boundary import BoundaryCondition 
from pybeamnlfea.model.load import NodalLoad 

# Author: James Whiteley (github.com/jamesalexwhiteley)

# Create a beam structure 
n = 10
L = 4 # m
beam = Frame() 
beam.add_nodes([[L*i/n, 0, 0] for i in range(n+1)])

# Steel properties
E = 210e9   # N/m2
G = 80e9    # N/m2 
rho = 7850  # kg/m3

# UB127x76x13 section properties
A = 1650e-6    # m2
Iy = 4.73e-6   # m4 
Iz = 0.557e-6  # m4
J = 0.0285e-6  # m4
Iw = 0.002e-6  # m6 

beam.add_material("steel", LinearElastic(rho=rho, E=E, G=G))
beam.add_section("UB127x76x13", Section(A=A, Iy=Iy, Iz=Iz, J=J, Iw=Iw, y0=0, z0=0))

# Add element 
beam.add_elements([[i, i+1] for i in range(n)], "steel", "UB127x76x13", element_class=ThinWalledBeamElement) 

# Add boundary conditions; Global (ux, uy, uz, θx, θy, θz, φ); 0=fixed, 1=free 
beam.add_boundary_condition(0, [0, 0, 0, 0, 1, 1, 0], BoundaryCondition) 
beam.add_boundary_condition(n, [0, 0, 0, 0, 1, 1, 0], BoundaryCondition)

# Add loads; Global (ux, uy, uz, θx, θy, θz, φ)
# beam.add_nodal_load(n/2, [0, 0, -1e3, 0, 0, 0, 0], NodalLoad) # N 
beam.add_gravity_load()

# Midspan deflection 
disp_analytic = -5 * (rho * A * 9.81) * L**4 / (384 * E * Iy) # m 

# Solve the model 
results = beam.solve() 
print(results.get_nodal_displacements(n/2))
disp_fea = results.get_nodal_displacements(n/2)[2] 
print(f"disp analytic = {disp_analytic * 1000:.4e} mm | disp fea = {disp_fea * 1000:.4e} mm | error = {(np.abs(disp_fea - disp_analytic)) / disp_analytic * 100:.4f} %") 
beam.show_deformed_shape(scale=5e2, cross_section_scale=2) 