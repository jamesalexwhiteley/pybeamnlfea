import numpy as np
from pybeamnlfea.model.frame import Frame
from pybeamnlfea.model.material import LinearElastic
from pybeamnlfea.model.section import Section 
from pybeamnlfea.model.element import ThinWalledBeamElement 
from pybeamnlfea.model.boundary import BoundaryCondition 
from pybeamnlfea.model.load import NodalLoad 

# Author: James Whiteley (github.com/jamesalexwhiteley)

# Create a beam structure 
beam = Frame() 
nodes = [
    [0.00, 0.0, 1.50 ],   # 0: left tip (5kN load)
    [1.25, 0.0, 1.50 ],   # 1: top chord
    [2.50, 0.0, 1.50 ],   # 2: top chord
    [3.75, 0.0, 1.50 ],   # 3: top chord
    [5.00, 0.0, 1.50 ],   # 4: top right — R1 (pin)
    [1.25, 0.0, 1.125],   # 5: bottom chord
    [2.50, 0.0, 0.75 ],   # 6: bottom chord
    [3.75, 0.0, 0.375],   # 7: bottom chord
    [5.00, 0.0, 0.00 ],   # 8: bottom right — R2 (roller)
]
beam.add_nodes([node for node in nodes])

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
elements = [
    [0, 1], [1, 2], [2, 3], [3, 4], # Top chord 
    [0, 5], [5, 6], [6, 7], [7, 8], # Bottom chord 
    [4, 8],                         # Right side 
    [1, 5], [2, 6], [3, 7],         # Verticals 
    [1, 6], [2, 7], [3, 8],         # Diagonals 
]
beam.add_elements([element for element in elements], "steel", "UB127x76x13", element_class=ThinWalledBeamElement) 

# Add boundary conditions; Global (ux, uy, uz, θx, θy, θz, φ); 0=fixed, 1=free 
beam.add_boundary_condition(4, [0, 0, 0, 0, 0, 0, 0], BoundaryCondition) 
beam.add_boundary_condition(8, [0, 0, 0, 0, 0, 0, 0], BoundaryCondition)

# Add loads; Global (ux, uy, uz, θx, θy, θz, φ)
beam.add_nodal_load(0, [0, 0, -5e3, 0, 0, 0, 0], NodalLoad) # N 
# beam.add_gravity_load()

# Solve the model 
results = beam.solve() 
print(results.get_nodal_forces(2))
# disp_fea = results.get_nodal_displacements(n/2)[2] 
# print(f"disp analytic = {disp_analytic * 1000:.4e} mm | disp fea = {disp_fea * 1000:.4e} mm | error = {(np.abs(disp_fea - disp_analytic)) / disp_analytic * 100:.4f} %") 
beam.show_deformed_shape(scale=1e2, cross_section_scale=1.5) 