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

n = 15
radius = 1 
theta_max = np.pi/2 
nodes = []
for i in range(n + 1):
    theta = (i / n) * theta_max  
    x = radius * np.cos(theta)  
    y = radius * np.sin(theta)  
    nodes.append([x, y, 0])  
beam.add_nodes(nodes)

# Add material and section (UB127x76x13) 
beam.add_material("material", LinearElastic(rho=1, E=1, G=1)) 
beam.add_section("section", Section(A=1, Iy=1, Iz=1, J=1, Iw=1, y0=0, z0=0)) 

# Add element 
beam.add_elements([[i, i+1] for i in range(n)], "material", "section", element_class=ThinWalledBeamElement) 

# Add boundary conditions and loads
beam.add_boundary_condition(0, [0, 0, 0, 0, 0, 0, 1], BoundaryCondition) 
beam.add_nodal_load(n, [0, 0, -1, 0, 0, 0, 0], NodalLoad) 

# End deflection 
disp_analytic = -1 * (np.pi / (4 * 1) + (3 * np.pi / 4 - 2) / (1))

# Solve the model 
results = beam.solve() 
disp_fea = results.get_nodal_displacements(n)[2]
print(f"disp analytic = {disp_analytic * 1000:.4e} mm | disp fea = {disp_fea * 1000:.4e} mm | error = {(np.abs(disp_fea - disp_analytic)) / disp_analytic * 100:.4f} %") 
results = beam.show_deformed_shape(scale=0.15, cross_section_scale=0.75) 

