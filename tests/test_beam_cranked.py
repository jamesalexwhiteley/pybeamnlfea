import numpy as np
from pybeamnlfea.model.frame import Frame
from pybeamnlfea.model.material import LinearElastic
from pybeamnlfea.model.section import Section 
from pybeamnlfea.model.element import ThinWalledBeamElement 
from pybeamnlfea.model.boundary import BoundaryCondition 
from pybeamnlfea.model.load import NodalLoad 

# Author: James Whiteley (github.com/jamesalexwhiteley)

# Create a beam structure 
n = 5
beam = Frame() 
beam.add_nodes([[i/n, 0, 0] for i in range(n)] + [[1.0, i/n, 0] for i in range(n+1)])

# Add material and section (UB127x76x13) 
beam.add_material("material", LinearElastic(rho=1, E=1, G=1)) 
beam.add_section("section", Section(A=1, Iy=1, Iz=1, J=1, Iw=1, y0=0, z0=0)) 

# Add element 
beam.add_elements([[i, i+1] for i in range(n*2)], "material", "section", element_class=ThinWalledBeamElement) 

# Add boundary conditions and loads
beam.add_boundary_condition(0, [0, 0, 0, 0, 0, 0, 1], BoundaryCondition) 
beam.add_boundary_condition(n*2, [0, 0, 0, 1, 1, 1, 1], BoundaryCondition) 
beam.add_nodal_load(n, [0, 0, -1, 0, 0, 0, 0], NodalLoad) 

# Solve the model
results = beam.solve() 
moment_fea = results.get_nodal_forces(0)[4]
moment_analytic = -5/7  # test_beam.pdf 
print(f"moment analytic = {moment_analytic:.4e} Nm | moment fea = {moment_fea:.4e} Nm | error = {(np.abs(moment_fea - moment_analytic)) / moment_analytic * 100:.4f} %") 
beam.show_deformed_shape(scale=1, cross_section_scale=0.75) 