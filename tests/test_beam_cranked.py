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
beam.add_nodes([[i/10, 0, 0] for i in range(10)] + [[1.0, i/10, 0] for i in range(11)])

# Add material and section (UB127x76x13) 
beam.add_material("material", LinearElastic(E=1, G=1)) 
beam.add_section("section", Section(A=1, Iy=1, Iz=1, J=1, Iw=1, y0=0, z0=0)) 

# Add element 
beam.add_elements([[i, i+1] for i in range(20)], "material", "section", element_class=ThinWalledBeamElement) 

# Add boundary conditions and loads
beam.add_boundary_condition(0, [0, 0, 0, 0, 0, 0, 1], BoundaryCondition) 
beam.add_boundary_condition(20, [0, 0, 0, 1, 1, 1, 1], BoundaryCondition) 
beam.add_nodal_load(10, [0, 0, -1, 0, 0, 0, 0], NodalLoad) 

# Solve the model
results = beam.solve() 
print(results.get_nodal_forces(0))
results = beam.show() 
