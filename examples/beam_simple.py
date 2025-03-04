from pybeamnlfea.model.frame import Frame
from pybeamnlfea.model.material import LinearElastic
from pybeamnlfea.model.section import Section 
from pybeamnlfea.model.element import ThinWalledBeamElement 

# Author: James Whiteley (github.com/jamesalexwhiteley)

# Create a beam structure
beam = Frame()

# Add nodes
beam.add_node(0, 0, 0)
beam.add_node(1, 0, 0)

# Add material and section (UB127x76x13)
beam.add_material("mat1", LinearElastic(210e9, 0.3))
beam.add_section("sec1", Section(A=16.5e-4, Ix=74.6e-6, Iy=14.7e-4, J=2.85e-8, Iw=2e-9))

# Add element 
beam.add_element([1, 2], "mat1", "sec1", element_class=ThinWalledBeamElement)

print(beam.nodes)
print(beam.elements)

# # Add boundary conditions and loads
# beam.add_boundary_condition(1, {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0})  # Fixed support
# beam.add_load(2, [0, -10000, 0, 0, 0, 0])  # Vertical load at node 2

# # Solve the model
# results = beam.solve()

# # Get results
# disp_node2 = beam.get_displacement_at_node(2)
# print(f"Displacement at node 2: {disp_node2}")