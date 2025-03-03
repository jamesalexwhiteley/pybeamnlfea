from pybeamnlfea.model.frame import Frame

# Author: James Whiteley (github.com/jamesalexwhiteley)

# Create a beam structure
beam = Frame()

# Add nodes
beam.add_node(1, 0, 0, 0)
beam.add_node(2, 1000, 0, 0)

# Add material and section
beam.add_material("steel", 210000, 0.3)
beam.add_section("I-beam", {"area": 100, "Iy": 1000, "Iz": 2000, "J": 50})

# Add element
beam.add_element([1, 2], "steel", "I-beam")

# Add boundary conditions and loads
beam.add_boundary_condition(1, {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0})  # Fixed support
beam.add_load(2, [0, -10000, 0, 0, 0, 0])  # Vertical load at node 2

# Solve the model
results = beam.solve()

# Get results
disp_node2 = beam.get_displacement_at_node(2)
print(f"Displacement at node 2: {disp_node2}")