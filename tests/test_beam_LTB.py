from pybeamnlfea.model.frame import Frame
from pybeamnlfea.model.material import LinearElastic
from pybeamnlfea.model.section import Section 
from pybeamnlfea.model.element import ThinWalledBeamElement 
from pybeamnlfea.model.boundary import BoundaryCondition 
from pybeamnlfea.model.load import UniformLoad, NodalLoad

# Author: James Whiteley (github.com/jamesalexwhiteley) 

# Create a beam structure 
n = 10
L = 8.0 # m 
beam = Frame() 
beam.add_nodes([[i*L/n, 0, 0] for i in range(n+1)])

# Material and section properties for steel UB127x76x13
# Steel properties
E = 210e9   # Pa
G = 80e9    # Pa
rho = 7850  # kg/m³

# UB127x76x13 section properties
A = 1662e-6   # m2
Iy = 4.98e-6  # m4
Iz = 0.65e-6  # m4
J = 0.025e-6  # m4
Iw = 0.63e-9  # m6

beam.add_material("steel", LinearElastic(rho=rho, E=E, G=G))
beam.add_section("UB127x76x13", Section(A=A, Iy=Iy, Iz=Iz, J=J, Iw=Iw, y0=0, z0=0))

# Add element 
beam.add_elements([[i, i+1] for i in range(n)], "steel", "UB127x76x13", element_class=ThinWalledBeamElement) 

# Add boundary conditions 
beam.add_boundary_condition(0, [0, 0, 0, 1, 1, 1, 0], BoundaryCondition) 
beam.add_boundary_condition(n, [0, 0, 0, 1, 1, 1, 0], BoundaryCondition) 

# Add gravity load
beam.add_gravity_load([0, 0, -100])

# moment_magnitude = 1e6  # Nm
# beam.add_nodal_load(0, [0, 0, 0, 0, moment_magnitude, 0, 0], NodalLoad)
# beam.add_nodal_load(n, [0, 0, 0, 0, -moment_magnitude, 0, 0], NodalLoad)

# Linear analysis 
beam.solve() 
beam.show() 

# Run eigenvalue buckling analysis
print("Running eigenvalue buckling analysis...")
eigenvalues, eigenvectors = beam.eigen_solve(num_modes=5)  # Calculate first 5 buckling modes
print("Buckling eigenvalues (load factors):")
for i, val in enumerate(eigenvalues):
    print(f"Mode {i+1}: {val}")

print("\nAnalyzing buckling modes to identify LTB:")
for mode in range(len(eigenvalues)):
    # Extract the mode shape data
    mode_data = beam.get_mode_shape(mode)
    max_lateral_disp = max(abs(node_data[1]) for node_data in mode_data)
    max_torsion = max(abs(node_data[3]) for node_data in mode_data)
    
    # If both lateral displacement and torsional rotation are significant, classify as LTB
    print(f"Mode {mode+1}:")
    print(f"  Max lateral displacement: {max_lateral_disp:.4e}")
    print(f"  Max torsional rotation: {max_torsion:.4e}")
    
    if max_lateral_disp > 1e-6 and max_torsion > 1e-6:
        print(f"  ✓ Mode {mode+1} exhibits characteristics of lateral torsional buckling")
    else:
        print(f"  ✗ Mode {mode+1} does not appear to be a lateral torsional buckling mode")

# Visualization of the critical buckling mode 
beam.plot_mode_shape(0, scale=0.5)