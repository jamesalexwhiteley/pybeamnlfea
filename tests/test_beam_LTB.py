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
E = 210e9   # N/m2
G = 80e9    # N/m2 
rho = 7850  # kg/m3

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

# # Linear analysis 
# beam.solve() 
# beam.show() 

# Run eigenvalue buckling analysis 
print("Running eigenvalue buckling analysis...")
eigenvalues, eigenvectors = beam.solve_eigen(num_modes=20) 
print("Buckling eigenvalues (load factors):")
for i, val in enumerate(eigenvalues):
    print(f"Mode {i+1}: {val}")

print("\nAnalyzing buckling modes to identify LTB:")
for mode in range(len(eigenvalues)):
    # Extract the mode shape data
    mode_data = beam.buckling_modes[mode]

    # Maximum (minor axis) lateral displacement 
    lateral_disps = [abs(mode_data.get((node, 1), 0)) for node in range(n+1)]
    max_lateral_disp = max(lateral_disps) if lateral_disps else 0
    
    # Maximum torsional rotation 
    torsional_rots = [abs(mode_data.get((node, 3), 0)) for node in range(n+1)]
    max_torsion = max(torsional_rots) if torsional_rots else 0
    
    if max_lateral_disp > 1e-6 and max_torsion > 1e-6:
        print(f"✓ Mode {mode+1} exhibits characteristics of lateral torsional buckling")
    else:
        print(f"✗ Mode {mode+1} does not appear to be a lateral torsional buckling mode")

# # Visualization of the critical buckling mode 
# beam.show_mode_shapes(scale=10)