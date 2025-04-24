from pybeamnlfea.model.frame import Frame
from pybeamnlfea.model.material import LinearElastic
from pybeamnlfea.model.section import Section 
from pybeamnlfea.model.element import ThinWalledBeamElement 
from pybeamnlfea.model.boundary import BoundaryCondition 
from pybeamnlfea.model.load import UniformLoad, NodalLoad

# Author: James Whiteley (github.com/jamesalexwhiteley) 

# Create a beam structure 
n = 20
L = 10 # m 
beam = Frame() 
beam.add_nodes([[i*L/n, 0, 0] for i in range(n+1)])
    
# Material and section properties for steel UB127x76x13
# Steel properties
E = 210e9   # N/m2
G = 80e9    # N/m2 
rho = 7850  # kg/m3

# UB127x76x13 section properties
A = 1650e-6    # m2
Iy = 0.746e-6  # m4
Iz = 0.147e-6  # m4
J = 0.0285e-6  # m4
Iw = 0.002e-12 # m6

beam.add_material("steel", LinearElastic(rho=rho, E=E, G=G))
beam.add_section("UB127x76x13", Section(A=A, Iy=Iy, Iz=Iz, J=J, Iw=Iw, y0=0, z0=0))

# Add element 
beam.add_elements([[i, i+1] for i in range(n)], "steel", "UB127x76x13", element_class=ThinWalledBeamElement) 

# Add boundary conditions 
# beam.add_boundary_condition(0, [0, 0, 0, 0, 0, 0, 0], BoundaryCondition) 
beam.add_boundary_condition(0, [0, 0, 0, 0, 1, 1, 0], BoundaryCondition) 
beam.add_boundary_condition(n, [1, 0, 0, 1, 1, 1, 0], BoundaryCondition) 

# Add load
beam.add_gravity_load([0, 1e-6, 1])
w = beam.get_self_weight()
# for i in range(n): 
#     beam.add_uniform_load(i, [0, 0, 1], UniformLoad)
# w = 1.0 * L

# # Linear analysis 
# results = beam.solve()
# print(results.get_nodal_displacements(10))
# beam.show(scale=1) 

# Run eigenvalue buckling analysis 
print("Running eigenvalue buckling analysis...") 
eigenvalues, eigenvectors = beam.solve_eigen(num_modes=5) 
print("Buckling eigenvalues (load factors):") 
for i, val in enumerate(eigenvalues):
    coeff = val * w / ((G * J * E * Iy)**0.5 / L**3)
    print(f"Mode {i+1}: wcr={w * val} coeff={coeff}") # wcr = 28.5 (Stratford and Burgoyne)

# NOTE check analytic buckling for simple case, e.g., Mcr = (π/L) * √(EIz * GJ) * √(1 + (π²*EIw)/(GJ*L²))

# print("\nAnalyzing buckling modes to identify LTB:")
# for mode in range(len(eigenvalues)):
#     # Extract the mode shape data
#     mode_data = beam.buckling_modes[mode]

#     # Maximum (major axis) lateral displacement 
#     lateral_disps = [abs(mode_data.get((node, 2), 0)) for node in range(n+1)]
#     max_lateral_major = max(lateral_disps) if lateral_disps else 0
#     if max_lateral_major > 1e-6: 
#         print(f"✓ Major {max_lateral_major}")
#     else: 
#         print(f"✗ Major {max_lateral_major}")

#     # Maximum (minor axis) lateral displacement 
#     lateral_disps = [abs(mode_data.get((node, 1), 0)) for node in range(n+1)]
#     max_lateral_minor = max(lateral_disps) if lateral_disps else 0
#     if max_lateral_minor > 1e-6: 
#         print(rf"✓ Minor {max_lateral_minor}")
#     else: 
#         print(f"✗ Minor {max_lateral_minor}")
    
#     # Maximum torsional rotation 
#     torsional_rots = [abs(mode_data.get((node, 3), 0)) for node in range(n+1)]
#     max_torsion = max(torsional_rots) if torsional_rots else 0
#     if max_torsion > 1e-6: 
#         print(f"✓ Torsion {max_torsion}")
#     else: 
#         print(f"✗ Torsion {max_torsion}")

#     print()

    # # Visualization of the critical buckling mode 
    # beam.show_mode_shape(mode_data, scale=10)