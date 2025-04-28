from pybeamnlfea.model.frame import Frame
from pybeamnlfea.model.material import LinearElastic
from pybeamnlfea.model.section import Section 
from pybeamnlfea.model.element import ThinWalledBeamElement 
from pybeamnlfea.model.boundary import BoundaryCondition 
from pybeamnlfea.model.load import NodalLoad 
import numpy as np

# Create a beam structure for axial-torsional buckling analysis
n = 20  # Number of elements
L = 5.0  # m - beam length
beam = Frame() 
beam.add_nodes([[i*L/n, 0, 0] for i in range(n+1)])

# Material properties - steel
E = 210e9   # N/m2 - Young's modulus
G = 80e9    # N/m2 - Shear modulus
rho = 7850  # kg/m3 - Density

# Section properties for UB127x76x13
A = 1650e-6    # m2 - Cross-sectional area
Iy = 0.746e-6  # m4 - Strong axis moment of inertia
Iz = 0.147e-6  # m4 - Weak axis moment of inertia
J = 0.0285e-6  # m4 - Torsional constant
Iw = 0.002e-12 # m6 - Warping constant

beam.add_material("steel", LinearElastic(rho=rho, E=E, G=G))
beam.add_section("UB127x76x13", Section(A=A, Iy=Iy, Iz=Iz, J=J, Iw=Iw, y0=0, z0=0))

# Add elements
beam.add_elements([[i, i+1] for i in range(n)], "steel", "UB127x76x13", element_class=ThinWalledBeamElement) 

# Boundary conditions for axial-torsional buckling
# Pin-ended conditions: restrain translations at both ends, but allow rotation
# At x=0: Fix all translations, allow all rotations
beam.add_boundary_condition(0, [0, 0, 0, 0, 1, 1, 1], BoundaryCondition)
# At x=L: Fix y and z translations, allow x-translation and all rotations
beam.add_boundary_condition(n, [1, 0, 0, 1, 1, 1, 1], BoundaryCondition)

# Apply axial compressive load at the free end
P = 1.0  # Reference load (will be scaled by eigenvalue)
beam.add_nodal_load(n, [-P, 0, 0, 0, 0, 0, 0], NodalLoad)  # Negative for compression

# Linear buckling analysis 
eigenvalues, eigenvectors = beam.solve_eigen(num_modes=10) 

# Calculate analytical solutions for comparison
# For a pin-ended column, the Euler buckling load is:
P_euler_y = np.pi**2 * E * Iy / L**2  # Buckling about y-axis (strong)
P_euler_z = np.pi**2 * E * Iz / L**2  # Buckling about z-axis (weak)

# For torsional buckling of a doubly symmetric section:
P_torsional = (1/A) * (G*J + np.pi**2*E*Iw/L**2)

# Display results
print(f"Numerical eigenvalues (critical load factors):")
for i in range(len(eigenvalues)):
    print(f"Mode {i+1}: Critical load = {eigenvalues[i]:.4e} N")
    mode_shape = eigenvectors[i]
    # Uncomment to visualise
    # beam.show_mode_shape(mode_shape, scale=1)

print("\nAnalytical buckling loads:")
print(f"Euler buckling load (weak axis): {P_euler_z:.4e} N")
print(f"Euler buckling load (strong axis): {P_euler_y:.4e} N")
print(f"Torsional buckling load: {P_torsional:.4e} N")

# The lowest eigenvalue should correspond to weak-axis buckling
print(f"\nRatio (Numerical/Analytical) for first mode: {eigenvalues[0]/min(P_euler_y, P_euler_z, P_torsional):.4f}")