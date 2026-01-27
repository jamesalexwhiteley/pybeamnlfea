"""
Reproduce Figures 5 and 6 from Anderson & Trahair (1972)
"Stability of Monosymmetric Beams and Cantilevers"

"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

from pybeamnlfea.model.frame import Frame
from pybeamnlfea.model.material import LinearElastic
from pybeamnlfea.model.section import Section 
from pybeamnlfea.model.element import ThinWalledBeamElement 
from pybeamnlfea.model.boundary import BoundaryCondition 
from pybeamnlfea.model.load import NodalLoad 

# =============================================================================
# A&T Table 1: Simply supported with midspan load
# =============================================================================
K_values = np.array([0, 0.1, 0.3, 1.0, 3.0])
delta_values = np.array([-0.6, -0.3, -0.1, 0, 0.1, 0.3, 0.6])
epsilon_values = np.array([0.6, 0.3, 0, -0.3, -0.6])

table1_data = np.array([
    [[4.05, 4.89, 5.56, 5.93, 6.32, 7.17, 8.61],
     [5.47, 7.27, 8.70, 9.47, 10.27, 11.93, 14.49],
     [8.61, 12.52, 15.58, 16.94, 18.23, 20.67, 24.05],
     [13.73, 18.01, 23.71, 26.43, 28.39, 31.52, 35.39],
     [14.30, 20.73, 26.81, 32.63, 36.52, 41.55, 46.54]],
    [[4.41, 5.22, 5.87, 6.22, 6.60, 7.44, 8.87],
     [6.07, 7.70, 9.04, 9.77, 10.53, 12.14, 14.70],
     [8.91, 12.92, 15.72, 17.04, 18.32, 20.77, 24.25],
     [14.09, 20.69, 25.52, 27.32, 28.95, 31.87, 35.80],
     [15.55, 21.17, 31.56, 35.43, 38.29, 42.51, 47.35]],
    [[5.24, 6.03, 6.65, 7.00, 7.37, 8.19, 9.63],
     [7.29, 8.78, 9.99, 10.67, 11.38, 12.94, 15.53],
     [11.13, 14.23, 16.57, 17.78, 19.01, 21.47, 25.15],
     [18.28, 23.91, 27.11, 28.62, 30.10, 32.96, 37.10],
     [24.17, 33.60, 37.80, 39.67, 41.45, 44.79, 49.40]],
    [[9.72, 10.80, 11.62, 12.07, 12.54, 13.58, 15.36],
     [12.96, 14.70, 16.04, 16.76, 17.53, 19.19, 21.97],
     [18.21, 21.01, 23.10, 24.22, 25.37, 27.78, 31.62],
     [26.45, 30.49, 33.34, 34.80, 36.27, 39.25, 43.75],
     [37.80, 42.70, 45.95, 47.57, 49.18, 52.38, 57.12]],
    [[34.74, 36.82, 38.29, 39.05, 39.83, 41.45, 44.02],
     [40.64, 43.19, 44.98, 45.91, 46.86, 48.82, 51.91],
     [47.86, 50.94, 53.09, 54.20, 55.33, 57.64, 61.25],
     [56.54, 60.16, 62.67, 63.95, 65.25, 67.89, 71.97],
     [66.71, 70.79, 73.62, 75.05, 76.49, 79.41, 83.87]]
])

interp_table1 = RegularGridInterpolator(
    (K_values, epsilon_values, delta_values),
    table1_data, method='cubic', bounds_error=False, fill_value=None
)

# =============================================================================
# A&T Table 3: Cantilevers with end load
# =============================================================================
table3_data = np.array([
    [[2.49, 1.94, 1.65, 1.53, 1.42, 1.23, 1.00],
     [5.39, 3.65, 2.83, 2.50, 2.22, 1.77, 1.30],
     [9.57, 6.47, 4.74, 4.01, 3.39, 2.43, 1.55],
     [11.11, 7.69, 5.67, 4.78, 3.99, 2.75, 1.63],
     [11.71, 8.15, 6.02, 5.06, 4.21, 3.44, 1.64]],
    [[2.56, 2.00, 1.71, 1.59, 1.47, 1.28, 1.06],
     [5.58, 3.79, 2.95, 2.62, 2.33, 1.88, 1.41],
     [9.88, 6.78, 5.03, 4.30, 3.66, 2.68, 1.80],
     [11.42, 8.05, 6.03, 5.14, 4.35, 3.11, 2.01],
     [12.02, 8.52, 6.40, 5.45, 4.60, 3.27, 2.09]],
    [[2.71, 2.13, 1.84, 1.71, 1.60, 1.41, 1.18],
     [5.95, 4.09, 3.22, 2.88, 2.58, 2.11, 1.62],
     [10.64, 7.50, 5.72, 4.96, 4.30, 3.25, 2.26],
     [12.36, 8.95, 6.93, 6.04, 5.24, 3.96, 2.70],
     [13.03, 9.48, 7.36, 6.42, 5.58, 4.21, 2.88]],
    [[3.46, 2.77, 2.43, 2.29, 2.16, 1.93, 1.67],
     [7.48, 5.33, 4.33, 3.93, 3.59, 3.04, 2.44],
     [13.61, 10.42, 8.51, 7.64, 6.85, 5.51, 4.05],
     [16.18, 12.99, 11.06, 10.17, 9.33, 7.82, 5.98],
     [17.24, 13.94, 11.96, 11.05, 10.20, 8.67, 6.81]],
    [[8.06, 6.75, 6.06, 5.76, 5.48, 5.00, 4.40],
     [13.92, 11.15, 9.68, 9.04, 8.46, 7.47, 6.29],
     [22.12, 18.51, 16.24, 15.17, 14.15, 12.27, 9.95],
     [27.74, 24.53, 22.43, 21.40, 20.37, 18.36, 15.46],
     [30.79, 27.80, 25.88, 24.94, 24.02, 22.22, 19.62]]
])

interp_table3 = RegularGridInterpolator(
    (K_values, epsilon_values, delta_values),
    table3_data, method='cubic', bounds_error=False, fill_value=None
)

# =============================================================================
# BEAM PROPERTIES matching A&T Figures 5 & 6
# =============================================================================
# Target A&T parameters
K_target = 0.247                                         
delta_target = 0.180                                        
sqrt_EIy_GJ_over_L2_lb = 8.80  # lb
lb_to_N = 4.44822

# Material: Aluminum
E = 7.000e+10   # Pa
G = 2.600e+10   # Pa
rho = 2700      # kg/m³

# Beam length
L = 1.678852    # m

# Section properties (adjusted to exactly match A&T parameters)
Iy = 9.209763e-09   # m^4 (minor axis - for lateral buckling)
Ix = 5.0e-08        # m^4 (major axis - approximate, for bending under load)
J = 7.262188e-10    # m^4 (torsion constant)
Iw = 4.699609e-12   # m^6 (warping constant)
A = 5.0e-04         # m^2 (approximate cross-sectional area)

# Monosymmetric properties
y0_positive_delta = -0.023669     # m (shear center offset for δ = +0.180)
beta_z_positive_delta = 0.051717  # m (monosymmetry property for δ = +0.180)
y0_negative_delta = 0.023669
beta_z_negative_delta = -0.051717

# Section depth (approximate, for calculating load height)
D = 0.0756  # m (75.6 mm)

# Scale factor for converting gamma_2 to P_c
scale_N = np.sqrt(E * Iy * G * J) / L**2   # should be ≈ 39.14 N
scale_lb = scale_N / lb_to_N               # should be ≈ 8.80 lb

# K and detla factors
K_factor = np.sqrt(np.pi**2 * E * Iw /(G * J * L**2))                   # should be ≈ 0.247
delta_factor = (beta_z_positive_delta / L) * np.sqrt(E * Iy /(G * J))   # should be ≈ 0.180

# Conversion factor: epsilon to load height a (in meters)
# epsilon = (a / L) * sqrt(E * Iy / (G * J))
# a = epsilon * L / sqrt(E * Iy / (G * J))
eps_to_a = L / np.sqrt(E * Iy / (G * J))

# =============================================================================
# PyBeamNLFEA  
# =============================================================================
def run_simply_supported_beam(E, G, rho, L, A, Ix, Iy, J, Iw, y0, beta_z, load_height_above_shear_center, n_elements=10):
    n = n_elements
    beam = Frame()
    beam.add_nodes([[L*i/n, 0, 0] for i in range(n+1)])
    
    # Section properties 
    beam.add_material("aluminum", LinearElastic(rho=rho, E=E, G=G))
    beam.add_section("monosym_I", Section(A=A, Iy=Ix, Iz=Iy, J=J, Iw=Iw, y0=y0, z0=0, beta_z=beta_z))
    beam.add_elements([[i, i+1] for i in range(n)], "aluminum", "monosym_I", element_class=ThinWalledBeamElement)
    
    # Simply supported, free to warp; (ux, uy, uz, θx, θy, θz, φ) with 0=fixed, 1=free
    beam.add_boundary_condition(0, [0, 0, 0, 0, 1, 1, 1], BoundaryCondition)
    beam.add_boundary_condition(n, [0, 0, 0, 0, 1, 1, 1], BoundaryCondition)
    beam.add_nodal_load(n//2, [0, 0, -1, 0, 0, 0, 0], NodalLoad)  # central point load
    
    # NOTE element doesn't handle load height internally; add torque:
    torque = load_height_above_shear_center * 1  # for unit load
    beam.add_nodal_load(n//2, [0, 0, 0, torque, 0, 0, 0], NodalLoad)
    eigenvalues, eigenvectors = beam.solve_eigen(num_modes=1)
    # beam.show_mode_shape(eigenvectors[0], scale=2, cross_section_scale=1)

    P_c = eigenvalues[0]
    
    # # A&T theoretical value 
    # epsilon = load_height_above_shear_center / eps_to_a
    # delta = beta_z / L * np.sqrt(E * Iy / (G * J))
    # gamma_2 = float(interp_table1((K_target, epsilon, delta)))
    # P_c = gamma_2 * scale_N
    
    return P_c

def run_cantilever(E, G, rho, L, A, Ix, Iy, J, Iw, y0, beta_z, load_height_above_shear_center, n_elements=10):
    n = n_elements
    beam = Frame()
    beam.add_nodes([[L*i/n, 0, 0] for i in range(n+1)])
    
    # Section properties 
    beam.add_material("aluminum", LinearElastic(rho=rho, E=E, G=G))
    beam.add_section("monosym_I", Section(A=A, Iy=Ix, Iz=Iy, J=J, Iw=Iw, y0=y0, z0=0, beta_z=beta_z))
    beam.add_elements([[i, i+1] for i in range(n)], "aluminum", "monosym_I", element_class=ThinWalledBeamElement)
    
    # Cantilever; (ux, uy, uz, θx, θy, θz, φ) with 0=fixed, 1=free
    beam.add_boundary_condition(0, [0, 0, 0, 0, 0, 0, 0], BoundaryCondition)
    beam.add_nodal_load(n, [0, 0, -1, 0, 0, 0, 0], NodalLoad)  # end load 
    
    eigenvalues, eigenvectors = beam.solve_eigen(num_modes=1)
    P_c = eigenvalues[0]
    
    # # PLACEHOLDER: Return A&T theoretical value
    # epsilon = load_height_above_shear_center / eps_to_a
    # delta = beta_z / L * np.sqrt(E * Iy / (G * J))
    # gamma_2 = float(interp_table3((K_target, epsilon, delta)))
    # P_c = gamma_2 * scale_N
    
    return P_c

# Epsilon values to test
test_epsilons = np.array([0.3, 0.15, 0, -0.15, -0.3])
# test_epsilons = np.array([0])  # NOTE load applied at centroid 
results_ss_pos = {'eps': [], 'Pc_N': [], 'Pc_lb': []}  # δ = +0.180
results_ss_neg = {'eps': [], 'Pc_N': [], 'Pc_lb': []}  # δ = -0.180

for eps in test_epsilons:
    a = eps * eps_to_a  # convert epsilon to load height in meters
    
    # δ = +0.180 (bottom flange smaller)
    Pc_pos = run_simply_supported_beam(E, G, rho, L, A, Ix, Iy, J, Iw, y0_positive_delta, beta_z_positive_delta, a)
    results_ss_pos['eps'].append(eps)
    results_ss_pos['Pc_N'].append(Pc_pos)
    results_ss_pos['Pc_lb'].append(Pc_pos / lb_to_N)
    
    # δ = -0.180 (top flange smaller / beam flipped)
    Pc_neg = run_simply_supported_beam(E, G, rho, L, A, Ix, Iy, J, Iw, y0_negative_delta, beta_z_negative_delta, a)
    results_ss_neg['eps'].append(eps)
    results_ss_neg['Pc_N'].append(Pc_neg)
    results_ss_neg['Pc_lb'].append(Pc_neg / lb_to_N)
    
    print(f"ε = {eps:+.2f}: δ=+0.180 → Pc = {Pc_pos:.1f} N ({Pc_pos/lb_to_N:.1f} lb), "f"δ=-0.180 → Pc = {Pc_neg:.1f} N ({Pc_neg/lb_to_N:.1f} lb)")

results_cant_pos = {'eps': [], 'Pc_N': [], 'Pc_lb': []}
results_cant_neg = {'eps': [], 'Pc_N': [], 'Pc_lb': []}

for eps in test_epsilons:
    a = eps * eps_to_a
    
    Pc_pos = run_cantilever(E, G, rho, L, A, Ix, Iy, J, Iw, y0_positive_delta, beta_z_positive_delta, a)
    results_cant_pos['eps'].append(eps)
    results_cant_pos['Pc_N'].append(Pc_pos)
    results_cant_pos['Pc_lb'].append(Pc_pos / lb_to_N)
    
    Pc_neg = run_cantilever(E, G, rho, L, A, Ix, Iy, J, Iw, y0_negative_delta, beta_z_negative_delta, a)
    results_cant_neg['eps'].append(eps)
    results_cant_neg['Pc_N'].append(Pc_neg)
    results_cant_neg['Pc_lb'].append(Pc_neg / lb_to_N)
    
    print(f"ε = {eps:+.2f}: δ=+0.180 → Pc = {Pc_pos:.1f} N ({Pc_pos/lb_to_N:.1f} lb), "f"δ=-0.180 → Pc = {Pc_neg:.1f} N ({Pc_neg/lb_to_N:.1f} lb)")

# =============================================================================
# PLOT FIGURE 5: Simply Supported Beams
# =============================================================================
fig5, ax5 = plt.subplots(figsize=(10, 8))

# A&T theoretical curves
epsilon_fine = np.linspace(0.5, -0.5, 100)
gamma2_pos = np.array([float(interp_table1((K_target, eps, 0.180))) for eps in epsilon_fine])
gamma2_neg = np.array([float(interp_table1((K_target, eps, -0.180))) for eps in epsilon_fine])
Pc_curve_pos = gamma2_pos * scale_lb
Pc_curve_neg = gamma2_neg * scale_lb

ax5.plot(Pc_curve_pos, epsilon_fine, 'b-', linewidth=1, label=r'A&T: $\delta = +0.180$')
ax5.plot(Pc_curve_neg, epsilon_fine, 'r-', linewidth=1, label=r'A&T: $\delta = -0.180$')

# Simulator results
ax5.plot(results_ss_pos['Pc_lb'], results_ss_pos['eps'], 'bo', markersize=8, markerfacecolor='none', markeredgewidth=1, label=r'FEM: $\delta = +0.180$')
ax5.plot(results_ss_neg['Pc_lb'], results_ss_neg['eps'], 'ro', markersize=8, markerfacecolor='none', markeredgewidth=1, label=r'FEM: $\delta = -0.180$')

ax5.set_xlabel(r'Critical Load $P_c$ (lb)', fontsize=12)
ax5.set_ylabel(r'Parameter $\epsilon$ (load height above shear center)', fontsize=12)
ax5.set_title('Figure 5: Simply Supported Beams\n' + r'$\sqrt{EI_y GJ}/L^2 = 8.80$ lb, $K = 0.247$', fontsize=12)
ax5.legend(loc='upper left', fontsize=10)
ax5.grid(True, alpha=0.3)
ax5.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax5.set_xlim(0, 400)
ax5.set_ylim(-0.4, 0.4)

plt.tight_layout()
plt.show()
plt.close()

# =============================================================================
# PLOT FIGURE 6: Cantilevers
# =============================================================================
fig6, ax6 = plt.subplots(figsize=(10, 8))

# A&T theoretical curves
gamma2_cant_pos = np.array([float(interp_table3((K_target, eps, 0.180))) for eps in epsilon_fine])
gamma2_cant_neg = np.array([float(interp_table3((K_target, eps, -0.180))) for eps in epsilon_fine])
Pc_cant_curve_pos = gamma2_cant_pos * scale_lb
Pc_cant_curve_neg = gamma2_cant_neg * scale_lb

ax6.plot(Pc_cant_curve_pos, epsilon_fine, 'b-', linewidth=1, label=r'A&T: $\delta = +0.180$')
ax6.plot(Pc_cant_curve_neg, epsilon_fine, 'r-', linewidth=1, label=r'A&T: $\delta = -0.180$')

# Simulator results
ax6.plot(results_cant_pos['Pc_lb'], results_cant_pos['eps'], 'bo', markersize=8, markerfacecolor='none', markeredgewidth=1, label=r'FEM: $\delta = +0.180$')
ax6.plot(results_cant_neg['Pc_lb'], results_cant_neg['eps'], 'ro', markersize=8, markerfacecolor='none', markeredgewidth=1, label=r'FEM: $\delta = -0.180$')

ax6.set_xlabel(r'Critical Load $P_c$ (lb)', fontsize=12)
ax6.set_ylabel(r'Parameter $\epsilon$ (load height above shear center)', fontsize=12)
ax6.set_title('Figure 6: Cantilevers\n' + r'$\sqrt{EI_y GJ}/L^2 = 8.80$ lb, $K = 0.247$', fontsize=12)
ax6.legend(loc='upper left', fontsize=10)
ax6.grid(True, alpha=0.3)
ax6.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax6.set_xlim(0, 80)
ax6.set_ylim(-0.4, 0.4)

plt.tight_layout()
plt.show()
plt.close()