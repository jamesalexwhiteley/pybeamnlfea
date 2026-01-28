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
# A&T Table 2: Simply supported with uniform load
# =============================================================================
table2_data = np.array([
    [[ 8.81, 10.71, 12.20, 13.08, 13.86, 15.70, 18.75],
     [11.07, 14.51, 17.22, 18.68, 20.19, 23.35, 28.30],
     [13.87, 20.41, 25.70, 28.32, 30.90, 35.90, 42.96],
     [17.96, 27.77, 38.29, 42.67, 46.54, 53.37, 62.07],
     [18.61, 32.33, 53.32, 60.24, 65.53, 73.97, 83.90]],
    [[ 8.95, 10.83, 12.31, 13.12, 13.98, 15.84, 19.03],
     [11.36, 14.69, 17.36, 18.81, 20.32, 23.53, 28.70],
     [14.30, 20.79, 25.89, 28.47, 31.04, 36.12, 43.57],
     [21.35, 28.54, 38.61, 42.85, 46.68, 53.59, 62.85],
     [23.21, 34.24, 53.99, 60.49, 65.67, 74.17, 84.78]],
    [[ 9.83, 11.65, 13.14, 13.97, 14.86, 16.85, 20.35],
     [12.73, 15.83, 18.40, 19.84, 21.38, 24.74, 30.39],
     [17.09, 22.64, 27.18, 29.63, 32.16, 37.37, 45.39],
     [22.86, 32.78, 40.36, 44.09, 47.71, 54.67, 64.53],
     [28.43, 44.86, 56.84, 61.98, 66.65, 75.03, 86.20]],
    [[17.03, 19.50, 21.47, 22.56, 23.73, 26.32, 30.88],
     [21.53, 25.21, 28.15, 29.77, 31.51, 35.30, 41.76],
     [28.11, 33.60, 37.89, 40.22, 42.67, 47.89, 56.35],
     [37.54, 45.34, 51.20, 54.29, 57.47, 64.02, 74.15],
     [50.09, 60.39, 67.74, 71.49, 75.28, 82.89, 94.26]],
    [[59.14, 63.69, 66.97, 68.68, 70.45, 74.14, 80.07],
     [67.22, 72.62, 76.49, 78.52, 80.59, 84.91, 91.79],
     [76.75, 83.10, 87.62, 89.97, 92.37, 97.34,105.19],
     [87.86, 95.21,100.41,103.08,105.82,111.43,120.21],
     [100.59,108.96,114.81,117.81,120.86,127.08,136.72]]
])


interp_table2 = RegularGridInterpolator(
    (K_values, epsilon_values, delta_values),
    table2_data, method='cubic', bounds_error=False, fill_value=None
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
# A&T Table 4: Cantilevers with uniform load 
# =============================================================================
table4_data = np.array([
    [[ 7.64,  5.32,  4.27,  3.85,  3.49,  2.89,  2.25],
     [19.76, 11.24,  7.83,  6.60,  5.62,  4.18,  2.84],
     [41.67, 25.54, 16.54, 12.89,  9.95,  7.49,  3.26],
     [54.17, 35.60, 23.74, 18.27, 13.35,  8.66,  3.37],
     [62.72, 42.18, 28.36, 21.60, 15.12,  9.38,  3.41]],
    [[ 8.02,  5.63,  4.56,  4.13,  3.77,  3.17,  2.52],
     [20.60, 11.89,  8.39,  7.14,  6.14,  4.68,  3.35],
     [42.34, 26.77, 17.90, 14.29, 11.36,  7.44,  4.56],
     [54.57, 36.88, 25.63, 20.51, 16.07,  9.32,  5.65],
     [62.89, 43.42, 30.49, 24.35, 18.89, 11.37,  6.32]],
    [[ 8.89,  6.39,  5.27,  4.83,  4.44,  3.80,  3.10],
     [22.51, 13.40,  9.73,  8.41,  7.35,  5.78,  4.29],
     [46.21, 30.31, 21.26, 17.53, 14.42,  9.99,  6.40],
     [59.40, 41.76, 30.78, 25.83, 21.46, 14.73,  8.95],
     [68.19, 49.02, 36.62, 30.87, 25.67, 17.57, 10.63]],
    [[13.03, 10.08,  8.69,  8.11,  7.60,  6.74,  5.75],
     [29.98, 20.04, 15.76, 14.13, 12.76, 10.61,  8.39],
     [59.47, 43.60, 34.20, 30.04, 26.31, 20.25, 14.22],
     [76.30, 60.13, 50.24, 45.65, 41.32, 33.55, 24.18],
     [87.01, 69.92, 59.34, 54.40, 49.74, 41.33, 31.14]],
    [[33.29, 27.60, 24.66, 23.38, 22.24, 20.19, 17.71],
     [59.11, 46.55, 40.01, 37.22, 34.71, 30.43, 25.46],
     [97.15, 80.16, 69.55, 64.57, 59.85, 51.34, 40.97],
     [125.30,109.80, 99.64, 94.66, 89.74, 80.01, 66.38],
     [143.20,128.30,118.70,114.10,109.50,100.60, 87.84]]
])

interp_table4 = RegularGridInterpolator(
    (K_values, epsilon_values, delta_values),
    table4_data, method='cubic', bounds_error=False, fill_value=None
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
    beam.add_nodal_load(n//2, [0, 0, -1, 0, 0, 0, 0], NodalLoad, load_height=load_height_above_shear_center)  # central point load
    # [beam.add_uniform_load(element_id=e, forces=[0, 0, -1], load_height=load_height_above_shear_center) for e in range(n)] 

    eigenvalues, eigenvectors = beam.solve_eigen(num_modes=1)
    # beam.show_mode_shape(eigenvectors[0], scale=2, cross_section_scale=1)
    P_c = eigenvalues[0]
    
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
    beam.add_nodal_load(n, [0, 0, -1, 0, 0, 0, 0], NodalLoad, load_height=load_height_above_shear_center)  # end load 
    # [beam.add_uniform_load(element_id=e, forces=[0, 0, -1], load_height=load_height_above_shear_center) for e in range(n)] 

    eigenvalues, eigenvectors = beam.solve_eigen(num_modes=1)
    # beam.show_mode_shape(eigenvectors[0], scale=2, cross_section_scale=1)
    P_c = eigenvalues[0]
    
    return P_c

# Epsilon values to test
test_epsilons = np.array([0.3, 0.15, 0, -0.15, -0.3])
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
# gamma2_pos = np.array([float(interp_table2((K_target, eps, 0.180))) for eps in epsilon_fine])
# gamma2_neg = np.array([float(interp_table2((K_target, eps, -0.180))) for eps in epsilon_fine])
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
# gamma2_cant_pos = np.array([float(interp_table4((K_target, eps, 0.180))) for eps in epsilon_fine])
# gamma2_cant_neg = np.array([float(interp_table4((K_target, eps, -0.180))) for eps in epsilon_fine])
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
# ax6.set_xlim(0, 400)
# ax6.set_ylim(-0.4, 0.4)

plt.tight_layout()
plt.show()
plt.close()