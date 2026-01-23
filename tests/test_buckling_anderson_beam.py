import numpy as np
from scipy.interpolate import RegularGridInterpolator
from pybeamnlfea.model.frame import Frame
from pybeamnlfea.model.material import LinearElastic
from pybeamnlfea.model.section import Section 
from pybeamnlfea.model.element import ThinWalledBeamElement 
from pybeamnlfea.model.boundary import BoundaryCondition 
from pybeamnlfea.model.load import NodalLoad 
from pybeamnlfea.utils.section import monosymmetric_section_properties

# Author: James Whiteley (github.com/jamesalexwhiteley)

# SIMPLY SUPPORTED  
# Anderson & Trahair (1972) "Stability of Monosymmetric Beams and Cantilevers"
K_values = np.array([0, 0.1, 0.3, 1.0, 3.0])
epsilon_values = np.array([0.6, 0.3, 0.0, -0.3, -0.6])
delta_values = np.array([-0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6])

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
    table1_data,
    method="cubic",
    bounds_error=False,
    fill_value=None
)

def table1_value(K, eps, delta):
    return float(interp_table1(np.array([K, eps, delta]))[0])

K_values = np.array([0, 0.1, 0.3, 1.0, 3.0])
epsilon_values = np.array([0.6, 0.3, 0.0, -0.3, -0.6])
delta_values = np.array([-0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6])

table2_data = np.array([
    [[18.75, 15.70, 13.86, 13.08, 12.20, 10.71, 8.81],
     [28.30, 23.35, 20.19, 18.68, 17.22, 14.51, 11.07],
     [42.96, 35.90, 30.90, 28.32, 25.70, 20.41, 13.87],
     [62.07, 53.37, 46.54, 42.67, 38.29, 27.77, 17.96],
     [83.90, 73.97, 65.53, 60.24, 53.32, 32.33, 18.61]],

    [[19.03, 15.84, 13.98, 13.12, 12.31, 10.83, 8.95],
     [28.70, 23.53, 20.32, 18.81, 17.36, 14.69, 11.36],
     [43.57, 36.12, 31.04, 28.47, 25.89, 20.79, 14.30],
     [62.85, 53.59, 46.68, 42.85, 38.61, 28.54, 21.35],
     [84.78, 74.17, 65.67, 60.49, 53.99, 34.24, 23.21]],

    [[20.35, 16.85, 14.86, 13.97, 13.14, 11.65, 9.83],
     [30.39, 24.74, 21.38, 19.84, 18.40, 15.83, 12.73],
     [45.39, 37.37, 32.16, 29.63, 27.18, 22.64, 17.09],
     [64.53, 54.67, 47.71, 44.09, 40.36, 32.78, 22.86],
     [86.20, 75.03, 66.65, 61.98, 56.84, 44.86, 28.43]],

    [[30.88, 26.32, 23.73, 22.56, 21.47, 19.50, 17.03],
     [41.76, 35.30, 31.51, 29.77, 28.15, 25.21, 21.53],
     [56.35, 47.89, 42.67, 40.22, 37.89, 33.60, 28.11],
     [74.15, 64.02, 57.47, 54.29, 51.20, 45.34, 37.54],
     [94.26, 82.89, 75.28, 71.49, 67.74, 60.39, 50.09]],

    [[80.07, 74.14, 70.45, 68.68, 66.97, 63.69, 59.14],
     [91.79, 84.91, 80.59, 78.52, 76.49, 72.62, 67.22],
     [105.19, 97.34, 92.37, 89.97, 87.62, 83.10, 76.75],
     [120.21, 111.43, 105.82, 103.08, 100.41, 95.21, 87.86],
     [136.72, 127.08, 120.86, 117.81, 114.81, 108.96, 100.59]]
])

interp_table2 = RegularGridInterpolator(
    (K_values, epsilon_values, delta_values),
    table2_data, method='cubic', bounds_error=False, fill_value=None
)

def table2_value(K, eps, delta):
    return float(interp_table2(np.array([K, eps, delta]))[0])

# Create a beam structure 
n = 10 
# L = 1     # m 
L = 1.651   # m 
beam = Frame() 
beam.add_nodes([[L*i/n, 0, 0] for i in range(n+1)]) 

# Steel properties
E = 210e9   # N/m2
G = 80e9    # N/m2 
rho = 7850  # kg/m3

# # UB127x76x13 section properties
# A = 1650e-6    # m2
# Iy = 4.73e-6   # m4 
# Iz = 0.557e-6  # m4
# J = 0.0285e-6  # m4
# Iw = 0.002e-6  # m6   
# beta_z = 0
# y0 = 0

# E, G, rho = 1, 1, 1 
# rho, E, G = 1, 1, 1 
# A, Iy, Iz, J, Iw = 1, 1, 1, 1, 1 

# Monosymmetric I-section 
B1 = 0.15   # m top flange
T1 = 0.025  # m 
B2 = 0.08   # m bottom flange
T2 = 0.012  # m 
D = 0.4    # m depth
t = 0.02    # m web
props = monosymmetric_section_properties(B1, T1, B2, T2, D, t)
A = props['A']    # m2
Iy = props['Iy']  # m4 
Iz = props['Iz']  # m4
J = props['J']    # m4
Iw = props['Iw']  # m6
# beta_y=0
beta_z = props['beta_z']   
# beta_w=0
y0 = props['y0']
# z0=0
# r1=0

beam.add_material("steel", LinearElastic(rho=rho, E=E, G=G))
beam.add_section("UB127x76x13", Section(A=A, Iy=Iy, Iz=Iz, J=J, Iw=Iw,
                                        y0=y0, z0=0, beta_y=0, beta_z=beta_z, beta_w=0, r1=0))

# Add element 
beam.add_elements([[i, i+1] for i in range(n)], "steel", "UB127x76x13", element_class=ThinWalledBeamElement) 

# Add boundary conditions; Global (ux, uy, uz, θx, θy, θz, φ); 0=fixed, 1=free 
beam.add_boundary_condition(0, [0, 0, 0, 0, 1, 1, 1], BoundaryCondition) 
beam.add_boundary_condition(n, [0, 0, 0, 0, 1, 1, 1], BoundaryCondition)

# Add loads; Global (ux, uy, uz, θx, θy, θz, φ)
beam.add_nodal_load(n/2, [0, 0, -1, 0, 0, 0, 0], NodalLoad) # N 
# [beam.add_uniform_load(element_id=e, forces=[0, 0, -1]) for e in range(n)] # NOTE:UDL

# Buckling capacity 
K = np.sqrt(np.pi**2 * E * Iw / (G * J * L**2))     # torsion parameter 
delta = (beta_z / L) * np.sqrt(E * Iy / (G * J))    # monosymmetry parameters (0 if symmetric) 
eps = 0                                             # load height parameters (0 since load applied at centroid)
gamma = table1_value(K, eps, delta) 
P_c = gamma * np.sqrt(E * Iz * G * J) / L**2  
# gamma = table2_value(K, eps, delta)               # NOTE:UDL          
# P_c = gamma * np.sqrt(E * Iz * G * J) / L**3      # NOTE:UDL 

# Linear eigenvalue analysis 
eigenvalues, eigenvectors = beam.solve_eigen(num_modes=1) 
for n in range(len(eigenvalues)):
    print(f"mode {n+1}: Pcr analytic = {P_c:.4e} | Pcr fea {eigenvalues[n]:.4e} | error = {(P_c-eigenvalues[n])/P_c*100:.2f} %")          
    # beam.show_mode_shape(eigenvectors[n], scale=5, cross_section_scale=3/4)

# # CANTILEVER 
# # Anderson & Trahair (1972) "Stability of Monosymmetric Beams and Cantilevers"
# K_values = np.array([0, 0.1, 0.3, 1.0, 3.0])
# epsilon_values = np.array([0.6, 0.3, 0.0, -0.3, -0.6])
# delta_values = np.array([-0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6])

# table3_data = np.array([
#     [[2.49, 1.94, 1.65, 1.53, 1.42, 1.23, 1.00],
#      [5.39, 3.65, 2.83, 2.50, 2.22, 1.77, 1.30],
#      [9.57, 6.47, 4.74, 4.01, 3.39, 2.43, 1.55],
#      [11.11, 7.69, 5.67, 4.78, 3.99, 2.75, 1.63],
#      [11.71, 8.15, 6.02, 5.06, 4.21, 3.44, 1.64]],
#     [[2.56, 2.00, 1.71, 1.59, 1.47, 1.28, 1.06],
#      [5.58, 3.79, 2.95, 2.62, 2.33, 1.88, 1.41],
#      [9.88, 6.78, 5.03, 4.30, 3.66, 2.68, 1.80],
#      [11.42, 8.05, 6.03, 5.14, 4.35, 3.11, 2.01],
#      [12.02, 8.52, 6.40, 5.45, 4.60, 3.27, 2.09]],
#     [[2.71, 2.13, 1.84, 1.71, 1.60, 1.41, 1.18],
#      [5.95, 4.09, 3.22, 2.88, 2.58, 2.11, 1.62],
#      [10.64, 7.50, 5.72, 4.96, 4.30, 3.25, 2.26],
#      [12.36, 8.95, 6.93, 6.04, 5.24, 3.96, 2.70],
#      [13.03, 9.48, 7.36, 6.42, 5.58, 4.21, 2.88]],
#     [[3.46, 2.77, 2.43, 2.29, 2.16, 1.93, 1.67],
#      [7.48, 5.33, 4.33, 3.93, 3.59, 3.04, 2.44],
#      [13.61, 10.42, 8.51, 7.64, 6.85, 5.51, 4.05],
#      [16.18, 12.99, 11.06, 10.17, 9.33, 7.82, 5.98],
#      [17.24, 13.94, 11.96, 11.05, 10.20, 8.67, 6.81]],
#     [[8.06, 6.75, 6.06, 5.76, 5.48, 5.00, 4.40],
#      [13.92, 11.15, 9.68, 9.04, 8.46, 7.47, 6.29],
#      [22.12, 18.51, 16.24, 15.17, 14.15, 12.27, 9.95],
#      [27.74, 24.53, 22.43, 21.40, 20.37, 18.36, 15.46],
#      [30.79, 27.80, 25.88, 24.94, 24.02, 22.22, 19.62]]
# ])

# interp_table3 = RegularGridInterpolator(
#     (K_values, epsilon_values, delta_values),
#     table3_data, method='cubic', bounds_error=False, fill_value=None
# )

# def table3_value(K, eps, delta):
#     return float(interp_table3(np.array([K, eps, delta]))[0])

# table4_data = np.array([
#     [[2.25, 2.89, 3.49, 3.85, 4.27, 5.32, 7.64],
#      [2.84, 4.18, 5.62, 6.60, 7.83, 11.24, 19.76],
#      [3.26, 7.49, 9.95, 12.89, 16.54, 25.54, 41.67],
#      [3.37, 8.66, 13.35, 18.27, 23.74, 35.60, 54.17],
#      [3.41, 9.38, 15.12, 21.60, 28.36, 42.18, 62.72]],

#     [[2.52, 3.17, 3.77, 4.13, 4.56, 5.63, 8.02],
#      [3.35, 4.68, 6.14, 7.14, 8.39, 11.89, 20.60],
#      [4.56, 7.44, 11.36, 14.29, 17.90, 26.77, 42.34],
#      [5.65, 9.32, 16.07, 20.51, 25.63, 36.88, 54.57],
#      [6.32, 11.37, 18.89, 24.35, 30.49, 43.42, 62.89]],

#     [[3.10, 3.80, 4.44, 4.83, 5.27, 6.39, 8.89],
#      [4.29, 5.78, 7.35, 8.41, 9.73, 13.40, 22.51],
#      [6.40, 9.99, 14.42, 17.53, 21.26, 30.31, 46.21],
#      [8.95, 14.73, 21.46, 25.83, 30.78, 41.76, 59.40],
#      [10.63, 17.57, 25.67, 30.87, 36.62, 49.02, 68.19]],

#     [[5.75, 6.74, 7.60, 8.11, 8.69, 10.08, 13.03],
#      [8.39, 10.61, 12.76, 14.13, 15.76, 20.04, 29.98],
#      [14.22, 20.25, 26.31, 30.04, 34.20, 43.60, 59.47],
#      [24.18, 33.55, 41.32, 45.65, 50.24, 60.13, 76.30],
#      [31.14, 41.33, 49.74, 54.40, 59.34, 69.92, 87.01]],

#     [[17.71, 20.19, 22.24, 23.38, 24.66, 27.60, 33.29],
#      [25.46, 30.43, 34.71, 37.22, 40.01, 46.55, 59.11],
#      [40.97, 51.34, 59.85, 64.57, 69.55, 80.16, 97.15],
#      [66.38, 80.01, 89.74, 94.66, 99.64, 109.8, 125.3],
#      [87.84, 100.6, 109.5, 114.1, 118.7, 128.3, 143.2]]
# ])

# interp_table4 = RegularGridInterpolator(
#     (K_values, epsilon_values, delta_values),
#     table4_data, method='cubic', bounds_error=False, fill_value=None
# )

# def table4_value(K, eps, delta):
#     return float(interp_table4(np.array([K, eps, delta]))[0])

# def table1_value(K, eps, delta):
#     point = np.array([K, eps, delta])
#     # print(interp_table1(point))
#     return float(interp_table3(point)[0])

# # Create a beam structure 
# n = 10
# L = 3 # m3
# beam = Frame() 
# beam.add_nodes([[L*i/n, 0, 0] for i in range(n+1)])

# # Steel properties
# E = 210e9   # N/m2
# G = 80e9    # N/m2 
# rho = 7850  # kg/m3

# # UB127x76x13 section properties
# A = 1650e-6    # m2
# Iy = 4.73e-6   # m4 
# Iz = 0.557e-6  # m4
# J = 0.0285e-6  # m4
# Iw = 0.002e-6  # m6

# beam.add_material("steel", LinearElastic(rho=rho, E=E, G=G))
# beam.add_section("UB127x76x13", Section(A=A, Iy=Iy, Iz=Iz, J=J, Iw=Iw, y0=0, z0=0))

# # Add element 
# beam.add_elements([[i, i+1] for i in range(n)], "steel", "UB127x76x13", element_class=ThinWalledBeamElement) 

# # Add boundary conditions; Global (ux, uy, uz, θx, θy, θz, φ); 0=fixed, 1=free 
# beam.add_boundary_condition(0, [0, 0, 0, 0, 0, 0, 0], BoundaryCondition) 

# # Add loads; Global (ux, uy, uz, θx, θy, θz, φ)
# beam.add_nodal_load(n, [0, 0, -1, 0, 0, 0, 0], NodalLoad) # N 
# # [beam.add_uniform_load(element_id=e, forces=[0, 0, -1]) for e in range(n)] # NOTE:UDL

# # Buckling capacity 
# K = np.sqrt(np.pi**2 * E * Iw / (G * J * L**2))
# delta = 0  # monosymmetry parameters (0 since symmetric) 
# eps = 0    # load height parameters (0 since load applied at centroid)
# gamma = table3_value(K, eps, delta)
# # gamma = table4_value(K, eps, delta) # NOTE:UDL
# P_c = gamma * np.sqrt(E * Iz * G * J) / L**2 
# # P_c = gamma * np.sqrt(E * Iz * G * J) / L**3 # NOTE:UDL

# # Linear eigenvalue analysis 
# eigenvalues, eigenvectors = beam.solve_eigen(num_modes=1) 
# for n in range(len(eigenvalues)):
#     print(f"mode {n+1}: Pcr analytic = {P_c:.4e} | Pcr fea {eigenvalues[n]:.4e} | error = {(P_c-eigenvalues[n])/P_c*100:.2f} %") 
#     beam.show_mode_shape(eigenvectors[n], scale=2, cross_section_scale=1.5)

