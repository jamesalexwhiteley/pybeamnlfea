import sympy as sp

# Author: James Whiteley (github.com/jamesalexwhiteley)

# ============================================== #
# Force vector  
# ============================================== # 

z, L = sp.symbols('z L')
xi = z/L

# Node 1: [w, u, v, θ, -v', u', θ'] indices [0, 1, 2, 3, 4, 5, 6]
# Node 2: [w, u, v, θ, -v', u', θ'] indices [7, 8, 9, 10, 11, 12, 13]

# Polynomial coefficients a, b, c and b in terms of nodal DOFs 
A_w = sp.Matrix([
    [1, 0], 
    [1, 1]   
])

# For u and θ
A_cubic = sp.Matrix([
    [1, 0, 0, 0],       # f(0)
    [0, 1, 0, 0],       # f'(0)
    [1, 1, 1, 1],       # f(L)
    [0, 1, 2, 3]        # f'(L)
])

# For v (and -v')
A_cubic_v = sp.Matrix([
    [1, 0, 0, 0],        # v(0)
    [0, -1, 0, 0],       # -v'(0)  
    [1, 1, 1, 1],        # v(L)
    [0, -1, -2, -3]      # -v'(L)  
])

A = sp.zeros(14, 14)
w_indices = [0, 7]         
u_indices = [1, 5, 8, 12]   
v_indices = [2, 4, 9, 11]   
theta_indices = [3, 6, 10, 13] 

def fill_block(A, start_indices, block):
    n = len(start_indices)
    for i in range(n):
        for j in range(n):
            A[start_indices[i], start_indices[j]] = block[i,j]

fill_block(A, w_indices, A_w)
fill_block(A, u_indices, A_cubic)
fill_block(A, v_indices, A_cubic_v) 
fill_block(A, theta_indices, A_cubic)

# Boundary condition vectors
def create_unit_vector(sise, pos):
    vec = sp.zeros(sise, 1)
    vec[pos] = 1
    return vec

b_vectors = [create_unit_vector(14, i) for i in range(14)]
c_vectors = [A.inv() * b for b in b_vectors]

# Shape functions
def create_linear_shape_function(coeff):
    return coeff[0] + coeff[1]*xi

def create_cubic_shape_function(coeff, with_L=False):
    base = coeff[0] + coeff[1]*xi + coeff[2]*xi**2 + coeff[3]*xi**3
    return L*base if with_L else base

N = [None] * 14

def extract_coeffs(c_vector, indices):
    return [c_vector[i] for i in indices]

# Using correct indices 
for i in range(14):
    if i in w_indices:
        coeffs = extract_coeffs(c_vectors[i], w_indices)
        N[i] = create_linear_shape_function(coeffs)
    elif i in u_indices:
        coeffs = extract_coeffs(c_vectors[i], u_indices)
        N[i] = create_cubic_shape_function(coeffs, i in [5, 12])
    elif i in v_indices:
        coeffs = extract_coeffs(c_vectors[i], v_indices)
        N[i] = create_cubic_shape_function(coeffs, i in [4, 11])
    elif i in theta_indices:
        coeffs = extract_coeffs(c_vectors[i], theta_indices)
        N[i] = create_cubic_shape_function(coeffs, i in [6, 13])

N = [sp.simplify(n) for n in N]

# Compute derivatives of shape functions
def get_derivatives(N_list, max_order=2):
    derivatives = []
    for N_func in N_list:
        d_list = [N_func]
        for order in range(1, max_order+1):
            d_list.append(sp.diff(N_func, z, order))
        derivatives.append(d_list)
    return derivatives

all_derivs = get_derivatives(N, 2)

E, G = sp.symbols('E G')
A = sp.Symbol('A')
Ix, Iy = sp.symbols('I_x I_y')
Is = sp.Symbol('J')
Iw = sp.Symbol('I_w')

P0 = sp.Symbol('P0')
Mx0, My0 = sp.symbols('Mx0, My0')
y0, x0 = sp.symbols('y0, x0')
B0 = sp.Symbol('B0')

# Generalised coordinates 
q = sp.symbols('q0:14')

# Express displacements in terms of shape functions
w_expr = sum(N[i] * q[i] for i in w_indices)
u_expr = sum(N[i] * q[i] for i in u_indices)
v_expr = sum(N[i] * q[i] for i in v_indices)
theta_expr = sum(N[i] * q[i] for i in theta_indices)

# Derivatives for strain energy
w_prime = sp.diff(w_expr, z)
u_prime = sp.diff(u_expr, z)
v_prime = sp.diff(v_expr, z)
u_double_prime = sp.diff(u_expr, z, 2)
v_double_prime = sp.diff(v_expr, z, 2)
theta_prime = sp.diff(theta_expr, z)
theta_double_prime = sp.diff(theta_expr, z, 2)

# Force vector
F = sp.zeros(14, 1)

# ============================================== #
# 1. Linear elastic terms 
# ============================================== #

# Axial strain energy contribution (w terms)
for i in w_indices: 
    dNi = all_derivs[i][1]  
    integrand = E*A*w_prime*dNi
    F[i] += sp.integrate(integrand, (z, 0, L))  

# Bending strain energy contribution (u terms)
for i in u_indices:
    d2Ni = all_derivs[i][2]
    integrand = E*Iy*u_double_prime*d2Ni
    F[i] += sp.integrate(integrand, (z, 0, L))

# Bending strain energy contribution (v terms)
for i in v_indices:
    d2Ni = all_derivs[i][2]  
    integrand = E*Ix*v_double_prime*d2Ni
    F[i] += sp.integrate(integrand, (z, 0, L))

# Torsional and warping strain energy contribution (θ terms)
for i in theta_indices:
    dNi = all_derivs[i][1]  
    d2Ni = all_derivs[i][2] 
    
    integrand_torsion = G*Is*theta_prime*dNi
    integrand_warping = E*Iw*theta_double_prime*d2Ni
    
    F_torsion = sp.integrate(integrand_torsion, (z, 0, L))
    F_warping = sp.integrate(integrand_warping, (z, 0, L))
    
    F[i] += sp.simplify(F_torsion + F_warping)

# ============================================== #
# 2. Initial load terms 
# ============================================== #

# P0 terms - initial axial force contribution
for i in w_indices:
    dNi = all_derivs[i][1]
    integrand = P0*dNi
    F[i] += sp.integrate(integrand, (z, 0, L))

# Mx0 terms - initial moment contribution
for i in v_indices:
    d2Ni = all_derivs[i][2]
    integrand = -Mx0*d2Ni
    F[i] += sp.integrate(integrand, (z, 0, L))

# My0 terms - initial moment contribution
for i in u_indices:
    d2Ni = all_derivs[i][2]
    integrand = My0*d2Ni
    F[i] += sp.integrate(integrand, (z, 0, L))

# B0 terms - initial bimoment contribution
for i in theta_indices:
    d2Ni = all_derivs[i][2]
    integrand = B0*d2Ni
    F[i] += sp.integrate(integrand, (z, 0, L))

# ============================================== #
# 3. P-Delta effects 
# ============================================== #

# P-Delta effects for u (lateral displacement)
for i in u_indices:
    dNi = all_derivs[i][1]
    integrand = P0*u_prime*dNi  # P-Delta term
    F[i] += sp.integrate(integrand, (z, 0, L))

# P-Delta effects for v (lateral displacement)
for i in v_indices:
    dNi = all_derivs[i][1]
    integrand = P0*v_prime*dNi  # P-Delta term
    F[i] += sp.integrate(integrand, (z, 0, L))

# ============================================== #
# 4. Load eccentricity effects 
# ============================================== #

if False:  
    # Eccentricity coupling - u and θ
    for i in u_indices:
        dNi = all_derivs[i][1]
        integrand = P0*y0*theta_prime*dNi
        F[i] += sp.integrate(integrand, (z, 0, L))
    
    for i in theta_indices:
        dNi = all_derivs[i][1]
        integrand = P0*y0*u_prime*dNi
        F[i] += sp.integrate(integrand, (z, 0, L))
    
    # Eccentricity coupling - v and θ
    for i in v_indices:
        dNi = all_derivs[i][1]
        integrand = -P0*x0*theta_prime*dNi
        F[i] += sp.integrate(integrand, (z, 0, L))
    
    for i in theta_indices:
        dNi = all_derivs[i][1]
        integrand = -P0*x0*v_prime*dNi
        F[i] += sp.integrate(integrand, (z, 0, L))

for i in range(14):
    F[i] = sp.simplify(F[i])
print("Simplified Internal Force Vector F(q):")
for i in range(14):
    sp.pprint(F[i]) 