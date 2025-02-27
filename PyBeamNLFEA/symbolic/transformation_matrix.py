import sympy as sp
# from sympy import MatrixSymbol, Matrix, eye, zeros
from sympy.physics.quantum import TensorProduct

x0, y0 = sp.symbols('x0, y0')

# Relating shear centre variables to centroid variables via matrix C, we have 
# ξ = ξ̄^ + θy₀        
# ξ' = ξ̄^ + θ'y₀     
# etc.
C1 = sp.eye(7)
C1[1, 3] = y0  
C1[5, 6] = y0  
C1[2, 3] = x0  
C1[4, 6] = x0  
C = TensorProduct(sp.eye(2), C1)
# sp.pprint(C)

# Rotation matrix R of direction cosines 
ax, ay, az = sp.symbols('ax, ay, az')
bx, by, bz = sp.symbols('bx, by, bz')
cx, cy, cz = sp.symbols('cx, cy, cz')
R = sp.Matrix([[ax, ay, az], 
               [bx, by, bz], 
               [cx, cy, cz]])

# Transformation matrix Q 
Q1 = sp.zeros(7)
Q1[0:3, 0:3] = R
Q1[3:6, 3:6] = R
Q1[6,6] = 1
Q = TensorProduct(sp.eye(2), Q1)
sp.pprint(Q)

T = C * Q * C.inv()
sp.pprint(sp.simplify(T)) 



