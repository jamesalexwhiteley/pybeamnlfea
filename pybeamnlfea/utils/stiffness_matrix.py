import numpy as np 
from scipy.sparse import lil_matrix

# Author: James Whiteley (github.com/jamesalexwhiteley)

# def thin_wall_stiffness_matrix(E, G, A, Iy, Iz, Iw, J, L, 
#                                 P0=0, My0=0, Mz0=0, B0_bar=0, 
#                                 W_bar=0, y0=0, z0=0, beta_y=0, beta_z=0, r=0):
    
#     """
#     Create the element stiffness matrix for a 3D thin-walled beam element.
    
#     Args:
#         E : float
#             Young's modulus
#         G : float
#             Shear modulus
#         A : float
#             Cross-sectional area
#         Iy, Iz : float
#             Second moments of area about y and z axes
#         Iw : float
#             Warping constant
#         J : float
#             Torsion constant
#         L : float
#             Element length
#         P0 : float, optional
#             Axial force
#         My0, Mz0 : float, optional
#             Bending moments about y and z axes
#         B0_bar, W_bar : float, optional
#             Bimoment and warping parameter 
#         y0, z0 : float, optional
#             Coordinates of the shear center
#         beta_y, beta_z : float, optional
#             Rotation Args:
#         r : float, optional
#             Polar radius of gyration
        
#     """

#     # Initialise a sparse matrix (using lil_matrix for efficient assembly)
#     K = lil_matrix((14, 14))
    
#     def set_symmetric(i, j, value):
#         K[i, j] = value
#         if i != j:
#             K[j, i] = value
    
#     # Axial terms
#     set_symmetric(0, 0, A*E/L)
#     set_symmetric(0, 7, -A*E/L)
#     set_symmetric(7, 7, A*E/L)
    
#     # Z-direction bending terms
#     set_symmetric(1, 1, 12*E*Iz/(L**3) + 6*P0/(5*L))
#     set_symmetric(1, 8, -12*E*Iz/(L**3) - 6*P0/(5*L))
#     set_symmetric(8, 8, 12*E*Iz/(L**3) + 6*P0/(5*L))
    
#     set_symmetric(1, 3, -3*Mz0/(5*L) + 3*P0*z0/(5*L))
#     set_symmetric(1, 10, 3*Mz0/(5*L) - 3*P0*z0/(5*L))
#     set_symmetric(8, 10, -3*Mz0/(5*L) + 3*P0*z0/(5*L))
#     set_symmetric(3, 8, -3*Mz0/(5*L) + 3*P0*z0/(5*L))
    
#     set_symmetric(1, 5, 6*E*Iz/(L**2) + P0/10)
#     set_symmetric(1, 12, 6*E*Iz/(L**2) + P0/10)
#     set_symmetric(5, 8, -6*E*Iz/(L**2) - P0/10)
#     set_symmetric(8, 12, -6*E*Iz/(L**2) - P0/10)
    
#     cross_term_val = -Mz0/20 + P0*z0/20
#     set_symmetric(1, 6, cross_term_val)
#     set_symmetric(1, 13, cross_term_val)
#     set_symmetric(8, 13, -cross_term_val)
#     set_symmetric(5, 10, -cross_term_val)
#     set_symmetric(3, 5, cross_term_val)
#     set_symmetric(3, 12, cross_term_val)
#     set_symmetric(6, 8, -cross_term_val)
#     set_symmetric(10, 12, -cross_term_val)
    
#     # Y-direction bending terms
#     set_symmetric(2, 2, 12*E*Iy/(L**3) + 6*P0/(5*L))
#     set_symmetric(2, 9, -12*E*Iy/(L**3) - 6*P0/(5*L))
#     set_symmetric(9, 9, 12*E*Iy/(L**3) + 6*P0/(5*L))
    
#     set_symmetric(2, 3, -3*My0/(5*L) - 3*P0*y0/(5*L))
#     set_symmetric(2, 10, 3*My0/(5*L) + 3*P0*y0/(5*L))
#     set_symmetric(3, 9, -3*My0/(5*L) - 3*P0*y0/(5*L))
#     set_symmetric(9, 10, 3*My0/(5*L) + 3*P0*y0/(5*L))
    
#     set_symmetric(2, 4, -6*E*Iy/(L**2) - P0/10)
#     set_symmetric(2, 11, -6*E*Iy/(L**2) - P0/10)
#     set_symmetric(4, 9, 6*E*Iy/(L**2) + P0/10)
#     set_symmetric(9, 11, 6*E*Iy/(L**2) + P0/10)
    
#     cross_term_val2 = -My0/20 - P0*y0/20
#     set_symmetric(2, 6, cross_term_val2)
#     set_symmetric(2, 13, cross_term_val2)
#     set_symmetric(9, 13, -cross_term_val2)
#     set_symmetric(3, 4, -cross_term_val2)
#     set_symmetric(3, 11, -cross_term_val2)
#     set_symmetric(10, 11, cross_term_val2)
#     set_symmetric(4, 10, cross_term_val2)
#     set_symmetric(6, 9, -cross_term_val2)
    
#     # Warping and torsion terms
#     warping_term = 12*E*Iw/(L**3) + 6*G*J/(5*L) + 6*P0*r**2/(5*L) + 3*Mz0*beta_z/(5*L) - 3*My0*beta_y/(5*L) - 6*B0_bar*W_bar/(5*L)
#     set_symmetric(3, 3, warping_term)
#     set_symmetric(3, 10, -warping_term)
#     set_symmetric(10, 10, warping_term)
    
#     warping_term2 = 6*E*Iw/(L**2) + G*J/10 + P0*r**2/10 + Mz0*beta_z/20 - My0*beta_y/20 - B0_bar*W_bar/10
#     set_symmetric(3, 6, warping_term2)
#     set_symmetric(3, 13, warping_term2)
#     set_symmetric(6, 10, -warping_term2)
#     set_symmetric(10, 13, -warping_term2)
    
#     warping_term3 = 4*E*Iw/L + 2*G*J*L/15 + 2*L*P0*r**2/15 + L*Mz0*beta_z/15 - L*My0*beta_y/15 - 2*B0_bar*L*W_bar/15
#     set_symmetric(6, 6, warping_term3)
#     set_symmetric(13, 13, warping_term3)
    
#     warping_term4 = 2*E*Iw/L - G*J*L/30 - L*P0*r**2/30 - L*Mz0*beta_z/60 + L*My0*beta_y/60 + B0_bar*L*W_bar/30
#     set_symmetric(6, 13, warping_term4)

#     # Additional terms
#     set_symmetric(4, 4, 4*E*Iy/L + 2*L*P0/15)
#     set_symmetric(11, 11, 4*E*Iy/L + 2*L*P0/15)
    
#     set_symmetric(4, 6, L*My0/15 + L*P0*y0/15)
#     set_symmetric(11, 13, L*My0/15 + L*P0*y0/15)
    
#     set_symmetric(4, 13, -L*My0/60 - L*P0*y0/60)
#     set_symmetric(6, 11, -L*My0/60 - L*P0*y0/60)
    
#     set_symmetric(4, 11, 2*E*Iy/L - L*P0/30)
    
#     set_symmetric(5, 5, 4*E*Iz/L + 2*L*P0/15)
#     set_symmetric(12, 12, 4*E*Iz/L + 2*L*P0/15)
    
#     set_symmetric(5, 6, -L*Mz0/15 + L*P0*z0/15)
#     set_symmetric(12, 13, -L*Mz0/15 + L*P0*z0/15)
    
#     set_symmetric(5, 12, 2*E*Iz/L - L*P0/30)
    
#     set_symmetric(5, 13, L*Mz0/60 - L*P0*z0/60)
#     set_symmetric(6, 12, L*Mz0/60 - L*P0*z0/60)

#     return K.tocsr()

# def thin_wall_stiffness_matrix(E, G, A, Iy, Iz, Iw, J, L, 
#                                 P0=0, My0=0, Mz0=0, B0_bar=0, 
#                                 W_bar=0, y0=0, z0=0, beta_y=0, beta_z=0, r=0):
    
#     """
#     Create the element stiffness matrix for a 3D thin-walled beam element.
    
#     Args:
#         E : float
#             Young's modulus
#         G : float
#             Shear modulus
#         A : float
#             Cross-sectional area
#         Iy, Iz : float
#             Second moments of area about y and z axes
#         Iw : float
#             Warping constant
#         J : float
#             Torsion constant
#         L : float
#             Element length
#         P0 : float, optional
#             Axial force
#         My0, Mz0 : float, optional
#             Bending moments about y and z axes
#         B0_bar, W_bar : float, optional
#             Bimoment and warping parameter 
#         y0, z0 : float, optional
#             Coordinates of the shear center
#         beta_y, beta_z : float, optional
#             Rotation Args:
#         r : float, optional
#             Polar radius of gyration
        
#     """

#     # Initialise a sparse matrix (using lil_matrix for efficient assembly)
#     K = lil_matrix((14, 14))
    
#     def set_symmetric(i, j, value):
#         K[i-1, j-1] = value
#         if (i-1) != (j-1):
#             K[j-1, i-1] = value
    
#     # Axial terms
#     set_symmetric(1, 1, A*E/L)
#     set_symmetric(1, 8, -A*E/L)
#     set_symmetric(8, 8, A*E/L)
    
#     # Z-direction bending terms
#     set_symmetric(2, 2, 12*E*Iz/(L**3) + 6*P0/(5*L))
#     set_symmetric(2, 9, -12*E*Iz/(L**3) - 6*P0/(5*L))
#     set_symmetric(9, 9, 12*E*Iz/(L**3) + 6*P0/(5*L))
    
#     set_symmetric(2, 4, -3*Mz0/(5*L) + 3*P0*z0/(5*L))
#     set_symmetric(2, 11, 3*Mz0/(5*L) - 3*P0*z0/(5*L))
#     set_symmetric(9, 11, -3*Mz0/(5*L) + 3*P0*z0/(5*L))
#     set_symmetric(4, 9, -3*Mz0/(5*L) + 3*P0*z0/(5*L))
    
#     set_symmetric(2, 6, 6*E*Iz/(L**2) + P0/10)
#     set_symmetric(2, 13, 6*E*Iz/(L**2) + P0/10)
#     set_symmetric(6, 9, -6*E*Iz/(L**2) - P0/10)
#     set_symmetric(9, 13, -6*E*Iz/(L**2) - P0/10)
    
#     cross_term_val = -Mz0/20 + P0*z0/20
#     set_symmetric(2, 7, cross_term_val)
#     set_symmetric(2, 14, cross_term_val)
#     set_symmetric(9, 14, -cross_term_val)
#     set_symmetric(6, 11, -cross_term_val)
#     set_symmetric(4, 6, cross_term_val)
#     set_symmetric(4, 13, cross_term_val)
#     set_symmetric(7, 9, -cross_term_val)
#     set_symmetric(11, 13, -cross_term_val)
    
#     # Y-direction bending terms
#     set_symmetric(3, 3, 12*E*Iy/(L**3) + 6*P0/(5*L))
#     set_symmetric(3, 10, -12*E*Iy/(L**3) - 6*P0/(5*L))
#     set_symmetric(10, 10, 12*E*Iy/(L**3) + 6*P0/(5*L))
    
#     set_symmetric(3, 4, -3*My0/(5*L) - 3*P0*y0/(5*L))
#     set_symmetric(3, 11, 3*My0/(5*L) + 3*P0*y0/(5*L))
#     set_symmetric(4, 10, -3*My0/(5*L) - 3*P0*y0/(5*L))
#     set_symmetric(10, 11, 3*My0/(5*L) + 3*P0*y0/(5*L))
    
#     set_symmetric(3, 5, -6*E*Iy/(L**2) - P0/10)
#     set_symmetric(3, 12, -6*E*Iy/(L**2) - P0/10)
#     set_symmetric(5, 10, 6*E*Iy/(L**2) + P0/10)
#     set_symmetric(10, 12, 6*E*Iy/(L**2) + P0/10)
    
#     cross_term_val2 = -My0/20 - P0*y0/20
#     set_symmetric(3, 7, cross_term_val2)
#     set_symmetric(3, 14, cross_term_val2)
#     set_symmetric(10, 14, -cross_term_val2)
#     set_symmetric(4, 5, -cross_term_val2)
#     set_symmetric(4, 12, -cross_term_val2)
#     set_symmetric(11, 12, cross_term_val2)
#     set_symmetric(5, 11, cross_term_val2)
#     set_symmetric(7, 10, -cross_term_val2)
    
#     # Warping and torsion terms
#     warping_term = 12*E*Iw/(L**3) + 6*G*J/(5*L) + 6*P0*r**2/(5*L) + 3*Mz0*beta_z/(5*L) - 3*My0*beta_y/(5*L) - 6*B0_bar*W_bar/(5*L)
#     set_symmetric(4, 4, warping_term)
#     set_symmetric(4, 11, -warping_term)
#     set_symmetric(11, 11, warping_term)
    
#     warping_term2 = 6*E*Iw/(L**2) + G*J/10 + P0*r**2/10 + Mz0*beta_z/20 - My0*beta_y/20 - B0_bar*W_bar/10
#     set_symmetric(4, 7, warping_term2)
#     set_symmetric(4, 14, warping_term2)
#     set_symmetric(7, 11, -warping_term2)
#     set_symmetric(11, 14, -warping_term2)
    
#     warping_term3 = 4*E*Iw/L + 2*G*J*L/15 + 2*L*P0*r**2/15 + L*Mz0*beta_z/15 - L*My0*beta_y/15 - 2*B0_bar*L*W_bar/15
#     set_symmetric(7, 7, warping_term3)
#     set_symmetric(14, 14, warping_term3)
    
#     warping_term4 = 2*E*Iw/L - G*J*L/30 - L*P0*r**2/30 - L*Mz0*beta_z/60 + L*My0*beta_y/60 + B0_bar*L*W_bar/30
#     set_symmetric(7, 14, warping_term4)

#     # Additional terms
#     set_symmetric(5, 5, 4*E*Iy/L + 2*L*P0/15)
#     set_symmetric(12, 12, 4*E*Iy/L + 2*L*P0/15)
    
#     set_symmetric(5, 7, L*My0/15 + L*P0*y0/15)
#     set_symmetric(12, 14, L*My0/15 + L*P0*y0/15)
    
#     set_symmetric(5, 14, -L*My0/60 - L*P0*y0/60)
#     set_symmetric(7, 12, -L*My0/60 - L*P0*y0/60)
    
#     set_symmetric(5, 12, 2*E*Iy/L - L*P0/30)
    
#     set_symmetric(6, 6, 4*E*Iz/L + 2*L*P0/15)
#     set_symmetric(13, 13, 4*E*Iz/L + 2*L*P0/15)
    
#     set_symmetric(6, 7, -L*Mz0/15 + L*P0*z0/15)
#     set_symmetric(13, 14, -L*Mz0/15 + L*P0*z0/15)
    
#     set_symmetric(6, 13, 2*E*Iz/L - L*P0/30)
    
#     set_symmetric(6, 14, L*Mz0/60 - L*P0*z0/60)
#     set_symmetric(7, 13, L*Mz0/60 - L*P0*z0/60)

#     return K.tocsr()

from scipy.sparse import lil_matrix

# def thin_wall_stiffness_matrix(E, G, A, Iy, Iz, Iw, J, L, 
#                                 P=0, My1=0, My2=0, Mz1=0, Mz2=0,
#                                 Mw=0, y0=0, z0=0, beta_y=0, beta_z=0, beta_w=0, r1=0,
#                                 include_elastic=True, include_geometric=True):
    
#     K = lil_matrix((14, 14))
    
#     def set_symmetric(i, j, value):
#         K[i-1, j-1] = K[i-1, j-1] + value
#         if (i-1) != (j-1):
#             K[j-1, i-1] = K[j-1, i-1] + value

#     # Shear terms not implemented/verified 
#     Vy, Vz = 0, 0
    
#     # =========================================================================
#     # ELASTIC STIFFNESS MATRIX 
#     # =========================================================================
#     if include_elastic:
#         set_symmetric(1, 1, A*E/L)
#         set_symmetric(1, 8, -A*E/L)
#         set_symmetric(8, 8, A*E/L)
        
#         set_symmetric(2, 2, 12*E*Iz/(L**3))
#         set_symmetric(2, 9, -12*E*Iz/(L**3))
#         set_symmetric(9, 9, 12*E*Iz/(L**3))
        
#         set_symmetric(2, 6, 6*E*Iz/(L**2))
#         set_symmetric(2, 13, 6*E*Iz/(L**2))
#         set_symmetric(6, 9, -6*E*Iz/(L**2))
#         set_symmetric(9, 13, -6*E*Iz/(L**2))
        
#         set_symmetric(3, 3, 12*E*Iy/(L**3))
#         set_symmetric(3, 10, -12*E*Iy/(L**3))
#         set_symmetric(10, 10, 12*E*Iy/(L**3))
        
#         set_symmetric(3, 5, -6*E*Iy/(L**2))
#         set_symmetric(3, 12, -6*E*Iy/(L**2))
#         set_symmetric(5, 10, 6*E*Iy/(L**2))
#         set_symmetric(10, 12, 6*E*Iy/(L**2))
        
#         set_symmetric(4, 4, 12*E*Iw/(L**3) + 6*G*J/(5*L))
#         set_symmetric(4, 11, -12*E*Iw/(L**3) - 6*G*J/(5*L))
#         set_symmetric(11, 11, 12*E*Iw/(L**3) + 6*G*J/(5*L))
        
#         set_symmetric(4, 7, 6*E*Iw/(L**2) + G*J/10)
#         set_symmetric(4, 14, 6*E*Iw/(L**2) + G*J/10)
#         set_symmetric(7, 11, -6*E*Iw/(L**2) - G*J/10)
#         set_symmetric(11, 14, -6*E*Iw/(L**2) - G*J/10)
        
#         set_symmetric(7, 7, 4*E*Iw/L + 2*G*J*L/15)
#         set_symmetric(14, 14, 4*E*Iw/L + 2*G*J*L/15)
        
#         set_symmetric(7, 14, 2*E*Iw/L - G*J*L/30)
        
#         set_symmetric(5, 5, 4*E*Iy/L)
#         set_symmetric(12, 12, 4*E*Iy/L)
#         set_symmetric(5, 12, 2*E*Iy/L)
        
#         set_symmetric(6, 6, 4*E*Iz/L)
#         set_symmetric(13, 13, 4*E*Iz/L)
#         set_symmetric(6, 13, 2*E*Iz/L)
    
#     # =========================================================================
#     # GEOMETRIC STIFFNESS MATRIX 
#     # =========================================================================
#     if include_geometric:
#         set_symmetric(2, 2, 6*P/(5*L))
#         set_symmetric(2, 4, -6*P*z0/(5*L) + 3*(My1 - My2)/(5*L) + Vz/2)
#         set_symmetric(2, 6, P/10)
#         set_symmetric(2, 7, -(1/10)*(P*z0 + My2 - Vz*L))
#         set_symmetric(2, 9, -6*P/(5*L))
#         set_symmetric(2, 11, 6*P*z0/(5*L) - 3*(My1 - My2)/(5*L) - Vz/2)
#         set_symmetric(2, 13, P/10)
#         set_symmetric(2, 14, -(1/10)*(P*z0 - My1 + Vz*L))
        
#         set_symmetric(3, 3, 6*P/(5*L))
#         set_symmetric(3, 4, 3*(2*P*y0 + Mz1 - Mz2)/(5*L) - Vy/2)
#         set_symmetric(3, 5, -P/10)
#         set_symmetric(3, 7, (1/10)*(P*y0 - Mz2 - Vy*L))
#         set_symmetric(3, 10, -6*P/(5*L))
#         set_symmetric(3, 11, -3*(2*P*y0 + Mz1 - Mz2)/(5*L) - Vy/2)
#         set_symmetric(3, 12, -P/10)
#         set_symmetric(3, 14, (1/10)*(P*y0 + Mz1 + Vy*L))
        
#         set_symmetric(4, 4, 6*P*r1/(5*L) + 3*beta_z*(Mz1 - Mz2)/(5*L) 
#                       - 3*beta_y*(My1 - My2)/(5*L) + 6*Mw*beta_w/(5*L)
#                       - Vz*z0/2 - Vy*y0/2)
#         set_symmetric(4, 5, -(1/10)*(P*y0 - Mz2 + Vy*L))
#         set_symmetric(4, 6, -(1/10)*(P*z0 + My2 + Vz*L))
#         set_symmetric(4, 7, (1/10)*(P*r1 - Mz2*beta_z + My2*beta_y + Mw*beta_w))
#         set_symmetric(4, 9, 6*P*z0/(5*L) - 3*(My1 - My2)/(5*L) - Vz/2)
#         set_symmetric(4, 10, -6*P*y0/(5*L) - 3*(Mz1 - Mz2)/(5*L) + Vy/2)
#         set_symmetric(4, 11, -6*P*r1/(5*L) - 3*beta_z*(Mz1 - Mz2)/(5*L)
#                       + 3*beta_y*(My1 - My2)/(5*L) - 6*Mw*beta_w/(5*L))
#         set_symmetric(4, 12, -(1/10)*(P*y0 + Mz1 - Vy*L))
#         set_symmetric(4, 13, -(1/10)*(P*z0 - My1 - Vz*L))
#         set_symmetric(4, 14, (1/10)*(P*r1 + Mz1*beta_z - My1*beta_y + Mw*beta_w))
        
#         set_symmetric(5, 5, 2*P*L/15)
#         set_symmetric(5, 7, -2*P*y0*L/15 - L*(3*Mz1 - Mz2)/30)
#         set_symmetric(5, 9, P/10)
#         set_symmetric(5, 11, (1/10)*(P*y0 - Mz2 + Vy*L))
#         set_symmetric(5, 12, -P*L/30)
#         set_symmetric(5, 14, L*(2*P*y0 + Mz1 - Mz2 - Vy*L)/60)
        
#         set_symmetric(6, 6, 2*P*L/15)
#         set_symmetric(6, 7, -2*P*z0*L/15 + L*(3*My1 - My2)/30)
#         set_symmetric(6, 9, -P/10)
#         set_symmetric(6, 11, -(1/10)*(P*z0 + My1 + Vz*L)) 
#         set_symmetric(6, 13, -P*L/30)
#         set_symmetric(6, 14, L*(2*P*z0 - My1 + My2 - Vz*L)/60)
        
#         set_symmetric(7, 7, 2*P*r1*L/15 + beta_z*L*(3*Mz1 - Mz2)/30
#                       - beta_y*L*(3*My1 - My2)/30 + 2*Mw*beta_w*L/15)
#         set_symmetric(7, 9, -(1/10)*(P*z0 + My1 - Vz*L))  
#         set_symmetric(7, 10, -(1/10)*(P*y0 - Mz1 - Vy*L))
#         set_symmetric(7, 11, -(1/10)*(P*r1 - Mz2*beta_z + My2*beta_y + Mw*beta_w))
#         set_symmetric(7, 12, L*(2*P*y0 + Mz1 - Mz2 + Vy*L)/60)
#         set_symmetric(7, 13, L*(2*P*z0 - My1 + My2 + Vz*L)/60)
#         set_symmetric(7, 14, -P*r1*L/30 - beta_z*L*(Mz1 - Mz2)/60
#                       + beta_y*L*(My1 - My2)/60 + Mw*beta_w*L/30)
        
#         set_symmetric(9, 9, 6*P/(5*L))
#         set_symmetric(9, 11, -6*P*z0/(5*L) + 3*(My1 - My2)/(5*L) - Vz/2)  
#         set_symmetric(9, 13, -P/10)
#         set_symmetric(9, 14, (1/10)*(P*z0 - My1 + Vz*L))
        
#         set_symmetric(10, 10, 6*P/(5*L))
#         set_symmetric(10, 11, 6*P*y0/(5*L) + 3*(Mz1 - Mz2)/(5*L) + Vy/2)
#         set_symmetric(10, 12, P/10)
#         set_symmetric(10, 14, -(1/10)*(P*y0 + Mz1 + Vy*L))
        
#         set_symmetric(11, 11, 6*P*r1/(5*L) + 3*beta_z*(Mz1 - Mz2)/(5*L)
#                       - 3*beta_y*(My1 - My2)/(5*L) + (Vz*z0 + Vy*y0)/2
#                       + 6*Mw*beta_w/(5*L))
#         set_symmetric(11, 12, (1/10)*(P*y0 + Mz2 - Vy*L))
#         set_symmetric(11, 13, (1/10)*(P*z0 - My1 - Vz*L))
#         set_symmetric(11, 14, -(1/10)*(P*r1 + Mz1*beta_z - My1*beta_y + Mw*beta_w))
        
#         set_symmetric(12, 12, 2*P*L/15)
#         set_symmetric(12, 14, 2*P*y0*L/15 - L*(Mz1 - 3*Mz2)/30)
        
#         set_symmetric(13, 13, 2*P*L/15)
#         set_symmetric(13, 14, -2*P*z0*L/15 + L*(My1 - 3*My2)/30)
        
#         set_symmetric(14, 14, 2*P*r1*L/15 + beta_z*L*(Mz1 - 3*Mz2)/30
#                       - beta_y*L*(My1 - 3*My2)/30 + 2*Mw*beta_w*L/15)

#     return K.tocsr()

from scipy.sparse import lil_matrix

def thin_wall_stiffness_matrix(E, G, A, Iy, Iz, Iw, J, L, 
                                P=0, My1=0, My2=0, Mz1=0, Mz2=0,
                                Mw=0, y0=0, z0=0, beta_y=0, beta_z=0, beta_w=0, r1=0,
                                Vy=0, Vz=0,
                                include_elastic=True, include_geometric=True):
    
    K = lil_matrix((14, 14))
    
    def set_symmetric(i, j, value):
        K[i-1, j-1] = K[i-1, j-1] + value
        if (i-1) != (j-1):
            K[j-1, i-1] = K[j-1, i-1] + value
    
    # =========================================================================
    # ELASTIC STIFFNESS MATRIX 
    # =========================================================================
    if include_elastic:
        set_symmetric(1, 1, A*E/L)
        set_symmetric(1, 8, -A*E/L)
        set_symmetric(8, 8, A*E/L)
        
        set_symmetric(2, 2, 12*E*Iz/(L**3))
        set_symmetric(2, 9, -12*E*Iz/(L**3))
        set_symmetric(9, 9, 12*E*Iz/(L**3))
        
        set_symmetric(2, 6, 6*E*Iz/(L**2))
        set_symmetric(2, 13, 6*E*Iz/(L**2))
        set_symmetric(6, 9, -6*E*Iz/(L**2))
        set_symmetric(9, 13, -6*E*Iz/(L**2))
        
        set_symmetric(3, 3, 12*E*Iy/(L**3))
        set_symmetric(3, 10, -12*E*Iy/(L**3))
        set_symmetric(10, 10, 12*E*Iy/(L**3))
        
        set_symmetric(3, 5, -6*E*Iy/(L**2))
        set_symmetric(3, 12, -6*E*Iy/(L**2))
        set_symmetric(5, 10, 6*E*Iy/(L**2))
        set_symmetric(10, 12, 6*E*Iy/(L**2))
        
        set_symmetric(4, 4, 12*E*Iw/(L**3) + 6*G*J/(5*L))
        set_symmetric(4, 11, -12*E*Iw/(L**3) - 6*G*J/(5*L))
        set_symmetric(11, 11, 12*E*Iw/(L**3) + 6*G*J/(5*L))
        
        set_symmetric(4, 7, 6*E*Iw/(L**2) + G*J/10)
        set_symmetric(4, 14, 6*E*Iw/(L**2) + G*J/10)
        set_symmetric(7, 11, -6*E*Iw/(L**2) - G*J/10)
        set_symmetric(11, 14, -6*E*Iw/(L**2) - G*J/10)
        
        set_symmetric(7, 7, 4*E*Iw/L + 2*G*J*L/15)
        set_symmetric(14, 14, 4*E*Iw/L + 2*G*J*L/15)
        
        set_symmetric(7, 14, 2*E*Iw/L - G*J*L/30)
        
        set_symmetric(5, 5, 4*E*Iy/L)
        set_symmetric(12, 12, 4*E*Iy/L)
        set_symmetric(5, 12, 2*E*Iy/L)
        
        set_symmetric(6, 6, 4*E*Iz/L)
        set_symmetric(13, 13, 4*E*Iz/L)
        set_symmetric(6, 13, 2*E*Iz/L)
    
    # =========================================================================
    # GEOMETRIC STIFFNESS MATRIX 
    # =========================================================================
    if include_geometric:
        set_symmetric(2, 2, 6*P/(5*L))
        set_symmetric(2, 4, -6*P*z0/(5*L) + 3*(My1 - My2)/(5*L) + Vz/2)
        set_symmetric(2, 6, P/10)
        set_symmetric(2, 7, -(1/10)*(P*z0 + My2 - Vz*L))
        set_symmetric(2, 9, -6*P/(5*L))
        set_symmetric(2, 11, 6*P*z0/(5*L) - 3*(My1 - My2)/(5*L) - Vz/2)
        set_symmetric(2, 13, P/10)
        set_symmetric(2, 14, -(1/10)*(P*z0 - My1 + Vz*L))
        
        set_symmetric(3, 3, 6*P/(5*L))
        set_symmetric(3, 4, 3*(2*P*y0 + Mz1 - Mz2)/(5*L) - Vy/2)
        set_symmetric(3, 5, -P/10)
        set_symmetric(3, 7, (1/10)*(P*y0 - Mz2 - Vy*L))
        set_symmetric(3, 10, -6*P/(5*L))
        set_symmetric(3, 11, -3*(2*P*y0 + Mz1 - Mz2)/(5*L) - Vy/2)
        set_symmetric(3, 12, -P/10)
        set_symmetric(3, 14, (1/10)*(P*y0 + Mz1 + Vy*L))
        
        set_symmetric(4, 4, 6*P*r1/(5*L) + 3*beta_z*(Mz1 - Mz2)/(5*L) 
                      - 3*beta_y*(My1 - My2)/(5*L) + 6*Mw*beta_w/(5*L)
                      - Vz*z0/2 - Vy*y0/2)
        set_symmetric(4, 5, -(1/10)*(P*y0 - Mz2 + Vy*L))
        set_symmetric(4, 6, -(1/10)*(P*z0 + My2 + Vz*L))
        set_symmetric(4, 7, (1/10)*(P*r1 - Mz2*beta_z + My2*beta_y + Mw*beta_w))
        set_symmetric(4, 9, 6*P*z0/(5*L) - 3*(My1 - My2)/(5*L) - Vz/2)
        set_symmetric(4, 10, -6*P*y0/(5*L) - 3*(Mz1 - Mz2)/(5*L) + Vy/2)
        set_symmetric(4, 11, -6*P*r1/(5*L) - 3*beta_z*(Mz1 - Mz2)/(5*L)
                      + 3*beta_y*(My1 - My2)/(5*L) - 6*Mw*beta_w/(5*L))
        set_symmetric(4, 12, -(1/10)*(P*y0 + Mz1 - Vy*L))
        set_symmetric(4, 13, -(1/10)*(P*z0 - My1 - Vz*L))
        set_symmetric(4, 14, (1/10)*(P*r1 + Mz1*beta_z - My1*beta_y + Mw*beta_w))
        
        set_symmetric(5, 5, 2*P*L/15)
        set_symmetric(5, 7, -2*P*y0*L/15 - L*(3*Mz1 - Mz2)/30)
        set_symmetric(5, 9, P/10)
        set_symmetric(5, 11, -(1/10)*(P*y0 - Mz2 + Vy*L))  # SIGN FIX: was +
        set_symmetric(5, 12, -P*L/30)
        set_symmetric(5, 14, L*(2*P*y0 + Mz1 - Mz2 - Vy*L)/60)
        
        set_symmetric(6, 6, 2*P*L/15)
        set_symmetric(6, 7, -2*P*z0*L/15 + L*(3*My1 - My2)/30)
        set_symmetric(6, 9, -P/10)
        set_symmetric(6, 11, -(1/10)*(P*z0 + My1 + Vz*L))  # SIGN FIX: was +
        set_symmetric(6, 13, -P*L/30)
        set_symmetric(6, 14, L*(2*P*z0 - My1 + My2 - Vz*L)/60)
        
        set_symmetric(7, 7, 2*P*r1*L/15 + beta_z*L*(3*Mz1 - Mz2)/30
                      - beta_y*L*(3*My1 - My2)/30 + 2*Mw*beta_w*L/15)
        set_symmetric(7, 9, -(1/10)*(P*z0 + My1 - Vz*L))  # SIGN FIX: was +
        set_symmetric(7, 10, (1/10)*(P*y0 - Mz1 - Vy*L))  # SIGN FIX: was -
        set_symmetric(7, 11, -(1/10)*(P*r1 - Mz2*beta_z + My2*beta_y + Mw*beta_w))
        set_symmetric(7, 12, L*(2*P*y0 + Mz1 - Mz2 + Vy*L)/60)
        set_symmetric(7, 13, L*(2*P*z0 - My1 + My2 + Vz*L)/60)
        set_symmetric(7, 14, -P*r1*L/30 - beta_z*L*(Mz1 - Mz2)/60
                      + beta_y*L*(My1 - My2)/60 + Mw*beta_w*L/30)
        
        set_symmetric(9, 9, 6*P/(5*L))
        set_symmetric(9, 11, -6*P*z0/(5*L) + 3*(My1 - My2)/(5*L) - Vz/2)  # SIGN FIX: was -3*
        set_symmetric(9, 13, -P/10)
        set_symmetric(9, 14, (1/10)*(P*z0 - My1 + Vz*L))
        
        set_symmetric(10, 10, 6*P/(5*L))
        set_symmetric(10, 11, 6*P*y0/(5*L) - 3*(Mz1 - Mz2)/(5*L) + Vy/2)  # SIGN FIX: was +3*
        set_symmetric(10, 12, P/10)
        set_symmetric(10, 14, -(1/10)*(P*y0 + Mz1 + Vy*L))
        
        set_symmetric(11, 11, 6*P*r1/(5*L) + 3*beta_z*(Mz1 - Mz2)/(5*L)
                      - 3*beta_y*(My1 - My2)/(5*L) + (Vz*z0 + Vy*y0)/2
                      + 6*Mw*beta_w/(5*L))
        set_symmetric(11, 12, (1/10)*(P*y0 + Mz2 - Vy*L))
        set_symmetric(11, 13, (1/10)*(P*z0 - My1 - Vz*L))
        set_symmetric(11, 14, -(1/10)*(P*r1 + Mz1*beta_z - My1*beta_y + Mw*beta_w))
        
        set_symmetric(12, 12, 2*P*L/15)
        set_symmetric(12, 14, 2*P*y0*L/15 - L*(Mz1 - 3*Mz2)/30)
        
        set_symmetric(13, 13, 2*P*L/15)
        set_symmetric(13, 14, -2*P*z0*L/15 + L*(My1 - 3*My2)/30)
        
        set_symmetric(14, 14, 2*P*r1*L/15 + beta_z*L*(Mz1 - 3*Mz2)/30
                      - beta_y*L*(My1 - 3*My2)/30 + 2*Mw*beta_w*L/15)

    return K.tocsr()
