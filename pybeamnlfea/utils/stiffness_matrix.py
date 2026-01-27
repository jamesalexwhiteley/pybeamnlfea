import numpy as np 
from scipy.sparse import lil_matrix
from numpy.polynomial.legendre import leggauss

# Author: James Whiteley (github.com/jamesalexwhiteley)

# def thin_wall_stiffness_matrix_bazant(E, G, A, Iy, Iz, Iw, J, L, 
#                                 P0=0, My0=0, Mz0=0, B0_bar=0, 
#                                 W_bar=0, y0=0, z0=0, beta_y=0, beta_z=0, r=0,
#                                 include_geometric=False
#                                 ):
    
#     """
#     Create the element stiffness matrix for a 3D thin-walled beam element.
        
#     """
#     # Initialise a sparse matrix (using lil_matrix for efficient assembly)
#     K = lil_matrix((14, 14))
    
#     def set_symmetric(i, j, value):
#         K[i, j] += value
#         if i != j:
#             K[j, i] += value
    
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
#     set_symmetric(3, 8, 3*Mz0/(5*L) - 3*P0*z0/(5*L))
    
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
    
#     set_symmetric(2, 3, 3*My0/(5*L) - 3*P0*y0/(5*L))
#     set_symmetric(2, 10, -3*My0/(5*L) + 3*P0*y0/(5*L))
#     set_symmetric(3, 9, -3*My0/(5*L) + 3*P0*y0/(5*L))
#     set_symmetric(9, 10, 3*My0/(5*L) - 3*P0*y0/(5*L))
    
#     set_symmetric(2, 4, -6*E*Iy/(L**2) - P0/10)
#     set_symmetric(2, 11, -6*E*Iy/(L**2) - P0/10)
#     set_symmetric(4, 9, 6*E*Iy/(L**2) + P0/10)
#     set_symmetric(9, 11, 6*E*Iy/(L**2) + P0/10)
    
#     cross_term_val2 = -My0/20 + P0*y0/20
#     set_symmetric(2, 6, cross_term_val2)
#     set_symmetric(2, 13, -cross_term_val2)
#     set_symmetric(9, 13, -cross_term_val2)
#     set_symmetric(3, 4, -cross_term_val2)
#     set_symmetric(3, 11, cross_term_val2)
#     set_symmetric(10, 11, cross_term_val2)
#     set_symmetric(4, 10, -cross_term_val2)
#     set_symmetric(6, 9, cross_term_val2)

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
    
#     set_symmetric(4, 6, -L*My0/15 + L*P0*y0/15)
#     set_symmetric(11, 13, -L*My0/15 + L*P0*y0/15)
    
#     set_symmetric(4, 13, L*My0/60 - L*P0*y0/60)
#     set_symmetric(6, 11, L*My0/60 - L*P0*y0/60)
    
#     set_symmetric(4, 11, 2*E*Iy/L - L*P0/30)
    
#     set_symmetric(5, 5, 4*E*Iz/L + 2*L*P0/15)
#     set_symmetric(12, 12, 4*E*Iz/L + 2*L*P0/15)
    
#     set_symmetric(5, 6, -L*Mz0/15 + L*P0*z0/15)
#     set_symmetric(12, 13, -L*Mz0/15 + L*P0*z0/15)
    
#     set_symmetric(5, 12, 2*E*Iz/L - L*P0/30)
    
#     set_symmetric(5, 13, L*Mz0/60 - L*P0*z0/60)
#     set_symmetric(6, 12, L*Mz0/60 - L*P0*z0/60)

#     return K.tocsr()

# def thin_wall_stiffness_matrix_bazant(E, G, A, Iy, Iz, Iw, J, L, 
#                                 P0=0, My0=0, Mz0=0, B0_bar=0, 
#                                 W_bar=0, y0=0, z0=0, beta_y=0, beta_z=0, r=0,
#                                 include_geometric=False
#                                 ):
    
#     """
#     Create the element stiffness matrix for a 3D thin-walled beam element.
        
#     """

#     # Initialise a sparse matrix (using lil_matrix for efficient assembly)
#     K = lil_matrix((14, 14))
    
#     def set_symmetric(i, j, value):
#         K[i, j] += value
#         if i != j:
#             K[j, i] += value
    
#     # Axial terms
#     set_symmetric(0, 0, A*E/L)
#     set_symmetric(0, 7, -A*E/L)
#     set_symmetric(7, 7, A*E/L)
    
#     # Z-direction bending terms
#     set_symmetric(1, 1, 12*E*Iz/(L**3))
#     set_symmetric(1, 8, -12*E*Iz/(L**3))
#     set_symmetric(8, 8, 12*E*Iz/(L**3))
    
#     set_symmetric(1, 3, -3*Mz0/(5*L))
#     set_symmetric(1, 10, 3*Mz0/(5*L))
#     set_symmetric(8, 10, -3*Mz0/(5*L))
#     set_symmetric(3, 8, 3*Mz0/(5*L))
    
#     set_symmetric(1, 5, 6*E*Iz/(L**2))
#     set_symmetric(1, 12, 6*E*Iz/(L**2))
#     set_symmetric(5, 8, -6*E*Iz/(L**2))
#     set_symmetric(8, 12, -6*E*Iz/(L**2))
    
#     cross_term_val = -Mz0/20 
#     set_symmetric(1, 6, cross_term_val)
#     set_symmetric(1, 13, cross_term_val)
#     set_symmetric(8, 13, -cross_term_val)
#     set_symmetric(5, 10, -cross_term_val)
#     set_symmetric(3, 5, cross_term_val)
#     set_symmetric(3, 12, cross_term_val)
#     set_symmetric(6, 8, -cross_term_val)
#     set_symmetric(10, 12, -cross_term_val)
    
#     # Y-direction bending terms
#     set_symmetric(2, 2, 12*E*Iy/(L**3))
#     set_symmetric(2, 9, -12*E*Iy/(L**3))
#     set_symmetric(9, 9, 12*E*Iy/(L**3))
    
#     set_symmetric(2, 3, 3*My0/(5*L))
#     set_symmetric(2, 10, -3*My0/(5*L))
#     set_symmetric(3, 9, -3*My0/(5*L))
#     set_symmetric(9, 10, 3*My0/(5*L))
    
#     set_symmetric(2, 4, -6*E*Iy/(L**2))
#     set_symmetric(2, 11, -6*E*Iy/(L**2))
#     set_symmetric(4, 9, 6*E*Iy/(L**2))
#     set_symmetric(9, 11, 6*E*Iy/(L**2))
    
#     cross_term_val2 = -My0/20 
#     set_symmetric(2, 6, cross_term_val2)
#     set_symmetric(2, 13, -cross_term_val2) 
#     set_symmetric(9, 13, -cross_term_val2)
#     set_symmetric(3, 4, -cross_term_val2)
#     set_symmetric(3, 11, cross_term_val2)
#     set_symmetric(10, 11, cross_term_val2)
#     set_symmetric(4, 10, -cross_term_val2)
#     set_symmetric(6, 9, cross_term_val2)
    
#     # Warping and torsion terms
#     warping_term = 12*E*Iw/(L**3) + 6*G*J/(5*L) 
#     set_symmetric(3, 3, warping_term)
#     set_symmetric(3, 10, -warping_term)
#     set_symmetric(10, 10, warping_term)
    
#     warping_term2 = 6*E*Iw/(L**2) + G*J/10 
#     set_symmetric(3, 6, warping_term2)
#     set_symmetric(3, 13, warping_term2)
#     set_symmetric(6, 10, -warping_term2)
#     set_symmetric(10, 13, -warping_term2)
    
#     warping_term3 = 4*E*Iw/L + 2*G*J*L/15 
#     set_symmetric(6, 6, warping_term3)
#     set_symmetric(13, 13, warping_term3)
    
#     warping_term4 = 2*E*Iw/L - G*J*L/30 
#     set_symmetric(6, 13, warping_term4)

#     # Additional terms
#     set_symmetric(4, 4, 4*E*Iy/L)
#     set_symmetric(11, 11, 4*E*Iy/L)
    
#     set_symmetric(4, 6, -L*My0/15)
#     set_symmetric(11, 13, -L*My0/15)
    
#     set_symmetric(4, 13, L*My0/60)
#     set_symmetric(6, 11, L*My0/60)
    
#     set_symmetric(4, 11, 2*E*Iy/L)
    
#     set_symmetric(5, 5, 4*E*Iz/L)
#     set_symmetric(12, 12, 4*E*Iz/L)
    
#     set_symmetric(5, 6, -L*Mz0/15)
#     set_symmetric(12, 13, -L*Mz0/15)
    
#     set_symmetric(5, 12, 2*E*Iz/L)
    
#     set_symmetric(5, 13, L*Mz0/60)
#     set_symmetric(6, 12, L*Mz0/60)

#     return K.tocsr()

def thin_wall_stiffness_matrix_bazant(E, G, A, Iy, Iz, Iw, J, L, 
                                P0=0, My0=0, Mz0=0, B0_bar=0, 
                                W_bar=0, y0=0, z0=0, beta_y=0, beta_z=0, r=0,
                                include_geometric=False
                                ):
    
    """
    Create the element stiffness matrix for a 3D thin-walled beam element.
    
    DOF order (0-indexed):
    Node 1: [u1, v1, w1, θx1, θy1, θz1, θx1']  = DOFs 0-6
    Node 2: [u2, v2, w2, θx2, θy2, θz2, θx2']  = DOFs 7-13
    
    """

    K = lil_matrix((14, 14))
    
    def set_symmetric(i, j, value):
        K[i, j] += value
        if i != j:
            K[j, i] += value
    
    # =========================================================================
    # ELASTIC STIFFNESS MATRIX 
    # =========================================================================
    
    # Axial terms
    set_symmetric(0, 0, A*E/L)
    set_symmetric(0, 7, -A*E/L)
    set_symmetric(7, 7, A*E/L)
    
    # Z-direction bending (v displacement, θz rotation)
    set_symmetric(1, 1, 12*E*Iz/L**3)
    set_symmetric(1, 8, -12*E*Iz/L**3)
    set_symmetric(8, 8, 12*E*Iz/L**3)
    
    set_symmetric(1, 5, 6*E*Iz/L**2)
    set_symmetric(1, 12, 6*E*Iz/L**2)
    set_symmetric(5, 8, -6*E*Iz/L**2)
    set_symmetric(8, 12, -6*E*Iz/L**2)
    
    set_symmetric(5, 5, 4*E*Iz/L)
    set_symmetric(12, 12, 4*E*Iz/L)
    set_symmetric(5, 12, 2*E*Iz/L)
    
    # Y-direction bending (w displacement, θy rotation)
    set_symmetric(2, 2, 12*E*Iy/L**3)
    set_symmetric(2, 9, -12*E*Iy/L**3)
    set_symmetric(9, 9, 12*E*Iy/L**3)
    
    set_symmetric(2, 4, -6*E*Iy/L**2)
    set_symmetric(2, 11, -6*E*Iy/L**2)
    set_symmetric(4, 9, 6*E*Iy/L**2)
    set_symmetric(9, 11, 6*E*Iy/L**2)
    
    set_symmetric(4, 4, 4*E*Iy/L)
    set_symmetric(11, 11, 4*E*Iy/L)
    set_symmetric(4, 11, 2*E*Iy/L)
    
    # Torsion and warping
    set_symmetric(3, 3, 12*E*Iw/L**3 + 6*G*J/(5*L))
    set_symmetric(3, 10, -12*E*Iw/L**3 - 6*G*J/(5*L))
    set_symmetric(10, 10, 12*E*Iw/L**3 + 6*G*J/(5*L))
    
    set_symmetric(3, 6, 6*E*Iw/L**2 + G*J/10)
    set_symmetric(3, 13, 6*E*Iw/L**2 + G*J/10)
    set_symmetric(6, 10, -6*E*Iw/L**2 - G*J/10)
    set_symmetric(10, 13, -6*E*Iw/L**2 - G*J/10)
    
    set_symmetric(6, 6, 4*E*Iw/L + 2*G*J*L/15)
    set_symmetric(13, 13, 4*E*Iw/L + 2*G*J*L/15)
    set_symmetric(6, 13, 2*E*Iw/L - G*J*L/30)

    # =========================================================================
    # GEOMETRIC STIFFNESS MATRIX 
    # =========================================================================
    # if include_geometric:
        
    # -----------------------------------------------------------------
    # Axial load P0 effects on bending
    # -----------------------------------------------------------------
    
    # v-direction (z-bending)
    set_symmetric(1, 1, 6*P0/(5*L))
    set_symmetric(1, 8, -6*P0/(5*L))
    set_symmetric(8, 8, 6*P0/(5*L))
    
    set_symmetric(1, 5, P0/10)
    set_symmetric(1, 12, P0/10)
    set_symmetric(5, 8, -P0/10)
    set_symmetric(8, 12, -P0/10)
    
    set_symmetric(5, 5, 2*L*P0/15)
    set_symmetric(12, 12, 2*L*P0/15)
    set_symmetric(5, 12, -L*P0/30)
    
    # w-direction (y-bending)
    set_symmetric(2, 2, 6*P0/(5*L))
    set_symmetric(2, 9, -6*P0/(5*L))
    set_symmetric(9, 9, 6*P0/(5*L))
    
    set_symmetric(2, 4, -P0/10)
    set_symmetric(2, 11, -P0/10)
    set_symmetric(4, 9, P0/10)
    set_symmetric(9, 11, P0/10)
    
    set_symmetric(4, 4, 2*L*P0/15)
    set_symmetric(11, 11, 2*L*P0/15)
    set_symmetric(4, 11, -L*P0/30)
    
    # -----------------------------------------------------------------
    # Axial load P0 effects on torsion (polar radius term)
    # -----------------------------------------------------------------
    set_symmetric(3, 3, 6*P0*r**2/(5*L))
    set_symmetric(3, 10, -6*P0*r**2/(5*L))
    set_symmetric(10, 10, 6*P0*r**2/(5*L))
    
    set_symmetric(3, 6, P0*r**2/10)
    set_symmetric(3, 13, P0*r**2/10)
    set_symmetric(6, 10, -P0*r**2/10)
    set_symmetric(10, 13, -P0*r**2/10)
    
    set_symmetric(6, 6, 2*L*P0*r**2/15)
    set_symmetric(13, 13, 2*L*P0*r**2/15)
    set_symmetric(6, 13, -L*P0*r**2/30)
    
    # -----------------------------------------------------------------
    # Wagner effect (monosymmetry) from moments
    # -----------------------------------------------------------------
    set_symmetric(3, 3, 3*Mz0*beta_z/(5*L) - 3*My0*beta_y/(5*L))
    set_symmetric(3, 10, -3*Mz0*beta_z/(5*L) + 3*My0*beta_y/(5*L))
    set_symmetric(10, 10, 3*Mz0*beta_z/(5*L) - 3*My0*beta_y/(5*L))
    
    set_symmetric(3, 6, Mz0*beta_z/20 - My0*beta_y/20)
    set_symmetric(3, 13, Mz0*beta_z/20 - My0*beta_y/20)
    set_symmetric(6, 10, -Mz0*beta_z/20 + My0*beta_y/20)
    set_symmetric(10, 13, -Mz0*beta_z/20 + My0*beta_y/20)
    
    set_symmetric(6, 6, L*Mz0*beta_z/15 - L*My0*beta_y/15)
    set_symmetric(13, 13, L*Mz0*beta_z/15 - L*My0*beta_y/15)
    set_symmetric(6, 13, -L*Mz0*beta_z/60 + L*My0*beta_y/60)
    
    # -----------------------------------------------------------------
    # Bimoment terms
    # -----------------------------------------------------------------
    set_symmetric(3, 3, -6*B0_bar*W_bar/(5*L))
    set_symmetric(3, 10, 6*B0_bar*W_bar/(5*L))
    set_symmetric(10, 10, -6*B0_bar*W_bar/(5*L))
    
    set_symmetric(3, 6, -B0_bar*W_bar/10)
    set_symmetric(3, 13, -B0_bar*W_bar/10)
    set_symmetric(6, 10, B0_bar*W_bar/10)
    set_symmetric(10, 13, B0_bar*W_bar/10)
    
    set_symmetric(6, 6, -2*B0_bar*L*W_bar/15)
    set_symmetric(13, 13, -2*B0_bar*L*W_bar/15)
    set_symmetric(6, 13, B0_bar*L*W_bar/30)
    
    # -----------------------------------------------------------------
    # LTB coupling: ∫M·v''·θ dx for Mz0 (v-θx coupling)
    # -----------------------------------------------------------------
    # v DOFs [1, 5, 8, 12] × θx DOFs [3, 6, 10, 13]
    
    set_symmetric(1, 3, -6*Mz0/(5*L))
    set_symmetric(1, 6, -Mz0/10)
    set_symmetric(1, 10, 6*Mz0/(5*L))
    set_symmetric(1, 13, -Mz0/10)
    
    set_symmetric(5, 3, -11*Mz0/10)
    set_symmetric(5, 6, -2*L*Mz0/15)
    set_symmetric(5, 10, Mz0/10)
    set_symmetric(5, 13, L*Mz0/30)
    
    set_symmetric(8, 3, 6*Mz0/(5*L))
    set_symmetric(8, 6, Mz0/10)
    set_symmetric(8, 10, -6*Mz0/(5*L))
    set_symmetric(8, 13, Mz0/10)
    
    set_symmetric(12, 3, -Mz0/10)
    set_symmetric(12, 6, L*Mz0/30)
    set_symmetric(12, 10, 11*Mz0/10)
    set_symmetric(12, 13, -2*L*Mz0/15)
    
    # -----------------------------------------------------------------
    # LTB coupling: ∫M·w''·θ dx for My0 (w-θx coupling)
    # -----------------------------------------------------------------
    # w DOFs [2, 4, 9, 11] × θx DOFs [3, 6, 10, 13]
    
    set_symmetric(2, 3, -6*My0/(5*L))
    set_symmetric(2, 6, -My0/10)
    set_symmetric(2, 10, 6*My0/(5*L))
    set_symmetric(2, 13, -My0/10)
    
    set_symmetric(4, 3, -11*My0/10)
    set_symmetric(4, 6, -2*L*My0/15)
    set_symmetric(4, 10, My0/10)
    set_symmetric(4, 13, L*My0/30)
    
    set_symmetric(9, 3, 6*My0/(5*L))
    set_symmetric(9, 6, My0/10)
    set_symmetric(9, 10, -6*My0/(5*L))
    set_symmetric(9, 13, My0/10)
    
    set_symmetric(11, 3, -My0/10)
    set_symmetric(11, 6, L*My0/30)
    set_symmetric(11, 10, 11*My0/10)
    set_symmetric(11, 13, -2*L*My0/15)

    return K.tocsr()

def thin_wall_stiffness_matrix_chan(E, G, A, Iy, Iz, Iw, J, L, 
                                P=0, My1=0, My2=0, Mz1=0, Mz2=0,
                                Mw=0, y0=0, z0=0, beta_y=0, beta_z=0, beta_w=0, r1=0,
                                beta_x=0,           # Monosymmetry parameter (Wagner coefficient)
                                wagner_sign=-1.0,   # Sign for Wagner effect (-1 for UDL, +1 for point load)
                                q1=0, q2=0,         # Transverse load intensity at element ends
                                load_height=0,      # Height of load above shear center
                                include_geometric=True):
    """
    Thin-walled beam element stiffness matrix (14x14).
    
    DOF order (1-indexed as in Chan & Kitipornchai):
    Node 1: [u1, v1, w1, θx1, θy1, θz1, θx1']  = DOFs 1-7
    Node 2: [u2, v2, w2, θx2, θy2, θz2, θx2']  = DOFs 8-14

    """
    
    K = lil_matrix((14, 14))
    Vy, Vz = 0, 0
    
    def set_symmetric(i, j, value):
        K[i-1, j-1] = K[i-1, j-1] + value
        if (i-1) != (j-1):
            K[j-1, i-1] = K[j-1, i-1] + value
    
    # =========================================================================
    # ELASTIC STIFFNESS MATRIX 
    # =========================================================================
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
        # Axial load terms (P-related)
        set_symmetric(2, 2, 6*P/(5*L))
        set_symmetric(2, 6, P/10)
        set_symmetric(2, 9, -6*P/(5*L))
        set_symmetric(2, 13, P/10)
        
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
        set_symmetric(4, 7, (1/10)*(P*r1 - Mz2*beta_z + My2*beta_y + Mw*beta_w))
        set_symmetric(4, 10, -6*P*y0/(5*L) - 3*(Mz1 - Mz2)/(5*L) + Vy/2)
        set_symmetric(4, 11, -6*P*r1/(5*L) - 3*beta_z*(Mz1 - Mz2)/(5*L)
                      + 3*beta_y*(My1 - My2)/(5*L) - 6*Mw*beta_w/(5*L))
        set_symmetric(4, 12, -(1/10)*(P*y0 + Mz1 - Vy*L))
        set_symmetric(4, 14, (1/10)*(P*r1 + Mz1*beta_z - My1*beta_y + Mw*beta_w))
        
        set_symmetric(5, 5, 2*P*L/15)
        set_symmetric(5, 7, -2*P*y0*L/15 - L*(3*Mz1 - Mz2)/30)
        set_symmetric(5, 9, P/10)
        set_symmetric(5, 11, (1/10)*(P*y0 - Mz2 - Vy*L))
        set_symmetric(5, 12, -P*L/30)
        set_symmetric(5, 14, L*(2*P*y0 + Mz1 - Mz2 - Vy*L)/60)
        
        set_symmetric(6, 6, 2*P*L/15)
        set_symmetric(6, 9, -P/10)
        set_symmetric(6, 13, -P*L/30)
        
        set_symmetric(7, 7, 2*P*r1*L/15 + beta_z*L*(3*Mz1 - Mz2)/30
                      - beta_y*L*(3*My1 - My2)/30 + 2*Mw*beta_w*L/15)
        set_symmetric(7, 10, (1/10)*(P*y0 - Mz1 - Vy*L))  
        set_symmetric(7, 11, -(1/10)*(P*r1 - Mz2*beta_z + My2*beta_y + Mw*beta_w))
        set_symmetric(7, 12, L*(2*P*y0 + Mz1 - Mz2 + Vy*L)/60)
        set_symmetric(7, 14, -P*r1*L/30 - beta_z*L*(Mz1 - Mz2)/60
                      + beta_y*L*(My1 - My2)/60 + Mw*beta_w*L/30)
        
        set_symmetric(9, 9, 6*P/(5*L))
        set_symmetric(9, 13, -P/10)
        
        set_symmetric(10, 10, 6*P/(5*L))
        set_symmetric(10, 11, 6*P*y0/(5*L) + 3*(Mz1 - Mz2)/(5*L) + Vy/2)  
        set_symmetric(10, 12, P/10)
        set_symmetric(10, 14, -(1/10)*(P*y0 + Mz1 + Vy*L))
        
        set_symmetric(11, 11, 6*P*r1/(5*L) + 3*beta_z*(Mz1 - Mz2)/(5*L)
                      - 3*beta_y*(My1 - My2)/(5*L) + (Vz*z0 + Vy*y0)/2
                      + 6*Mw*beta_w/(5*L))
        set_symmetric(11, 12, (1/10)*(P*y0 - Mz2 - Vy*L)) 
        set_symmetric(11, 14, -(1/10)*(P*r1 + Mz1*beta_z - My1*beta_y + Mw*beta_w))
        
        set_symmetric(12, 12, 2*P*L/15)
        set_symmetric(12, 14, 2*P*y0*L/15 - L*(Mz1 - 3*Mz2)/30)
        
        set_symmetric(13, 13, 2*P*L/15)
        
        set_symmetric(14, 14, 2*P*r1*L/15 + beta_z*L*(Mz1 - 3*Mz2)/30
                      - beta_y*L*(My1 - 3*My2)/30 + 2*Mw*beta_w*L/15)

        # =====================================================================
        # LTB COUPLING: ∫M·v''·θ dx
        # Couples lateral bending with torsion
        # =====================================================================
        ltb_coupling = _compute_ltb_coupling(L, My1, My2)

        v_dofs = [2, 6, 9, 13]
        theta_dofs = [4, 7, 11, 14]
        
        for i, vi in enumerate(v_dofs):
            for j, tj in enumerate(theta_dofs):
                K[vi-1, tj-1] = K[vi-1, tj-1] + ltb_coupling[i, j]
                K[tj-1, vi-1] = K[tj-1, vi-1] + ltb_coupling[i, j]

        # =====================================================================
        # WAGNER EFFECT (MONOSYMMETRY): ∫(β_x/2)·M·θ'·θ' dx
        # This adds to the torsional geometric stiffness (θ-θ coupling)
        # The factor of 1/2 comes from the energy formulation
        # =====================================================================
        beta_x = -beta_z                                                                # NOTE 
        if abs(beta_x) > 1e-16:
            wagner_matrix = _compute_wagner_coupling(L, My1, My2, beta_x, sign=wagner_sign)
            
            # Torsion DOFs: θx1=4, θx1'=7, θx2=11, θx2'=14 (1-indexed)
            theta_dofs = [4, 7, 11, 14]
            
            for i, ti in enumerate(theta_dofs):
                for j, tj in enumerate(theta_dofs):
                    K[ti-1, tj-1] = K[ti-1, tj-1] + wagner_matrix[i, j]

        # =====================================================================
        # LOAD HEIGHT EFFECT: ∫q·a·θ² dx
        # Destabilizing when load is above shear center (a > 0)
        # Stabilizing when load is below shear center (a < 0)
        # =====================================================================
        if abs(load_height) > 1e-16 and (abs(q1) > 1e-16 or abs(q2) > 1e-16):
            load_height_matrix = _compute_load_height_coupling(L, q1, q2, load_height)
            
            # Torsion DOFs: θx1=4, θx1'=7, θx2=11, θx2'=14 (1-indexed)
            theta_dofs = [4, 7, 11, 14]
            
            for i, ti in enumerate(theta_dofs):
                for j, tj in enumerate(theta_dofs):
                    K[ti-1, tj-1] = K[ti-1, tj-1] + load_height_matrix[i, j]

    return K.tocsr()

def _compute_ltb_coupling(L, My1, My2):
    """
    Compute the LTB coupling matrix from ∫M·v''·θ dx.
    
    Returns 4x4 matrix coupling:
    - v DOFs [v1, θz1, v2, θz2] 
    - with θ DOFs [θx1, θx1', θx2, θx2']
    """
    from numpy.polynomial.legendre import leggauss
    
    xi_g, w_g = leggauss(4)
    xi_g = (xi_g + 1) / 2
    w_g = w_g / 2
    
    def N_v_2prime(xi):
        """Second derivatives of Hermite shape functions for v''"""
        return np.array([
            (-6 + 12*xi) / L**2,
            (-4 + 6*xi) / L,
            (6 - 12*xi) / L**2,
            (-2 + 6*xi) / L
        ])
    
    def N_theta(xi):
        """Hermite shape functions for θ"""
        return np.array([
            1 - 3*xi**2 + 2*xi**3,
            L * xi * (1-xi)**2,
            3*xi**2 - 2*xi**3,
            L * xi**2 * (xi-1)
        ])
    
    coupling = np.zeros((4, 4))
    for k in range(4):
        xi = xi_g[k]
        M = My1 * (1 - xi) + My2 * xi
        coupling += w_g[k] * M * np.outer(N_v_2prime(xi), N_theta(xi)) * L

    return coupling

# def _compute_ltb_coupling(L, My1, My2):
#     """Analytical LTB coupling matrix."""

#     coupling = np.array([
#         [(-11*My1 - My2)/(10*L),  -My1/10,              (My1 + 11*My2)/(10*L),   -My2/10             ],
#         [-9*My1/10 - My2/5,        L*(-3*My1 - My2)/30, -My1/10 + My2/5,          L*My1/30           ],
#         [(11*My1 + My2)/(10*L),    My1/10,             (-My1 - 11*My2)/(10*L),    My2/10             ],
#         [-My1/5 + My2/10,          L*My2/30,            My1/5 + 9*My2/10,         L*(-My1 - 3*My2)/30]
#     ])

#     # print(coupling) 
#     return coupling 

def _compute_wagner_coupling(L, My1, My2, beta_x, sign=1.0):
    """
    Compute the Wagner (monosymmetry) coupling matrix.
    
    The monosymmetry effect is captured through:
    ∫ β_x · M · θ'² dx
    
    The sign parameter allows calibration for different load types:
    - sign=-1.0 for UDL (Table 2 convention)
    - sign=+1.0 for point load (Table 1 convention)
    
    Returns 4x4 matrix for θ DOFs [θx1, θx1', θx2, θx2']
    """
    from numpy.polynomial.legendre import leggauss
    
    xi_g, w_g = leggauss(4)
    xi_g = (xi_g + 1) / 2
    w_g = w_g / 2
    
    def N_theta_prime(xi):
        """First derivatives of Hermite shape functions for θ'"""
        return np.array([
            (-6*xi + 6*xi**2) / L,
            1 - 4*xi + 3*xi**2,
            (6*xi - 6*xi**2) / L,
            -2*xi + 3*xi**2
        ])
    
    wagner = np.zeros((4, 4))
    
    for k in range(len(xi_g)):
        xi = xi_g[k]
        M = My1 * (1 - xi) + My2 * xi
        Ntp = N_theta_prime(xi)
        
        # Wagner term with configurable sign
        wagner += w_g[k] * sign * beta_x * M * np.outer(Ntp, Ntp) * L
    
    return wagner

def _compute_load_height_coupling(L, q1, q2, a):
    """
    Compute the load height coupling matrix.
    
    The load height effect comes from the potential energy change when the
    applied load moves during buckling:
    
    ∫ q · a · θ² dx
    
    where:
    - q is the transverse load intensity (varies linearly from q1 to q2)
    - a is the height of load application above the shear center
    - θ is the twist angle
    
    For load above shear center (a > 0), this is destabilizing (reduces Kg).
    For load below shear center (a < 0), this is stabilizing (increases Kg).
    
    Parameters:
    -----------
    L : float
        Element length
    q1, q2 : float
        Load intensity at start and end of element
    a : float
        Load height above shear center (positive = above)
        
    Returns 4x4 matrix for θ DOFs [θx1, θx1', θx2, θx2']
    """
    from numpy.polynomial.legendre import leggauss
    
    xi_g, w_g = leggauss(4)
    xi_g = (xi_g + 1) / 2
    w_g = w_g / 2
    
    def N_theta(xi):
        """Hermite shape functions for θ"""
        return np.array([
            1 - 3*xi**2 + 2*xi**3,
            L * xi * (1-xi)**2,
            3*xi**2 - 2*xi**3,
            L * xi**2 * (xi-1)
        ])
    
    load_height = np.zeros((4, 4))
    
    for k in range(len(xi_g)):
        xi = xi_g[k]
        q = q1 * (1 - xi) + q2 * xi
        Nt = N_theta(xi)
        
        # Load height term: -q · a · θ²
        # Negative sign because load above shear center is destabilizing
        # (reduces the geometric stiffness, making buckling easier)
        load_height += w_g[k] * (-1.0) * q * a * np.outer(Nt, Nt) * L
    
    return load_height

def thin_wall_stiffness_matrix_derived(E, G, A, Iy, Iz, Iw, J, L,
                                     P=0, My1=0, My2=0, Mz1=0, Mz2=0,
                                     Mw=0, y0=0, z0=0, beta_y=0, beta_z=0, beta_w=0, r1=0,
                                     beta_x=0,
                                     include_geometric=True,
                                     n_gauss=4):
    """Thin-walled beam element stiffness matrix (14x14)."""
    
    K = lil_matrix((14, 14))
    
    def add_sym(i, j, val):
        """Add value symmetrically (1-indexed)."""
        K[i-1, j-1] += val
        if i != j:
            K[j-1, i-1] += val
    
    # =========================================================================
    # ELASTIC STIFFNESS MATRIX
    # =========================================================================
    
    # Axial
    add_sym(1, 1, A*E/L)
    add_sym(1, 8, -A*E/L)
    add_sym(8, 8, A*E/L)
    
    # Lateral bending v (about z-axis, using Iz)
    add_sym(2, 2, 12*E*Iz/L**3)
    add_sym(2, 6, 6*E*Iz/L**2)
    add_sym(2, 9, -12*E*Iz/L**3)
    add_sym(2, 13, 6*E*Iz/L**2)
    add_sym(6, 6, 4*E*Iz/L)
    add_sym(6, 9, -6*E*Iz/L**2)
    add_sym(6, 13, 2*E*Iz/L)
    add_sym(9, 9, 12*E*Iz/L**3)
    add_sym(9, 13, -6*E*Iz/L**2)
    add_sym(13, 13, 4*E*Iz/L)
    
    # Vertical bending w (about y-axis, using Iy)
    add_sym(3, 3, 12*E*Iy/L**3)
    add_sym(3, 5, -6*E*Iy/L**2)
    add_sym(3, 10, -12*E*Iy/L**3)
    add_sym(3, 12, -6*E*Iy/L**2)
    add_sym(5, 5, 4*E*Iy/L)
    add_sym(5, 10, 6*E*Iy/L**2)
    add_sym(5, 12, 2*E*Iy/L)
    add_sym(10, 10, 12*E*Iy/L**3)
    add_sym(10, 12, 6*E*Iy/L**2)
    add_sym(12, 12, 4*E*Iy/L)
    
    # Torsion with warping
    add_sym(4, 4, 12*E*Iw/L**3 + 6*G*J/(5*L))
    add_sym(4, 7, 6*E*Iw/L**2 + G*J/10)
    add_sym(4, 11, -12*E*Iw/L**3 - 6*G*J/(5*L))
    add_sym(4, 14, 6*E*Iw/L**2 + G*J/10)
    add_sym(7, 7, 4*E*Iw/L + 2*G*J*L/15)
    add_sym(7, 11, -6*E*Iw/L**2 - G*J/10)
    add_sym(7, 14, 2*E*Iw/L - G*J*L/30)
    add_sym(11, 11, 12*E*Iw/L**3 + 6*G*J/(5*L))
    add_sym(11, 14, -6*E*Iw/L**2 - G*J/10)
    add_sym(14, 14, 4*E*Iw/L + 2*G*J*L/15)
    
    if not include_geometric:
        return K.tocsr()
    
    # =========================================================================
    # GEOMETRIC STIFFNESS MATRIX
    # =========================================================================
    
    # Compute r₀² if not provided
    if r1 != 0:
        r0_sq = r1
    else:
        r0_sq = (Iy + Iz) / A + y0**2 + z0**2
    
    # Wagner parameter (use beta_z as the monosymmetry parameter)
    beta_mono = beta_z if beta_x == 0 else beta_x
    
    # Gauss quadrature on [0, 1]
    xi_g, w_g = leggauss(n_gauss)
    xi_g = (xi_g + 1) / 2
    w_g = w_g / 2
    
    def N_hermite(xi):
        return np.array([
            1 - 3*xi**2 + 2*xi**3,
            L * xi * (1 - xi)**2,
            3*xi**2 - 2*xi**3,
            L * xi**2 * (xi - 1)
        ])
    
    def N_prime(xi):
        return np.array([
            (-6*xi + 6*xi**2) / L,
            1 - 4*xi + 3*xi**2,
            (6*xi - 6*xi**2) / L,
            -2*xi + 3*xi**2
        ])
    
    def N_dprime(xi):
        return np.array([
            (-6 + 12*xi) / L**2,
            (-4 + 6*xi) / L,
            (6 - 12*xi) / L**2,
            (-2 + 6*xi) / L
        ])
    
    # Initialize sub-matrices
    K_Np_Np = np.zeros((4, 4))
    K_My_Ndp_N = np.zeros((4, 4))
    K_Mz_Ndp_N = np.zeros((4, 4))
    K_My_Np_Np = np.zeros((4, 4))
    
    for k in range(n_gauss):
        xi = xi_g[k]
        wt = w_g[k]
        
        N = N_hermite(xi)
        Np = N_prime(xi)
        Ndp = N_dprime(xi)
        
        My = My1 * (1 - xi) + My2 * xi
        Mz = Mz1 * (1 - xi) + Mz2 * xi
        
        K_Np_Np += wt * np.outer(Np, Np) * L
        K_My_Ndp_N += wt * My * np.outer(Ndp, N) * L
        K_Mz_Ndp_N += wt * Mz * np.outer(Ndp, N) * L
        K_My_Np_Np += wt * My * np.outer(Np, Np) * L
    
    w_sign = np.array([1, -1, 1, -1])
    
    # DOF mappings (0-indexed)
    v_dofs = np.array([1, 5, 8, 12])
    w_dofs = np.array([2, 4, 9, 11])
    t_dofs = np.array([3, 6, 10, 13])
    
    # 1. P·∫v'² dx
    K_Pv = P * K_Np_Np
    for i in range(4):
        for j in range(4):
            K[v_dofs[i], v_dofs[j]] += K_Pv[i, j]
    
    # 2. P·∫w'² dx
    K_Pw = P * np.outer(w_sign, w_sign) * K_Np_Np
    for i in range(4):
        for j in range(4):
            K[w_dofs[i], w_dofs[j]] += K_Pw[i, j]
    
    # 3. P·r₀²·∫θx'² dx
    K_Pt = P * r0_sq * K_Np_Np
    for i in range(4):
        for j in range(4):
            K[t_dofs[i], t_dofs[j]] += K_Pt[i, j]
    
    # 4. P·y₀·∫w'·θx' dx
    if abs(y0) > 1e-16:
        K_Py0 = P * y0 * K_Np_Np
        for i in range(4):
            for j in range(4):
                val = w_sign[i] * K_Py0[i, j]
                K[w_dofs[i], t_dofs[j]] += val
                K[t_dofs[j], w_dofs[i]] += val
    
    # 5. P·z₀·∫v'·θx' dx
    if abs(z0) > 1e-16:
        K_Pz0 = P * z0 * K_Np_Np
        for i in range(4):
            for j in range(4):
                val = K_Pz0[i, j]
                K[v_dofs[i], t_dofs[j]] += val
                K[t_dofs[j], v_dofs[i]] += val
    
    # 6. ∫My·v''·θx dx (LTB coupling)
    for i in range(4):
        for j in range(4):
            K[v_dofs[i], t_dofs[j]] += K_My_Ndp_N[i, j]
            K[t_dofs[j], v_dofs[i]] += K_My_Ndp_N[i, j]
    
    # 7. ∫Mz·w''·θx dx (minor axis LTB)
    for i in range(4):
        for j in range(4):
            val = w_sign[i] * K_Mz_Ndp_N[i, j]
            K[w_dofs[i], t_dofs[j]] += val
            K[t_dofs[j], w_dofs[i]] += val
    
    # 8. Wagner effect with calibrated factor
    K_Wagner = beta_mono * K_My_Np_Np
    for i in range(4):
        for j in range(4):
            K[t_dofs[i], t_dofs[j]] += K_Wagner[i, j]
    
    return K.tocsr()

if __name__ == "__main__":

    # # Stiffness matrix 
    keb = thin_wall_stiffness_matrix_bazant(E=1, G=1, A=1, Iy=1, Iz=1, Iw=1, J=1, L=1, 
                                            P0=0, My0=0, Mz0=0, B0_bar=0, 
                                            W_bar=0, y0=0, z0=0, beta_y=0, beta_z=0, r=0)
    ktb = thin_wall_stiffness_matrix_bazant(E=1, G=1, A=1, Iy=1, Iz=1, Iw=1, J=1, L=1, 
                                            P0=0, My0=0, Mz0=-2, B0_bar=0, 
                                            W_bar=0, y0=0, z0=0, beta_y=0, beta_z=0, r=0)
    kgb = ktb - keb

    # Stiffness matrix
    kec = thin_wall_stiffness_matrix_chan(E=1, G=1, A=1, Iy=1, Iz=1, Iw=1, J=1, L=1, 
                                        P=0, My1=0, My2=0, Mz1=0, Mz2=0, Mw=0, 
                                        y0=0, z0=0, beta_y=0, beta_z=0, beta_w=0, r1=0,
                                        include_geometric=False)
    
    ktc = thin_wall_stiffness_matrix_chan(E=1, G=1, A=1, Iy=1, Iz=1, Iw=1, J=1, L=1,  
                                        P=0, My1=-1, My2=-1, Mz1=0, Mz2=0, Mw=0, 
                                        y0=0, z0=0, beta_y=0, beta_z=0, beta_w=0, r1=0,
                                        include_geometric=True)
    kgc = ktc - kec

    np.set_printoptions(
        precision=6,
        suppress=True,
        linewidth=200,
        threshold=np.inf
    )

    print((kgb).todense())
    print((kgc).todense())
    print((kgb - kgc).todense())
    
    # Save to file
    import os
    
    def dict_to_string(sparse_dict):
        """Convert sparse dictionary to formatted string"""
        lines = []
        for (i, j), value in sparse_dict.items():
            if abs(value) > 1e-10: # filter out near-zero values
                lines.append(f"K[{i+1}, {j+1}] = {value:.6e}")
        return "\n".join(lines)
    
    # Create output directory
    output_dir = "output_files"
    os.makedirs(output_dir, exist_ok=True)
    
    # Write to file
    file_path = os.path.join(output_dir, "stiffness_matrix.txt")
    with open(file_path, 'w') as f:
        f.write(dict_to_string(K.todok()))
    