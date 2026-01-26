import numpy as np 
from scipy.sparse import lil_matrix

# Author: James Whiteley (github.com/jamesalexwhiteley)

def thin_wall_stiffness_matrix_bazant(E, G, A, Iy, Iz, Iw, J, L, 
                                P0=0, My0=0, Mz0=0, B0_bar=0, 
                                W_bar=0, y0=0, z0=0, beta_y=0, beta_z=0, r=0,
                                include_geometric=False
                                ):
    
    """
    Create the element stiffness matrix for a 3D thin-walled beam element.
        
    """
    # Initialise a sparse matrix (using lil_matrix for efficient assembly)
    K = lil_matrix((14, 14))
    
    def set_symmetric(i, j, value):
        K[i, j] = value
        if i != j:
            K[j, i] = value
    
    # Axial terms
    set_symmetric(0, 0, A*E/L)
    set_symmetric(0, 7, -A*E/L)
    set_symmetric(7, 7, A*E/L)
    
    # Z-direction bending terms
    set_symmetric(1, 1, 12*E*Iz/(L**3) + 6*P0/(5*L))
    set_symmetric(1, 8, -12*E*Iz/(L**3) - 6*P0/(5*L))
    set_symmetric(8, 8, 12*E*Iz/(L**3) + 6*P0/(5*L))
    
    set_symmetric(1, 3, -3*Mz0/(5*L) + 3*P0*z0/(5*L))
    set_symmetric(1, 10, 3*Mz0/(5*L) - 3*P0*z0/(5*L))
    set_symmetric(8, 10, -3*Mz0/(5*L) + 3*P0*z0/(5*L))   
    set_symmetric(3, 8, 3*Mz0/(5*L) - 3*P0*z0/(5*L))
    # set_symmetric(1, 3, -5*Mz0/(5*L))
    # set_symmetric(1, 10, 5*Mz0/(5*L))
    # set_symmetric(8, 10, -5*Mz0/(5*L))
    # set_symmetric(3, 8, 5*Mz0/(5*L))
    
    set_symmetric(1, 5, 6*E*Iz/(L**2) + P0/10)
    set_symmetric(1, 12, 6*E*Iz/(L**2) + P0/10)
    set_symmetric(5, 8, -6*E*Iz/(L**2) - P0/10)
    set_symmetric(8, 12, -6*E*Iz/(L**2) - P0/10)
    
    cross_term_val = -Mz0/20 + P0*z0/20
    set_symmetric(1, 6, cross_term_val)
    set_symmetric(1, 13, cross_term_val)
    set_symmetric(8, 13, -cross_term_val)
    set_symmetric(5, 10, -cross_term_val)
    set_symmetric(3, 5, cross_term_val)
    set_symmetric(3, 12, cross_term_val)
    set_symmetric(6, 8, -cross_term_val)
    set_symmetric(10, 12, -cross_term_val)
    
    # Y-direction bending terms
    set_symmetric(2, 2, 12*E*Iy/(L**3) + 6*P0/(5*L))
    set_symmetric(2, 9, -12*E*Iy/(L**3) - 6*P0/(5*L))
    set_symmetric(9, 9, 12*E*Iy/(L**3) + 6*P0/(5*L))
    
    set_symmetric(2, 3, 3*My0/(5*L) - 3*P0*y0/(5*L))
    set_symmetric(2, 10, -3*My0/(5*L) + 3*P0*y0/(5*L))
    set_symmetric(3, 9, -3*My0/(5*L) + 3*P0*y0/(5*L))
    set_symmetric(9, 10, 3*My0/(5*L) - 3*P0*y0/(5*L))
    
    set_symmetric(2, 4, -6*E*Iy/(L**2) - P0/10)
    set_symmetric(2, 11, -6*E*Iy/(L**2) - P0/10)
    set_symmetric(4, 9, 6*E*Iy/(L**2) + P0/10)
    set_symmetric(9, 11, 6*E*Iy/(L**2) + P0/10)
    
    cross_term_val2 = -My0/20 + P0*y0/20
    set_symmetric(2, 6, cross_term_val2)
    set_symmetric(2, 13, -cross_term_val2)
    set_symmetric(9, 13, -cross_term_val2)
    set_symmetric(3, 4, -cross_term_val2)
    set_symmetric(3, 11, cross_term_val2)
    set_symmetric(10, 11, cross_term_val2)
    set_symmetric(4, 10, -cross_term_val2)
    set_symmetric(6, 9, cross_term_val2)

    # Warping and torsion terms
    warping_term = 12*E*Iw/(L**3) + 6*G*J/(5*L) + 6*P0*r**2/(5*L) + 3*Mz0*beta_z/(5*L) - 3*My0*beta_y/(5*L) - 6*B0_bar*W_bar/(5*L)
    set_symmetric(3, 3, warping_term)
    set_symmetric(3, 10, -warping_term)
    set_symmetric(10, 10, warping_term)
    
    warping_term2 = 6*E*Iw/(L**2) + G*J/10 + P0*r**2/10 + Mz0*beta_z/20 - My0*beta_y/20 - B0_bar*W_bar/10
    set_symmetric(3, 6, warping_term2)
    set_symmetric(3, 13, warping_term2)
    set_symmetric(6, 10, -warping_term2)
    set_symmetric(10, 13, -warping_term2)
    
    warping_term3 = 4*E*Iw/L + 2*G*J*L/15 + 2*L*P0*r**2/15 + L*Mz0*beta_z/15 - L*My0*beta_y/15 - 2*B0_bar*L*W_bar/15
    set_symmetric(6, 6, warping_term3)
    set_symmetric(13, 13, warping_term3)
    
    warping_term4 = 2*E*Iw/L - G*J*L/30 - L*P0*r**2/30 - L*Mz0*beta_z/60 + L*My0*beta_y/60 + B0_bar*L*W_bar/30
    set_symmetric(6, 13, warping_term4)

    # Additional terms
    set_symmetric(4, 4, 4*E*Iy/L + 2*L*P0/15)
    set_symmetric(11, 11, 4*E*Iy/L + 2*L*P0/15)
    
    set_symmetric(4, 6, -L*My0/15 + L*P0*y0/15)
    set_symmetric(11, 13, -L*My0/15 + L*P0*y0/15)
    
    set_symmetric(4, 13, L*My0/60 - L*P0*y0/60)
    set_symmetric(6, 11, L*My0/60 - L*P0*y0/60)
    
    set_symmetric(4, 11, 2*E*Iy/L - L*P0/30)
    
    set_symmetric(5, 5, 4*E*Iz/L + 2*L*P0/15)
    set_symmetric(12, 12, 4*E*Iz/L + 2*L*P0/15)
    
    set_symmetric(5, 6, -L*Mz0/15 + L*P0*z0/15)
    set_symmetric(12, 13, -L*Mz0/15 + L*P0*z0/15)
    
    set_symmetric(5, 12, 2*E*Iz/L - L*P0/30)
    
    set_symmetric(5, 13, L*Mz0/60 - L*P0*z0/60)
    set_symmetric(6, 12, L*Mz0/60 - L*P0*z0/60)

    return K.tocsr()

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
#         K[i, j] = value
#         if i != j:
#             K[j, i] = value
    
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

# def thin_wall_stiffness_matrix_bazant(E, G, A, Iy, Iz, Iw, J, L, 
#                                 P0=0, My0=0, Mz0=0, B0_bar=0, 
#                                 W_bar=0, y0=0, z0=0, beta_y=0, beta_z=0, r=0,
#                                 include_geometric=False
#                                 ):
    
#     """
#     Create the element stiffness matrix for a 3D thin-walled beam element.
    
#     DOF order (0-indexed):
#     Node 1: [u1, v1, w1, θx1, θy1, θz1, θx1']  = DOFs 0-6
#     Node 2: [u2, v2, w2, θx2, θy2, θz2, θx2']  = DOFs 7-13
    
#     """

#     K = lil_matrix((14, 14))
    
#     def set_symmetric(i, j, value):
#         K[i, j] = value
#         if i != j:
#             K[j, i] = value
    
#     # =========================================================================
#     # ELASTIC STIFFNESS MATRIX 
#     # =========================================================================
    
#     # Axial terms
#     set_symmetric(0, 0, A*E/L)
#     set_symmetric(0, 7, -A*E/L)
#     set_symmetric(7, 7, A*E/L)
    
#     # Z-direction bending (v displacement, θz rotation)
#     set_symmetric(1, 1, 12*E*Iz/L**3)
#     set_symmetric(1, 8, -12*E*Iz/L**3)
#     set_symmetric(8, 8, 12*E*Iz/L**3)
    
#     set_symmetric(1, 5, 6*E*Iz/L**2)
#     set_symmetric(1, 12, 6*E*Iz/L**2)
#     set_symmetric(5, 8, -6*E*Iz/L**2)
#     set_symmetric(8, 12, -6*E*Iz/L**2)
    
#     set_symmetric(5, 5, 4*E*Iz/L)
#     set_symmetric(12, 12, 4*E*Iz/L)
#     set_symmetric(5, 12, 2*E*Iz/L)
    
#     # Y-direction bending (w displacement, θy rotation)
#     set_symmetric(2, 2, 12*E*Iy/L**3)
#     set_symmetric(2, 9, -12*E*Iy/L**3)
#     set_symmetric(9, 9, 12*E*Iy/L**3)
    
#     set_symmetric(2, 4, -6*E*Iy/L**2)
#     set_symmetric(2, 11, -6*E*Iy/L**2)
#     set_symmetric(4, 9, 6*E*Iy/L**2)
#     set_symmetric(9, 11, 6*E*Iy/L**2)
    
#     set_symmetric(4, 4, 4*E*Iy/L)
#     set_symmetric(11, 11, 4*E*Iy/L)
#     set_symmetric(4, 11, 2*E*Iy/L)
    
#     # Torsion and warping
#     set_symmetric(3, 3, 12*E*Iw/L**3 + 6*G*J/(5*L))
#     set_symmetric(3, 10, -12*E*Iw/L**3 - 6*G*J/(5*L))
#     set_symmetric(10, 10, 12*E*Iw/L**3 + 6*G*J/(5*L))
    
#     set_symmetric(3, 6, 6*E*Iw/L**2 + G*J/10)
#     set_symmetric(3, 13, 6*E*Iw/L**2 + G*J/10)
#     set_symmetric(6, 10, -6*E*Iw/L**2 - G*J/10)
#     set_symmetric(10, 13, -6*E*Iw/L**2 - G*J/10)
    
#     set_symmetric(6, 6, 4*E*Iw/L + 2*G*J*L/15)
#     set_symmetric(13, 13, 4*E*Iw/L + 2*G*J*L/15)
#     set_symmetric(6, 13, 2*E*Iw/L - G*J*L/30)

#     # =========================================================================
#     # GEOMETRIC STIFFNESS MATRIX 
#     # =========================================================================
#     # if include_geometric:
        
#     # # -----------------------------------------------------------------
#     # # Axial load P0 effects on bending
#     # # -----------------------------------------------------------------
    
#     # # v-direction (z-bending)
#     # set_symmetric(1, 1, 6*P0/(5*L))
#     # set_symmetric(1, 8, -6*P0/(5*L))
#     # set_symmetric(8, 8, 6*P0/(5*L))
    
#     # set_symmetric(1, 5, P0/10)
#     # set_symmetric(1, 12, P0/10)
#     # set_symmetric(5, 8, -P0/10)
#     # set_symmetric(8, 12, -P0/10)
    
#     # set_symmetric(5, 5, 2*L*P0/15)
#     # set_symmetric(12, 12, 2*L*P0/15)
#     # set_symmetric(5, 12, -L*P0/30)
    
#     # # w-direction (y-bending)
#     # set_symmetric(2, 2, 6*P0/(5*L))
#     # set_symmetric(2, 9, -6*P0/(5*L))
#     # set_symmetric(9, 9, 6*P0/(5*L))
    
#     # set_symmetric(2, 4, -P0/10)
#     # set_symmetric(2, 11, -P0/10)
#     # set_symmetric(4, 9, P0/10)
#     # set_symmetric(9, 11, P0/10)
    
#     # set_symmetric(4, 4, 2*L*P0/15)
#     # set_symmetric(11, 11, 2*L*P0/15)
#     # set_symmetric(4, 11, -L*P0/30)
    
#     # # -----------------------------------------------------------------
#     # # Axial load P0 effects on torsion (polar radius term)
#     # # -----------------------------------------------------------------
#     # set_symmetric(3, 3, 6*P0*r**2/(5*L))
#     # set_symmetric(3, 10, -6*P0*r**2/(5*L))
#     # set_symmetric(10, 10, 6*P0*r**2/(5*L))
    
#     # set_symmetric(3, 6, P0*r**2/10)
#     # set_symmetric(3, 13, P0*r**2/10)
#     # set_symmetric(6, 10, -P0*r**2/10)
#     # set_symmetric(10, 13, -P0*r**2/10)
    
#     # set_symmetric(6, 6, 2*L*P0*r**2/15)
#     # set_symmetric(13, 13, 2*L*P0*r**2/15)
#     # set_symmetric(6, 13, -L*P0*r**2/30)
    
#     # # -----------------------------------------------------------------
#     # # Wagner effect (monosymmetry) from moments
#     # # -----------------------------------------------------------------
#     # set_symmetric(3, 3, 3*Mz0*beta_z/(5*L) - 3*My0*beta_y/(5*L))
#     # set_symmetric(3, 10, -3*Mz0*beta_z/(5*L) + 3*My0*beta_y/(5*L))
#     # set_symmetric(10, 10, 3*Mz0*beta_z/(5*L) - 3*My0*beta_y/(5*L))
    
#     # set_symmetric(3, 6, Mz0*beta_z/20 - My0*beta_y/20)
#     # set_symmetric(3, 13, Mz0*beta_z/20 - My0*beta_y/20)
#     # set_symmetric(6, 10, -Mz0*beta_z/20 + My0*beta_y/20)
#     # set_symmetric(10, 13, -Mz0*beta_z/20 + My0*beta_y/20)
    
#     # set_symmetric(6, 6, L*Mz0*beta_z/15 - L*My0*beta_y/15)
#     # set_symmetric(13, 13, L*Mz0*beta_z/15 - L*My0*beta_y/15)
#     # set_symmetric(6, 13, -L*Mz0*beta_z/60 + L*My0*beta_y/60)
    
#     # # -----------------------------------------------------------------
#     # # Bimoment terms
#     # # -----------------------------------------------------------------
#     # set_symmetric(3, 3, -6*B0_bar*W_bar/(5*L))
#     # set_symmetric(3, 10, 6*B0_bar*W_bar/(5*L))
#     # set_symmetric(10, 10, -6*B0_bar*W_bar/(5*L))
    
#     # set_symmetric(3, 6, -B0_bar*W_bar/10)
#     # set_symmetric(3, 13, -B0_bar*W_bar/10)
#     # set_symmetric(6, 10, B0_bar*W_bar/10)
#     # set_symmetric(10, 13, B0_bar*W_bar/10)
    
#     # set_symmetric(6, 6, -2*B0_bar*L*W_bar/15)
#     # set_symmetric(13, 13, -2*B0_bar*L*W_bar/15)
#     # set_symmetric(6, 13, B0_bar*L*W_bar/30)
    
#     # -----------------------------------------------------------------
#     # LTB coupling: ∫M·v''·θ dx for Mz0 (v-θx coupling)
#     # -----------------------------------------------------------------
#     # v DOFs [1, 5, 8, 12] × θx DOFs [3, 6, 10, 13]
    
#     set_symmetric(1, 3, -6*Mz0/(5*L))
#     set_symmetric(1, 6, -Mz0/10)
#     set_symmetric(1, 10, 6*Mz0/(5*L))
#     set_symmetric(1, 13, -Mz0/10)
    
#     set_symmetric(5, 3, -11*Mz0/10)
#     set_symmetric(5, 6, -2*L*Mz0/15)
#     set_symmetric(5, 10, Mz0/10)
#     set_symmetric(5, 13, L*Mz0/30)
    
#     set_symmetric(8, 3, 6*Mz0/(5*L))
#     set_symmetric(8, 6, Mz0/10)
#     set_symmetric(8, 10, -6*Mz0/(5*L))
#     set_symmetric(8, 13, Mz0/10)
    
#     set_symmetric(12, 3, -Mz0/10)
#     set_symmetric(12, 6, L*Mz0/30)
#     set_symmetric(12, 10, 11*Mz0/10)
#     set_symmetric(12, 13, -2*L*Mz0/15)
    
#     # # -----------------------------------------------------------------
#     # # LTB coupling: ∫M·w''·θ dx for My0 (w-θx coupling)
#     # # -----------------------------------------------------------------
#     # # w DOFs [2, 4, 9, 11] × θx DOFs [3, 6, 10, 13]
    
#     # set_symmetric(2, 3, -6*My0/(5*L))
#     # set_symmetric(2, 6, -My0/10)
#     # set_symmetric(2, 10, 6*My0/(5*L))
#     # set_symmetric(2, 13, -My0/10)
    
#     # set_symmetric(4, 3, -11*My0/10)
#     # set_symmetric(4, 6, -2*L*My0/15)
#     # set_symmetric(4, 10, My0/10)
#     # set_symmetric(4, 13, L*My0/30)
    
#     # set_symmetric(9, 3, 6*My0/(5*L))
#     # set_symmetric(9, 6, My0/10)
#     # set_symmetric(9, 10, -6*My0/(5*L))
#     # set_symmetric(9, 13, My0/10)
    
#     # set_symmetric(11, 3, -My0/10)
#     # set_symmetric(11, 6, L*My0/30)
#     # set_symmetric(11, 10, 11*My0/10)
#     # set_symmetric(11, 13, -2*L*My0/15)

#     return K.tocsr()

# from scipy.sparse import lil_matrix

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

# def thin_wall_stiffness_matrix_old(E, G, A, Iz, Iy, Iw, J, L, 
#                                 P=0, My1=0, My2=0, Mz1=0, Mz2=0,
#                                 Mw=0, y0=0, z0=0, beta_y=0, beta_z=0, beta_w=0, r1=0,
#                                 # Vy=0, Vz=0,
#                                 include_elastic=True, include_geometric=True):
    
#     K = lil_matrix((14, 14))
#     Vy, Vz = 0, 0
    
#     def set_symmetric(i, j, value):
#         K[i-1, j-1] = K[i-1, j-1] + value
#         if (i-1) != (j-1):
#             K[j-1, i-1] = K[j-1, i-1] + value
    
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
#         set_symmetric(5, 11, (1/10)*(P*y0 - Mz2 - Vy*L))
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
#         set_symmetric(7, 10, (1/10)*(P*y0 - Mz1 - Vy*L))  
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
#         set_symmetric(11, 12, (1/10)*(P*y0 - Mz2 - Vy*L)) 
#         set_symmetric(11, 13, (1/10)*(P*z0 - My1 - Vz*L))
#         set_symmetric(11, 14, -(1/10)*(P*r1 + Mz1*beta_z - My1*beta_y + Mw*beta_w))
        
#         set_symmetric(12, 12, 2*P*L/15)
#         set_symmetric(12, 14, 2*P*y0*L/15 - L*(Mz1 - 3*Mz2)/30)
        
#         set_symmetric(13, 13, 2*P*L/15)
#         set_symmetric(13, 14, -2*P*z0*L/15 + L*(My1 - 3*My2)/30)
        
#         set_symmetric(14, 14, 2*P*r1*L/15 + beta_z*L*(Mz1 - 3*Mz2)/30
#                       - beta_y*L*(My1 - 3*My2)/30 + 2*Mw*beta_w*L/15)

#     return K.tocsr()

# def thin_wall_stiffness_matrix(E, G, A, Iy, Iz, Iw, J, L, 
#                                 P=0, My1=0, My2=0, Mz1=0, Mz2=0, Mw=0, 
#                                 y0=0, z0=0, beta_y=0, beta_z=0, beta_w=0, r1=0,
#                                 include_elastic=True, include_geometric=True):
#     """
#     Thin-walled beam element stiffness matrix (14x14).
    
#     DOF order (1-indexed as in Chan & Kitipornchai):
#     Node 1: [u1, v1, w1, θx1, θy1, θz1, θx1']  = DOFs 1-7
#     Node 2: [u2, v2, w2, θx2, θy2, θz2, θx2']  = DOFs 8-14

#     """
    
#     K = lil_matrix((14, 14))
#     Vy, Vz = 0, 0
    
#     def set_symmetric(i, j, value):
#         K[i-1, j-1] = K[i-1, j-1] + value
#         if (i-1) != (j-1):
#             K[j-1, i-1] = K[j-1, i-1] + value
    
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
#         # Note: The original Chan & Kitipornchai terms that couple v with θx 
#         # (DOFs 2,6,9,13 with 4,7,11,14) use ∫M·v'·θ' type coupling.
#         # For LTB, we need ∫M·v''·θ coupling instead.
#         # So we skip the original v-θx coupling terms and add the correct ones below.
        
#         # Axial load terms (P-related, not involving v-θx coupling)
#         set_symmetric(2, 2, 6*P/(5*L))
#         set_symmetric(2, 6, P/10)
#         set_symmetric(2, 9, -6*P/(5*L))
#         set_symmetric(2, 13, P/10)
        
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
#         # Skip 4,6 - part of v-θx coupling
#         set_symmetric(4, 7, (1/10)*(P*r1 - Mz2*beta_z + My2*beta_y + Mw*beta_w))
#         # Skip 4,9 - part of v-θx coupling  
#         set_symmetric(4, 10, -6*P*y0/(5*L) - 3*(Mz1 - Mz2)/(5*L) + Vy/2)
#         set_symmetric(4, 11, -6*P*r1/(5*L) - 3*beta_z*(Mz1 - Mz2)/(5*L)
#                       + 3*beta_y*(My1 - My2)/(5*L) - 6*Mw*beta_w/(5*L))
#         set_symmetric(4, 12, -(1/10)*(P*y0 + Mz1 - Vy*L))
#         # Skip 4,13 - part of v-θx coupling
#         set_symmetric(4, 14, (1/10)*(P*r1 + Mz1*beta_z - My1*beta_y + Mw*beta_w))
        
#         set_symmetric(5, 5, 2*P*L/15)
#         set_symmetric(5, 7, -2*P*y0*L/15 - L*(3*Mz1 - Mz2)/30)
#         set_symmetric(5, 9, P/10)
#         set_symmetric(5, 11, (1/10)*(P*y0 - Mz2 - Vy*L))
#         set_symmetric(5, 12, -P*L/30)
#         set_symmetric(5, 14, L*(2*P*y0 + Mz1 - Mz2 - Vy*L)/60)
        
#         set_symmetric(6, 6, 2*P*L/15)
#         # Skip 6,7 - involves My which should go to LTB coupling
#         set_symmetric(6, 9, -P/10)
#         # Skip 6,11 - involves My
#         set_symmetric(6, 13, -P*L/30)
#         # Skip 6,14 - involves My
        
#         set_symmetric(7, 7, 2*P*r1*L/15 + beta_z*L*(3*Mz1 - Mz2)/30
#                       - beta_y*L*(3*My1 - My2)/30 + 2*Mw*beta_w*L/15)
#         # Skip 7,9 - involves My
#         set_symmetric(7, 10, (1/10)*(P*y0 - Mz1 - Vy*L))  
#         set_symmetric(7, 11, -(1/10)*(P*r1 - Mz2*beta_z + My2*beta_y + Mw*beta_w))
#         set_symmetric(7, 12, L*(2*P*y0 + Mz1 - Mz2 + Vy*L)/60)
#         # Skip 7,13 - involves My
#         set_symmetric(7, 14, -P*r1*L/30 - beta_z*L*(Mz1 - Mz2)/60
#                       + beta_y*L*(My1 - My2)/60 + Mw*beta_w*L/30)
        
#         set_symmetric(9, 9, 6*P/(5*L))
#         # Skip 9,11 - involves My
#         set_symmetric(9, 13, -P/10)
#         # Skip 9,14 - involves My
        
#         set_symmetric(10, 10, 6*P/(5*L))
#         set_symmetric(10, 11, 6*P*y0/(5*L) + 3*(Mz1 - Mz2)/(5*L) + Vy/2)  
#         set_symmetric(10, 12, P/10)
#         set_symmetric(10, 14, -(1/10)*(P*y0 + Mz1 + Vy*L))
        
#         set_symmetric(11, 11, 6*P*r1/(5*L) + 3*beta_z*(Mz1 - Mz2)/(5*L)
#                       - 3*beta_y*(My1 - My2)/(5*L) + (Vz*z0 + Vy*y0)/2
#                       + 6*Mw*beta_w/(5*L))
#         set_symmetric(11, 12, (1/10)*(P*y0 - Mz2 - Vy*L)) 
#         # Skip 11,13 - involves My
#         set_symmetric(11, 14, -(1/10)*(P*r1 + Mz1*beta_z - My1*beta_y + Mw*beta_w))
        
#         set_symmetric(12, 12, 2*P*L/15)
#         set_symmetric(12, 14, 2*P*y0*L/15 - L*(Mz1 - 3*Mz2)/30)
        
#         set_symmetric(13, 13, 2*P*L/15)
#         # Skip 13,14 - involves My
        
#         set_symmetric(14, 14, 2*P*r1*L/15 + beta_z*L*(Mz1 - 3*Mz2)/30
#                       - beta_y*L*(My1 - 3*My2)/30 + 2*Mw*beta_w*L/15)

#         # =====================================================================
#         # ADDITIONAL LTB COUPLING: ∫M·v''·θ dx
#         # This is the key term missing from Chan & Kitipornchai for pure LTB
#         # Couples lateral bending (v: DOFs 2,6,9,13) with torsion (θx: DOFs 4,7,11,14)
#         # =====================================================================
        
#         # Compute coupling matrix via Gauss quadrature
#         # For moment My (bending about y-axis, causing LTB in v-θx plane)
#         ltb_coupling = _compute_ltb_coupling(L, My1, My2)
        
#         # Map to 14x14 matrix
#         # Lateral bending DOFs: v1=2, θz1=6, v2=9, θz2=13 (1-indexed)
#         # Torsion DOFs: θx1=4, θx1'=7, θx2=11, θx2'=14 (1-indexed)
#         v_dofs = [2, 6, 9, 13]
#         theta_dofs = [4, 7, 11, 14]
        
#         # Add coupling terms (these couple different DOF types, so add both K[i,j] and K[j,i])
#         for i, vi in enumerate(v_dofs):
#             for j, tj in enumerate(theta_dofs):
#                 K[vi-1, tj-1] = K[vi-1, tj-1] + ltb_coupling[i, j]
#                 K[tj-1, vi-1] = K[tj-1, vi-1] + ltb_coupling[i, j]

#         # =====================================================================
#         # WAGNER EFFECT (MONOSYMMETRY): ∫(β_x/2)·M·θ'·θ' dx
#         # This adds to the torsional geometric stiffness (θ-θ coupling)
#         # The factor of 1/2 comes from the energy formulation
#         # =====================================================================
#         if abs(beta_z) > 1e-16:
#             wagner_matrix = _compute_wagner_coupling(L, My1, My2, -beta_z)
            
#             # Torsion DOFs: θx1=4, θx1'=7, θx2=11, θx2'=14 (1-indexed)
#             theta_dofs = [4, 7, 11, 14]
            
#             for i, ti in enumerate(theta_dofs):
#                 for j, tj in enumerate(theta_dofs):
#                     K[ti-1, tj-1] = K[ti-1, tj-1] + wagner_matrix[i, j]

#     return K.tocsr()

# def _compute_ltb_coupling(L, My1, My2):
#     """
#     Compute the LTB coupling matrix from ∫M·v''·θ dx.
    
#     Returns 4x4 matrix coupling:
#     - v DOFs [v1, θz1, v2, θz2] 
#     - with θ DOFs [θx1, θx1', θx2, θx2']
#     """
#     from numpy.polynomial.legendre import leggauss
    
#     # Gauss quadrature points
#     xi_g, w_g = leggauss(4)
#     xi_g = (xi_g + 1) / 2  # Map to [0, 1]
#     w_g = w_g / 2
    
#     def N_v_2prime(xi):
#         """Second derivatives of Hermite shape functions for v'' (w.r.t. x)"""
#         return np.array([
#             (-6 + 12*xi) / L**2,      # d²N1/dx²
#             (-4 + 6*xi) / L,           # d²N2/dx² (N2 = L*ξ*(1-ξ)²)
#             (6 - 12*xi) / L**2,        # d²N3/dx²
#             (-2 + 6*xi) / L            # d²N4/dx² (N4 = L*ξ²*(ξ-1))
#         ])
    
#     def N_theta(xi):
#         """Hermite shape functions for θ"""
#         return np.array([
#             1 - 3*xi**2 + 2*xi**3,     # N1
#             L * xi * (1-xi)**2,         # N2
#             3*xi**2 - 2*xi**3,          # N3
#             L * xi**2 * (xi-1)          # N4
#         ])
    
#     coupling = np.zeros((4, 4))
#     for k in range(4):
#         xi = xi_g[k]
#         M = My1 * (1 - xi) + My2 * xi  # Linear moment variation
#         Nv_pp = N_v_2prime(xi)
#         Nth = N_theta(xi)
#         coupling += w_g[k] * M * np.outer(Nv_pp, Nth) * L  # L is Jacobian
    
#     return coupling

# def _compute_wagner_coupling(L, My1, My2, beta_y):
#     """
#     Compute the Wagner (monosymmetry) coupling matrix.
    
#     The monosymmetry effect is captured through:
#     ∫ β_x · M · θ'² dx
    
#     The sign and magnitude are calibrated to match Anderson & Trahair (1972).
    
#     Returns 4x4 matrix for θ DOFs [θx1, θx1', θx2, θx2']
#     """
#     from numpy.polynomial.legendre import leggauss
    
#     xi_g, w_g = leggauss(4)
#     xi_g = (xi_g + 1) / 2
#     w_g = w_g / 2
    
#     def N_theta_prime(xi):
#         """First derivatives of Hermite shape functions for θ'"""
#         return np.array([
#             (-6*xi + 6*xi**2) / L,
#             1 - 4*xi + 3*xi**2,
#             (6*xi - 6*xi**2) / L,
#             -2*xi + 3*xi**2
#         ])
    
#     wagner = np.zeros((4, 4))
    
#     for k in range(len(xi_g)):
#         xi = xi_g[k]
#         M = My1 * (1 - xi) + My2 * xi
#         Ntp = N_theta_prime(xi)
        
#         # Wagner term: -β_x · M · θ'²
#         # Sign: negative beta_x (delta<0) should DECREASE geometric stiffness -> HIGHER eigenvalue
#         # This matches: negative delta -> higher gamma in Table 2 (UDL)
#         wagner += w_g[k] * (-1.0) * beta_y * M * np.outer(Ntp, Ntp) * L
    
#     return wagner

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
    
    Parameters:
    -----------
    beta_x : float
        Wagner coefficient / monosymmetry parameter.
        For I-sections: β_x = (1/Iy)·∫y(x²+y²)dA - 2·y0
        Positive when larger flange is in compression (reduces buckling capacity).
    
    q1, q2 : float
        Transverse load intensity at element start and end nodes.
        Used for load height effect calculation.
        
    load_height : float
        Height of transverse load above shear center.
        Positive = load above shear center (destabilizing)
        Negative = load below shear center (stabilizing)
        
    The geometric stiffness includes:
    1. Axial load effects (P terms)
    2. LTB coupling: ∫M·v''·θ dx
    3. Wagner effect: ∫β_x·M·θ'·θ' dx (monosymmetry)
    4. Load height effect: ∫q·a·θ² dx
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

# def _compute_ltb_coupling(L, My1, My2):
#     """
#     Compute the LTB coupling matrix from ∫M·v''·θ dx.
    
#     Returns 4x4 matrix coupling:
#     - v DOFs [v1, θz1, v2, θz2] 
#     - with θ DOFs [θx1, θx1', θx2, θx2']
#     """
#     from numpy.polynomial.legendre import leggauss
    
#     xi_g, w_g = leggauss(4)
#     xi_g = (xi_g + 1) / 2
#     w_g = w_g / 2
    
#     def N_v_2prime(xi):
#         """Second derivatives of Hermite shape functions for v''"""
#         return np.array([
#             (-6 + 12*xi) / L**2,
#             (-4 + 6*xi) / L,
#             (6 - 12*xi) / L**2,
#             (-2 + 6*xi) / L
#         ])
    
#     def N_theta(xi):
#         """Hermite shape functions for θ"""
#         return np.array([
#             1 - 3*xi**2 + 2*xi**3,
#             L * xi * (1-xi)**2,
#             3*xi**2 - 2*xi**3,
#             L * xi**2 * (xi-1)
#         ])
    
#     coupling = np.zeros((4, 4))
#     for k in range(4):
#         xi = xi_g[k]
#         M = My1 * (1 - xi) + My2 * xi
#         coupling += w_g[k] * M * np.outer(N_v_2prime(xi), N_theta(xi)) * L

#     print(coupling)
#     return coupling

def _compute_ltb_coupling(L, My1, My2):
    """Analytical LTB coupling matrix."""

    coupling = np.array([
        [(-11*My1 - My2)/(10*L),  -My1/10,              (My1 + 11*My2)/(10*L),   -My2/10             ],
        [-9*My1/10 - My2/5,        L*(-3*My1 - My2)/30, -My1/10 + My2/5,          L*My1/30           ],
        [(11*My1 + My2)/(10*L),    My1/10,             (-My1 - 11*My2)/(10*L),    My2/10             ],
        [-My1/5 + My2/10,          L*My2/30,            My1/5 + 9*My2/10,         L*(-My1 - 3*My2)/30]
    ])

    # print(coupling) 
    return coupling 

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
    
    # # Save to file
    # import os
    
    # def dict_to_string(sparse_dict):
    #     """Convert sparse dictionary to formatted string"""
    #     lines = []
    #     for (i, j), value in sparse_dict.items():
    #         if abs(value) > 1e-10: # filter out near-zero values
    #             lines.append(f"K[{i+1}, {j+1}] = {value:.6e}")
    #     return "\n".join(lines)
    
    # # Create output directory
    # output_dir = "output_files"
    # os.makedirs(output_dir, exist_ok=True)
    
    # # Write to file
    # file_path = os.path.join(output_dir, "stiffness_matrix.txt")
    # with open(file_path, 'w') as f:
    #     f.write(dict_to_string(K.todok()))
    