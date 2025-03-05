import numpy as np 
from scipy.sparse import lil_matrix

# Author: James Whiteley (github.com/jamesalexwhiteley)
# Modified to use 0-based indexing

def thin_wall_stiffness_matrix(E, G, A, Ix, Iy, Iw, J, L, 
                                P0=0, Mx0=0, My0=0, B0_bar=0, W_bar=0,
                                x0=0, y0=0, beta_x=0, beta_y=0, r=0):
    """
    Create the element stiffness matrix for a 3D thin-walled beam element.
    
    Parameters:
    -----------
    E : floats
        Young's modulus
    G : float
        Shear modulus
    A : float
        Cross-sectional area
    Ix, Iy : float
        Second moments of area about x and y axes
    Iw : float
        Warping constant
    J : float
        Torsion constant
    L : float
        Element length
    P0 : float, optional
        Axial force
    Mx0, My0 : float, optional
        Bending moments about x and y axes
    B0_bar, W_bar : float, optional
        Bimoment and warping parameters
    x0, y0 : float, optional
        Coordinates of the shear center
    beta_x, beta_y : float, optional
        Rotation parameters
    r : float, optional
        Polar radius of gyration
        
    Returns:
    --------
    K : scipy.sparse.csr_matrix
        Element stiffness matrix (14x14)
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
    
    # Y-direction bending terms
    set_symmetric(1, 1, 12*E*Iy/(L**3) + 6*P0/(5*L))
    set_symmetric(1, 8, -12*E*Iy/(L**3) - 6*P0/(5*L))
    set_symmetric(8, 8, 12*E*Iy/(L**3) + 6*P0/(5*L))
    
    set_symmetric(1, 3, -3*Mx0/(5*L) + 3*P0*y0/(5*L))
    set_symmetric(1, 10, 3*Mx0/(5*L) - 3*P0*y0/(5*L))
    set_symmetric(8, 10, -3*Mx0/(5*L) + 3*P0*y0/(5*L))
    set_symmetric(3, 8, -3*Mx0/(5*L) + 3*P0*y0/(5*L))
    
    set_symmetric(1, 5, 6*E*Iy/(L**2) + P0/10)
    set_symmetric(1, 12, 6*E*Iy/(L**2) + P0/10)
    set_symmetric(5, 8, -6*E*Iy/(L**2) - P0/10)
    set_symmetric(8, 12, -6*E*Iy/(L**2) - P0/10)
    
    cross_term_val = -Mx0/20 + P0*y0/20
    set_symmetric(1, 6, cross_term_val)
    set_symmetric(1, 13, cross_term_val)
    set_symmetric(8, 13, -cross_term_val)
    set_symmetric(5, 10, -cross_term_val)
    set_symmetric(3, 5, cross_term_val)
    set_symmetric(3, 12, cross_term_val)
    set_symmetric(6, 8, -cross_term_val)
    set_symmetric(10, 12, -cross_term_val)
    
    # X-direction bending terms
    set_symmetric(2, 2, 12*E*Ix/(L**3) + 6*P0/(5*L))
    set_symmetric(2, 9, -12*E*Ix/(L**3) - 6*P0/(5*L))
    set_symmetric(9, 9, 12*E*Ix/(L**3) + 6*P0/(5*L))
    
    set_symmetric(2, 3, -3*My0/(5*L) - 3*P0*x0/(5*L))
    set_symmetric(2, 10, 3*My0/(5*L) + 3*P0*x0/(5*L))
    set_symmetric(3, 9, -3*My0/(5*L) - 3*P0*x0/(5*L))
    set_symmetric(9, 10, 3*My0/(5*L) + 3*P0*x0/(5*L))
    
    set_symmetric(2, 4, -6*E*Ix/(L**2) - P0/10)
    set_symmetric(2, 11, -6*E*Ix/(L**2) - P0/10)
    set_symmetric(4, 9, 6*E*Ix/(L**2) + P0/10)
    set_symmetric(9, 11, 6*E*Ix/(L**2) + P0/10)
    
    cross_term_val2 = -My0/20 - P0*x0/20
    set_symmetric(2, 6, cross_term_val2)
    set_symmetric(2, 13, cross_term_val2)
    set_symmetric(9, 13, -cross_term_val2)
    set_symmetric(3, 4, -cross_term_val2)
    set_symmetric(3, 11, -cross_term_val2)
    set_symmetric(10, 11, cross_term_val2)
    set_symmetric(4, 10, cross_term_val2)
    set_symmetric(6, 9, -cross_term_val2)
    
    # Warping and torsion terms
    warping_term = 12*E*Iw/(L**3) + 6*G*J/(5*L) + 6*P0*r**2/(5*L) + 3*Mx0*beta_y/(5*L) - 3*My0*beta_x/(5*L) - 6*B0_bar*W_bar/(5*L)
    set_symmetric(3, 3, warping_term)
    set_symmetric(3, 10, -warping_term)
    set_symmetric(10, 10, warping_term)
    
    warping_term2 = 6*E*Iw/(L**2) + G*J/10 + P0*r**2/10 + Mx0*beta_y/20 - My0*beta_x/20 - B0_bar*W_bar/10
    set_symmetric(3, 6, warping_term2)
    set_symmetric(3, 13, warping_term2)
    set_symmetric(6, 10, -warping_term2)
    set_symmetric(10, 13, -warping_term2)
    
    warping_term3 = 4*E*Iw/L + 2*G*J*L/15 + 2*L*P0*r**2/15 + L*Mx0*beta_y/15 - L*My0*beta_x/15 - 2*B0_bar*L*W_bar/15
    set_symmetric(6, 6, warping_term3)
    set_symmetric(13, 13, warping_term3)
    
    warping_term4 = 2*E*Iw/L - G*J*L/30 - L*P0*r**2/30 - L*Mx0*beta_y/60 + L*My0*beta_x/60 + B0_bar*L*W_bar/30
    set_symmetric(6, 13, warping_term4)

    # Additional terms
    set_symmetric(4, 4, 4*E*Ix/L + 2*L*P0/15)
    set_symmetric(11, 11, 4*E*Ix/L + 2*L*P0/15)
    
    set_symmetric(4, 6, L*My0/15 + L*P0*x0/15)
    set_symmetric(11, 13, L*My0/15 + L*P0*x0/15)
    
    set_symmetric(4, 13, -L*My0/60 - L*P0*x0/60)
    set_symmetric(6, 11, -L*My0/60 - L*P0*x0/60)
    
    set_symmetric(4, 11, 2*E*Ix/L - L*P0/30)
    
    set_symmetric(5, 5, 4*E*Iy/L + 2*L*P0/15)
    set_symmetric(12, 12, 4*E*Iy/L + 2*L*P0/15)
    
    set_symmetric(5, 6, -L*Mx0/15 + L*P0*y0/15)
    set_symmetric(12, 13, -L*Mx0/15 + L*P0*y0/15)
    
    set_symmetric(5, 12, 2*E*Iy/L - L*P0/30)
    
    set_symmetric(5, 13, L*Mx0/60 - L*P0*y0/60)
    set_symmetric(6, 12, L*Mx0/60 - L*P0*y0/60)
    
    # Convert to CSR format for fast arithmetic 
    return K.tocsr()