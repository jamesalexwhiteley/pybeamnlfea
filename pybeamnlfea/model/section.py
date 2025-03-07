# import numpy as np 
# from typing import List, Dict, Tuple, Optional, Union

# # Author: James Whiteley (github.com/jamesalexwhiteley)

# class Section:
#     """
#     Cross-section properties. 
#     """
#     def __init__(self, A: float, Ix: float, Iy: float, J: float, Iw: float, x0: float, y0: float, **kwargs):
#         """
#         Initialise a cross-section with its geometric properties.
        
#         Args:
#         
#         A : float
#             Cross-sectional area.
#         Ixx : float
#             Second moment of area about x-x axis.
#         Iyy : float
#             Second moment of area about y-y axis.
#         J : float
#             Torsional constant.
#         Iw : float
#             Warping constant. 
#         **kwargs : dict
#             Additional section properties like plastic moduli, section class, etc.
#         """
#         self.A = A
#         self.Ix = Ix
#         self.Iy = Iy
#         self.J = J
#         self.Iw = Iw

#         self.x0 = x0
#         self.y0 = y0
        
#         # Store any additional properties
#         for key, value in kwargs.items():
#             setattr(self, key, value)

import numpy as np 
from typing import List, Dict, Tuple, Optional, Union

# Author: James Whiteley (github.com/jamesalexwhiteley)

class Section:
    """
    Cross-section properties. 
    """
    def __init__(self, A: float, Iy: float, Iz: float, J: float, Iw: float, y0: float, z0: float, **kwargs):
        """
        Initialise a cross-section with its geometric properties.
        
        Args:
            A : float
                Cross-sectional area.
            Iy : float
                Second moment of area about y-y axis.
            Iz : float
                Second moment of area about z-z axis.
            J : float
                Torsional constant.
            Iw : float
                Warping constant. 
            y0 : float
                Y-coordinate of shear center relative to centroid.
            z0 : float
                Z-coordinate of shear center relative to centroid.
            **kwargs : dict
                Additional section properties like plastic moduli, section class, etc.
        """
        self.A = A
        self.Iy = Iy
        self.Iz = Iz
        self.J = J
        self.Iw = Iw

        self.y0 = y0
        self.z0 = z0
        
        # Store any additional properties
        for key, value in kwargs.items():
            setattr(self, key, value)