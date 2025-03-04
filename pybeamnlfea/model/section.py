import numpy as np 
from typing import List, Dict, Tuple, Optional, Union

# Author: James Whiteley (github.com/jamesalexwhiteley)

class Section:
    """
    Cross-section properties. 
    """
    def __init__(self, A: float, Ix: float, Iy: float, J: float, Iw: float = 0.0, **kwargs):
        """
        Initialize a cross-section with its geometric properties.
        
        Parameters
        ----------
        A : float
            Cross-sectional area.
        Ixx : float
            Second moment of area about x-x axis.
        Iyy : float
            Second moment of area about y-y axis.
        J : float
            Torsional constant.
        Iw : float
            Warping constant. 
        **kwargs : dict
            Additional section properties like plastic moduli, section class, etc.
        """
        self.A = A
        self.Ix = Ix
        self.Iy = Iy
        self.J = J
        self.Iw = Iw
        
        # Store any additional properties
        for key, value in kwargs.items():
            setattr(self, key, value)