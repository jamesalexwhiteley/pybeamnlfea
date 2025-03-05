import numpy as np 
from typing import List, Dict, Tuple, Optional, Union

# Author: James Whiteley (github.com/jamesalexwhiteley)

class Material:
    """
    Base material class. 
    """
    def __init__(self, **kwargs):
        """
        Initialise a generic material.
        
        Parameters
        ----------
        **kwargs : dict
            Material properties
        """
        # Store additional properties
        for key, value in kwargs.items():
            setattr(self, key, value)

class LinearElastic(Material):
    """
    Linear elastic material.
    """
    def __init__(self, young_modulus: float, poisson_ratio: float, **kwargs):
        """        
        Parameters
        ----------
        young_modulus : float
            Young's modulus (E) 
        poisson_ratio : float
            Poisson's ratio (nu) 
        **kwargs : dict
            Additional material properties
        """
        super().__init__(**kwargs)
        self.E = young_modulus
        self.nu = poisson_ratio
        
    @property
    def G(self) -> float:
        """
        Calculate and return the shear modulus (G) 
        G = E / (2 * (1 + nu))
        """
        return self.E / (2 * (1 + self.nu))