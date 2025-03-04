import numpy as np 
from typing import List, Dict, Tuple, Optional, Union

# Author: James Whiteley (github.com/jamesalexwhiteley)

class Material:
    """
    Base material class. 
    """
    def __init__(self, **kwargs):
        """
        Initialize a generic material.
        
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
        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio
        
    @property
    def shear_modulus(self) -> float:
        """
        Calculate and return the shear modulus (G) 
        G = E / (2 * (1 + nu))
        """
        return self.young_modulus / (2 * (1 + self.poisson_ratio))