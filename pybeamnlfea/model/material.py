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
        
        Args:
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
    def __init__(self, E: float, nu: float = None, G: float = None, **kwargs):
        """        
        Args: 
            E : float
                Young's modulus 
            nu : float
                Poisson's ratio 
            G : float
                Shear modulus 
            **kwargs : dict
                Additional material properties
        """
        super().__init__(**kwargs)
        self.E = E
        self.nu = nu
        
        if G == None: 
            self.G = self.shear_modulus()
        else: 
            self.G = G

    def shear_modulus(self) -> float:
        """
        Calculate and return the shear modulus (G) 
        G = E / (2 * (1 + nu))
        """
        if self.nu == None: 
            raise ValueError("You must provide either poisson ratio or shear modulus.")
        
        return self.E / (2 * (1 + self.nu))