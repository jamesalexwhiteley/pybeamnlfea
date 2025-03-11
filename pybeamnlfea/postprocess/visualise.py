# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Author: James Whiteley (github.com/jamesalexwhiteley)

# class Visualiser:
#     def __init__(self, results):
#         """
#         Initialize the visualizer with analysis results.
        
#         Args:
#             results: Results object containing displacement data
#         """
#         self.results = results
#         self.frame = results.frame
    
#     def plot_deformed_shape(self, scale_factor=1.0, show_undeformed=True, num_points=20):
#         """
#         Plot the deformed and undeformed shape of the structure.
        
#         Args:
#             scale_factor: Factor to amplify displacements for better visualization
#             show_undeformed: Whether to show the undeformed shape
#             num_points: Number of points to use for curved elements
#         """
#         # Create a 3D plot
#         fig = plt.figure(figsize=(10, 8))
#         ax = fig.add_subplot(111, projection='3d')
        
#         # Plot undeformed structure
#         if show_undeformed:
#             self._plot_nodes_and_elements(ax, deformed=False, color='gray', alpha=0.3)
        
#         # Plot deformed structure with nodes
#         self._plot_nodes_and_elements(ax, deformed=True, scale_factor=scale_factor, color='blue')
        
#         # Plot deformed structure with curved elements
#         self._plot_curved_elements(ax, scale_factor=scale_factor, num_points=num_points, color='blue')
        
#         # Set labels and title
#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_zlabel('Z')
#         ax.set_title(f'Structural Deformation (scale factor: {scale_factor})')
        
#         # Add legend
#         from matplotlib.lines import Line2D
#         legend_elements = [
#             Line2D([0], [0], color='gray', lw=1.5, alpha=0.3, label='Undeformed'),
#             Line2D([0], [0], color='blue', lw=1.5, label='Deformed')
#         ]
#         ax.legend(handles=legend_elements, loc='best')

#         # Set equal aspect ratio for all axes
#         ax.set_box_aspect([1, 1, 1]) 

#         # Remove background
#         ax.xaxis.pane.fill = False
#         ax.yaxis.pane.fill = False
#         ax.zaxis.pane.fill = False

#         # Make panes transparent
#         ax.xaxis.pane.set_edgecolor('w')
#         ax.yaxis.pane.set_edgecolor('w')
#         ax.zaxis.pane.set_edgecolor('w')
        
#         plt.tight_layout()
#         plt.show()
#         return fig, ax
    
#     def _plot_nodes_and_elements(self, ax, deformed=False, scale_factor=1.0, color='blue', alpha=1.0):
#         """
#         Helper method to plot the nodes and elements.
#         """
#         # Get node coordinates (original and deformed)
#         node_coords = {}
#         for node_id, node in self.frame.nodes.items():
#             original_coords = node.coords
            
#             if deformed:
#                 # Get displacements
#                 ux = self.results.get_nodal_displacement(node_id, 0) * scale_factor  # ux - local x
#                 uy_bar = self.results.get_nodal_displacement(node_id, 1) * scale_factor  # uȳ - local y
#                 uz_bar = self.results.get_nodal_displacement(node_id, 2) * scale_factor  # uz̄ - local z
    
#                 deformed_coords = original_coords + np.array([ux, uy_bar, uz_bar])
                
#                 node_coords[node_id] = deformed_coords
#             else:
#                 node_coords[node_id] = original_coords
        
#         # Plot nodes
#         for node_id, coords in node_coords.items():
#             ax.scatter(coords[0], coords[1], coords[2], s=50, color=color, alpha=alpha)
        
#         # Plot elements as straight lines between nodes
#         for element_id, element in self.frame.elements.items():
#             # Get coordinates of element nodes
#             element_nodes = [node.id for node in element.nodes]
#             node_points = [node_coords[node_id] for node_id in element_nodes]
            
#             # Extract x, y, z coordinates for plotting
#             x_coords = [point[0] for point in node_points]
#             y_coords = [point[1] for point in node_points]
#             z_coords = [point[2] for point in node_points]
            
#             ax.plot(x_coords, y_coords, z_coords, linewidth=1.5, color=color, alpha=alpha)
    
#     def _plot_curved_elements(self, ax, scale_factor=1.0, num_points=20, color='blue', alpha=1.0):
#         """
#         Plot elements with their deflected curved shape.
#         """
#         for element_id, element in self.frame.elements.items():
#             # Get deflection data in local coordinates
#             c, deflection = self.results.calculate_element_deflection(element_id, num_points)
            
#             # Get nodes
#             start_node = element.nodes[0]
#             end_node = element.nodes[1]
            
#             # Get node coordinates
#             start_pos = start_node.coords
#             end_pos = end_node.coords
            
#             # Calculate element direction and length
#             element_dir = end_pos - start_pos
#             element_length = np.linalg.norm(element_dir)
            
#             # Create array for deformed points
#             global_points = np.zeros((num_points, 3))
            
#             # Get local coordinate system (rotation matrix)
#             R = element.R
            
#             for i in range(num_points):
#                 # Normalized position along element
#                 xi = c[i] / element_length
                
#                 # Base point along undeformed element
#                 base_point = start_pos + xi * element_dir
                
#                 # Local displacement vector (uy, uz, ux) at this point
#                 local_disp = np.array([
#                     deflection[i, 0],  # ux (local x)
#                     deflection[i, 1],  # uy (local y)
#                     deflection[i, 2]   # uz (local z)
#                 ])
                
#                 # Transform local displacement to global coordinates
#                 global_disp = R.T @ local_disp  # R.T transforms from local to global

#                 # Apply displacement scaling
#                 global_disp *= scale_factor
                
#                 # Final position = base point + displacement
#                 global_points[i] = base_point + global_disp
            
#             # Plot the curved element
#             ax.plot(global_points[:, 0], global_points[:, 1], global_points[:, 2], 
#                     linewidth=2, color=color, alpha=alpha)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Visualiser:
    def __init__(self, results):
        """
        Initialize the visualizer with analysis results.
        
        Args:
            results: Results object containing displacement data
        """
        self.results = results
        self.frame = results.frame
    
    def plot_deformed_shape(self, scale_factor=1.0, show_undeformed=True, 
                           num_points=20, figsize=(10, 8), show_warping=True):
        """
        Plot the deformed and undeformed shape of the structure.
        Using the improved shape function approach from static_fem.
        
        Args:
            scale_factor: Factor to amplify displacements for better visualization
            show_undeformed: Whether to show the undeformed shape
            num_points: Number of points to use for curved elements
            figsize: Figure size
            show_warping: Whether to visualize warping effects with color
        """
        # Delegate to the Results class plotting method which now uses the same logic
        # as the static_fem script
        return self.results.plot_deformed_shape(
            scale=scale_factor,
            npoints=num_points,
            figsize=figsize,
            show_warping=show_warping,
            show_undeformed=show_undeformed
        )
    
    def plot_internal_forces(self, element_ids=None, force_type='all', 
                            num_points=50, figsize=(15, 10)):
        """
        Plot internal forces along selected elements.
        
        Args:
            element_ids: List of element IDs to plot (None = all elements)
            force_type: Type of internal forces to plot ('axial', 'shear', 'moment', 'torsion', 'all')
            num_points: Number of points along each element
            figsize: Figure size for the plot
        """
        if element_ids is None:
            element_ids = list(self.frame.elements.keys())
            
        if isinstance(element_ids, int):
            element_ids = [element_ids]
            
        # Set up subplots based on what to display
        force_types = []
        if force_type == 'all':
            force_types = ['axial', 'shear_y', 'shear_z', 'torsion', 'moment_y', 'moment_z']
            n_rows, n_cols = 3, 2
        elif force_type == 'axial':
            force_types = ['axial']
            n_rows, n_cols = 1, 1
        elif force_type == 'shear':
            force_types = ['shear_y', 'shear_z']
            n_rows, n_cols = 1, 2
        elif force_type == 'moment':
            force_types = ['moment_y', 'moment_z']
            n_rows, n_cols = 1, 2
        elif force_type == 'torsion':
            force_types = ['torsion']
            n_rows, n_cols = 1, 1
        else:
            # Specific force type
            force_types = [force_type]
            n_rows, n_cols = 1, 1
            
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Handle case of single subplot
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        
        # Flatten axes array for easier iteration
        axes = axes.flatten()
        
        # Dictionary to map force types to subplot indices and labels
        force_info = {
            'axial': {'idx': 0, 'label': 'Axial Force (N)'},
            'shear_y': {'idx': 1, 'label': 'Shear Force Y (N)'},
            'shear_z': {'idx': 2, 'label': 'Shear Force Z (N)'},
            'torsion': {'idx': 3, 'label': 'Torsion Moment (N⋅m)'},
            'moment_y': {'idx': 4, 'label': 'Bending Moment Y (N⋅m)'},
            'moment_z': {'idx': 5, 'label': 'Bending Moment Z (N⋅m)'}
        }
        
        # Collect data for each element
        for element_id in element_ids:
            element = self.frame.elements[element_id]
            element_forces = self.results.get_element_forces(element_id)
            
            if element_forces is None:
                continue
                
            # Create array of positions along element
            element_length = element.length
            x = np.linspace(0, element_length, num_points)
            
            # Get element node IDs for labeling
            start_node_id = element.nodes[0].id
            end_node_id = element.nodes[1].id
            element_label = f"Element {element_id} (Node {start_node_id} to {end_node_id})"
            
            # Plot each force type in its respective subplot
            for ft in force_types:
                ax_idx = force_info[ft]['idx']
                ax = axes[ax_idx]
                
                # Extract force values and interpolate if needed
                # For demonstration, using simple linear interpolation between end values
                # This could be enhanced with proper shape functions for force interpolation
                start_force = element_forces[ft + '_start']
                end_force = element_forces[ft + '_end']
                
                # Linear interpolation
                forces = start_force + (end_force - start_force) * x / element_length
                
                # Plot
                ax.plot(x, forces, label=element_label, linewidth=2)
                ax.set_xlabel('Position along element (m)')
                ax.set_ylabel(force_info[ft]['label'])
                ax.set_title(force_info[ft]['label'])
                ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legends if multiple elements
        if len(element_ids) > 1:
            for ax in axes:
                ax.legend()
                
        plt.tight_layout()
        plt.show()
        return fig, axes
        
    def plot_cross_section_results(self, element_id, position=0.5, stress_type='normal', 
                                  num_points=100, figsize=(8, 6)):
        """
        Plot stress distribution on a cross-section.
        
        Args:
            element_id: ID of the element to plot
            position: Normalized position along element (0 to 1)
            stress_type: Type of stress ('normal', 'shear', 'von_mises')
            num_points: Resolution of the cross-section plot
            figsize: Figure size
        """
        # Get the element
        element = self.frame.elements[element_id]
        
        # Get cross-section properties
        cross_section = element.section
        
        # This is a placeholder for now - in a real implementation, 
        # you would use cross-section information to calculate stress distribution
        fig, ax = plt.subplots(figsize=figsize)
        
        # Placeholder for drawing cross-section shape
        # This would depend on the actual cross-section data structure
        if hasattr(cross_section, 'height') and hasattr(cross_section, 'width'):
            # Rectangle section
            height = cross_section.height
            width = cross_section.width
            rectangle = plt.Rectangle((-width/2, -height/2), width, height, 
                                     fill=False, color='black')
            ax.add_patch(rectangle)
            
            # Generate a stress field (placeholder)
            x = np.linspace(-width/2, width/2, num_points)
            y = np.linspace(-height/2, height/2, num_points)
            X, Y = np.meshgrid(x, y)
            
            # Get forces at this position
            forces = self.results.get_element_forces(element_id)
            if forces:
                # Calculate position along element
                L = element.length
                pos = position * L
                
                # For demonstration, use linear interpolation of forces
                axial_force = forces['axial_start'] + (forces['axial_end'] - forces['axial_start']) * position
                moment_y = forces['moment_y_start'] + (forces['moment_y_end'] - forces['moment_y_start']) * position
                moment_z = forces['moment_z_start'] + (forces['moment_z_end'] - forces['moment_z_start']) * position
                
                # Simple stress calculation for rectangular section
                A = width * height
                Iy = width * height**3 / 12
                Iz = height * width**3 / 12
                
                # Normal stress from axial force and bending
                sigma = axial_force / A + moment_y * X / Iy + moment_z * Y / Iz
                
                # Create contour plot
                c = ax.contourf(X, Y, sigma, cmap='coolwarm')
                plt.colorbar(c, ax=ax, label='Normal Stress (Pa)')
        
        ax.set_aspect('equal')
        ax.set_xlabel('y (m)')
        ax.set_ylabel('z (m)')
        ax.set_title(f'Stress at Element {element_id}, Position {position:.2f}')
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()
        return fig, ax