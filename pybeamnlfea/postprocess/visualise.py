import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Author: James Whiteley (github.com/jamesalexwhiteley)

class Visualiser:
    def __init__(self, results):
        """
        Initialize the visualizer with analysis results.
        
        Args:
            results: Results object containing displacement data
        """
        self.results = results
        self.frame = results.frame
        
    def plot_deformed_shape(self, scale=1.0, fig=None, ax=None, show_undeformed=True, 
                           color_undeformed='k', color_deformed='r', 
                           node_size=20, show_nodes=True, num_points=20):
        """
        Plot the original and curved deflected shape of the structure.
        
        Args:
            scale: Scale factor for the deformation
            fig: Existing figure (optional)
            ax: Existing axis (optional)
            show_undeformed: Whether to show the undeformed shape
            color_undeformed: Color for the undeformed structure
            color_deformed: Color for the deformed structure
            node_size: Size of node markers
            show_nodes: Whether to show nodes
            num_points: Number of points along each element for smooth curves
            
        Returns:
            fig, ax: The figure and axis objects
        """
        # Determine if we're dealing with a 2D or 3D structure
        is_3d = True
        for node in self.frame.nodes.values():
            if hasattr(node, 'z') and node.z != 0:
                is_3d = True
                break
        
        # Create or get figure and axis
        if fig is None or ax is None:
            fig = plt.figure(figsize=(10, 8))
            if is_3d:
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(111)
        
        # Plot undeformed structure
        if show_undeformed:
            self._plot_structure_straight(ax, deformed=False, color=color_undeformed, 
                                      node_size=node_size, show_nodes=show_nodes)
        
        # Plot deformed structure using curved element deflections
        self._plot_curved_structure(ax, scale=scale, color=color_deformed, 
                                  node_size=node_size, show_nodes=show_nodes,
                                  num_points=num_points)
        
        # Set axis properties
        if is_3d:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_box_aspect([1.0, 1.0, 1.0])
        else:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_aspect('equal')
            
        ax.grid(True)
        plt.title(f'Structure Deformation (scale: {scale}X)')
        legend_items = []
        if show_undeformed:
            legend_items.append(('Original', color_undeformed))
        legend_items.append(('Deformed', color_deformed))
        ax.legend([plt.Line2D([0], [0], color=color, lw=2) for _, color in legend_items],
                 [label for label, _ in legend_items])
        
        plt.tight_layout()
        plt.show()
        return fig, ax
    
    def _plot_structure_straight(self, ax, deformed=False, scale=1.0, color='k', 
                              node_size=20, show_nodes=True):
        """
        Helper method to plot the structure with straight elements.
        
        Args:
            ax: Matplotlib axis to plot on
            deformed: Whether to plot the deformed shape
            scale: Deformation scale factor (only used if deformed=True)
            color: Line and marker color
            node_size: Size of node markers
            show_nodes: Whether to show nodes
        """
        is_3d = isinstance(ax, Axes3D)
        
        # Get all node coordinates (original or deformed)
        node_coords = {}
        for node_id, node in self.frame.nodes.items():
            if deformed:
                # Get displacements from results
                dx = self.results.get_nodal_displacement(node_id, 0) if hasattr(node, 'x') else 0
                dy = self.results.get_nodal_displacement(node_id, 1) if hasattr(node, 'y') else 0
                dz = self.results.get_nodal_displacement(node_id, 2) if hasattr(node, 'z') else 0
                
                # Apply scale factor
                dx *= scale
                dy *= scale
                dz *= scale
                
                # Apply displacement to original coordinates
                x = node.x + dx if hasattr(node, 'x') else dx
                y = node.y + dy if hasattr(node, 'y') else dy
                z = node.z + dz if hasattr(node, 'z') else dz
            else:
                # Use original node coordinates
                x = node.x if hasattr(node, 'x') else 0
                y = node.y if hasattr(node, 'y') else 0
                z = node.z if hasattr(node, 'z') else 0
            
            node_coords[node_id] = (x, y, z)
        
        # Plot elements
        for element_id, element in self.frame.elements.items():
            # Get node IDs for this element
            node_ids = [node.id for node in element.nodes]
            
            # Extract coordinates for each node in the element
            x_vals = [node_coords[node_id][0] for node_id in node_ids]
            y_vals = [node_coords[node_id][1] for node_id in node_ids]
            
            if is_3d:
                z_vals = [node_coords[node_id][2] for node_id in node_ids]
                ax.plot(x_vals, y_vals, z_vals, color=color, linewidth=2)
            else:
                ax.plot(x_vals, y_vals, color=color, linewidth=2)
        
        # Plot nodes
        if show_nodes:
            x_vals = [coords[0] for coords in node_coords.values()]
            y_vals = [coords[1] for coords in node_coords.values()]
            
            if is_3d:
                z_vals = [coords[2] for coords in node_coords.values()]
                ax.scatter(x_vals, y_vals, z_vals, color=color, s=node_size)
            else:
                ax.scatter(x_vals, y_vals, color=color, s=node_size)
    
    def _plot_curved_structure(self, ax, scale=1.0, color='r', node_size=20, 
                             show_nodes=True, num_points=20):
        """
        Plot the deformed structure with curved elements using the calculate_element_deflection method.
        
        Args:
            ax: Matplotlib axis to plot on
            scale: Deformation scale factor
            color: Line and marker color
            node_size: Size of node markers
            show_nodes: Whether to show nodes
            num_points: Number of points to calculate along each element
        """
        is_3d = isinstance(ax, Axes3D)
        
        # Get deformed node coordinates for plotting nodes
        node_coords = {}
        for node_id, node in self.frame.nodes.items():
            # Get displacements from results
            dx = self.results.get_nodal_displacement(node_id, 0) if hasattr(node, 'x') else 0
            dy = self.results.get_nodal_displacement(node_id, 1) if hasattr(node, 'y') else 0
            dz = self.results.get_nodal_displacement(node_id, 2) if hasattr(node, 'z') else 0
            
            # Apply scale factor
            dx *= scale
            dy *= scale
            dz *= scale
            
            # Apply displacement to original coordinates
            x = node.x + dx if hasattr(node, 'x') else dx
            y = node.y + dy if hasattr(node, 'y') else dy
            z = node.z + dz if hasattr(node, 'z') else dz
            
            node_coords[node_id] = (x, y, z)
        
        # Plot each element with curved deflection
        for element_id, element in self.frame.elements.items():
            # Get element orientation and data
            try:
                # Get element direction vector (local x-axis)
                start_node = element.nodes[0]
                end_node = element.nodes[-1]
                
                # Calculate element direction vector in global coordinates
                dir_vector = np.array([
                    end_node.x - start_node.x if hasattr(start_node, 'x') else 0,
                    end_node.y - start_node.y if hasattr(start_node, 'y') else 0,
                    end_node.z - start_node.z if hasattr(start_node, 'z') else 0
                ])
                
                # Normalize direction vector
                dir_length = np.linalg.norm(dir_vector)
                if dir_length > 0:
                    local_x = dir_vector / dir_length
                else:
                    raise ValueError("Element has zero length")
                
                # Get global coordinates of element start point
                start_global = np.array([
                    start_node.x if hasattr(start_node, 'x') else 0,
                    start_node.y if hasattr(start_node, 'y') else 0,
                    start_node.z if hasattr(start_node, 'z') else 0
                ])
                
                # Define local coordinate system
                # For local y-axis, use the element's orientation vector if available
                if hasattr(element, 'orientation_vector'):
                    # If orientation vector is provided, use it to define local y
                    local_y_temp = element.orientation_vector
                    # Ensure local_y is perpendicular to local_x
                    local_y = local_y_temp - np.dot(local_y_temp, local_x) * local_x
                    norm_y = np.linalg.norm(local_y)
                    if norm_y > 0:
                        local_y = local_y / norm_y
                    else:
                        raise ValueError("Invalid orientation vector")
                else:
                    # Determine a suitable reference vector for establishing the local coordinate system
                    # For vertical elements, use global y as reference
                    if abs(local_x[2]) > 0.99:  # Almost parallel to global z
                        ref_vector = np.array([0, 1, 0])  # Global y
                    else:
                        ref_vector = np.array([0, 0, 1])  # Global z
                    
                    # Local y is perpendicular to local x
                    local_y = np.cross(ref_vector, local_x)
                    norm_y = np.linalg.norm(local_y)
                    if norm_y > 0:
                        local_y = local_y / norm_y
                    else:
                        # If local_y is zero, try a different reference vector
                        ref_vector = np.array([1, 0, 0])  # Global x
                        local_y = np.cross(ref_vector, local_x)
                        local_y = local_y / np.linalg.norm(local_y)
                
                # Local z is perpendicular to both local x and y
                local_z = np.cross(local_x, local_y)
                local_z = local_z / np.linalg.norm(local_z)
                
                # Get element deflection in local coordinates
                z_local, deflection = self.results.calculate_element_deflection(
                    element_id, num_points=num_points)
                
                # Convert local deflection to global coordinates
                global_points = np.zeros((num_points, 3))
                for i in range(num_points):
                    # Position along local x-axis
                    pos_local_x = z_local[i]
                    
                    # Get deflection components (w, u, v, θz, θx, θy) at this point
                    # w: Axial deflection (along local x)
                    # u: Transverse deflection along local y
                    # v: Transverse deflection along local z
                    w = deflection[i, 0] * scale  # Axial
                    u = deflection[i, 1] * scale  # Transverse in y
                    v = deflection[i, 2] * scale  # Transverse in z
                    
                    # Convert to global coordinates
                    # Start at the element's starting position
                    # Move along the local x-axis by pos_local_x + axial deflection
                    # Add transverse deflections in local y and z directions
                    global_points[i] = (
                        start_global + 
                        (pos_local_x + w) * local_x + 
                        u * local_y + 
                        v * local_z
                    )
                
                # Plot the curved element
                if is_3d:
                    ax.plot(global_points[:, 0], global_points[:, 1], global_points[:, 2], 
                            color=color, linewidth=2)
                else:
                    ax.plot(global_points[:, 0], global_points[:, 1], color=color, linewidth=2)
                    
            except (KeyError, ValueError, AttributeError) as e:
                # If error in calculation, fall back to straight line between deformed nodes
                print(f"Warning: Unable to plot curved element {element_id}. Error: {e}")
                node_ids = [node.id for node in element.nodes]
                x_vals = [node_coords[node_id][0] for node_id in node_ids]
                y_vals = [node_coords[node_id][1] for node_id in node_ids]
                
                if is_3d:
                    z_vals = [node_coords[node_id][2] for node_id in node_ids]
                    ax.plot(x_vals, y_vals, z_vals, color=color, linewidth=2, linestyle='--')
                else:
                    ax.plot(x_vals, y_vals, color=color, linewidth=2, linestyle='--')
        
        # Plot nodes
        if show_nodes:
            x_vals = [coords[0] for coords in node_coords.values()]
            y_vals = [coords[1] for coords in node_coords.values()]
            
            if is_3d:
                z_vals = [coords[2] for coords in node_coords.values()]
                ax.scatter(x_vals, y_vals, z_vals, color=color, s=node_size)
            else:
                ax.scatter(x_vals, y_vals, color=color, s=node_size)
