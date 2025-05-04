import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Author: James Whiteley (github.com/jamesalexwhiteley)

class Visualiser:
    def __init__(self, model=None, results=None):
        """
        Initialise visualiser with model and optional results.

        """
        self.model = model
        self.results = results
        self.fig = None
        self.ax = None
        self.local_axes_scale = 0.3

    def initialize_plot(self, figsize=(10, 8), projection='3d'):
        """Create figure and axis if they don't exist."""
        if self.fig is None or self.ax is None:
            self._initialize_plot(figsize=figsize, projection=projection)
        
    def _initialize_plot(self, figsize=(10, 8), projection='3d'):
        """Create figure and axis if they don't exist."""
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection=projection)
        
        # Configure plot settings
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        # Remove background 
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('w')
        self.ax.yaxis.pane.set_edgecolor('w')
        self.ax.zaxis.pane.set_edgecolor('w')
            
        return self.fig, self.ax
    
    def draw_local_axes(self, origin, R, scale=0.2, linewidth=2, labels=True):
        """Draw local coordinate axes at a given origin with rotation matrix R."""
        self.initialize_plot()
        
        # Extract basis vectors
        x_axis = R[0, :] # Local x-axis in global coordinates
        y_axis = R[1, :] # Local y-axis in global coordinates
        z_axis = R[2, :] # Local z-axis in global coordinates
        
        # Plot the three axes with different colors
        self.ax.quiver(origin[0], origin[1], origin[2], 
                scale*x_axis[0], scale*x_axis[1], scale*x_axis[2], 
                color='r', linewidth=linewidth, arrow_length_ratio=0.2)
        
        self.ax.quiver(origin[0], origin[1], origin[2], 
                scale*y_axis[0], scale*y_axis[1], scale*y_axis[2], 
                color='g', linewidth=linewidth, arrow_length_ratio=0.2)
        
        self.ax.quiver(origin[0], origin[1], origin[2], 
                scale*z_axis[0], scale*z_axis[1], scale*z_axis[2], 
                color='b', linewidth=linewidth, arrow_length_ratio=0.2)
        
        if labels:
            # Add labels at the end of each axis
            self.ax.text(origin[0] + scale*x_axis[0]*1.1, 
                    origin[1] + scale*x_axis[1]*1.1, 
                    origin[2] + scale*x_axis[2]*1.1, "x", color='r')
            self.ax.text(origin[0] + scale*y_axis[0]*1.1, 
                    origin[1] + scale*y_axis[1]*1.1, 
                    origin[2] + scale*y_axis[2]*1.1, "y", color='g')
            self.ax.text(origin[0] + scale*z_axis[0]*1.1, 
                    origin[1] + scale*z_axis[1]*1.1, 
                    origin[2] + scale*z_axis[2]*1.1, "z", color='b')
                
    def plot_undeformed_model(self, nodes=True, dashed=True, show_local_axes=True, node_labels=True):
        """Visualise the show_undeformed model with local axes for each element."""

        if self.model is None:
            raise ValueError("No model provided to visualiser")
            
        self.initialize_plot()
        
        # Plot elements
        for elem_id, element in self.model.elements.items():
            start_node, end_node = element.nodes
            
            # Plot element
            if dashed: 
                line_style = 'k--'
            else: 
                line_style = 'k-'
            self.ax.plot([start_node.coords[0], end_node.coords[0]],
                         [start_node.coords[1], end_node.coords[1]],
                         [start_node.coords[2], end_node.coords[2]], line_style, lw=1, alpha=0.7)
            
            # Plot local axes at midpoint
            if show_local_axes: 
                midpoint = (start_node.coords + end_node.coords) / 2
                self.draw_local_axes(midpoint, element.R, scale=self.local_axes_scale*element.L)
            
            # Plot nodes
            if nodes: 
                self.ax.scatter(start_node.coords[0], start_node.coords[1], start_node.coords[2], 
                            color='gray', s=20, alpha=0.7)
                self.ax.scatter(end_node.coords[0], end_node.coords[1], end_node.coords[2], 
                            color='gray', s=20, alpha=0.7)
            
            # Add node labels 
            if node_labels:
                self.ax.text(start_node.coords[0], start_node.coords[1], start_node.coords[2], 
                          f" {start_node.id}", fontsize=10)
                self.ax.text(end_node.coords[0], end_node.coords[1], end_node.coords[2], 
                          f" {end_node.id}", fontsize=10)

        return self.fig, self.ax

    # NOTE STRAIGHT LINE PLOTTING
    # def plot_deformed_shape(self, scale=1.0, show_undeformed=False, show_previous=False, 
    #                         show_local_axes=False, node_labels=True, npoints=20):
    #     """Visualize the model with deformation between previous and current states."""

    #     if self.model is None:
    #         raise ValueError("No model provided to visualiser")
        
    #     self.initialize_plot()
        
    #     if show_undeformed:
    #         self.plot_undeformed_model(nodes=True, dashed=True, show_local_axes=False, node_labels=False)
        
    #     # Plot elements
    #     for elem_id, element in self.model.elements.items():
            
    #         # Initial state 
    #         init_coords = element.initial_state['coords']

    #         # Current step
    #         prev_coords = element.previous_state['coords']
    #         prev_R = element.previous_state['R']
    #         prev_L = element.previous_state['L']
            
    #         curr_coords = element.current_state['coords']
    #         curr_R = element.current_state['R'] 
    #         curr_L = element.current_state['L']

    #         # Scale deformations 
    #         prev_coords = [
    #             init_coords[i] + scale * (prev_coords[i] - init_coords[i]) 
    #             for i in range(2)
    #         ]
            
    #         curr_coords = [
    #             init_coords[i] + scale * (curr_coords[i] - init_coords[i])
    #             for i in range(2)
    #         ]
            
    #         # For previous state (if we scale displacements, we also need to scale local axes)
    #         prev_L = np.linalg.norm(prev_coords[1] - prev_coords[0])
    #         prev_R = prev_R.copy()
    #         if prev_L > 1e-10:
    #             # Update x-axis (tangent) to match scaled direction
    #             prev_R[0] = (prev_coords[1] - prev_coords[0]) / prev_L
                
    #             # Ensure y and z axes remain orthogonal to new x-axis
    #             y_temp = prev_R[1]
    #             y_local = y_temp - np.dot(y_temp, prev_R[0]) * prev_R[0]
    #             y_local = y_local / np.linalg.norm(y_local)
                
    #             z_local = np.cross(prev_R[0], y_local)
    #             z_local = z_local / np.linalg.norm(z_local)
                
    #             prev_R[1] = y_local
    #             prev_R[2] = z_local
            
    #         # For current state (if we scale displacements, we also need to scale local axes)
    #         curr_L = np.linalg.norm(curr_coords[1] - curr_coords[0])
    #         curr_R = curr_R.copy()
    #         if curr_L > 1e-10:
    #             # Update x-axis (tangent) to match scaled direction
    #             curr_R[0] = (curr_coords[1] - curr_coords[0]) / curr_L
                
    #             # Ensure y and z axes remain orthogonal to new x-axis
    #             y_temp = curr_R[1]
    #             y_local = y_temp - np.dot(y_temp, curr_R[0]) * curr_R[0]
    #             y_local = y_local / np.linalg.norm(y_local)
                
    #             z_local = np.cross(curr_R[0], y_local)
    #             z_local = z_local / np.linalg.norm(z_local)
                
    #             curr_R[1] = y_local
    #             curr_R[2] = z_local

    #         # Plot previous state if requested
    #         if show_previous:
    #             # Plot previous state (dotted)
    #             self.ax.plot([prev_coords[0][0], prev_coords[1][0]],
    #                     [prev_coords[0][1], prev_coords[1][1]],
    #                     [prev_coords[0][2], prev_coords[1][2]],
    #                     'g--', lw=1.5, alpha=0.7)
                
    #             # Show nodes in previous state
    #             self.ax.scatter(prev_coords[0][0], prev_coords[0][1], prev_coords[0][2], 
    #                         color='g', s=25, alpha=0.7)
    #             self.ax.scatter(prev_coords[1][0], prev_coords[1][1], prev_coords[1][2], 
    #                         color='g', s=25, alpha=0.7)
                
    #             # Show local axes at previous state if requested
    #             if show_local_axes:
    #                 midpoint = (prev_coords[0] + prev_coords[1]) / 2
    #                 self.draw_local_axes(midpoint, prev_R, 
    #                                 scale=self.local_axes_scale*prev_L,
    #                                 linewidth=1, labels=False)
            
    #         # Plot current state (solid)
    #         self.ax.plot([curr_coords[0][0], curr_coords[1][0]],
    #                 [curr_coords[0][1], curr_coords[1][1]],
    #                 [curr_coords[0][2], curr_coords[1][2]],
    #                 'b-', lw=2)
            
    #         # Show nodes in current state
    #         self.ax.scatter(curr_coords[0][0], curr_coords[0][1], curr_coords[0][2], 
    #                     color='b', s=30)
    #         self.ax.scatter(curr_coords[1][0], curr_coords[1][1], curr_coords[1][2], 
    #                     color='b', s=30)
            
    #         # Show local axes for current state
    #         if show_local_axes:
    #             midpoint = (curr_coords[0] + curr_coords[1]) / 2
    #             self.draw_local_axes(midpoint, curr_R, 
    #                             scale=self.local_axes_scale*curr_L,
    #                             linewidth=2)
            
    #         # Add node labels if requested
    #         if node_labels:
    #             self.ax.text(curr_coords[0][0], curr_coords[0][1], curr_coords[0][2], 
    #                     f" {element.nodes[0].id}", fontsize=10)
    #             self.ax.text(curr_coords[1][0], curr_coords[1][1], curr_coords[1][2], 
    #                     f" {element.nodes[1].id}", fontsize=10)
        
    #     return self.fig, self.ax

    # NOTE SHAPE FUNCTION PLOTTING (INITIAL -> CURRENT)
    def plot_deformed_shape(self, scale=1.0, show_undeformed=False, npoints=20, show_local_axes=False): 
        """Visualize the model with properly interpolated deformations using shape functions."""
        
        if self.model is None or self.results is None:
            raise ValueError("Both model and results must be provided")
        
        self.initialize_plot()
        
        if show_undeformed:
            self.plot_undeformed_model(nodes=True, dashed=True, show_local_axes=False, node_labels=False)
        
        # Plot elements with shape function interpolation
        for elem_id, element in self.model.elements.items():
            # Get initial state
            init_coords = element.initial_state['coords']
            init_R = element.initial_state['R']
            L = element.initial_state['L']
            
            # Extract local DOFs from global displacements
            local_dofs = self.results.extract_local_dofs(element, init_R)
            
            # Create points along element for shape function evaluation
            xi_values = np.linspace(0, 1, npoints)
            deformed_points = []
            
            for xi in xi_values:
                # Evaluate shape functions
                u_xl, v_yl, w_zl, rx_xl, phi_xl = self.results.shape_thin_walled_beam(xi, L, local_dofs)
                
                # Local displacement vector
                local_disp = np.array([u_xl, v_yl, w_zl])
                
                # Convert to global coordinates
                global_disp = init_R.T @ local_disp
                
                # Initial position along element (linear interpolation)
                x0 = init_coords[0] * (1-xi) + init_coords[1] * xi
                
                # Apply scaled displacement
                deformed_point = x0 + scale * global_disp
                deformed_points.append(deformed_point)
            
            # Convert to array for plotting
            deformed_points = np.array(deformed_points)
            
            # Plot the deformed shape
            self.ax.plot(deformed_points[:, 0], deformed_points[:, 1], deformed_points[:, 2], 
                    'b-', lw=2)
            
            # Plot nodes
            self.ax.scatter(deformed_points[0, 0], deformed_points[0, 1], deformed_points[0, 2], 
                        color='b', s=30)
            self.ax.scatter(deformed_points[-1, 0], deformed_points[-1, 1], deformed_points[-1, 2], 
                        color='b', s=30)
        
        return self.fig, self.ax

    # NOTE SHAPE FUNCTION PLOTTING (PREVIOUS -> CURRENT)
    # def plot_deformed_shape(self, scale=1.0, show_undeformed=False, show_previous=False, 
    #                         show_local_axes=False, node_labels=True, npoints=20):
    #     """Visualize the model with deformation between previous and current states using shape functions."""

    #     if self.model is None:
    #         raise ValueError("No model provided to visualiser")
        
    #     self.initialize_plot()
        
    #     if show_undeformed:
    #         self.plot_undeformed_model(nodes=True, dashed=True, show_local_axes=False, node_labels=False)
        
    #     # Plot elements
    #     for elem_id, element in self.model.elements.items():
            
    #         # Initial state 
    #         init_coords = element.initial_state['coords']
    #         init_R = element.initial_state['R']
    #         init_L = element.initial_state['L']

    #         # Current step
    #         prev_coords = element.previous_state['coords']
    #         prev_R = element.previous_state['R']
    #         prev_L = element.previous_state['L']
            
    #         curr_coords = element.current_state['coords']
    #         curr_R = element.current_state['R'] 
    #         curr_L = element.current_state['L']

    #         # Scale deformations 
    #         prev_coords = [
    #             init_coords[i] + scale * (prev_coords[i] - init_coords[i]) 
    #             for i in range(2)
    #         ]
            
    #         curr_coords = [
    #             init_coords[i] + scale * (curr_coords[i] - init_coords[i])
    #             for i in range(2)
    #         ]
            
    #         # For previous state (if we scale displacements, we also need to scale local axes)
    #         prev_L = np.linalg.norm(prev_coords[1] - prev_coords[0])
    #         prev_R = prev_R.copy()
    #         if prev_L > 1e-10:
    #             # Update x-axis (tangent) to match scaled direction
    #             prev_R[0] = (prev_coords[1] - prev_coords[0]) / prev_L
                
    #             # Ensure y and z axes remain orthogonal to new x-axis
    #             y_temp = prev_R[1]
    #             y_local = y_temp - np.dot(y_temp, prev_R[0]) * prev_R[0]
    #             y_local = y_local / np.linalg.norm(y_local)
                
    #             z_local = np.cross(prev_R[0], y_local)
    #             z_local = z_local / np.linalg.norm(z_local)
                
    #             prev_R[1] = y_local
    #             prev_R[2] = z_local
            
    #         # For current state (if we scale displacements, we also need to scale local axes)
    #         curr_L = np.linalg.norm(curr_coords[1] - curr_coords[0])
    #         curr_R = curr_R.copy()
    #         if curr_L > 1e-10:
    #             # Update x-axis (tangent) to match scaled direction
    #             curr_R[0] = (curr_coords[1] - curr_coords[0]) / curr_L
                
    #             # Ensure y and z axes remain orthogonal to new x-axis
    #             y_temp = curr_R[1]
    #             y_local = y_temp - np.dot(y_temp, curr_R[0]) * curr_R[0]
    #             y_local = y_local / np.linalg.norm(y_local)
                
    #             z_local = np.cross(curr_R[0], y_local)
    #             z_local = z_local / np.linalg.norm(z_local)
                
    #             curr_R[1] = y_local
    #             curr_R[2] = z_local

    #         # Plot previous state if requested
    #         if show_previous:
    #             # Get local DOFs for previous state
    #             prev_disps = [prev_coords[i] - init_coords[i] for i in range(2)]
    #             prev_local_dofs = self.results.extract_local_dofs(element, prev_R)
                
    #             # Create points along element for shape function evaluation
    #             xi_values = np.linspace(0, 1, npoints)
    #             prev_points = []
                
    #             for xi in xi_values:
    #                 # Evaluate shape functions for previous state
    #                 u_xl, v_yl, w_zl, rx_xl, phi_xl = self.results.shape_thin_walled_beam(xi, prev_L, prev_local_dofs)
                    
    #                 # Local displacement vector
    #                 local_disp = np.array([u_xl, v_yl, w_zl])
                    
    #                 # Convert to global coordinates (using prev_R since we're in scaled previous state)
    #                 global_disp = prev_R.T @ local_disp
                    
    #                 # Initial position along element (linear interpolation)
    #                 x0 = init_coords[0] * (1-xi) + init_coords[1] * xi
                    
    #                 # Apply displacement directly (already scaled earlier)
    #                 point = x0 + global_disp
    #                 prev_points.append(point)
                
    #             # Convert to array for plotting
    #             prev_points = np.array(prev_points)
                
    #             # Plot previous state (dotted)
    #             self.ax.plot(prev_points[:,0], prev_points[:,1], prev_points[:,2],
    #                     'g--', lw=1.5, alpha=0.7)
                
    #             # Show nodes in previous state
    #             self.ax.scatter(prev_coords[0][0], prev_coords[0][1], prev_coords[0][2], 
    #                         color='g', s=25, alpha=0.7)
    #             self.ax.scatter(prev_coords[1][0], prev_coords[1][1], prev_coords[1][2], 
    #                         color='g', s=25, alpha=0.7)
                
    #             # Show local axes at previous state if requested
    #             if show_local_axes:
    #                 midpoint = (prev_coords[0] + prev_coords[1]) / 2
    #                 self.draw_local_axes(midpoint, prev_R, 
    #                                 scale=self.local_axes_scale*prev_L,
    #                                 linewidth=1, labels=False)
            
    #         # Plot current state with shape function interpolation
    #         # Get local DOFs for current state
    #         curr_disps = [curr_coords[i] - init_coords[i] for i in range(2)]
    #         curr_local_dofs = self.results.extract_local_dofs(element, curr_R)
            
    #         # Create points along element for shape function evaluation
    #         xi_values = np.linspace(0, 1, npoints)
    #         curr_points = []
            
    #         for xi in xi_values:
    #             # Evaluate shape functions for current state
    #             u_xl, v_yl, w_zl, rx_xl, phi_xl = self.results.shape_thin_walled_beam(xi, curr_L, curr_local_dofs)
                
    #             # Local displacement vector
    #             local_disp = np.array([u_xl, v_yl, w_zl])
                
    #             # Convert to global coordinates (using curr_R since we're in scaled current state)
    #             global_disp = curr_R.T @ local_disp
                
    #             # Initial position along element (linear interpolation)
    #             x0 = init_coords[0] * (1-xi) + init_coords[1] * xi
                
    #             # Apply displacement directly (already scaled earlier)
    #             point = x0 + global_disp
    #             curr_points.append(point)
            
    #         # Convert to array for plotting
    #         curr_points = np.array(curr_points)
            
    #         # Plot current state (solid)
    #         self.ax.plot(curr_points[:,0], curr_points[:,1], curr_points[:,2],
    #                 'b-', lw=2)
            
    #         # Show nodes in current state
    #         self.ax.scatter(curr_coords[0][0], curr_coords[0][1], curr_coords[0][2], 
    #                     color='b', s=30)
    #         self.ax.scatter(curr_coords[1][0], curr_coords[1][1], curr_coords[1][2], 
    #                     color='b', s=30)
            
    #         # Show local axes for current state
    #         if show_local_axes:
    #             midpoint = (curr_coords[0] + curr_coords[1]) / 2
    #             self.draw_local_axes(midpoint, curr_R, 
    #                             scale=self.local_axes_scale*curr_L,
    #                             linewidth=2)
            
    #         # Add node labels if requested
    #         if node_labels:
    #             self.ax.text(curr_coords[0][0], curr_coords[0][1], curr_coords[0][2], 
    #                     f" {element.nodes[0].id}", fontsize=10)
    #             self.ax.text(curr_coords[1][0], curr_coords[1][1], curr_coords[1][2], 
    #                     f" {element.nodes[1].id}", fontsize=10)
        
    #     return self.fig, self.ax

    # NOTE SHAPE FUNCTION PLOTTING (PREVIOUS -> CURRENT)
    # def plot_deformed_shape(self, scale=1.0, show_undeformed=True, show_previous=True, show_current=False,
    #                         show_local_axes=False, node_labels=True, npoints=20):
    #     """Visualize the model with deformation between previous and current states using shape functions."""

    #     if self.model is None:
    #         raise ValueError("No model provided to visualiser")
        
    #     self.initialize_plot()
        
    #     # if show_undeformed:
    #     #     self.plot_undeformed_model(nodes=True, dashed=True, show_local_axes=False, node_labels=False)
        
    #     # Plot elements
    #     for elem_id, element in self.model.elements.items():
            
    #         # Initial state 
    #         init_coords = element.initial_state['coords']
    #         # init_R = element.initial_state['R']
    #         # init_L = element.initial_state['L']

    #         # Current step
    #         prev_coords = element.previous_state['coords']
    #         prev_R = element.previous_state['R']
    #         prev_L = element.previous_state['L']
            
    #         curr_coords = element.current_state['coords']
    #         curr_R = element.current_state['R'] 
    #         curr_L = element.current_state['L']

    #         # Scale deformations 
    #         prev_coords = [
    #             init_coords[i] + scale * (prev_coords[i] - init_coords[i]) 
    #             for i in range(2)
    #         ]
            
    #         curr_coords = [
    #             init_coords[i] + scale * (curr_coords[i] - init_coords[i])
    #             for i in range(2)
    #         ]
            
    #         # Update rotation matrices based on scaled deformations
    #         prev_L, prev_R = self._update_rotation_matrix(prev_coords, prev_R)
    #         curr_L, curr_R = self._update_rotation_matrix(curr_coords, curr_R)

    #         # Plot previous state if requested
    #         if show_previous:
    #             # Get interpolated points using shape functions for previous state
    #             prev_local_dofs = self.results.extract_local_dofs(element, prev_R)
    #             prev_points = self._get_interpolated_points(xi_values=np.linspace(0, 1, npoints),
    #                                                     init_coords=init_coords,
    #                                                     rotation_matrix=prev_R,
    #                                                     element_length=prev_L,
    #                                                     local_dofs=prev_local_dofs)
                
    #             # Plot previous state (dotted)
    #             self.ax.plot(prev_points[:,0], prev_points[:,1], prev_points[:,2],
    #                     'g--', lw=1.5, alpha=0.7)
                
    #             # Show nodes in previous state
    #             self.ax.scatter(prev_coords[0][0], prev_coords[0][1], prev_coords[0][2], 
    #                         color='g', s=25, alpha=0.7)
    #             self.ax.scatter(prev_coords[1][0], prev_coords[1][1], prev_coords[1][2], 
    #                         color='g', s=25, alpha=0.7)
                
    #             # Show local axes at previous state if requested
    #             if show_local_axes:
    #                 midpoint = (prev_coords[0] + prev_coords[1]) / 2
    #                 self.draw_local_axes(midpoint, prev_R, 
    #                                 scale=self.local_axes_scale*prev_L,
    #                                 linewidth=1, labels=False)

    #         # Plot current state with shape function interpolation
    #         # Get interpolated points using shape functions for current state
    #         if show_current:
    #             curr_local_dofs = self.results.extract_local_dofs(element, curr_R)
    #             curr_points = self._get_interpolated_points(xi_values=np.linspace(0, 1, npoints),
    #                                                     init_coords=init_coords,
    #                                                     rotation_matrix=curr_R,
    #                                                     element_length=curr_L,
    #                                                     local_dofs=curr_local_dofs)
                
    #             # Plot current state (solid)
    #             self.ax.plot(curr_points[:,0], curr_points[:,1], curr_points[:,2],
    #                     'b-', lw=2)
                
    #             # Show nodes in current state
    #             self.ax.scatter(curr_coords[0][0], curr_coords[0][1], curr_coords[0][2], 
    #                         color='b', s=30)
    #             self.ax.scatter(curr_coords[1][0], curr_coords[1][1], curr_coords[1][2], 
    #                         color='b', s=30)
                
    #             # Show local axes for current state at both ends and middle to see if they're flipping
    #             if show_local_axes:                    
    #                 midpoint = (curr_coords[0] + curr_coords[1]) / 2
    #                 self.draw_local_axes(midpoint, curr_R, 
    #                                 scale=self.local_axes_scale*curr_L,
    #                                 linewidth=2)
            
    #         # Add node labels if requested
    #         if node_labels:
    #             self.ax.text(curr_coords[0][0], curr_coords[0][1], curr_coords[0][2], 
    #                     f" {element.nodes[0].id}", fontsize=10)
    #             self.ax.text(curr_coords[1][0], curr_coords[1][1], curr_coords[1][2], 
    #                     f" {element.nodes[1].id}", fontsize=10)
        
    #     return self.fig, self.ax

    # def plot_deformed_shape(self, scale=1.0, show_undeformed=True, 
    #                         show_local_axes=False, node_labels=True, npoints=20):
    #     """Visualize the model with undeformed geometry and current state using shape functions."""

    #     if self.model is None:
    #         raise ValueError("No model provided to visualiser")
        
    #     self.initialize_plot()
        
    #     # Plot elements
    #     for elem_id, element in self.model.elements.items():
            
    #         # Initial state 
    #         init_coords = element.initial_state['coords']
    #         init_R = element.initial_state['R']
    #         init_L = element.initial_state['L']

    #         # Current state
    #         curr_coords = element.current_state['coords']
    #         curr_R = element.current_state['R'] 
    #         curr_L = element.current_state['L']

    #         # Scale deformations 
    #         curr_coords = [
    #             init_coords[i] + scale * (curr_coords[i] - init_coords[i])
    #             for i in range(2)
    #         ]
            
    #         # Update rotation matrix based on scaled deformations
    #         curr_L, curr_R = self._update_rotation_matrix(curr_coords, curr_R)

    #         # Plot undeformed model if requested
    #         if show_undeformed:
    #             # Plot undeformed element (dashed)
    #             self.ax.plot([init_coords[0][0], init_coords[1][0]],
    #                     [init_coords[0][1], init_coords[1][1]],
    #                     [init_coords[0][2], init_coords[1][2]],
    #                     'k--', lw=1.0, alpha=0.7)
                
    #             # Show nodes in undeformed state
    #             self.ax.scatter(init_coords[0][0], init_coords[0][1], init_coords[0][2], 
    #                         color='k', s=25, alpha=0.7)
    #             self.ax.scatter(init_coords[1][0], init_coords[1][1], init_coords[1][2], 
    #                         color='k', s=25, alpha=0.7)
                
    #             # Show local axes at undeformed state if requested
    #             if show_local_axes:
    #                 midpoint = (init_coords[0] + init_coords[1]) / 2
    #                 self.draw_local_axes(midpoint, init_R, 
    #                                 scale=self.local_axes_scale*init_L,
    #                                 linewidth=1, labels=False)

    #         # Plot current state with shape function interpolation
    #         # Get interpolated points using shape functions for current state
    #         curr_local_dofs = self.results.extract_local_dofs(element, element.initial_state['R'])
    #         curr_points = self._get_interpolated_points(xi_values=np.linspace(0, 1, npoints),
    #                                                 init_coords=init_coords,
    #                                                 rotation_matrix=curr_R,
    #                                                 element_length=curr_L,
    #                                                 local_dofs=curr_local_dofs)
            
    #         # Plot current state (solid)
    #         self.ax.plot(curr_points[:,0], curr_points[:,1], curr_points[:,2],
    #                 'b-', lw=2)
            
    #         # Show nodes in current state
    #         self.ax.scatter(curr_coords[0][0], curr_coords[0][1], curr_coords[0][2], 
    #                     color='b', s=30)
    #         self.ax.scatter(curr_coords[1][0], curr_coords[1][1], curr_coords[1][2], 
    #                     color='b', s=30)
            
    #         # Show local axes for current state
    #         if show_local_axes:                    
    #             midpoint = (curr_coords[0] + curr_coords[1]) / 2
    #             self.draw_local_axes(midpoint, curr_R, 
    #                             scale=self.local_axes_scale*curr_L,
    #                             linewidth=2)
            
    #         # Add node labels if requested
    #         if node_labels:
    #             self.ax.text(curr_coords[0][0], curr_coords[0][1], curr_coords[0][2], 
    #                     f" {element.nodes[0].id}", fontsize=10)
    #             self.ax.text(curr_coords[1][0], curr_coords[1][1], curr_coords[1][2], 
    #                     f" {element.nodes[1].id}", fontsize=10)
        
    #     return self.fig, self.ax

    # def _update_rotation_matrix(self, coords, R):
    #     """Update rotation matrix based on scaled deformations."""
    #     R = R.copy()
    #     L = np.linalg.norm(coords[1] - coords[0])
        
    #     if L > 1e-10:
    #         # Update x-axis (tangent) to match scaled direction
    #         R[0] = (coords[1] - coords[0]) / L
            
    #         # Ensure y and z axes remain orthogonal to new x-axis
    #         y_temp = R[1]
    #         y_local = y_temp - np.dot(y_temp, R[0]) * R[0]
    #         y_local = y_local / np.linalg.norm(y_local)
            
    #         z_local = np.cross(R[0], y_local)
    #         z_local = z_local / np.linalg.norm(z_local)
            
    #         R[1] = y_local
    #         R[2] = z_local
        
    #     return L, R

    # def _get_interpolated_points(self, xi_values, init_coords, rotation_matrix, element_length, local_dofs):
    #     """Get interpolated points along the element using shape functions."""
    #     interpolated_points = []
        
    #     for xi in xi_values:
    #         # Evaluate shape functions
    #         u_xl, v_yl, w_zl, rx_xl, phi_xl = self.results.shape_thin_walled_beam(xi, element_length, local_dofs)
            
    #         # Local displacement vector
    #         local_disp = np.array([u_xl, v_yl, w_zl])
            
    #         # Convert to global coordinates
    #         global_disp = rotation_matrix.T @ local_disp
            
    #         # Initial position along element (linear interpolation)
    #         x0 = init_coords[0] * (1-xi) + init_coords[1] * xi
            
    #         # Apply displacement
    #         point = x0 + global_disp
    #         interpolated_points.append(point)
        
    #     return np.array(interpolated_points)

    def plot_deformed_shape(self, scale=1.0, show_undeformed=False, npoints=20, show_local_axes=False): 
        """Visualize the model with properly interpolated deformations using shape functions."""
        
        if self.model is None or self.results is None:
            raise ValueError("Both model and results must be provided")
        
        self.initialize_plot()
        
        if show_undeformed:
            self.plot_undeformed_model(nodes=True, dashed=True, show_local_axes=False, node_labels=False)
        
        # Plot elements with shape function interpolation
        for elem_id, element in self.model.elements.items():
            # Get initial state
            init_coords = element.initial_state['coords']
            init_R = element.initial_state['R']
            L = element.initial_state['L']
            
            # Get current state for local axes visualization
            curr_coords = element.current_state['coords']
            curr_R = element.current_state['R']
            curr_L = element.current_state['L']
            
            # Apply scaling to current state coordinates
            curr_coords = [
                init_coords[i] + scale * (curr_coords[i] - init_coords[i])
                for i in range(2)
            ]
            
            # Extract local DOFs from global displacements
            local_dofs = self.results.extract_local_dofs(element, init_R)
            
            # Create points along element for shape function evaluation
            xi_values = np.linspace(0, 1, npoints)
            deformed_points = []
            
            for xi in xi_values:
                # Evaluate shape functions
                u_xl, v_yl, w_zl, rx_xl, phi_xl = self.results.shape_thin_walled_beam(xi, L, local_dofs)
                
                # Local displacement vector
                local_disp = np.array([u_xl, v_yl, w_zl])
                
                # Convert to global coordinates
                global_disp = init_R.T @ local_disp
                
                # Initial position along element (linear interpolation)
                x0 = init_coords[0] * (1-xi) + init_coords[1] * xi
                
                # Apply scaled displacement
                deformed_point = x0 + scale * global_disp
                deformed_points.append(deformed_point)
            
            # Convert to array for plotting
            deformed_points = np.array(deformed_points)
            
            # Plot the deformed shape
            self.ax.plot(deformed_points[:, 0], deformed_points[:, 1], deformed_points[:, 2], 
                    'b-', lw=2)
            
            # Plot nodes
            self.ax.scatter(deformed_points[0, 0], deformed_points[0, 1], deformed_points[0, 2], 
                        color='b', s=30)
            self.ax.scatter(deformed_points[-1, 0], deformed_points[-1, 1], deformed_points[-1, 2], 
                        color='b', s=30)
                
            # For current state (if we scale displacements, we also need to scale local axes)
            curr_L = np.linalg.norm(curr_coords[1] - curr_coords[0])
            curr_R = curr_R.copy()
            if curr_L > 1e-10:
                # Update x-axis (tangent) to match scaled direction
                curr_R[0] = (curr_coords[1] - curr_coords[0]) / curr_L
                
                # Ensure y and z axes remain orthogonal to new x-axis
                y_temp = curr_R[1]
                y_local = y_temp - np.dot(y_temp, curr_R[0]) * curr_R[0]
                y_local = y_local / np.linalg.norm(y_local)
                
                z_local = np.cross(curr_R[0], y_local)
                z_local = z_local / np.linalg.norm(z_local)
                
                curr_R[1] = y_local
                curr_R[2] = z_local

            # Show local axes at midpoint of current state if requested
            if show_local_axes:
                midpoint = (curr_coords[0] + curr_coords[1]) / 2
                self.draw_local_axes(midpoint, curr_R, 
                                scale=self.local_axes_scale*curr_L,
                                linewidth=2)
        
        return self.fig, self.ax

    def show(self):
        """Show the current plot."""
        if self.fig is not None:
            plt.tight_layout()
            # plt.axis('off') 
            plt.axis('equal') 
            plt.show()
            self.fig, self.ax = None, None 
        else:
            print("No plot has been created yet.")
    
    def save(self, filename, dpi=300):
        """Save the current plot to a file."""
        if self.fig is not None:
            plt.tight_layout()
            plt.savefig(filename, dpi=dpi)
            print(f"Plot saved to {filename}")
        else:
            print("No plot has been created yet.")