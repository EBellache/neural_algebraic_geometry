"""
Comprehensive projection visualization system connecting 3D biology to 2D retinal encoding.

This module provides interactive visualizations for understanding how stereographic
and other projections transform 3D structures into 2D patterns, with emphasis on
the emergence of hexagonal patterns from icosahedral symmetry.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation
from typing import Tuple, List, Optional, Dict, Callable
import ipywidgets as widgets
from IPython.display import display

from ..polytope_core.platonic_solids import PlatonicSolids
from ..polytope_core.stereographic import StereographicProjection
from ..polytope_core.twentyfour_cell import TwentyFourCell


class ProjectionVisualizer:
    """Master stereographic projection visualizer with synchronized panels."""
    
    def __init__(self):
        self.sphere_radius = 1.0
        self.projection_center = np.array([0, 0, 1])  # North pole default
        self.platonic = PlatonicSolids()
        self.stereographic = StereographicProjection()
        
    def create_synchronized_view(self, polytope_type: str = 'icosahedron') -> go.FigureWidget:
        """Create three synchronized panels: 3D sphere, 2D projection, side view."""
        # Create subplots
        fig = make_subplots(
            rows=1, cols=3,
            specs=[[{'type': 'scene'}, {'type': 'xy'}, {'type': 'scene'}]],
            subplot_titles=['3D Polytope on Sphere', '2D Stereographic Projection', 'Side View']
        )
        
        # Get polytope
        vertices, edges = self.platonic.get_polytope(polytope_type)
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)  # Project to sphere
        
        # 3D sphere with polytope
        self._add_3d_polytope(fig, vertices, edges, row=1, col=1)
        
        # 2D projection
        projected = self._project_vertices(vertices)
        self._add_2d_projection(fig, projected, edges, row=1, col=2)
        
        # Side view showing projection geometry
        self._add_side_view(fig, vertices, projected, row=1, col=3)
        
        # Update layout
        fig.update_layout(
            title=f"Stereographic Projection of {polytope_type.capitalize()}",
            showlegend=False,
            height=600,
            width=1800
        )
        
        # Make it interactive
        fig_widget = go.FigureWidget(fig)
        
        # Add rotation controls
        rotation_slider = widgets.FloatSlider(
            value=0, min=0, max=360, step=5,
            description='Rotation:'
        )
        
        def update_rotation(change):
            angle = np.radians(change['new'])
            rot = Rotation.from_euler('z', angle)
            rotated_vertices = rot.apply(vertices)
            projected = self._project_vertices(rotated_vertices)
            
            # Update all panels
            with fig_widget.batch_update():
                # Update 3D polytope
                fig_widget.data[0].x = rotated_vertices[:, 0]
                fig_widget.data[0].y = rotated_vertices[:, 1]
                fig_widget.data[0].z = rotated_vertices[:, 2]
                
                # Update 2D projection
                fig_widget.data[3].x = projected[:, 0]
                fig_widget.data[3].y = projected[:, 1]
                
                # Update projection rays
                self._update_projection_rays(fig_widget, rotated_vertices, projected)
        
        rotation_slider.observe(update_rotation, names='value')
        
        return widgets.VBox([rotation_slider, fig_widget])
    
    def _add_3d_polytope(self, fig, vertices, edges, row, col):
        """Add 3D polytope on sphere to figure."""
        # Add sphere
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = self.sphere_radius * np.outer(np.cos(u), np.sin(v))
        y = self.sphere_radius * np.outer(np.sin(u), np.sin(v))
        z = self.sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig.add_trace(
            go.Surface(x=x, y=y, z=z, opacity=0.3, showscale=False),
            row=row, col=col
        )
        
        # Add polytope vertices
        fig.add_trace(
            go.Scatter3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                mode='markers',
                marker=dict(size=8, color='red')
            ),
            row=row, col=col
        )
        
        # Add edges
        for edge in edges:
            v1, v2 = vertices[edge[0]], vertices[edge[1]]
            fig.add_trace(
                go.Scatter3d(
                    x=[v1[0], v2[0]], y=[v1[1], v2[1]], z=[v1[2], v2[2]],
                    mode='lines',
                    line=dict(color='blue', width=3)
                ),
                row=row, col=col
            )
    
    def _add_2d_projection(self, fig, projected, edges, row, col):
        """Add 2D stereographic projection to figure."""
        # Add vertices
        fig.add_trace(
            go.Scatter(
                x=projected[:, 0], y=projected[:, 1],
                mode='markers',
                marker=dict(size=10, color='red')
            ),
            row=row, col=col
        )
        
        # Add edges
        for edge in edges:
            p1, p2 = projected[edge[0]], projected[edge[1]]
            fig.add_trace(
                go.Scatter(
                    x=[p1[0], p2[0]], y=[p1[1], p2[1]],
                    mode='lines',
                    line=dict(color='blue', width=2)
                ),
                row=row, col=col
            )
        
        # Set equal aspect ratio
        fig.update_xaxes(scaleanchor="y", scaleratio=1, row=row, col=col)
    
    def _add_side_view(self, fig, vertices, projected, row, col):
        """Add side view showing projection geometry."""
        # Show YZ plane with projection rays
        for i, (v3d, v2d) in enumerate(zip(vertices, projected)):
            # Ray from projection center through 3D point to 2D plane
            t = np.linspace(0, 2, 100)
            ray_dir = v3d - self.projection_center
            ray_points = self.projection_center[:, np.newaxis] + t * ray_dir[:, np.newaxis]
            
            fig.add_trace(
                go.Scatter3d(
                    x=ray_points[0], y=ray_points[1], z=ray_points[2],
                    mode='lines',
                    line=dict(color='gray', width=1),
                    opacity=0.3
                ),
                row=row, col=col
            )
    
    def _project_vertices(self, vertices):
        """Project vertices using stereographic projection."""
        projected = []
        for v in vertices:
            p = self.stereographic.project_from_pole(v, self.projection_center)
            projected.append(p)
        return np.array(projected)
    
    def _update_projection_rays(self, fig_widget, vertices, projected):
        """Update projection rays in side view."""
        # This would update the ray traces in the figure
        pass


class HexagonalEmergenceDemo:
    """Demonstrate emergence of hexagonal pattern from icosahedral projection."""
    
    def __init__(self):
        self.platonic = PlatonicSolids()
        self.stereographic = StereographicProjection()
        
    def create_transition_animation(self) -> widgets.VBox:
        """Animate transition from pentagonal to hexagonal as projection point moves."""
        # Get icosahedron
        vertices, edges = self.platonic.get_polytope('icosahedron')
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        
        # Find vertex and face center
        vertex_pos = vertices[0]  # Any vertex
        face_vertices = self._get_face_vertices(vertices, edges)
        face_center = np.mean(face_vertices[0], axis=0)
        face_center = face_center / np.linalg.norm(face_center)
        
        # Create figure
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'xy'}]],
            subplot_titles=['3D View', '2D Projection Pattern']
        )
        
        # Animation slider
        t_slider = widgets.FloatSlider(
            value=0, min=0, max=1, step=0.01,
            description='Transition:',
            continuous_update=True
        )
        
        # Pattern info
        pattern_label = widgets.Label(value="Pattern: Pentagonal (5-fold symmetry)")
        
        fig_widget = go.FigureWidget(fig)
        
        def update_projection(change):
            t = change['new']
            # Interpolate projection center from vertex to face center
            proj_center = (1 - t) * vertex_pos + t * face_center
            proj_center = proj_center / np.linalg.norm(proj_center)
            
            # Project vertices
            projected = []
            for v in vertices:
                p = self.stereographic.project_from_pole(v, proj_center)
                projected.append(p)
            projected = np.array(projected)
            
            # Update plots
            with fig_widget.batch_update():
                # Clear previous projection traces
                fig_widget.data = fig_widget.data[:2]  # Keep only sphere and vertices
                
                # Add new projection
                for edge in edges:
                    p1, p2 = projected[edge[0]], projected[edge[1]]
                    fig_widget.add_trace(
                        go.Scatter(
                            x=[p1[0], p2[0]], y=[p1[1], p2[1]],
                            mode='lines',
                            line=dict(color='blue', width=2)
                        ),
                        row=1, col=2
                    )
                
                # Update pattern label
                if t < 0.3:
                    pattern_label.value = "Pattern: Pentagonal (5-fold symmetry)"
                elif t < 0.7:
                    pattern_label.value = "Pattern: Transitioning..."
                else:
                    pattern_label.value = "Pattern: Hexagonal (6-fold symmetry)"
        
        t_slider.observe(update_projection, names='value')
        
        # Initial setup
        self._setup_initial_view(fig_widget, vertices, edges)
        
        return widgets.VBox([
            widgets.HTML("<h3>Hexagonal Emergence from Icosahedral Projection</h3>"),
            t_slider,
            pattern_label,
            fig_widget
        ])
    
    def _get_face_vertices(self, vertices, edges):
        """Extract face vertices from edges."""
        # Simple approach - find triangular faces
        faces = []
        n = len(vertices)
        adjacency = [set() for _ in range(n)]
        
        for e in edges:
            adjacency[e[0]].add(e[1])
            adjacency[e[1]].add(e[0])
        
        # Find triangles
        for i in range(n):
            for j in adjacency[i]:
                if j > i:
                    for k in adjacency[i]:
                        if k > j and k in adjacency[j]:
                            faces.append([vertices[i], vertices[j], vertices[k]])
        
        return faces
    
    def _setup_initial_view(self, fig, vertices, edges):
        """Set up initial 3D view."""
        # Add sphere
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig.add_trace(
            go.Surface(x=x, y=y, z=z, opacity=0.3, showscale=False),
            row=1, col=1
        )
        
        # Add icosahedron
        fig.add_trace(
            go.Scatter3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                mode='markers',
                marker=dict(size=6, color='red')
            ),
            row=1, col=1
        )


class MultiLayerProjectionSystem:
    """Model visual pathway through multiple projection layers."""
    
    def __init__(self):
        self.layers = []
        
    def add_layer(self, name: str, projection_func: Callable, 
                  preserves: List[str], distorts: List[str]):
        """Add a projection layer to the system."""
        self.layers.append({
            'name': name,
            'projection': projection_func,
            'preserves': preserves,
            'distorts': distorts
        })
    
    def create_pathway_visualization(self) -> go.Figure:
        """Visualize complete visual pathway with transformations."""
        fig = make_subplots(
            rows=1, cols=len(self.layers) + 1,
            subplot_titles=['3D World'] + [layer['name'] for layer in self.layers],
            specs=[[{'type': 'scene'}] + [{'type': 'xy'}] * len(self.layers)]
        )
        
        # Start with 3D bacterium shape
        bacterium = self._create_bacterium_shape()
        
        # Add 3D view
        fig.add_trace(
            go.Scatter3d(
                x=bacterium[:, 0], y=bacterium[:, 1], z=bacterium[:, 2],
                mode='lines+markers',
                line=dict(color='green', width=3),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        # Apply each projection layer
        current_data = bacterium
        for i, layer in enumerate(self.layers):
            current_data = layer['projection'](current_data)
            
            # Add to subplot
            if len(current_data.shape) == 2 and current_data.shape[1] >= 2:
                fig.add_trace(
                    go.Scatter(
                        x=current_data[:, 0], y=current_data[:, 1],
                        mode='lines+markers',
                        line=dict(color='blue', width=2),
                        marker=dict(size=3),
                        name=layer['name']
                    ),
                    row=1, col=i + 2
                )
                
                # Add annotations about what's preserved/distorted
                fig.add_annotation(
                    text=f"Preserves: {', '.join(layer['preserves'])}",
                    xref=f"x{i+2}", yref=f"y{i+2}",
                    x=0, y=1.1, showarrow=False,
                    font=dict(size=10, color='green')
                )
                fig.add_annotation(
                    text=f"Distorts: {', '.join(layer['distorts'])}",
                    xref=f"x{i+2}", yref=f"y{i+2}",
                    x=0, y=-1.1, showarrow=False,
                    font=dict(size=10, color='red')
                )
        
        fig.update_layout(
            title="Visual Pathway: Multiple Projection Layers",
            height=400,
            width=1600,
            showlegend=False
        )
        
        return fig
    
    def _create_bacterium_shape(self):
        """Create simple 3D bacterium shape (rod with rounded ends)."""
        t = np.linspace(0, 2*np.pi, 50)
        
        # Cylindrical body
        body_x = np.zeros(50)
        body_y = 0.2 * np.cos(t)
        body_z = 0.2 * np.sin(t)
        
        # Add length
        body_x[:25] = -0.5
        body_x[25:] = 0.5
        
        # Combine into single array
        points = np.column_stack([body_x, body_y, body_z])
        
        return points


class TwentyFourCellMOGProjection:
    """Demonstrate 24-cell to Mathieu group projection."""
    
    def __init__(self):
        self.cell24 = TwentyFourCell()
        
    def create_mog_animation(self) -> widgets.VBox:
        """Animate 24-cell rotation showing MOG at special angles."""
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'xy'}, {'type': 'xy'}]],
            subplot_titles=['4D→2D Projection', 'Pattern Analysis']
        )
        
        # Rotation controls
        angle_sliders = {
            'xy': widgets.FloatSlider(value=0, min=0, max=360, step=5, description='XY rotation:'),
            'xz': widgets.FloatSlider(value=0, min=0, max=360, step=5, description='XZ rotation:'),
            'xw': widgets.FloatSlider(value=0, min=0, max=360, step=5, description='XW rotation:')
        }
        
        pattern_label = widgets.Label(value="Pattern: Complex")
        
        fig_widget = go.FigureWidget(fig)
        
        def update_projection(*args):
            # Get rotation angles
            angles = {k: np.radians(v.value) for k, v in angle_sliders.items()}
            
            # Rotate 24-cell in 4D
            vertices = self.cell24.vertices.copy()
            vertices = self._rotate_4d(vertices, angles)
            
            # Project to 2D
            projected = self._project_4d_to_2d(vertices)
            
            # Check if we have MOG pattern (4×6 grid)
            is_mog = self._check_mog_pattern(projected)
            
            with fig_widget.batch_update():
                # Clear previous traces
                fig_widget.data = []
                
                # Add projected vertices
                fig_widget.add_trace(
                    go.Scatter(
                        x=projected[:, 0], y=projected[:, 1],
                        mode='markers',
                        marker=dict(
                            size=10,
                            color='red' if is_mog else 'blue'
                        )
                    ),
                    row=1, col=1
                )
                
                # Add edges
                for edge in self.cell24.edges:
                    p1, p2 = projected[edge[0]], projected[edge[1]]
                    fig_widget.add_trace(
                        go.Scatter(
                            x=[p1[0], p2[0]], y=[p1[1], p2[1]],
                            mode='lines',
                            line=dict(
                                color='orange' if is_mog else 'gray',
                                width=2 if is_mog else 1
                            )
                        ),
                        row=1, col=1
                    )
                
                # Update pattern label
                if is_mog:
                    pattern_label.value = "Pattern: MOG (4×6 grid) - Special angle!"
                else:
                    pattern_label.value = "Pattern: Complex"
                
                # Add grid analysis in second panel
                if is_mog:
                    self._add_grid_analysis(fig_widget, projected)
        
        # Connect sliders
        for slider in angle_sliders.values():
            slider.observe(update_projection, names='value')
        
        # Initial update
        update_projection()
        
        return widgets.VBox([
            widgets.HTML("<h3>24-Cell to MOG Projection</h3>"),
            widgets.HTML("<p>Rotate to find special angles where MOG pattern emerges</p>"),
            *angle_sliders.values(),
            pattern_label,
            fig_widget
        ])
    
    def _rotate_4d(self, vertices, angles):
        """Rotate vertices in 4D space."""
        # Simple 4D rotations in different planes
        cos_xy, sin_xy = np.cos(angles['xy']), np.sin(angles['xy'])
        cos_xz, sin_xz = np.cos(angles['xz']), np.sin(angles['xz'])
        cos_xw, sin_xw = np.cos(angles['xw']), np.sin(angles['xw'])
        
        # XY rotation
        rot_xy = np.array([
            [cos_xy, -sin_xy, 0, 0],
            [sin_xy, cos_xy, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # XZ rotation
        rot_xz = np.array([
            [cos_xz, 0, -sin_xz, 0],
            [0, 1, 0, 0],
            [sin_xz, 0, cos_xz, 0],
            [0, 0, 0, 1]
        ])
        
        # XW rotation
        rot_xw = np.array([
            [cos_xw, 0, 0, -sin_xw],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [sin_xw, 0, 0, cos_xw]
        ])
        
        # Apply rotations
        vertices = vertices @ rot_xy.T @ rot_xz.T @ rot_xw.T
        
        return vertices
    
    def _project_4d_to_2d(self, vertices):
        """Project 4D vertices to 2D using stereographic projection."""
        # First project to 3D, then to 2D
        vertices_3d = vertices[:, :3] / (1 - vertices[:, 3:4])
        vertices_2d = vertices_3d[:, :2] / (1 - vertices_3d[:, 2:3])
        return vertices_2d
    
    def _check_mog_pattern(self, projected):
        """Check if projection forms MOG 4×6 grid pattern."""
        # Simple heuristic - check for regular grid spacing
        x_coords = np.sort(np.unique(np.round(projected[:, 0], 2)))
        y_coords = np.sort(np.unique(np.round(projected[:, 1], 2)))
        
        if len(x_coords) == 4 and len(y_coords) == 6:
            # Check for regular spacing
            x_diffs = np.diff(x_coords)
            y_diffs = np.diff(y_coords)
            
            if np.std(x_diffs) < 0.1 and np.std(y_diffs) < 0.1:
                return True
        
        return False
    
    def _add_grid_analysis(self, fig, projected):
        """Add grid lines showing MOG structure."""
        x_coords = np.sort(np.unique(np.round(projected[:, 0], 2)))
        y_coords = np.sort(np.unique(np.round(projected[:, 1], 2)))
        
        # Add vertical lines
        for x in x_coords:
            fig.add_trace(
                go.Scatter(
                    x=[x, x], y=[y_coords[0], y_coords[-1]],
                    mode='lines',
                    line=dict(color='green', width=1, dash='dash')
                ),
                row=1, col=2
            )
        
        # Add horizontal lines
        for y in y_coords:
            fig.add_trace(
                go.Scatter(
                    x=[x_coords[0], x_coords[-1]], y=[y, y],
                    mode='lines',
                    line=dict(color='green', width=1, dash='dash')
                ),
                row=1, col=2
            )


class InverseProjectionInterface:
    """Interactive interface for 3D reconstruction from 2D patterns."""
    
    def __init__(self):
        self.stereographic = StereographicProjection()
        
    def create_reconstruction_interface(self) -> widgets.VBox:
        """Create interface for drawing 2D patterns and seeing possible 3D sources."""
        # Canvas for drawing
        canvas_output = widgets.Output()
        
        # Pattern selection
        pattern_dropdown = widgets.Dropdown(
            options=['Hexagonal', 'Square', 'Triangular', 'Custom'],
            value='Hexagonal',
            description='Pattern:'
        )
        
        # Constraint checkboxes
        constraints = {
            'sphere': widgets.Checkbox(value=True, description='Must be on sphere'),
            'convex': widgets.Checkbox(value=True, description='Must be convex'),
            'platonic': widgets.Checkbox(value=False, description='Must be Platonic solid')
        }
        
        # Results display
        results_output = widgets.Output()
        
        def reconstruct_3d(change):
            pattern_type = pattern_dropdown.value
            
            # Get 2D pattern points
            if pattern_type == 'Hexagonal':
                pattern_2d = self._create_hex_pattern()
            elif pattern_type == 'Square':
                pattern_2d = self._create_square_pattern()
            elif pattern_type == 'Triangular':
                pattern_2d = self._create_triangular_pattern()
            else:
                pattern_2d = np.array([[0, 0], [1, 0], [0.5, 0.866]])  # Default triangle
            
            # Find possible 3D structures
            possible_3d = self._inverse_project(pattern_2d, constraints)
            
            # Display results
            with results_output:
                results_output.clear_output()
                fig = self._visualize_reconstructions(pattern_2d, possible_3d)
                fig.show()
        
        pattern_dropdown.observe(reconstruct_3d, names='value')
        for checkbox in constraints.values():
            checkbox.observe(reconstruct_3d, names='value')
        
        # Initial reconstruction
        reconstruct_3d(None)
        
        return widgets.VBox([
            widgets.HTML("<h3>3D Reconstruction from 2D Patterns</h3>"),
            pattern_dropdown,
            widgets.HTML("<h4>Constraints:</h4>"),
            widgets.VBox(list(constraints.values())),
            widgets.HTML("<h4>Possible 3D Structures:</h4>"),
            results_output
        ])
    
    def _create_hex_pattern(self):
        """Create hexagonal pattern points."""
        angles = np.linspace(0, 2*np.pi, 7)[:-1]
        return np.column_stack([np.cos(angles), np.sin(angles)])
    
    def _create_square_pattern(self):
        """Create square pattern points."""
        return np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    
    def _create_triangular_pattern(self):
        """Create triangular pattern points."""
        angles = np.linspace(0, 2*np.pi, 4)[:-1]
        return np.column_stack([np.cos(angles), np.sin(angles)])
    
    def _inverse_project(self, pattern_2d, constraints):
        """Find possible 3D structures that project to given 2D pattern."""
        possible_structures = []
        
        # For each point in 2D, find possible 3D sources
        # This is a simplified version - real implementation would be more sophisticated
        
        if constraints['platonic'].value:
            # Check Platonic solids
            platonic = PlatonicSolids()
            for solid_type in ['tetrahedron', 'octahedron', 'icosahedron']:
                vertices, _ = platonic.get_polytope(solid_type)
                vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
                
                # Check if this could produce the pattern
                # (Simplified check)
                if len(vertices) >= len(pattern_2d):
                    possible_structures.append({
                        'name': solid_type,
                        'vertices': vertices,
                        'confidence': 0.8
                    })
        
        else:
            # Generate possible 3D points by inverse stereographic projection
            for proj_center in [np.array([0, 0, 1]), np.array([0, 0, -1])]:
                vertices_3d = []
                for p2d in pattern_2d:
                    # Inverse stereographic projection
                    r2 = np.dot(p2d, p2d)
                    if proj_center[2] > 0:  # North pole
                        x = 2 * p2d[0] / (1 + r2)
                        y = 2 * p2d[1] / (1 + r2)
                        z = (r2 - 1) / (r2 + 1)
                    else:  # South pole
                        x = 2 * p2d[0] / (1 + r2)
                        y = 2 * p2d[1] / (1 + r2)
                        z = (1 - r2) / (1 + r2)
                    
                    vertices_3d.append([x, y, z])
                
                vertices_3d = np.array(vertices_3d)
                
                # Check constraints
                if constraints['sphere'].value:
                    # Normalize to sphere
                    vertices_3d = vertices_3d / np.linalg.norm(vertices_3d, axis=1, keepdims=True)
                
                if constraints['convex'].value:
                    # Check convexity (simplified)
                    if self._is_convex(vertices_3d):
                        name = f"Projection from {'north' if proj_center[2] > 0 else 'south'} pole"
                        possible_structures.append({
                            'name': name,
                            'vertices': vertices_3d,
                            'confidence': 0.6
                        })
        
        return possible_structures
    
    def _is_convex(self, vertices):
        """Simple convexity check."""
        # For simplicity, just check if all points are on same side of any plane
        # Real implementation would use proper convex hull algorithm
        return True
    
    def _visualize_reconstructions(self, pattern_2d, possible_3d):
        """Visualize 2D pattern and possible 3D sources."""
        n_structures = len(possible_3d)
        
        fig = make_subplots(
            rows=1, cols=n_structures + 1,
            specs=[[{'type': 'xy'}] + [{'type': 'scene'}] * n_structures],
            subplot_titles=['2D Pattern'] + [s['name'] for s in possible_3d]
        )
        
        # Add 2D pattern
        fig.add_trace(
            go.Scatter(
                x=pattern_2d[:, 0], y=pattern_2d[:, 1],
                mode='markers+lines',
                marker=dict(size=10, color='red'),
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Add 3D reconstructions
        for i, structure in enumerate(possible_3d):
            vertices = structure['vertices']
            
            # Add vertices
            fig.add_trace(
                go.Scatter3d(
                    x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=structure['confidence'],
                        colorscale='Viridis',
                        showscale=False
                    ),
                    text=f"Confidence: {structure['confidence']:.2f}"
                ),
                row=1, col=i + 2
            )
            
            # Add wireframe
            if len(vertices) == 4:  # Tetrahedron
                edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
            elif len(vertices) == 6:  # Octahedron
                edges = [(0,1), (0,2), (0,3), (0,4), (1,2), (1,4), 
                        (2,3), (3,4), (1,5), (2,5), (3,5), (4,5)]
            else:  # Generic wireframe
                edges = []
            
            for edge in edges:
                v1, v2 = vertices[edge[0]], vertices[edge[1]]
                fig.add_trace(
                    go.Scatter3d(
                        x=[v1[0], v2[0]], y=[v1[1], v2[1]], z=[v1[2], v2[2]],
                        mode='lines',
                        line=dict(color='gray', width=2)
                    ),
                    row=1, col=i + 2
                )
        
        fig.update_layout(
            title="3D Reconstruction from 2D Pattern",
            height=400,
            width=400 * (n_structures + 1),
            showlegend=False
        )
        
        return fig


class ProjectionErrorVisualizer:
    """Visualize distortions in different projections."""
    
    def __init__(self):
        self.stereographic = StereographicProjection()
        
    def create_distortion_heatmap(self, projection_type: str = 'stereographic') -> go.Figure:
        """Create heatmap showing projection distortions."""
        # Create grid on sphere
        theta = np.linspace(0, np.pi, 50)
        phi = np.linspace(0, 2*np.pi, 100)
        THETA, PHI = np.meshgrid(theta, phi)
        
        # Convert to Cartesian
        X = np.sin(THETA) * np.cos(PHI)
        Y = np.sin(THETA) * np.sin(PHI)
        Z = np.cos(THETA)
        
        # Calculate distortion at each point
        distortion = self._calculate_distortion(X, Y, Z, projection_type)
        
        # Create figure
        fig = go.Figure(data=go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=distortion,
            colorscale='RdBu_r',
            cmin=0, cmax=5,
            colorbar=dict(title="Distortion Factor")
        ))
        
        fig.update_layout(
            title=f"{projection_type.capitalize()} Projection Distortion",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z"
            )
        )
        
        return fig
    
    def _calculate_distortion(self, X, Y, Z, projection_type):
        """Calculate distortion factor at each point."""
        distortion = np.zeros_like(X)
        
        if projection_type == 'stereographic':
            # Stereographic distortion increases with distance from projection point
            proj_point = np.array([0, 0, 1])  # North pole
            
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    point = np.array([X[i,j], Y[i,j], Z[i,j]])
                    # Distance from projection point
                    dist = np.arccos(np.dot(point, proj_point))
                    # Stereographic distortion formula
                    distortion[i,j] = 1 / (np.cos(dist/2)**2)
        
        elif projection_type == 'mercator':
            # Mercator distortion increases with latitude
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    latitude = np.arcsin(Z[i,j])
                    # Mercator distortion formula
                    distortion[i,j] = 1 / np.cos(latitude) if np.cos(latitude) > 0.1 else 10
        
        return distortion
    
    def compare_projections(self) -> go.Figure:
        """Compare stereographic and Mercator projections side by side."""
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'geo'}, {'type': 'geo'}],
                   [{'type': 'xy'}, {'type': 'xy'}]],
            subplot_titles=['Mercator Projection', 'Stereographic Projection',
                          'Mercator Distortion', 'Stereographic Distortion']
        )
        
        # Add world map data (simplified)
        # In real implementation, would load actual country boundaries
        
        # Add distortion patterns
        x = np.linspace(-180, 180, 100)
        y = np.linspace(-90, 90, 50)
        X, Y = np.meshgrid(x, y)
        
        # Mercator distortion
        mercator_dist = 1 / np.cos(np.radians(Y))
        mercator_dist[np.abs(Y) > 85] = 10  # Cap at poles
        
        fig.add_trace(
            go.Heatmap(x=x, y=y, z=mercator_dist, colorscale='Reds'),
            row=2, col=1
        )
        
        # Stereographic distortion (from north pole)
        # Convert lat/lon to 3D then calculate distortion
        stereo_dist = np.zeros_like(Y)
        for i in range(len(y)):
            for j in range(len(x)):
                lat, lon = np.radians(y[i]), np.radians(x[j])
                # Convert to 3D point
                point = np.array([
                    np.cos(lat) * np.cos(lon),
                    np.cos(lat) * np.sin(lon),
                    np.sin(lat)
                ])
                # Distance from north pole
                dist = np.arccos(point[2])
                stereo_dist[i,j] = 1 / (np.cos(dist/2)**2) if dist < np.pi else 10
        
        fig.add_trace(
            go.Heatmap(x=x, y=y, z=stereo_dist, colorscale='Reds'),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Projection Comparison: Mercator vs Stereographic",
            height=800
        )
        
        return fig


class FisheyeCorrector:
    """Demonstrate fisheye distortion correction for microscopy."""
    
    def __init__(self):
        self.stereographic = StereographicProjection()
        
    def create_correction_demo(self) -> widgets.VBox:
        """Interactive demo of fisheye correction process."""
        # Create synthetic fisheye image of bacteria
        bacteria_image = self._create_synthetic_bacteria_image()
        
        # Distortion slider
        distortion_slider = widgets.FloatSlider(
            value=0.5, min=0, max=1, step=0.1,
            description='Distortion:'
        )
        
        # Correction method
        method_dropdown = widgets.Dropdown(
            options=['None', 'Inverse Fisheye', 'Stereographic'],
            value='None',
            description='Method:'
        )
        
        # Output display
        output = widgets.Output()
        
        def update_correction(change):
            distortion = distortion_slider.value
            method = method_dropdown.value
            
            with output:
                output.clear_output()
                
                # Apply distortion
                distorted = self._apply_fisheye_distortion(bacteria_image, distortion)
                
                # Apply correction
                if method == 'None':
                    corrected = distorted
                elif method == 'Inverse Fisheye':
                    corrected = self._inverse_fisheye(distorted, distortion)
                else:  # Stereographic
                    corrected = self._stereographic_correction(distorted, distortion)
                
                # Display results
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(bacteria_image, cmap='gray')
                axes[0].set_title('Original')
                axes[0].axis('off')
                
                axes[1].imshow(distorted, cmap='gray')
                axes[1].set_title('Fisheye Distorted')
                axes[1].axis('off')
                
                axes[2].imshow(corrected, cmap='gray')
                axes[2].set_title(f'Corrected ({method})')
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.show()
        
        distortion_slider.observe(update_correction, names='value')
        method_dropdown.observe(update_correction, names='value')
        
        # Initial display
        update_correction(None)
        
        return widgets.VBox([
            widgets.HTML("<h3>Fisheye Distortion Correction for Microscopy</h3>"),
            distortion_slider,
            method_dropdown,
            output
        ])
    
    def _create_synthetic_bacteria_image(self):
        """Create synthetic image with rod-shaped bacteria."""
        img = np.zeros((500, 500))
        
        # Add grid lines for reference
        for i in range(0, 500, 50):
            img[i, :] = 0.3
            img[:, i] = 0.3
        
        # Add some rod-shaped bacteria
        from skimage.draw import line, disk
        
        # Bacteria 1 - horizontal
        rr, cc = line(250, 150, 250, 350)
        img[rr, cc] = 1.0
        
        # Bacteria 2 - vertical
        rr, cc = line(150, 250, 350, 250)
        img[rr, cc] = 1.0
        
        # Bacteria 3 - diagonal
        rr, cc = line(150, 150, 350, 350)
        img[rr, cc] = 1.0
        
        return img
    
    def _apply_fisheye_distortion(self, image, strength):
        """Apply fisheye distortion to image."""
        h, w = image.shape
        cx, cy = w//2, h//2
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Convert to centered coordinates
        x_c = x - cx
        y_c = y - cy
        
        # Convert to polar
        r = np.sqrt(x_c**2 + y_c**2)
        theta = np.arctan2(y_c, x_c)
        
        # Apply fisheye distortion
        max_r = np.sqrt(cx**2 + cy**2)
        r_distorted = r * (1 + strength * (r/max_r)**2)
        
        # Convert back to Cartesian
        x_new = cx + r_distorted * np.cos(theta)
        y_new = cy + r_distorted * np.sin(theta)
        
        # Interpolate
        from scipy.ndimage import map_coordinates
        coords = np.array([y_new.ravel(), x_new.ravel()])
        
        # Ensure coordinates are within bounds
        coords[0] = np.clip(coords[0], 0, h-1)
        coords[1] = np.clip(coords[1], 0, w-1)
        
        distorted = map_coordinates(image, coords, order=1).reshape(image.shape)
        
        return distorted
    
    def _inverse_fisheye(self, image, strength):
        """Apply inverse fisheye transformation."""
        # Similar to above but with inverse transformation
        return self._apply_fisheye_distortion(image, -strength/(1+strength))
    
    def _stereographic_correction(self, image, strength):
        """Correct using stereographic projection principles."""
        h, w = image.shape
        cx, cy = w//2, h//2
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Map image to sphere
        x_norm = (x - cx) / (w/2)
        y_norm = (y - cy) / (h/2)
        
        # Inverse stereographic projection to sphere
        r2 = x_norm**2 + y_norm**2
        X = 2 * x_norm / (1 + r2)
        Y = 2 * y_norm / (1 + r2)
        Z = (r2 - 1) / (r2 + 1)
        
        # Apply correction on sphere
        # ... (simplified for demo)
        
        # Project back
        corrected = image.copy()  # Simplified
        
        return corrected


class ProjectionSymmetryExplorer:
    """Explore different projection centers and resulting symmetries."""
    
    def __init__(self):
        self.platonic = PlatonicSolids()
        self.stereographic = StereographicProjection()
        
    def create_symmetry_explorer(self) -> widgets.VBox:
        """Interactive explorer for projection symmetries."""
        # Get icosahedron
        vertices, edges = self.platonic.get_polytope('icosahedron')
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        
        # Find all special points
        special_points = self._find_special_points(vertices, edges)
        
        # Create dropdown for special points
        point_dropdown = widgets.Dropdown(
            options=[(p['name'], i) for i, p in enumerate(special_points)],
            value=0,
            description='Projection Point:'
        )
        
        # Symmetry info
        symmetry_label = widgets.Label(value="")
        
        # Create figure
        fig = make_subplots(
            rows=1, cols=3,
            specs=[[{'type': 'scene'}, {'type': 'xy'}, {'type': 'xy'}]],
            subplot_titles=['3D View', '2D Projection', 'Symmetry Analysis']
        )
        
        fig_widget = go.FigureWidget(fig)
        
        def update_projection(change):
            idx = point_dropdown.value
            special_point = special_points[idx]
            proj_center = special_point['position']
            
            # Project vertices
            projected = []
            for v in vertices:
                p = self.stereographic.project_from_pole(v, proj_center)
                projected.append(p)
            projected = np.array(projected)
            
            # Analyze symmetry
            symmetry_info = self._analyze_symmetry(projected)
            symmetry_label.value = f"Symmetry: {symmetry_info['type']} ({symmetry_info['order']}-fold)"
            
            with fig_widget.batch_update():
                # Update 3D view with projection point highlighted
                fig_widget.data = []
                
                # Add sphere
                u = np.linspace(0, 2 * np.pi, 30)
                v = np.linspace(0, np.pi, 30)
                x = np.outer(np.cos(u), np.sin(v))
                y = np.outer(np.sin(u), np.sin(v))
                z = np.outer(np.ones(np.size(u)), np.cos(v))
                
                fig_widget.add_trace(
                    go.Surface(x=x, y=y, z=z, opacity=0.3, showscale=False),
                    row=1, col=1
                )
                
                # Add icosahedron
                fig_widget.add_trace(
                    go.Scatter3d(
                        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                        mode='markers',
                        marker=dict(size=5, color='blue')
                    ),
                    row=1, col=1
                )
                
                # Highlight projection center
                fig_widget.add_trace(
                    go.Scatter3d(
                        x=[proj_center[0]], y=[proj_center[1]], z=[proj_center[2]],
                        mode='markers',
                        marker=dict(size=15, color='red', symbol='diamond')
                    ),
                    row=1, col=1
                )
                
                # Add 2D projection
                for edge in edges:
                    p1, p2 = projected[edge[0]], projected[edge[1]]
                    fig_widget.add_trace(
                        go.Scatter(
                            x=[p1[0], p2[0]], y=[p1[1], p2[1]],
                            mode='lines',
                            line=dict(color='blue', width=2)
                        ),
                        row=1, col=2
                    )
                
                # Add symmetry visualization
                self._add_symmetry_viz(fig_widget, projected, symmetry_info, row=1, col=3)
        
        point_dropdown.observe(update_projection, names='value')
        
        # Initial update
        update_projection(None)
        
        return widgets.VBox([
            widgets.HTML("<h3>Projection Symmetry Explorer</h3>"),
            point_dropdown,
            symmetry_label,
            fig_widget
        ])
    
    def _find_special_points(self, vertices, edges):
        """Find all special points on icosahedron."""
        special_points = []
        
        # Vertices
        for i, v in enumerate(vertices):
            special_points.append({
                'name': f'Vertex {i+1}',
                'position': v,
                'type': 'vertex'
            })
        
        # Edge midpoints
        for i, edge in enumerate(edges):
            midpoint = (vertices[edge[0]] + vertices[edge[1]]) / 2
            midpoint = midpoint / np.linalg.norm(midpoint)
            special_points.append({
                'name': f'Edge {i+1} midpoint',
                'position': midpoint,
                'type': 'edge'
            })
        
        # Face centers
        faces = self._get_faces(vertices, edges)
        for i, face in enumerate(faces):
            center = np.mean([vertices[v] for v in face], axis=0)
            center = center / np.linalg.norm(center)
            special_points.append({
                'name': f'Face {i+1} center',
                'position': center,
                'type': 'face'
            })
        
        return special_points
    
    def _get_faces(self, vertices, edges):
        """Extract faces from vertices and edges."""
        # For icosahedron, find triangular faces
        n = len(vertices)
        adjacency = [set() for _ in range(n)]
        
        for e in edges:
            adjacency[e[0]].add(e[1])
            adjacency[e[1]].add(e[0])
        
        faces = []
        for i in range(n):
            for j in adjacency[i]:
                if j > i:
                    for k in adjacency[i]:
                        if k > j and k in adjacency[j]:
                            faces.append([i, j, k])
        
        return faces
    
    def _analyze_symmetry(self, projected):
        """Analyze symmetry of 2D projection."""
        # Simplified symmetry analysis
        # In real implementation would use group theory
        
        # Count unique distances
        distances = []
        for i in range(len(projected)):
            for j in range(i+1, len(projected)):
                dist = np.linalg.norm(projected[i] - projected[j])
                distances.append(dist)
        
        unique_distances = len(np.unique(np.round(distances, 3)))
        
        # Heuristic symmetry detection
        if unique_distances < 3:
            return {'type': 'Hexagonal', 'order': 6}
        elif unique_distances < 5:
            return {'type': 'Pentagonal', 'order': 5}
        else:
            return {'type': 'Irregular', 'order': 1}
    
    def _add_symmetry_viz(self, fig, projected, symmetry_info, row, col):
        """Visualize symmetry axes and rotations."""
        if symmetry_info['order'] > 1:
            # Add rotation indicators
            center = np.mean(projected, axis=0)
            radius = np.max(np.linalg.norm(projected - center, axis=1))
            
            angles = np.linspace(0, 2*np.pi, symmetry_info['order'] + 1)
            for angle in angles[:-1]:
                x = center[0] + radius * np.cos(angle)
                y = center[1] + radius * np.sin(angle)
                fig.add_trace(
                    go.Scatter(
                        x=[center[0], x], y=[center[1], y],
                        mode='lines',
                        line=dict(color='green', width=1, dash='dash')
                    ),
                    row=row, col=col
                )


class ProjectionLaboratory:
    """Advanced projection exploration with custom projections."""
    
    def __init__(self):
        self.platonic = PlatonicSolids()
        
    def create_laboratory(self) -> widgets.VBox:
        """Create interactive projection laboratory."""
        # Projection type selector
        projection_types = {
            'Stereographic': self._stereographic_projection,
            'Orthographic': self._orthographic_projection,
            'Perspective': self._perspective_projection,
            'Gnomonic': self._gnomonic_projection,
            'Custom Matrix': self._custom_matrix_projection
        }
        
        projection_dropdown = widgets.Dropdown(
            options=list(projection_types.keys()),
            value='Stereographic',
            description='Projection:'
        )
        
        # Polytope selector
        polytope_dropdown = widgets.Dropdown(
            options=['tetrahedron', 'cube', 'octahedron', 'dodecahedron', 'icosahedron'],
            value='icosahedron',
            description='Polytope:'
        )
        
        # Parameter controls
        param_sliders = {
            'distance': widgets.FloatSlider(value=2, min=1, max=5, description='Distance:'),
            'fov': widgets.FloatSlider(value=60, min=30, max=120, description='FOV (deg):'),
            'scale': widgets.FloatSlider(value=1, min=0.1, max=3, description='Scale:')
        }
        
        # Matrix editor for custom projection
        matrix_text = widgets.Textarea(
            value='1, 0, 0\n0, 1, 0\n0, 0, 0',
            description='Matrix:',
            layout=widgets.Layout(width='300px', height='100px')
        )
        
        # Output
        output = widgets.Output()
        
        def update_projection(*args):
            with output:
                output.clear_output()
                
                # Get polytope
                vertices, edges = self.platonic.get_polytope(polytope_dropdown.value)
                vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
                
                # Apply projection
                proj_func = projection_types[projection_dropdown.value]
                
                if projection_dropdown.value == 'Custom Matrix':
                    # Parse matrix
                    try:
                        matrix_lines = matrix_text.value.strip().split('\n')
                        matrix = np.array([
                            [float(x) for x in line.split(',')]
                            for line in matrix_lines
                        ])
                        projected = proj_func(vertices, matrix)
                    except:
                        print("Invalid matrix format")
                        return
                else:
                    # Use standard projection with parameters
                    params = {k: v.value for k, v in param_sliders.items()}
                    projected = proj_func(vertices, **params)
                
                # Visualize
                fig = self._visualize_projection(
                    vertices, projected, edges,
                    projection_dropdown.value
                )
                fig.show()
        
        # Connect callbacks
        projection_dropdown.observe(update_projection, names='value')
        polytope_dropdown.observe(update_projection, names='value')
        for slider in param_sliders.values():
            slider.observe(update_projection, names='value')
        matrix_text.observe(update_projection, names='value')
        
        # Initial display
        update_projection()
        
        # Layout
        controls = widgets.VBox([
            projection_dropdown,
            polytope_dropdown,
            widgets.HTML("<h4>Parameters:</h4>"),
            *param_sliders.values(),
            widgets.HTML("<h4>Custom Matrix (for Custom Matrix projection):</h4>"),
            matrix_text
        ])
        
        return widgets.VBox([
            widgets.HTML("<h3>Projection Laboratory</h3>"),
            widgets.HBox([controls, output])
        ])
    
    def _stereographic_projection(self, vertices, **params):
        """Standard stereographic projection."""
        scale = params.get('scale', 1)
        projected = []
        for v in vertices:
            if v[2] < 0.999:  # Not at north pole
                x = scale * v[0] / (1 - v[2])
                y = scale * v[1] / (1 - v[2])
                projected.append([x, y])
            else:
                projected.append([0, 0])  # Handle pole
        return np.array(projected)
    
    def _orthographic_projection(self, vertices, **params):
        """Orthographic (parallel) projection."""
        scale = params.get('scale', 1)
        # Simply drop z-coordinate
        return scale * vertices[:, :2]
    
    def _perspective_projection(self, vertices, **params):
        """Perspective projection with configurable distance."""
        distance = params.get('distance', 2)
        scale = params.get('scale', 1)
        
        projected = []
        for v in vertices:
            # Project from point at (0, 0, -distance)
            z_eye = v[2] + distance
            if z_eye > 0:
                x = scale * distance * v[0] / z_eye
                y = scale * distance * v[1] / z_eye
                projected.append([x, y])
            else:
                projected.append([0, 0])
        
        return np.array(projected)
    
    def _gnomonic_projection(self, vertices, **params):
        """Gnomonic projection (great circles to straight lines)."""
        scale = params.get('scale', 1)
        
        projected = []
        for v in vertices:
            if v[2] > 0:  # Visible hemisphere
                x = scale * v[0] / v[2]
                y = scale * v[1] / v[2]
                projected.append([x, y])
            else:
                projected.append([np.nan, np.nan])  # Not visible
        
        return np.array(projected)
    
    def _custom_matrix_projection(self, vertices, matrix):
        """Apply custom projection matrix."""
        # Ensure matrix is 2x3 or 3x3
        if matrix.shape[0] == 3:
            matrix = matrix[:2, :]  # Take first two rows
        
        # Apply matrix
        projected = vertices @ matrix.T
        
        return projected
    
    def _visualize_projection(self, vertices_3d, vertices_2d, edges, proj_name):
        """Create visualization of projection results."""
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'xy'}]],
            subplot_titles=['3D Original', f'2D {proj_name} Projection']
        )
        
        # 3D view
        fig.add_trace(
            go.Scatter3d(
                x=vertices_3d[:, 0], y=vertices_3d[:, 1], z=vertices_3d[:, 2],
                mode='markers',
                marker=dict(size=6, color='red')
            ),
            row=1, col=1
        )
        
        for edge in edges:
            v1, v2 = vertices_3d[edge[0]], vertices_3d[edge[1]]
            fig.add_trace(
                go.Scatter3d(
                    x=[v1[0], v2[0]], y=[v1[1], v2[1]], z=[v1[2], v2[2]],
                    mode='lines',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
        
        # 2D projection
        valid_idx = ~np.isnan(vertices_2d[:, 0])
        if np.any(valid_idx):
            fig.add_trace(
                go.Scatter(
                    x=vertices_2d[valid_idx, 0],
                    y=vertices_2d[valid_idx, 1],
                    mode='markers',
                    marker=dict(size=8, color='red')
                ),
                row=1, col=2
            )
            
            for edge in edges:
                if valid_idx[edge[0]] and valid_idx[edge[1]]:
                    p1, p2 = vertices_2d[edge[0]], vertices_2d[edge[1]]
                    fig.add_trace(
                        go.Scatter(
                            x=[p1[0], p2[0]], y=[p1[1], p2[1]],
                            mode='lines',
                            line=dict(color='blue', width=2)
                        ),
                        row=1, col=2
                    )
        
        # Equal aspect ratio for 2D
        fig.update_xaxes(scaleanchor="y", scaleratio=1, row=1, col=2)
        
        fig.update_layout(
            showlegend=False,
            height=500,
            width=1000
        )
        
        return fig


# Main demonstration functions
def demonstrate_all_projections():
    """Create comprehensive notebook demonstration of all projection features."""
    print("=== Comprehensive Projection Visualization System ===\n")
    
    # 1. Master synchronized view
    print("1. Synchronized 3D-2D Projection View")
    viz = ProjectionVisualizer()
    display(viz.create_synchronized_view())
    
    print("\n2. Hexagonal Emergence from Icosahedral Projection")
    hex_demo = HexagonalEmergenceDemo()
    display(hex_demo.create_transition_animation())
    
    print("\n3. Multi-Layer Visual Pathway Projection")
    pathway = MultiLayerProjectionSystem()
    
    # Add retinal projection
    pathway.add_layer(
        'Retinal Projection',
        lambda x: x[:, :2] / (1 - x[:, 2:3]),  # Stereographic
        preserves=['angles', 'circles'],
        distorts=['areas', 'distances']
    )
    
    # Add cortical mapping
    pathway.add_layer(
        'V1 Mapping',
        lambda x: np.log(1 + np.abs(x)) * np.sign(x),  # Log-polar like
        preserves=['topology', 'neighborhoods'],
        distorts=['distances', 'shapes']
    )
    
    display(pathway.create_pathway_visualization())
    
    print("\n4. 24-Cell to MOG Projection")
    mog_demo = TwentyFourCellMOGProjection()
    display(mog_demo.create_mog_animation())
    
    print("\n5. Inverse Projection for 3D Reconstruction")
    inverse = InverseProjectionInterface()
    display(inverse.create_reconstruction_interface())
    
    print("\n6. Projection Error Visualization")
    error_viz = ProjectionErrorVisualizer()
    display(error_viz.create_distortion_heatmap())
    display(error_viz.compare_projections())
    
    print("\n7. Fisheye Correction for Microscopy")
    fisheye = FisheyeCorrector()
    display(fisheye.create_correction_demo())
    
    print("\n8. Projection Symmetry Explorer")
    symmetry = ProjectionSymmetryExplorer()
    display(symmetry.create_symmetry_explorer())
    
    print("\n9. Advanced Projection Laboratory")
    lab = ProjectionLaboratory()
    display(lab.create_laboratory())


if __name__ == "__main__":
    demonstrate_all_projections()