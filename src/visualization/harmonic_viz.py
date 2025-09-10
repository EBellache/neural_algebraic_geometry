"""
Visualization of spherical harmonics on polytope vertices.

This module provides comprehensive visualization tools for spherical harmonics
evaluated at polytope vertex positions. It includes color-mapped visualizations,
harmonic decomposition animations, interactive harmonic explorers, and
frequency-domain visualizations that map to biological interpretations.

Key features:
- Spherical harmonic evaluation on arbitrary polytopes
- Color-mapped visualization of harmonic values
- Animated transitions between harmonic modes
- Interactive harmonic coefficient adjustment
- Frequency band visualization (delta/theta/alpha/beta/gamma)
- Biologically-relevant harmonic patterns
- Power spectrum visualization
- Phase relationships between harmonics
- Harmonic synthesis visualization
"""

import jax
import jax.numpy as jnp
from typing import List, Dict, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors, animation
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets
from IPython.display import display, HTML
import plotly.graph_objects as go
import plotly.express as px
from scipy.special import sph_harm
import colorsys


@dataclass
class HarmonicVisualization:
    """Container for harmonic visualization data.
    
    Attributes:
        vertices: (N, 3) polytope vertices
        harmonic_values: (N,) complex harmonic values at vertices
        l: Harmonic degree
        m: Harmonic order
        frequency_band: Associated brain frequency band
        biological_interpretation: What this harmonic represents biologically
    """
    vertices: jnp.ndarray
    harmonic_values: jnp.ndarray
    l: int
    m: int
    frequency_band: str = ""
    biological_interpretation: str = ""


@dataclass
class HarmonicDecomposition:
    """Complete harmonic decomposition of a shape.
    
    Attributes:
        coefficients: Dict mapping (l,m) to complex coefficients
        power_spectrum: Power at each l value
        phase_spectrum: Phase relationships
        dominant_modes: List of dominant (l,m) modes
        biological_signature: Biological interpretation
    """
    coefficients: Dict[Tuple[int, int], complex]
    power_spectrum: Dict[int, float]
    phase_spectrum: Dict[Tuple[int, int], float]
    dominant_modes: List[Tuple[int, int]]
    biological_signature: str = ""


class SphericalHarmonicVisualizer:
    """Main visualizer for spherical harmonics on polytopes."""
    
    # Map harmonic orders to brain frequencies
    FREQUENCY_BANDS = {
        0: ("DC", "Constant/baseline"),
        1: ("Delta", "0.5-4 Hz - Deep sleep, healing"),
        2: ("Theta", "4-8 Hz - Meditation, memory"),
        3: ("Alpha", "8-12 Hz - Relaxation, creativity"),
        4: ("Beta", "12-30 Hz - Active thinking, focus"),
        5: ("Low Gamma", "30-50 Hz - Conscious awareness"),
        6: ("High Gamma", "50-100 Hz - Higher cognitive functions")
    }
    
    # Biological interpretations
    BIOLOGICAL_PATTERNS = {
        (0, 0): "Uniform distribution - cell volume",
        (1, 0): "Polar gradient - cell polarity", 
        (1, 1): "Lateral asymmetry - chirality",
        (2, 0): "Quadrupolar - elongation axis",
        (2, 1): "Oblique elongation - bending",
        (2, 2): "Cross-sectional ellipticity",
        (3, 0): "Octupolar - branching points",
        (3, 1): "Asymmetric branching",
        (4, 0): "Hexadecapolar - complex symmetry",
        (4, 4): "Tetrahedral symmetry - viral capsids"
    }
    
    def __init__(self, max_l: int = 6):
        """Initialize visualizer with maximum harmonic degree."""
        self.max_l = max_l
        self.current_decomposition = None
    
    @staticmethod
    def compute_spherical_harmonic(l: int, m: int, 
                                 vertices: jnp.ndarray) -> jnp.ndarray:
        """Compute spherical harmonic Y_lm at vertex positions.
        
        Args:
            l: Harmonic degree
            m: Harmonic order (-l <= m <= l)
            vertices: (N, 3) vertex positions
            
        Returns:
            (N,) complex harmonic values
        """
        # Normalize vertices to unit sphere
        norms = jnp.linalg.norm(vertices, axis=1, keepdims=True)
        normalized = vertices / (norms + 1e-10)
        
        # Convert to spherical coordinates
        x, y, z = normalized[:, 0], normalized[:, 1], normalized[:, 2]
        r = jnp.ones_like(x)  # Already normalized
        
        # theta: polar angle (0 to pi)
        theta = jnp.arccos(jnp.clip(z, -1, 1))
        
        # phi: azimuthal angle (0 to 2pi)
        phi = jnp.arctan2(y, x)
        
        # Compute spherical harmonic
        # Using scipy for accurate implementation
        Y_lm = sph_harm(m, l, phi, theta)
        
        # Weight by distance from origin (for non-normalized vertices)
        weighted = Y_lm * jnp.squeeze(norms)
        
        return jnp.array(weighted)
    
    def visualize_single_harmonic(self, vertices: jnp.ndarray,
                                l: int, m: int,
                                show_phase: bool = True,
                                colormap: str = 'RdBu') -> plt.Figure:
        """Visualize single spherical harmonic on polytope.
        
        Args:
            vertices: (N, 3) polytope vertices
            l: Harmonic degree
            m: Harmonic order
            show_phase: Whether to show phase in color
            colormap: Matplotlib colormap
            
        Returns:
            Figure with visualization
        """
        # Compute harmonic values
        Y_lm = self.compute_spherical_harmonic(l, m, vertices)
        
        # Create figure
        fig = plt.figure(figsize=(12, 5))
        
        # 3D visualization
        ax1 = fig.add_subplot(121, projection='3d')
        
        if show_phase:
            # Color by phase, size by magnitude
            phases = np.angle(Y_lm)
            magnitudes = np.abs(Y_lm)
            
            # Normalize for visualization
            norm_mag = magnitudes / (np.max(magnitudes) + 1e-10)
            
            # Create custom colormap for phase
            colors_rgb = [colorsys.hsv_to_rgb(p/(2*np.pi), 1, 1) 
                         for p in (phases + np.pi)]
            
            scatter = ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                                c=colors_rgb, s=100 + 300*norm_mag,
                                alpha=0.8, edgecolors='k', linewidth=0.5)
            
            ax1.set_title(f'Y_{l},{m} - Phase and Magnitude')
        else:
            # Color by real part
            values = np.real(Y_lm)
            
            scatter = ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                                c=values, cmap=colormap, s=200,
                                alpha=0.8, edgecolors='k', linewidth=0.5)
            
            fig.colorbar(scatter, ax=ax1, shrink=0.5)
            ax1.set_title(f'Y_{l},{m} - Real Part')
        
        # Add frequency band info
        if l in self.FREQUENCY_BANDS:
            band, desc = self.FREQUENCY_BANDS[l]
            ax1.text2D(0.02, 0.98, f'{band}: {desc}', 
                      transform=ax1.transAxes, fontsize=10,
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Biological interpretation
        if (l, m) in self.BIOLOGICAL_PATTERNS:
            ax1.text2D(0.02, 0.88, self.BIOLOGICAL_PATTERNS[(l, m)],
                      transform=ax1.transAxes, fontsize=10,
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_box_aspect([1,1,1])
        
        # 2D projection (spherical plot)
        ax2 = fig.add_subplot(122)
        
        # Project to 2D using stereographic projection
        proj_x, proj_y = self._stereographic_projection(vertices)
        
        if show_phase:
            scatter2 = ax2.scatter(proj_x, proj_y, c=colors_rgb, s=100 + 300*norm_mag,
                                 alpha=0.8, edgecolors='k', linewidth=0.5)
        else:
            scatter2 = ax2.scatter(proj_x, proj_y, c=values, cmap=colormap, s=200,
                                 alpha=0.8, edgecolors='k', linewidth=0.5)
            
        ax2.set_aspect('equal')
        ax2.set_title('Stereographic Projection')
        ax2.set_xlabel('u')
        ax2.set_ylabel('v')
        
        plt.tight_layout()
        return fig
    
    def _stereographic_projection(self, vertices: jnp.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project vertices to 2D using stereographic projection."""
        # Normalize first
        norms = np.linalg.norm(vertices, axis=1, keepdims=True)
        normalized = vertices / (norms + 1e-10)
        
        # Project from north pole
        X = normalized[:, 0] / (1 - normalized[:, 2] + 1e-10)
        Y = normalized[:, 1] / (1 - normalized[:, 2] + 1e-10)
        
        # Handle points at north pole
        at_pole = normalized[:, 2] > 0.99
        X[at_pole] = 0
        Y[at_pole] = 0
        
        return X, Y
    
    def decompose_shape(self, vertices: jnp.ndarray, 
                       values: Optional[jnp.ndarray] = None) -> HarmonicDecomposition:
        """Decompose shape into spherical harmonics.
        
        Args:
            vertices: (N, 3) vertex positions
            values: (N,) optional values at vertices (default: use radial distance)
            
        Returns:
            Complete harmonic decomposition
        """
        if values is None:
            # Use radial distance as default
            values = jnp.linalg.norm(vertices, axis=1)
        
        # Compute all harmonic coefficients
        coefficients = {}
        
        for l in range(self.max_l + 1):
            for m in range(-l, l + 1):
                # Compute Y_lm
                Y_lm = self.compute_spherical_harmonic(l, m, vertices)
                
                # Integrate against values
                coeff = jnp.mean(values * jnp.conj(Y_lm))
                coefficients[(l, m)] = complex(coeff)
        
        # Compute power spectrum
        power_spectrum = {}
        for l in range(self.max_l + 1):
            power = 0.0
            for m in range(-l, l + 1):
                power += np.abs(coefficients[(l, m)])**2
            power_spectrum[l] = np.sqrt(power)
        
        # Compute phase spectrum
        phase_spectrum = {}
        for (l, m), coeff in coefficients.items():
            phase_spectrum[(l, m)] = np.angle(coeff)
        
        # Find dominant modes
        sorted_modes = sorted(coefficients.items(), 
                            key=lambda x: np.abs(x[1]), reverse=True)
        dominant_modes = [mode for mode, _ in sorted_modes[:10]]
        
        # Biological signature
        bio_sig = self._interpret_biological_signature(power_spectrum, dominant_modes)
        
        self.current_decomposition = HarmonicDecomposition(
            coefficients=coefficients,
            power_spectrum=power_spectrum,
            phase_spectrum=phase_spectrum,
            dominant_modes=dominant_modes,
            biological_signature=bio_sig
        )
        
        return self.current_decomposition
    
    def _interpret_biological_signature(self, power_spectrum: Dict[int, float],
                                      dominant_modes: List[Tuple[int, int]]) -> str:
        """Interpret biological meaning from harmonic signature."""
        interpretations = []
        
        # Check for elongation (l=2 dominance)
        if power_spectrum.get(2, 0) > 2 * power_spectrum.get(0, 0):
            interpretations.append("Elongated (bacterial-like)")
        
        # Check for sphericity (l=0 dominance)
        elif power_spectrum.get(0, 0) > 2 * sum(power_spectrum.get(l, 0) for l in range(1, 4)):
            interpretations.append("Spherical (organelle-like)")
        
        # Check for branching (l=3 presence)
        if power_spectrum.get(3, 0) > 0.3 * max(power_spectrum.values()):
            interpretations.append("Branched structure")
        
        # Check for asymmetry (m != 0 modes)
        asymmetric = sum(1 for (l, m) in dominant_modes[:5] if m != 0)
        if asymmetric > 2:
            interpretations.append("Asymmetric")
        
        return "; ".join(interpretations) if interpretations else "Regular polytope"
    
    def visualize_decomposition(self, decomposition: Optional[HarmonicDecomposition] = None,
                              top_k: int = 6) -> plt.Figure:
        """Visualize harmonic decomposition results.
        
        Args:
            decomposition: Decomposition to visualize (or use current)
            top_k: Number of top modes to show
            
        Returns:
            Figure with decomposition visualization
        """
        if decomposition is None:
            decomposition = self.current_decomposition
        
        if decomposition is None:
            raise ValueError("No decomposition available")
        
        fig = plt.figure(figsize=(15, 10))
        
        # Power spectrum
        ax1 = fig.add_subplot(2, 3, 1)
        l_values = list(decomposition.power_spectrum.keys())
        powers = list(decomposition.power_spectrum.values())
        
        bars = ax1.bar(l_values, powers, color='steelblue', alpha=0.8)
        
        # Color bars by frequency band
        for l, bar in zip(l_values, bars):
            if l in self.FREQUENCY_BANDS:
                band = self.FREQUENCY_BANDS[l][0]
                color_map = {
                    "DC": "gray",
                    "Delta": "purple",
                    "Theta": "blue",
                    "Alpha": "green",
                    "Beta": "orange",
                    "Low Gamma": "red",
                    "High Gamma": "darkred"
                }
                bar.set_color(color_map.get(band, "black"))
        
        ax1.set_xlabel('Harmonic Degree (l)')
        ax1.set_ylabel('Power')
        ax1.set_title('Power Spectrum')
        ax1.grid(True, alpha=0.3)
        
        # Frequency band legend
        from matplotlib.patches import Patch
        legend_elements = []
        for l in range(min(7, self.max_l + 1)):
            if l in self.FREQUENCY_BANDS:
                band, _ = self.FREQUENCY_BANDS[l]
                color_map = {
                    "DC": "gray", "Delta": "purple", "Theta": "blue",
                    "Alpha": "green", "Beta": "orange",
                    "Low Gamma": "red", "High Gamma": "darkred"
                }
                legend_elements.append(
                    Patch(facecolor=color_map[band], label=f'l={l}: {band}')
                )
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        # Top coefficients
        ax2 = fig.add_subplot(2, 3, 2)
        top_modes = decomposition.dominant_modes[:top_k]
        top_coeffs = [decomposition.coefficients[mode] for mode in top_modes]
        
        mode_labels = [f'Y_{{{l},{m}}}' for (l, m) in top_modes]
        magnitudes = [np.abs(c) for c in top_coeffs]
        
        bars2 = ax2.bar(range(len(magnitudes)), magnitudes, color='coral', alpha=0.8)
        ax2.set_xticks(range(len(mode_labels)))
        ax2.set_xticklabels(mode_labels, rotation=45)
        ax2.set_ylabel('Magnitude')
        ax2.set_title(f'Top {top_k} Harmonic Modes')
        ax2.grid(True, alpha=0.3)
        
        # Phase wheel for top modes
        ax3 = fig.add_subplot(2, 3, 3, projection='polar')
        phases = [np.angle(c) for c in top_coeffs]
        radii = magnitudes / max(magnitudes)
        
        colors_phase = [colorsys.hsv_to_rgb(p/(2*np.pi), 1, 1) for p in (np.array(phases) + np.pi)]
        
        for i, (phase, radius, label) in enumerate(zip(phases, radii, mode_labels)):
            ax3.scatter(phase, radius, s=200, c=[colors_phase[i]], 
                       edgecolors='k', linewidth=1, alpha=0.8)
            ax3.annotate(label, (phase, radius), fontsize=8)
        
        ax3.set_ylim(0, 1.1)
        ax3.set_title('Phase Relationships', pad=20)
        
        # Biological interpretation
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.text(0.05, 0.95, "Biological Interpretation:", 
                transform=ax4.transAxes, fontsize=12, weight='bold',
                verticalalignment='top')
        ax4.text(0.05, 0.85, decomposition.biological_signature,
                transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', wrap=True)
        
        # List dominant patterns
        y_pos = 0.65
        for (l, m) in top_modes[:3]:
            if (l, m) in self.BIOLOGICAL_PATTERNS:
                ax4.text(0.05, y_pos, f"Y_{{{l},{m}}}: {self.BIOLOGICAL_PATTERNS[(l, m)]}",
                        transform=ax4.transAxes, fontsize=10,
                        verticalalignment='top')
                y_pos -= 0.1
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        # Reconstructed vs Original comparison
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.text(0.5, 0.5, "Reconstruction Error Analysis\n(Would show here)", 
                ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        ax5.axis('off')
        
        # Harmonic synthesis preview
        ax6 = fig.add_subplot(2, 3, 6, projection='3d')
        ax6.text2D(0.5, 0.5, "Harmonic Synthesis\n(Would show here)",
                  ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        ax6.set_box_aspect([1,1,1])
        
        plt.tight_layout()
        return fig
    
    def create_interactive_explorer(self, vertices: jnp.ndarray) -> widgets.VBox:
        """Create interactive harmonic explorer widget.
        
        Args:
            vertices: Polytope vertices to explore
            
        Returns:
            Interactive widget interface
        """
        # Sliders for l and m
        l_slider = widgets.IntSlider(
            value=0, min=0, max=self.max_l, 
            description='Degree (l):', continuous_update=False
        )
        
        m_slider = widgets.IntSlider(
            value=0, min=0, max=0,
            description='Order (m):', continuous_update=False
        )
        
        # Update m range when l changes
        def update_m_range(change):
            m_slider.min = -change['new']
            m_slider.max = change['new']
            m_slider.value = 0
        
        l_slider.observe(update_m_range, names='value')
        
        # Phase visualization toggle
        phase_toggle = widgets.ToggleButton(
            value=True, description='Show Phase',
            icon='check'
        )
        
        # Output widget
        output = widgets.Output()
        
        # Update function
        def update_visualization(change):
            with output:
                output.clear_output(wait=True)
                fig = self.visualize_single_harmonic(
                    vertices, l_slider.value, m_slider.value,
                    show_phase=phase_toggle.value
                )
                plt.show()
        
        # Connect observers
        l_slider.observe(update_visualization, names='value')
        m_slider.observe(update_visualization, names='value')
        phase_toggle.observe(update_visualization, names='value')
        
        # Initial plot
        update_visualization(None)
        
        # Layout
        controls = widgets.HBox([l_slider, m_slider, phase_toggle])
        
        return widgets.VBox([
            widgets.HTML("<h3>Interactive Spherical Harmonic Explorer</h3>"),
            controls,
            output
        ])
    
    def animate_harmonic_transitions(self, vertices: jnp.ndarray,
                                   mode_sequence: List[Tuple[int, int]],
                                   duration: float = 10.0,
                                   save_path: Optional[str] = None) -> animation.FuncAnimation:
        """Animate transitions between harmonic modes.
        
        Args:
            vertices: Polytope vertices
            mode_sequence: List of (l, m) modes to transition through
            duration: Total animation duration in seconds
            save_path: Optional path to save animation
            
        Returns:
            Matplotlib animation object
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Initial scatter plot
        scatter = ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                           c=np.zeros(len(vertices)), cmap='RdBu',
                           s=200, alpha=0.8, edgecolors='k', linewidth=0.5)
        
        # Color bar
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.5)
        cbar.set_label('Harmonic Value')
        
        # Title
        title = ax.text2D(0.5, 0.95, '', transform=ax.transAxes,
                         ha='center', fontsize=14, weight='bold')
        
        # Info box
        info = ax.text2D(0.02, 0.95, '', transform=ax.transAxes,
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1,1,1])
        
        # Animation function
        n_frames = len(mode_sequence) * 30  # 30 frames per mode
        
        def animate(frame):
            # Determine current mode
            mode_idx = frame // 30
            if mode_idx >= len(mode_sequence):
                mode_idx = len(mode_sequence) - 1
            
            l, m = mode_sequence[mode_idx]
            
            # Compute harmonic
            Y_lm = self.compute_spherical_harmonic(l, m, vertices)
            values = np.real(Y_lm)
            
            # Update scatter plot
            scatter.set_array(values)
            scatter.set_clim(vmin=np.min(values), vmax=np.max(values))
            
            # Update title
            title.set_text(f'Spherical Harmonic Y_{{{l},{m}}}')
            
            # Update info
            if l in self.FREQUENCY_BANDS:
                band, desc = self.FREQUENCY_BANDS[l]
                info_text = f'{band}: {desc}'
                if (l, m) in self.BIOLOGICAL_PATTERNS:
                    info_text += f'\n{self.BIOLOGICAL_PATTERNS[(l, m)]}'
                info.set_text(info_text)
            
            # Rotate view
            ax.view_init(elev=30, azim=frame * 2)
            
            return scatter, title, info
        
        anim = animation.FuncAnimation(
            fig, animate, frames=n_frames,
            interval=duration * 1000 / n_frames,
            blit=False
        )
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=30)
        
        return anim
    
    def visualize_harmonic_synthesis(self, vertices: jnp.ndarray,
                                   coefficients: Dict[Tuple[int, int], complex],
                                   show_components: bool = True) -> plt.Figure:
        """Visualize synthesis of multiple harmonics.
        
        Args:
            vertices: Polytope vertices
            coefficients: Harmonic coefficients to synthesize
            show_components: Show individual components
            
        Returns:
            Figure showing synthesis
        """
        # Synthesize total field
        total_field = np.zeros(len(vertices), dtype=complex)
        
        for (l, m), coeff in coefficients.items():
            Y_lm = self.compute_spherical_harmonic(l, m, vertices)
            total_field += coeff * Y_lm
        
        # Create figure
        n_components = len(coefficients) if show_components else 0
        n_rows = 1 + (n_components + 2) // 3
        
        fig = plt.figure(figsize=(15, 5 * n_rows))
        
        # Total synthesis
        ax_main = fig.add_subplot(n_rows, 3, 1, projection='3d')
        
        scatter = ax_main.scatter(
            vertices[:, 0], vertices[:, 1], vertices[:, 2],
            c=np.real(total_field), cmap='RdBu', s=200,
            alpha=0.8, edgecolors='k', linewidth=0.5
        )
        
        fig.colorbar(scatter, ax=ax_main, shrink=0.5)
        ax_main.set_title('Synthesized Field', fontsize=14, weight='bold')
        ax_main.set_box_aspect([1,1,1])
        
        # Individual components
        if show_components:
            for idx, ((l, m), coeff) in enumerate(coefficients.items()):
                if idx + 2 > n_rows * 3:
                    break
                
                ax = fig.add_subplot(n_rows, 3, idx + 2, projection='3d')
                
                Y_lm = self.compute_spherical_harmonic(l, m, vertices)
                component = coeff * Y_lm
                
                scatter = ax.scatter(
                    vertices[:, 0], vertices[:, 1], vertices[:, 2],
                    c=np.real(component), cmap='RdBu', s=100,
                    alpha=0.6, edgecolors='k', linewidth=0.5
                )
                
                ax.set_title(f'{np.abs(coeff):.2f} × Y_{{{l},{m}}}', fontsize=10)
                ax.set_box_aspect([1,1,1])
                
                # Minimal labels
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
        
        plt.tight_layout()
        return fig


class BiologicalHarmonicPatterns:
    """Generate biologically relevant harmonic patterns."""
    
    @staticmethod
    def bacteria_elongation_pattern() -> Dict[Tuple[int, int], complex]:
        """Typical bacterial elongation pattern."""
        return {
            (0, 0): 1.0 + 0j,      # Base spherical component
            (2, 0): 2.5 + 0j,      # Strong quadrupole (elongation)
            (2, 1): 0.3 + 0.1j,    # Slight bending
            (4, 0): 0.5 + 0j,      # Hexadecapole (end caps)
        }
    
    @staticmethod
    def organelle_pattern() -> Dict[Tuple[int, int], complex]:
        """Typical organelle (mitochondria) pattern."""
        return {
            (0, 0): 1.0 + 0j,      # Dominant spherical
            (2, 0): 0.8 + 0j,      # Moderate elongation
            (2, 2): 0.4 + 0j,      # Cross-sectional ellipticity
            (3, 1): 0.2 + 0.1j,    # Slight asymmetry
        }
    
    @staticmethod
    def viral_capsid_pattern() -> Dict[Tuple[int, int], complex]:
        """Icosahedral viral capsid pattern."""
        return {
            (0, 0): 1.0 + 0j,      # Spherical base
            (6, 0): 0.8 + 0j,      # Icosahedral symmetry
            (6, 5): 0.6 + 0j,      # 5-fold axes
            (10, 0): 0.4 + 0j,     # Higher order symmetry
        }
    
    @staticmethod
    def cell_division_pattern(phase: float = 0.5) -> Dict[Tuple[int, int], complex]:
        """Cell division pattern as function of phase."""
        # Phase: 0 = single cell, 0.5 = constriction, 1 = two cells
        constriction = 1 - 2 * abs(phase - 0.5)
        
        return {
            (0, 0): 1.0 + 0j,                    # Base volume
            (2, 0): 0.5 * constriction + 0j,     # Elongation reduces
            (1, 0): 0.8 * constriction + 0j,     # Dipole emerges
            (3, 0): 1.2 * constriction + 0j,     # Constriction point
        }


def create_plotly_3d_harmonic(vertices: np.ndarray, l: int, m: int,
                            show_mesh: bool = True) -> go.Figure:
    """Create interactive 3D harmonic visualization with Plotly.
    
    Args:
        vertices: Polytope vertices
        l: Harmonic degree
        m: Harmonic order
        show_mesh: Whether to show connecting mesh
        
    Returns:
        Plotly figure object
    """
    visualizer = SphericalHarmonicVisualizer()
    Y_lm = visualizer.compute_spherical_harmonic(l, m, vertices)
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(go.Scatter3d(
        x=vertices[:, 0],
        y=vertices[:, 1], 
        z=vertices[:, 2],
        mode='markers',
        marker=dict(
            size=10,
            color=np.real(Y_lm),
            colorscale='RdBu',
            showscale=True,
            colorbar=dict(title=f'Y_{{{l},{m}}} (Real)')
        ),
        text=[f'Y_{l},{m} = {val:.3f}' for val in Y_lm],
        hoverinfo='text'
    ))
    
    # Add mesh if requested
    if show_mesh and len(vertices) > 4:
        # Simple Delaunay triangulation for mesh
        from scipy.spatial import ConvexHull
        hull = ConvexHull(vertices)
        
        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=hull.simplices[:, 0],
            j=hull.simplices[:, 1],
            k=hull.simplices[:, 2],
            intensity=np.real(Y_lm),
            colorscale='RdBu',
            showscale=False,
            opacity=0.3
        ))
    
    # Update layout
    fig.update_layout(
        title=f'Spherical Harmonic Y<sub>{l},{m}</sub>',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        ),
        width=800,
        height=600
    )
    
    return fig


# Example usage functions
def example_basic_harmonic_visualization():
    """Basic example of harmonic visualization."""
    # Create icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    
    # Create visualizer
    viz = SphericalHarmonicVisualizer(max_l=6)
    
    # Visualize Y_2,0 (quadrupole - elongation)
    fig = viz.visualize_single_harmonic(vertices, l=2, m=0, show_phase=False)
    plt.show()
    
    return viz, vertices


def example_harmonic_decomposition():
    """Example of decomposing a shape."""
    # Create elongated shape (bacterial-like)
    theta = np.linspace(0, np.pi, 20)
    phi = np.linspace(0, 2*np.pi, 40)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    
    # Elongated ellipsoid
    a, b, c = 2.0, 0.7, 0.7
    x = a * np.sin(theta_grid.flatten()) * np.cos(phi_grid.flatten())
    y = b * np.sin(theta_grid.flatten()) * np.sin(phi_grid.flatten())
    z = c * np.cos(theta_grid.flatten())
    
    vertices = np.stack([x, y, z], axis=-1)
    
    # Decompose
    viz = SphericalHarmonicVisualizer(max_l=6)
    decomposition = viz.decompose_shape(vertices)
    
    # Visualize decomposition
    fig = viz.visualize_decomposition(decomposition)
    plt.show()
    
    print(f"Biological signature: {decomposition.biological_signature}")
    print(f"Dominant modes: {decomposition.dominant_modes[:5]}")
    
    return viz, decomposition


def example_interactive_explorer():
    """Example of interactive harmonic explorer."""
    # Create cube vertices
    vertices = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
    ]) / np.sqrt(3)
    
    # Create visualizer and explorer
    viz = SphericalHarmonicVisualizer(max_l=6)
    explorer = viz.create_interactive_explorer(vertices)
    
    display(explorer)
    
    return viz, explorer


def example_biological_patterns():
    """Example of biological harmonic patterns."""
    # Create octahedron (for bacteria base shape)
    vertices = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ])
    
    viz = SphericalHarmonicVisualizer()
    
    # Get biological patterns
    patterns = BiologicalHarmonicPatterns()
    
    # Visualize bacteria pattern
    bacteria_coeffs = patterns.bacteria_elongation_pattern()
    fig1 = viz.visualize_harmonic_synthesis(vertices, bacteria_coeffs)
    fig1.suptitle('Bacterial Elongation Pattern', fontsize=16)
    plt.show()
    
    # Visualize organelle pattern
    organelle_coeffs = patterns.organelle_pattern()
    fig2 = viz.visualize_harmonic_synthesis(vertices, organelle_coeffs)
    fig2.suptitle('Organelle (Mitochondria) Pattern', fontsize=16)
    plt.show()
    
    return viz, patterns


def example_harmonic_animation():
    """Example of harmonic mode animation."""
    # Create dodecahedron vertices (simplified)
    vertices = np.random.randn(20, 3)
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    
    viz = SphericalHarmonicVisualizer()
    
    # Define mode sequence (increasing complexity)
    mode_sequence = [
        (0, 0),  # Monopole
        (1, 0),  # Dipole
        (1, 1),  # Lateral dipole
        (2, 0),  # Quadrupole
        (2, 2),  # Cross quadrupole
        (3, 0),  # Octupole
        (4, 0),  # Hexadecapole
    ]
    
    # Create animation
    anim = viz.animate_harmonic_transitions(
        vertices, mode_sequence, duration=10.0
    )
    
    # Display in notebook
    from IPython.display import HTML
    HTML(anim.to_jshtml())
    
    return anim


def example_plotly_interactive():
    """Example of interactive Plotly visualization."""
    # Create tetrahedron
    vertices = np.array([
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
    ]) / np.sqrt(3)
    
    # Create interactive plot
    fig = create_plotly_3d_harmonic(vertices, l=2, m=1, show_mesh=True)
    fig.show()
    
    return fig


if __name__ == "__main__":
    print("Spherical Harmonic Visualization System")
    print("Available examples:")
    print("- example_basic_harmonic_visualization()")
    print("- example_harmonic_decomposition()")
    print("- example_interactive_explorer()")
    print("- example_biological_patterns()")
    print("- example_harmonic_animation()")
    print("- example_plotly_interactive()")