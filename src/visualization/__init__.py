"""Visualization modules for polytope-based bacteria segmentation."""

from .polytope_viz import (
    PolytopeVisualization,
    AnimationFrame,
    PolytopeRenderer,
    create_platonic_visualizations,
    create_24cell_visualization,
    visualize_bacteria_deformation,
    create_tiling_visualization,
    StereographicProjectionViewer,
    PolytopeSignatureViewer,
    InteractivePolytopeBuilder,
    MultiScaleTransparencyVisualizer,
    ARPolytopeViewer,
    example_basic_visualization,
    example_bacteria_animation,
    example_stereographic_projection,
    example_harmonic_signatures,
    example_interactive_builder,
    example_multiscale_visualization
)

from .harmonic_viz import (
    HarmonicVisualization,
    HarmonicDecomposition,
    SphericalHarmonicVisualizer,
    BiologicalHarmonicPatterns,
    create_plotly_3d_harmonic,
    example_basic_harmonic_visualization,
    example_harmonic_decomposition,
    example_interactive_explorer,
    example_biological_patterns,
    example_harmonic_animation,
    example_plotly_interactive
)

__all__ = [
    # polytope_viz
    'PolytopeVisualization',
    'AnimationFrame',
    'PolytopeRenderer',
    'create_platonic_visualizations',
    'create_24cell_visualization',
    'visualize_bacteria_deformation',
    'create_tiling_visualization',
    'StereographicProjectionViewer',
    'PolytopeSignatureViewer',
    'InteractivePolytopeBuilder',
    'MultiScaleTransparencyVisualizer',
    'ARPolytopeViewer',
    'example_basic_visualization',
    'example_bacteria_animation',
    'example_stereographic_projection',
    'example_harmonic_signatures',
    'example_interactive_builder',
    'example_multiscale_visualization',
    # harmonic_viz
    'HarmonicVisualization',
    'HarmonicDecomposition',
    'SphericalHarmonicVisualizer',
    'BiologicalHarmonicPatterns',
    'create_plotly_3d_harmonic',
    'example_basic_harmonic_visualization',
    'example_harmonic_decomposition',
    'example_interactive_explorer',
    'example_biological_patterns',
    'example_harmonic_animation',
    'example_plotly_interactive'
]