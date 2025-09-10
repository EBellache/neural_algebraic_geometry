"""Core polytope functionality for bacteria segmentation."""

from .platonic_solids import (
    PlatonicSolid,
    create_tetrahedron,
    create_cube,
    create_octahedron,
    create_icosahedron,
    create_dodecahedron,
    create_all_platonic_solids,
    compute_dual,
    cubic_tiling,
    icosahedral_quasicrystal_tiling
)

from .twentyfour_cell import (
    Cell24,
    create_24cell,
    generate_24cell_vertices,
    orthogonal_projection,
    stereographic_projection,
    schlegel_projection,
    platonic_cross_sections,
    vertex_to_golay_codeword,
    golay_codeword_to_vertex,
    golay_error_correction_via_24cell,
    generate_f4_symmetry_group,
    create_24cell_with_golay
)

from .stereographic import (
    ProjectionResult,
    HexagonalPattern,
    stereographic_forward,
    stereographic_inverse,
    project_icosahedron_from_face,
    extract_hexagonal_pattern,
    reconstruct_polytope_from_partial,
    full_segmentation_pipeline,
    compute_hexagon_areas,
    compute_symmetry_score,
    fit_polytope_to_points,
    create_retinal_model
)

from .dual_operations import (
    DualPolytope,
    HarmonicDuality,
    pole_reciprocation,
    construct_dual_polytope,
    dual_harmonic_analysis,
    fast_24cell_duality,
    bacteria_granule_dual_classification,
    iterative_dual_refinement,
    DualSpaceClassifier
)

__all__ = [
    'PlatonicSolid',
    'create_tetrahedron',
    'create_cube', 
    'create_octahedron',
    'create_icosahedron',
    'create_dodecahedron',
    'create_all_platonic_solids',
    'compute_dual',
    'cubic_tiling',
    'icosahedral_quasicrystal_tiling',
    'Cell24',
    'create_24cell',
    'generate_24cell_vertices',
    'orthogonal_projection',
    'stereographic_projection',
    'schlegel_projection',
    'platonic_cross_sections',
    'vertex_to_golay_codeword',
    'golay_codeword_to_vertex',
    'golay_error_correction_via_24cell',
    'generate_f4_symmetry_group',
    'create_24cell_with_golay',
    'ProjectionResult',
    'HexagonalPattern',
    'stereographic_forward',
    'stereographic_inverse',
    'project_icosahedron_from_face',
    'extract_hexagonal_pattern',
    'reconstruct_polytope_from_partial',
    'full_segmentation_pipeline',
    'compute_hexagon_areas',
    'compute_symmetry_score',
    'fit_polytope_to_points',
    'create_retinal_model',
    'DualPolytope',
    'HarmonicDuality',
    'pole_reciprocation',
    'construct_dual_polytope',
    'dual_harmonic_analysis',
    'fast_24cell_duality',
    'bacteria_granule_dual_classification',
    'iterative_dual_refinement',
    'DualSpaceClassifier'
]