"""Golay code and 24-cell based error correction."""

from .error_correction import (
    ErrorPattern,
    CorrectionResult,
    CELL24_VERTICES,
    embed_codeword_to_24cell,
    compute_geometric_syndrome,
    generate_f4_symmetries,
    apply_symmetry_to_codeword,
    correct_errors_geometric,
    fast_correction_lookup,
    metabolic_stress_pattern,
    synaptic_failure_pattern,
    soft_decision_decode,
    visualize_correction_geometry
)

from .mog_as_projection import (
    MOGProjection,
    Octad,
    create_mog_projection,
    find_hyperplane_octads,
    mog_lookup,
    compose_projections,
    generate_coset_projections,
    project_to_hexagonal_plane,
    mog_automorphism_correspondence,
    visualize_rotating_mog_shadow,
    identify_segmentation_boundaries,
    verify_classical_mog
)

__all__ = [
    'ErrorPattern',
    'CorrectionResult',
    'CELL24_VERTICES',
    'embed_codeword_to_24cell',
    'compute_geometric_syndrome',
    'generate_f4_symmetries',
    'apply_symmetry_to_codeword',
    'correct_errors_geometric',
    'fast_correction_lookup',
    'metabolic_stress_pattern',
    'synaptic_failure_pattern',
    'soft_decision_decode',
    'visualize_correction_geometry',
    'MOGProjection',
    'Octad',
    'create_mog_projection',
    'find_hyperplane_octads',
    'mog_lookup',
    'compose_projections',
    'generate_coset_projections',
    'project_to_hexagonal_plane',
    'mog_automorphism_correspondence',
    'visualize_rotating_mog_shadow',
    'identify_segmentation_boundaries',
    'verify_classical_mog'
]