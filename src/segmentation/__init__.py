"""Segmentation algorithms for bacteria and smooth objects."""

from .smooth_curve_segment import (
    SmoothObject,
    PolytopeFit,
    SegmentationResult,
    compute_curvature,
    fit_local_polytope,
    adaptive_segmentation,
    find_patch_boundaries,
    compute_confidence_map,
    harmonic_smoothing,
    parallel_segmentation,
    segment_with_topology
)

from .polytope_matching import (
    PolytopeTemplate,
    MatchResult,
    TemplateLibrary,
    create_bacteria_templates,
    create_organelle_templates,
    create_cell_templates,
    match_to_template,
    hierarchical_matching,
    update_template_library,
    create_default_library,
    save_library,
    load_library
)

from .one_shot_learning import (
    LabeledExample,
    LearnedSignature,
    OneShotSegmentation,
    BiologicalPrior,
    BACTERIA_PRIOR,
    ORGANELLE_PRIOR,
    extract_signature_from_example,
    generate_augmented_examples,
    segment_with_learned_signature,
    transfer_learn_signature,
    refine_signature_bayesian,
    select_active_queries
)

__all__ = [
    'SmoothObject',
    'PolytopeFit', 
    'SegmentationResult',
    'compute_curvature',
    'fit_local_polytope',
    'adaptive_segmentation',
    'find_patch_boundaries',
    'compute_confidence_map',
    'harmonic_smoothing',
    'parallel_segmentation',
    'segment_with_topology',
    'PolytopeTemplate',
    'MatchResult',
    'TemplateLibrary',
    'create_bacteria_templates',
    'create_organelle_templates',
    'create_cell_templates',
    'match_to_template',
    'hierarchical_matching',
    'update_template_library',
    'create_default_library',
    'save_library',
    'load_library',
    'LabeledExample',
    'LearnedSignature',
    'OneShotSegmentation',
    'BiologicalPrior',
    'BACTERIA_PRIOR',
    'ORGANELLE_PRIOR',
    'extract_signature_from_example',
    'generate_augmented_examples',
    'segment_with_learned_signature',
    'transfer_learn_signature',
    'refine_signature_bayesian',
    'select_active_queries'
]