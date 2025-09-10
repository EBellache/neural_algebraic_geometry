"""Spherical harmonics functionality for shape analysis."""

from .platonic_harmonics import (
    HarmonicBasis,
    HarmonicSpectrum,
    FREQUENCY_BANDS,
    spherical_harmonic,
    compute_harmonic_basis,
    analyze_shape_harmonics,
    bacteria_harmonic_signature,
    granule_harmonic_signature,
    compare_bacteria_granule_harmonics,
    map_harmonics_to_brain_frequencies,
    DiscreteSphericalTransform
)

from .harmonic_wavelets import (
    HarmonicWavelet,
    WaveletBasis,
    WaveletDecomposition,
    construct_mother_wavelet,
    construct_scale_specific_wavelets,
    detect_bacteria_features,
    wavelet_transform,
    FastWaveletTransform,
    multi_scale_bacteria_analysis,
    adaptive_wavelet_segmentation
)

from .signature_extraction import (
    ShapeSignature,
    IncrementalSignature,
    compute_power_spectrum,
    compute_power_ratios,
    wigner_3j,
    compute_bispectrum_features,
    extract_shape_signature,
    incremental_signature_extraction,
    compare_signatures,
    SignatureDatabase
)

__all__ = [
    'HarmonicBasis',
    'HarmonicSpectrum',
    'FREQUENCY_BANDS',
    'spherical_harmonic',
    'compute_harmonic_basis',
    'analyze_shape_harmonics',
    'bacteria_harmonic_signature',
    'granule_harmonic_signature',
    'compare_bacteria_granule_harmonics',
    'map_harmonics_to_brain_frequencies',
    'DiscreteSphericalTransform',
    'HarmonicWavelet',
    'WaveletBasis',
    'WaveletDecomposition',
    'construct_mother_wavelet',
    'construct_scale_specific_wavelets',
    'detect_bacteria_features',
    'wavelet_transform',
    'FastWaveletTransform',
    'multi_scale_bacteria_analysis',
    'adaptive_wavelet_segmentation',
    'ShapeSignature',
    'IncrementalSignature',
    'compute_power_spectrum',
    'compute_power_ratios',
    'wigner_3j',
    'compute_bispectrum_features',
    'extract_shape_signature',
    'incremental_signature_extraction',
    'compare_signatures',
    'SignatureDatabase'
]