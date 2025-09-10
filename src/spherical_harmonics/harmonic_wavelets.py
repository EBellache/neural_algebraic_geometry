"""
Wavelets constructed from spherical harmonics on polytope vertices.

This module creates localized basis functions by combining spherical harmonics
at different scales. Each frequency band (delta through ripples) gets its own
wavelet family tuned to detect specific shape features. The key insight is that
bacteria create characteristic patterns in the alpha (8-13 Hz) and beta (13-30 Hz)
bands due to their elongated structure.

Mathematical foundations:
- Wavelets as scale-localized spherical harmonics
- Mother wavelets for each Platonic solid symmetry
- Multiresolution analysis on discrete polytope vertices
- Fast transforms using pre-computed wavelet matrices
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Dict, List, Tuple, Callable
from functools import partial
import numpy as np


class HarmonicWavelet(NamedTuple):
    """Wavelet constructed from spherical harmonics.
    
    Attributes:
        name: Wavelet identifier
        scale: Primary scale/frequency band
        harmonic_coeffs: Coefficients for Y_lm combination
        spatial_values: Wavelet values at polytope vertices
        localization: Spatial localization measure
        symmetry_group: Compatible polytope symmetry
    """
    name: str
    scale: int
    harmonic_coeffs: Dict[Tuple[int, int], complex]
    spatial_values: jnp.ndarray
    localization: float
    symmetry_group: str


class WaveletBasis(NamedTuple):
    """Complete wavelet basis for a polytope.
    
    Attributes:
        polytope_name: Name of underlying polytope
        vertices: (N, 3) vertex positions
        wavelets: List of wavelets at different scales
        transform_matrix: (N, N) fast transform matrix
        frequency_bands: Mapping of scales to brain frequencies
    """
    polytope_name: str
    vertices: jnp.ndarray
    wavelets: List[HarmonicWavelet]
    transform_matrix: jnp.ndarray
    frequency_bands: Dict[int, str]


class WaveletDecomposition(NamedTuple):
    """Decomposition of a shape into wavelet coefficients.
    
    Attributes:
        coefficients: Dict mapping wavelet names to coefficients
        scale_energies: Energy at each scale
        spatial_map: Spatial distribution of energy
        dominant_features: List of most active wavelets
    """
    coefficients: Dict[str, float]
    scale_energies: jnp.ndarray
    spatial_map: jnp.ndarray
    dominant_features: List[Tuple[str, float]]


# Frequency band definitions (matching brain rhythms)
FREQUENCY_BANDS = {
    0: "delta",     # 0.5-4 Hz: Overall size
    1: "theta",     # 4-8 Hz: Orientation
    2: "alpha",     # 8-13 Hz: Elongation (bacteria)
    3: "beta",      # 13-30 Hz: Angular features
    4: "gamma",     # 30-50 Hz: Fine structure
    5: "high_gamma", # 50-150 Hz: Texture
    6: "ripples"    # 150-250 Hz: Microscale
}


@jax.jit
def gaussian_window(l: int, scale: float) -> float:
    """Gaussian windowing function for scale localization.
    
    Args:
        l: Spherical harmonic degree
        scale: Target scale/frequency
        
    Returns:
        Window weight for this l at given scale
    """
    sigma = scale / 2.0
    return jnp.exp(-(l - scale)**2 / (2 * sigma**2))


def construct_mother_wavelet(scale: int,
                           max_l: int = 6,
                           symmetry: str = "icosahedral") -> Dict[Tuple[int, int], complex]:
    """Construct mother wavelet at given scale.
    
    Mother wavelets are prototypes that generate wavelet families
    through rotation and translation on the polytope.
    
    Args:
        scale: Primary scale (0-6 corresponding to frequency bands)
        max_l: Maximum spherical harmonic degree
        symmetry: Polytope symmetry to respect
        
    Returns:
        Dictionary of (l, m) -> coefficient mappings
    """
    coeffs = {}
    
    # Apply Gaussian windowing centered at scale
    for l in range(max_l + 1):
        weight = gaussian_window(l, scale)
        
        if weight > 0.01:  # Threshold for sparsity
            # Select m values based on symmetry
            if symmetry == "icosahedral":
                # Icosahedral: specific m values for A5 symmetry
                if l == 0:
                    coeffs[(l, 0)] = weight
                elif l == 6:
                    for m in [0, 5, -5]:
                        coeffs[(l, m)] = weight * np.exp(2j * np.pi * m / 12)
            elif symmetry == "octahedral":
                # Octahedral: m divisible by 4 for S4 symmetry
                for m in range(-l, l + 1):
                    if m % 4 == 0:
                        coeffs[(l, m)] = weight
            elif symmetry == "tetrahedral":
                # Tetrahedral: specific selection for A4
                if l in [0, 3, 4, 6]:
                    for m in range(-l, l + 1):
                        if is_tetrahedral_compatible(l, m):
                            coeffs[(l, m)] = weight
            else:
                # No symmetry constraint
                for m in range(-l, l + 1):
                    coeffs[(l, m)] = weight
    
    # Normalize
    norm = jnp.sqrt(sum(abs(c)**2 for c in coeffs.values()))
    if norm > 0:
        coeffs = {lm: c/norm for lm, c in coeffs.items()}
    
    return coeffs


def is_tetrahedral_compatible(l: int, m: int) -> bool:
    """Check if (l,m) is compatible with tetrahedral symmetry."""
    if l == 0:
        return m == 0
    elif l == 3:
        return m in [0, 3, -3]
    elif l == 4:
        return m % 4 == 0
    elif l == 6:
        return m in [0, 6, -6]
    return False


@jax.jit
def evaluate_wavelet(wavelet_coeffs: Dict[Tuple[int, int], complex],
                    harmonic_values: Dict[Tuple[int, int], jnp.ndarray]) -> jnp.ndarray:
    """Evaluate wavelet at polytope vertices.
    
    Args:
        wavelet_coeffs: Harmonic coefficients defining the wavelet
        harmonic_values: Pre-computed Y_lm values at vertices
        
    Returns:
        (N,) array of wavelet values at vertices
    """
    result = jnp.zeros(len(next(iter(harmonic_values.values()))))
    
    for (l, m), coeff in wavelet_coeffs.items():
        if (l, m) in harmonic_values:
            result += coeff * harmonic_values[(l, m)]
    
    return jnp.real(result)


def create_wavelet_family(mother_wavelet: HarmonicWavelet,
                         vertices: jnp.ndarray,
                         n_rotations: int = 6) -> List[HarmonicWavelet]:
    """Generate family of wavelets through symmetry operations.
    
    Args:
        mother_wavelet: Prototype wavelet
        vertices: Polytope vertices
        n_rotations: Number of rotated copies
        
    Returns:
        List of wavelets forming complete family
    """
    family = [mother_wavelet]
    
    # Generate rotated versions
    for i in range(1, n_rotations):
        angle = 2 * np.pi * i / n_rotations
        
        # Rotate harmonic coefficients
        rotated_coeffs = {}
        for (l, m), coeff in mother_wavelet.harmonic_coeffs.items():
            # Rotation adds phase e^{im*angle} to Y_l^m
            rotated_coeffs[(l, m)] = coeff * np.exp(1j * m * angle)
        
        # Create new wavelet
        rotated = HarmonicWavelet(
            name=f"{mother_wavelet.name}_rot{i}",
            scale=mother_wavelet.scale,
            harmonic_coeffs=rotated_coeffs,
            spatial_values=mother_wavelet.spatial_values,  # Will recompute
            localization=mother_wavelet.localization,
            symmetry_group=mother_wavelet.symmetry_group
        )
        family.append(rotated)
    
    return family


def construct_scale_specific_wavelets(vertices: jnp.ndarray,
                                    harmonic_basis_values: Dict[Tuple[int, int], jnp.ndarray],
                                    scales: List[int] = None) -> Dict[int, List[HarmonicWavelet]]:
    """Construct wavelets for each frequency scale.
    
    Args:
        vertices: Polytope vertices
        harmonic_basis_values: Pre-computed Y_lm values
        scales: List of scales (default: all frequency bands)
        
    Returns:
        Dictionary mapping scale to wavelet list
    """
    if scales is None:
        scales = list(range(7))  # All frequency bands
    
    wavelets_by_scale = {}
    
    for scale in scales:
        # Create mother wavelet for this scale
        mother_coeffs = construct_mother_wavelet(scale)
        
        # Evaluate spatial values
        spatial_values = evaluate_wavelet(mother_coeffs, harmonic_basis_values)
        
        # Compute localization (spatial concentration)
        localization = compute_spatial_localization(spatial_values)
        
        mother = HarmonicWavelet(
            name=f"{FREQUENCY_BANDS[scale]}_mother",
            scale=scale,
            harmonic_coeffs=mother_coeffs,
            spatial_values=spatial_values,
            localization=localization,
            symmetry_group="icosahedral"
        )
        
        # Generate family
        family = create_wavelet_family(mother, vertices)
        wavelets_by_scale[scale] = family
    
    return wavelets_by_scale


@jax.jit
def compute_spatial_localization(values: jnp.ndarray) -> float:
    """Compute spatial localization measure of wavelet.
    
    Higher values indicate more localized (concentrated) wavelets.
    
    Args:
        values: Wavelet values at vertices
        
    Returns:
        Localization score
    """
    # Use ratio of L4 to L2 norm (kurtosis-like)
    l2_norm = jnp.sqrt(jnp.sum(values**2))
    l4_norm = jnp.sqrt(jnp.sqrt(jnp.sum(values**4)))
    
    if l2_norm > 0:
        localization = l4_norm / l2_norm
    else:
        localization = 0.0
    
    return localization


def detect_bacteria_features(shape_values: jnp.ndarray,
                           wavelet_basis: WaveletBasis) -> Dict[str, jnp.ndarray]:
    """Detect bacteria-specific features using wavelets.
    
    Bacteria have characteristic signatures:
    - Strong alpha band (elongation)
    - Moderate beta band (angular features)
    - Low delta band (not spherical)
    
    Args:
        shape_values: Function values at vertices
        wavelet_basis: Complete wavelet basis
        
    Returns:
        Dictionary of detected features
    """
    features = {}
    
    # Decompose into wavelets
    decomposition = wavelet_transform(shape_values, wavelet_basis)
    
    # Alpha band analysis (elongation)
    alpha_wavelets = [w for w in wavelet_basis.wavelets if w.scale == 2]
    alpha_coeffs = [abs(decomposition.coefficients.get(w.name, 0)) for w in alpha_wavelets]
    features['elongation_strength'] = jnp.max(jnp.array(alpha_coeffs))
    features['elongation_direction'] = jnp.argmax(jnp.array(alpha_coeffs))
    
    # Beta band analysis (angular features)
    beta_wavelets = [w for w in wavelet_basis.wavelets if w.scale == 3]
    beta_coeffs = [abs(decomposition.coefficients.get(w.name, 0)) for w in beta_wavelets]
    features['angular_complexity'] = jnp.mean(jnp.array(beta_coeffs))
    
    # Delta/alpha ratio (sphericity measure)
    delta_energy = decomposition.scale_energies[0]
    alpha_energy = decomposition.scale_energies[2]
    features['sphericity'] = delta_energy / (alpha_energy + 1e-10)
    
    # Spatial localization of features
    features['spatial_concentration'] = decomposition.spatial_map
    
    # Bacteria score combining features
    bacteria_score = (
        features['elongation_strength'] * 2.0 +
        features['angular_complexity'] * 1.0 -
        features['sphericity'] * 1.5
    )
    features['bacteria_likelihood'] = jax.nn.sigmoid(bacteria_score)
    
    return features


def wavelet_transform(shape_values: jnp.ndarray,
                     wavelet_basis: WaveletBasis) -> WaveletDecomposition:
    """Perform wavelet transform on shape.
    
    Args:
        shape_values: Function values at vertices
        wavelet_basis: Complete wavelet basis
        
    Returns:
        WaveletDecomposition with coefficients and analysis
    """
    coefficients = {}
    
    # Compute coefficients for each wavelet
    for wavelet in wavelet_basis.wavelets:
        # Inner product with shape
        coeff = jnp.sum(shape_values * wavelet.spatial_values)
        coefficients[wavelet.name] = float(coeff)
    
    # Scale-wise energy
    scale_energies = jnp.zeros(7)
    for wavelet in wavelet_basis.wavelets:
        energy = coefficients[wavelet.name]**2
        scale_energies = scale_energies.at[wavelet.scale].add(energy)
    
    # Spatial energy map
    spatial_map = jnp.zeros_like(shape_values)
    for wavelet in wavelet_basis.wavelets:
        contribution = coefficients[wavelet.name] * wavelet.spatial_values
        spatial_map += contribution**2
    
    # Find dominant features
    sorted_coeffs = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
    dominant_features = sorted_coeffs[:5]
    
    return WaveletDecomposition(
        coefficients=coefficients,
        scale_energies=scale_energies,
        spatial_map=spatial_map,
        dominant_features=dominant_features
    )


def create_fast_wavelet_transform(vertices: jnp.ndarray,
                                wavelets: List[HarmonicWavelet]) -> jnp.ndarray:
    """Create fast transform matrix for wavelet analysis.
    
    Pre-computes the transform matrix W where:
    coefficients = W @ shape_values
    
    Args:
        vertices: Polytope vertices
        wavelets: List of wavelets
        
    Returns:
        (M, N) transform matrix where M = number of wavelets, N = vertices
    """
    n_vertices = len(vertices)
    n_wavelets = len(wavelets)
    
    W = jnp.zeros((n_wavelets, n_vertices))
    
    for i, wavelet in enumerate(wavelets):
        W = W.at[i, :].set(wavelet.spatial_values)
    
    # Normalize rows for stable transform
    row_norms = jnp.linalg.norm(W, axis=1, keepdims=True)
    W = W / (row_norms + 1e-10)
    
    return W


class FastWaveletTransform:
    """Fast wavelet transform using pre-computed matrices."""
    
    def __init__(self, wavelet_basis: WaveletBasis):
        """Initialize with wavelet basis.
        
        Args:
            wavelet_basis: Complete wavelet basis
        """
        self.basis = wavelet_basis
        self.W = create_fast_wavelet_transform(
            wavelet_basis.vertices,
            wavelet_basis.wavelets
        )
        self.W_inv = jnp.linalg.pinv(self.W)
    
    @partial(jax.jit, static_argnums=(0,))
    def forward(self, shape_values: jnp.ndarray) -> jnp.ndarray:
        """Fast forward wavelet transform.
        
        Args:
            shape_values: (N,) function values
            
        Returns:
            (M,) wavelet coefficients
        """
        return jnp.dot(self.W, shape_values)
    
    @partial(jax.jit, static_argnums=(0,))
    def inverse(self, coefficients: jnp.ndarray) -> jnp.ndarray:
        """Fast inverse wavelet transform.
        
        Args:
            coefficients: (M,) wavelet coefficients
            
        Returns:
            (N,) reconstructed function values
        """
        return jnp.dot(self.W_inv, coefficients)
    
    def denoise(self, shape_values: jnp.ndarray, threshold: float = 0.1) -> jnp.ndarray:
        """Denoise shape by wavelet thresholding.
        
        Args:
            shape_values: Noisy shape values
            threshold: Coefficient threshold
            
        Returns:
            Denoised shape values
        """
        # Forward transform
        coeffs = self.forward(shape_values)
        
        # Soft thresholding
        coeffs_thresh = jnp.sign(coeffs) * jnp.maximum(jnp.abs(coeffs) - threshold, 0)
        
        # Inverse transform
        return self.inverse(coeffs_thresh)
    
    def extract_scale(self, shape_values: jnp.ndarray, scale: int) -> jnp.ndarray:
        """Extract specific frequency scale from shape.
        
        Args:
            shape_values: Input shape
            scale: Target scale (0-6)
            
        Returns:
            Shape component at given scale
        """
        coeffs = self.forward(shape_values)
        
        # Zero out other scales
        scale_mask = jnp.zeros_like(coeffs)
        for i, wavelet in enumerate(self.basis.wavelets):
            if wavelet.scale == scale:
                scale_mask = scale_mask.at[i].set(1.0)
        
        coeffs_filtered = coeffs * scale_mask
        
        return self.inverse(coeffs_filtered)


def multi_scale_bacteria_analysis(shape_values: jnp.ndarray,
                                 wavelet_basis: WaveletBasis) -> Dict[str, jnp.ndarray]:
    """Analyze bacteria shape at multiple scales.
    
    Args:
        shape_values: Shape function values
        wavelet_basis: Wavelet basis
        
    Returns:
        Multi-scale analysis results
    """
    fwt = FastWaveletTransform(wavelet_basis)
    analysis = {}
    
    # Extract each scale
    for scale in range(7):
        scale_component = fwt.extract_scale(shape_values, scale)
        band_name = FREQUENCY_BANDS[scale]
        
        analysis[f'{band_name}_component'] = scale_component
        analysis[f'{band_name}_energy'] = jnp.sum(scale_component**2)
    
    # Compute scale interactions
    alpha_component = analysis['alpha_component']
    beta_component = analysis['beta_component']
    
    # Alpha-beta coherence (elongation with angular features)
    coherence = jnp.abs(jnp.sum(alpha_component * beta_component))
    analysis['alpha_beta_coherence'] = coherence
    
    # Multi-scale bacteria score
    bacteria_score = (
        2.0 * analysis['alpha_energy'] +
        1.5 * analysis['beta_energy'] +
        1.0 * coherence -
        3.0 * analysis['delta_energy']
    )
    analysis['multi_scale_bacteria_score'] = bacteria_score
    
    return analysis


def adaptive_wavelet_segmentation(shape_values: jnp.ndarray,
                                wavelet_basis: WaveletBasis,
                                noise_level: float = 0.1) -> Dict[str, jnp.ndarray]:
    """Adaptive segmentation using wavelets.
    
    Args:
        shape_values: Input shape
        wavelet_basis: Wavelet basis
        noise_level: Estimated noise level
        
    Returns:
        Segmentation results
    """
    fwt = FastWaveletTransform(wavelet_basis)
    
    # Denoise first
    denoised = fwt.denoise(shape_values, threshold=noise_level)
    
    # Multi-scale analysis
    scale_analysis = multi_scale_bacteria_analysis(denoised, wavelet_basis)
    
    # Detect features
    features = detect_bacteria_features(denoised, wavelet_basis)
    
    # Segment based on dominant scales
    if scale_analysis['multi_scale_bacteria_score'] > 0:
        # Bacteria-like: use alpha and beta scales
        segmented = (
            scale_analysis['alpha_component'] +
            0.5 * scale_analysis['beta_component']
        )
        object_type = 'bacteria'
    else:
        # Granule-like: use delta scale
        segmented = scale_analysis['delta_component']
        object_type = 'granule'
    
    return {
        'segmented_shape': segmented,
        'object_type': object_type,
        'confidence': features['bacteria_likelihood'] if object_type == 'bacteria' else 1 - features['bacteria_likelihood'],
        'denoised_shape': denoised,
        'features': features,
        'scale_analysis': scale_analysis
    }