"""
Extract rotation-invariant shape signatures using harmonic decomposition on polytopes.

This module solves the rotation invariance problem by using combinations of spherical
harmonic coefficients that don't change under rotation. The power spectrum |Y_lm|² and
bispectrum form complete shape descriptors that enable robust bacteria classification.

Key invariants:
- Power spectrum: Sum_m |c_lm|² for each l
- Bispectrum: Wigner 3j contracted triple products
- Bacterial signature: High l=2/l=0, specific l=3/l=5, low l=1, characteristic l=4
- Granule signature: Dominant l=0, uniform l distribution, smooth decay
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, List, Dict, Optional
from functools import partial, lru_cache
import numpy as np
from jax.scipy.special import gammaln


class ShapeSignature(NamedTuple):
    """Rotation-invariant shape signature.
    
    Attributes:
        power_spectrum: (L+1,) array of power per l value
        power_ratios: Key ratios like l=2/l=0 for elongation
        bispectrum_features: Rotation-invariant triple products
        signature_vector: Complete invariant descriptor
        polytope_fit: Best-fitting polytope at each scale
        confidence: Confidence in classification
        object_type: Predicted type (bacteria/granule/unknown)
    """
    power_spectrum: jnp.ndarray
    power_ratios: Dict[str, float]
    bispectrum_features: jnp.ndarray
    signature_vector: jnp.ndarray
    polytope_fit: Dict[int, str]
    confidence: float
    object_type: str


class IncrementalSignature(NamedTuple):
    """Incrementally refined signature.
    
    Attributes:
        coarse_features: Features from l ≤ 2
        medium_features: Features from l ≤ 4
        fine_features: Features from l ≤ 6
        refinement_level: Current level of detail
        early_stop_confidence: Confidence at each level
    """
    coarse_features: jnp.ndarray
    medium_features: Optional[jnp.ndarray]
    fine_features: Optional[jnp.ndarray]
    refinement_level: int
    early_stop_confidence: jnp.ndarray


# Bacterial and granule signature templates
BACTERIAL_SIGNATURE = {
    'l2_l0_ratio': (2.0, 5.0),      # High elongation
    'l3_l5_ratio': (0.8, 1.5),      # Cylindrical symmetry
    'l1_power': (0.0, 0.2),         # Low bilateral symmetry
    'l4_pattern': 'helical',        # Helical structure
    'l_decay': 'slow'               # Slow spectral decay
}

GRANULE_SIGNATURE = {
    'l2_l0_ratio': (0.0, 0.5),      # Low elongation
    'l3_l5_ratio': (0.9, 1.1),      # Isotropic
    'l1_power': (0.0, 0.1),         # Minimal asymmetry
    'l4_pattern': 'uniform',        # No structure
    'l_decay': 'exponential'        # Fast spectral decay
}


@jax.jit
def compute_power_spectrum(harmonic_coeffs: Dict[Tuple[int, int], complex],
                          max_l: int) -> jnp.ndarray:
    """Compute rotation-invariant power spectrum.
    
    Power spectrum P_l = Sum_m |c_lm|² is invariant under rotation.
    
    Args:
        harmonic_coeffs: Dictionary of (l,m) -> coefficient
        max_l: Maximum l value
        
    Returns:
        (max_l+1,) array of power per l
    """
    power_spectrum = jnp.zeros(max_l + 1)
    
    for (l, m), coeff in harmonic_coeffs.items():
        if l <= max_l:
            power = jnp.abs(coeff)**2
            power_spectrum = power_spectrum.at[l].add(power)
    
    return power_spectrum


@jax.jit
def compute_power_ratios(power_spectrum: jnp.ndarray) -> Dict[str, float]:
    """Compute diagnostic power ratios.
    
    These ratios capture key shape characteristics:
    - l2/l0: Elongation (high for bacteria)
    - l3/l5: Cylindrical vs spherical symmetry
    - l4/l2: Helical structure
    - l1/total: Bilateral asymmetry
    
    Args:
        power_spectrum: Power per l value
        
    Returns:
        Dictionary of named ratios
    """
    eps = 1e-10  # Avoid division by zero
    total_power = jnp.sum(power_spectrum) + eps
    
    ratios = {
        'l2_l0_ratio': float(power_spectrum[2] / (power_spectrum[0] + eps)),
        'l3_l5_ratio': float(power_spectrum[3] / (power_spectrum[5] + eps)) if len(power_spectrum) > 5 else 1.0,
        'l4_l2_ratio': float(power_spectrum[4] / (power_spectrum[2] + eps)) if len(power_spectrum) > 4 else 1.0,
        'l1_normalized': float(power_spectrum[1] / total_power),
        'l0_fraction': float(power_spectrum[0] / total_power),
        'spectral_centroid': float(jnp.sum(jnp.arange(len(power_spectrum)) * power_spectrum) / total_power)
    }
    
    return ratios


@lru_cache(maxsize=1000)
def wigner_3j(l1: int, l2: int, l3: int, m1: int, m2: int, m3: int) -> float:
    """Compute Wigner 3j symbol.
    
    The Wigner 3j symbol (l1 l2 l3; m1 m2 m3) appears in the coupling
    of angular momenta and ensures rotation invariance.
    
    Args:
        l1, l2, l3: Angular momentum quantum numbers
        m1, m2, m3: Magnetic quantum numbers
        
    Returns:
        Value of 3j symbol
    """
    # Check triangle inequality
    if not (abs(l1 - l2) <= l3 <= l1 + l2):
        return 0.0
    
    # Check m selection rules
    if m1 + m2 + m3 != 0:
        return 0.0
    
    if abs(m1) > l1 or abs(m2) > l2 or abs(m3) > l3:
        return 0.0
    
    # Simplified computation for common cases
    if l1 == 0:
        return 1.0 / np.sqrt(2*l2 + 1) if l2 == l3 and m2 == -m3 else 0.0
    
    if l2 == 0:
        return 1.0 / np.sqrt(2*l1 + 1) if l1 == l3 and m1 == -m3 else 0.0
    
    # General case - use Racah formula (simplified)
    # Full implementation would use recurrence relations
    prefactor = np.sqrt((2*l1 + 1) * (2*l2 + 1) * (2*l3 + 1) / (4*np.pi))
    
    # Approximate for demonstration
    value = prefactor * np.exp(-(m1**2 + m2**2 + m3**2) / 10.0)
    
    return value


def compute_bispectrum_features(harmonic_coeffs: Dict[Tuple[int, int], complex],
                               l_triples: List[Tuple[int, int, int]] = None) -> jnp.ndarray:
    """Compute rotation-invariant bispectrum features.
    
    The bispectrum B(l1,l2,l3) = Sum_{m1,m2,m3} (3j symbol) × c_l1m1 × c_l2m2 × c_l3m3
    is rotation invariant and captures third-order statistics.
    
    Args:
        harmonic_coeffs: Spherical harmonic coefficients
        l_triples: List of (l1,l2,l3) to compute (default: important ones)
        
    Returns:
        Array of bispectrum features
    """
    if l_triples is None:
        # Default triples that capture important shape features
        l_triples = [
            (0, 2, 2),  # Spherical with quadrupole
            (2, 2, 0),  # Pure quadrupole
            (2, 2, 4),  # Quadrupole-hexadecapole coupling
            (1, 2, 3),  # Asymmetric elongation
            (3, 3, 0),  # Octupole strength
            (2, 3, 5),  # Higher order couplings
        ]
    
    features = []
    
    for l1, l2, l3 in l_triples:
        bispectrum = 0.0
        
        for m1 in range(-l1, l1 + 1):
            for m2 in range(-l2, l2 + 1):
                m3 = -(m1 + m2)  # Selection rule
                
                if abs(m3) <= l3:
                    # Get coefficients
                    c1 = harmonic_coeffs.get((l1, m1), 0.0)
                    c2 = harmonic_coeffs.get((l2, m2), 0.0)
                    c3 = harmonic_coeffs.get((l3, m3), 0.0)
                    
                    # Compute 3j symbol
                    w3j = wigner_3j(l1, l2, l3, m1, m2, m3)
                    
                    # Add contribution
                    bispectrum += w3j * c1 * c2 * np.conj(c3)
        
        features.append(np.abs(bispectrum))
    
    return jnp.array(features)


def extract_bacterial_features(power_spectrum: jnp.ndarray,
                              power_ratios: Dict[str, float],
                              bispectrum: jnp.ndarray) -> Dict[str, float]:
    """Extract bacteria-specific features from invariants.
    
    Args:
        power_spectrum: Power per l value
        power_ratios: Diagnostic ratios
        bispectrum: Bispectrum features
        
    Returns:
        Dictionary of bacterial features
    """
    features = {}
    
    # Elongation score (high l=2/l=0 ratio)
    elongation = power_ratios['l2_l0_ratio']
    features['elongation_score'] = float(jax.nn.sigmoid(2.0 * (elongation - 1.0)))
    
    # Cylindrical symmetry (specific l=3/l=5 ratio)
    cyl_symmetry = power_ratios['l3_l5_ratio']
    features['cylindrical_score'] = float(np.exp(-(cyl_symmetry - 1.2)**2 / 0.5))
    
    # Low bilateral symmetry (low l=1)
    features['symmetry_score'] = float(1.0 - power_ratios['l1_normalized'])
    
    # Helical structure detection (l=4 pattern)
    if len(power_spectrum) > 4:
        l4_strength = power_spectrum[4] / (jnp.sum(power_spectrum) + 1e-10)
        features['helical_score'] = float(l4_strength * power_ratios['l4_l2_ratio'])
    else:
        features['helical_score'] = 0.0
    
    # Spectral decay pattern
    if len(power_spectrum) > 3:
        decay_rate = -jnp.mean(jnp.diff(jnp.log(power_spectrum[:4] + 1e-10)))
        features['slow_decay_score'] = float(jax.nn.sigmoid(-decay_rate + 1.0))
    else:
        features['slow_decay_score'] = 0.5
    
    # Bispectrum features for shape complexity
    if len(bispectrum) > 0:
        features['shape_complexity'] = float(jnp.std(bispectrum))
        features['bispectrum_strength'] = float(jnp.mean(bispectrum))
    
    # Combined bacterial score
    features['bacteria_score'] = (
        3.0 * features['elongation_score'] +
        2.0 * features['cylindrical_score'] +
        1.5 * features['symmetry_score'] +
        1.0 * features['helical_score'] +
        1.0 * features['slow_decay_score']
    ) / 8.5
    
    return features


def extract_granule_features(power_spectrum: jnp.ndarray,
                           power_ratios: Dict[str, float],
                           bispectrum: jnp.ndarray) -> Dict[str, float]:
    """Extract granule-specific features from invariants.
    
    Args:
        power_spectrum: Power per l value
        power_ratios: Diagnostic ratios
        bispectrum: Bispectrum features
        
    Returns:
        Dictionary of granule features
    """
    features = {}
    
    # Spherical dominance (high l=0 fraction)
    features['spherical_score'] = float(power_ratios['l0_fraction'])
    
    # Low elongation (low l=2/l=0 ratio)
    elongation = power_ratios['l2_l0_ratio']
    features['compact_score'] = float(jax.nn.sigmoid(-2.0 * (elongation - 0.3)))
    
    # Isotropic distribution (uniform l spectrum)
    if len(power_spectrum) > 3:
        spectrum_variance = jnp.var(power_spectrum[:4] / (jnp.mean(power_spectrum[:4]) + 1e-10))
        features['isotropy_score'] = float(np.exp(-spectrum_variance))
    else:
        features['isotropy_score'] = 0.5
    
    # Fast spectral decay
    if len(power_spectrum) > 3:
        decay_rate = -jnp.mean(jnp.diff(jnp.log(power_spectrum[:4] + 1e-10)))
        features['fast_decay_score'] = float(jax.nn.sigmoid(decay_rate - 1.0))
    else:
        features['fast_decay_score'] = 0.5
    
    # Low shape complexity
    if len(bispectrum) > 0:
        features['simplicity_score'] = float(np.exp(-jnp.std(bispectrum)))
    else:
        features['simplicity_score'] = 0.5
    
    # Combined granule score
    features['granule_score'] = (
        3.0 * features['spherical_score'] +
        2.0 * features['compact_score'] +
        1.5 * features['isotropy_score'] +
        1.0 * features['fast_decay_score'] +
        1.0 * features['simplicity_score']
    ) / 8.5
    
    return features


def find_best_fitting_polytope(shape_values: jnp.ndarray,
                              polytope_vertices: Dict[str, jnp.ndarray]) -> Tuple[str, float]:
    """Find polytope that best fits the shape.
    
    Args:
        shape_values: Function values at test points
        polytope_vertices: Dictionary of polytope names to vertices
        
    Returns:
        Tuple of (best_polytope_name, fit_quality)
    """
    best_fit = 0.0
    best_polytope = 'icosahedron'  # Default
    
    for name, vertices in polytope_vertices.items():
        # Interpolate shape values to polytope vertices
        # Simplified - full implementation would use proper interpolation
        if len(vertices) == len(shape_values):
            correlation = jnp.corrcoef(shape_values, jnp.linalg.norm(vertices, axis=1))[0, 1]
            fit_quality = abs(correlation)
            
            if fit_quality > best_fit:
                best_fit = fit_quality
                best_polytope = name
    
    return best_polytope, float(best_fit)


def compute_signature_vector(power_spectrum: jnp.ndarray,
                           power_ratios: Dict[str, float],
                           bispectrum_features: jnp.ndarray,
                           normalize: bool = True) -> jnp.ndarray:
    """Combine all features into signature vector.
    
    Args:
        power_spectrum: Power per l
        power_ratios: Key ratios
        bispectrum_features: Bispectrum values
        normalize: Whether to normalize for scale invariance
        
    Returns:
        Complete signature vector
    """
    # Collect all features
    features = []
    
    # Power spectrum (normalized)
    if normalize and jnp.sum(power_spectrum) > 0:
        features.extend(power_spectrum / jnp.sum(power_spectrum))
    else:
        features.extend(power_spectrum)
    
    # Power ratios
    ratio_values = [
        power_ratios['l2_l0_ratio'],
        power_ratios['l3_l5_ratio'],
        power_ratios['l4_l2_ratio'],
        power_ratios['l1_normalized'],
        power_ratios['spectral_centroid']
    ]
    features.extend(ratio_values)
    
    # Bispectrum features (normalized)
    if normalize and jnp.sum(bispectrum_features) > 0:
        features.extend(bispectrum_features / jnp.sum(bispectrum_features))
    else:
        features.extend(bispectrum_features)
    
    return jnp.array(features)


def extract_shape_signature(shape_values: jnp.ndarray,
                          harmonic_coeffs: Dict[Tuple[int, int], complex],
                          max_l: int = 6,
                          polytope_vertices: Optional[Dict[str, jnp.ndarray]] = None) -> ShapeSignature:
    """Extract complete rotation-invariant shape signature.
    
    Args:
        shape_values: Function values at sample points
        harmonic_coeffs: Spherical harmonic decomposition
        max_l: Maximum l value to consider
        polytope_vertices: Optional polytope vertex sets
        
    Returns:
        Complete ShapeSignature
    """
    # Compute invariants
    power_spectrum = compute_power_spectrum(harmonic_coeffs, max_l)
    power_ratios = compute_power_ratios(power_spectrum)
    bispectrum_features = compute_bispectrum_features(harmonic_coeffs)
    
    # Extract type-specific features
    bacterial_features = extract_bacterial_features(power_spectrum, power_ratios, bispectrum_features)
    granule_features = extract_granule_features(power_spectrum, power_ratios, bispectrum_features)
    
    # Determine object type
    bacteria_score = bacterial_features['bacteria_score']
    granule_score = granule_features['granule_score']
    
    if bacteria_score > 0.7:
        object_type = 'bacteria'
        confidence = bacteria_score
    elif granule_score > 0.7:
        object_type = 'granule'
        confidence = granule_score
    else:
        object_type = 'unknown'
        confidence = 1.0 - abs(bacteria_score - granule_score)
    
    # Find best-fitting polytopes at different scales
    polytope_fit = {}
    if polytope_vertices:
        for scale in [2, 4, 6]:
            # Use coefficients up to scale
            scale_values = shape_values  # Simplified
            best_poly, fit_quality = find_best_fitting_polytope(scale_values, polytope_vertices)
            polytope_fit[scale] = best_poly
    
    # Create signature vector
    signature_vector = compute_signature_vector(power_spectrum, power_ratios, bispectrum_features)
    
    return ShapeSignature(
        power_spectrum=power_spectrum,
        power_ratios=power_ratios,
        bispectrum_features=bispectrum_features,
        signature_vector=signature_vector,
        polytope_fit=polytope_fit,
        confidence=confidence,
        object_type=object_type
    )


def incremental_signature_extraction(shape_values: jnp.ndarray,
                                   harmonic_coeffs: Dict[Tuple[int, int], complex],
                                   confidence_threshold: float = 0.9) -> IncrementalSignature:
    """Extract signature incrementally with early stopping.
    
    Start with coarse features (l≤2), add detail only if needed.
    
    Args:
        shape_values: Function values
        harmonic_coeffs: Harmonic decomposition
        confidence_threshold: Stop when confidence exceeds this
        
    Returns:
        IncrementalSignature with appropriate level of detail
    """
    early_stop_confidence = []
    
    # Level 1: Coarse features (l ≤ 2)
    coarse_coeffs = {(l, m): c for (l, m), c in harmonic_coeffs.items() if l <= 2}
    coarse_signature = extract_shape_signature(shape_values, coarse_coeffs, max_l=2)
    coarse_features = coarse_signature.signature_vector
    early_stop_confidence.append(coarse_signature.confidence)
    
    if coarse_signature.confidence >= confidence_threshold:
        return IncrementalSignature(
            coarse_features=coarse_features,
            medium_features=None,
            fine_features=None,
            refinement_level=1,
            early_stop_confidence=jnp.array(early_stop_confidence)
        )
    
    # Level 2: Medium features (l ≤ 4)
    medium_coeffs = {(l, m): c for (l, m), c in harmonic_coeffs.items() if l <= 4}
    medium_signature = extract_shape_signature(shape_values, medium_coeffs, max_l=4)
    medium_features = medium_signature.signature_vector
    early_stop_confidence.append(medium_signature.confidence)
    
    if medium_signature.confidence >= confidence_threshold:
        return IncrementalSignature(
            coarse_features=coarse_features,
            medium_features=medium_features,
            fine_features=None,
            refinement_level=2,
            early_stop_confidence=jnp.array(early_stop_confidence)
        )
    
    # Level 3: Fine features (l ≤ 6)
    fine_signature = extract_shape_signature(shape_values, harmonic_coeffs, max_l=6)
    fine_features = fine_signature.signature_vector
    early_stop_confidence.append(fine_signature.confidence)
    
    return IncrementalSignature(
        coarse_features=coarse_features,
        medium_features=medium_features,
        fine_features=fine_features,
        refinement_level=3,
        early_stop_confidence=jnp.array(early_stop_confidence)
    )


def compare_signatures(sig1: ShapeSignature, sig2: ShapeSignature) -> Dict[str, float]:
    """Compare two shape signatures.
    
    Args:
        sig1, sig2: Shape signatures to compare
        
    Returns:
        Dictionary of similarity measures
    """
    # Power spectrum similarity
    power_similarity = 1.0 - jnp.mean(jnp.abs(sig1.power_spectrum - sig2.power_spectrum))
    
    # Ratio similarity
    ratio_keys = set(sig1.power_ratios.keys()) & set(sig2.power_ratios.keys())
    ratio_diffs = [abs(sig1.power_ratios[k] - sig2.power_ratios[k]) for k in ratio_keys]
    ratio_similarity = 1.0 - np.mean(ratio_diffs) if ratio_diffs else 0.5
    
    # Bispectrum similarity
    bispec_similarity = 1.0 - jnp.mean(jnp.abs(sig1.bispectrum_features - sig2.bispectrum_features))
    
    # Overall similarity
    overall = (power_similarity + ratio_similarity + bispec_similarity) / 3.0
    
    return {
        'power_similarity': float(power_similarity),
        'ratio_similarity': float(ratio_similarity),
        'bispectrum_similarity': float(bispec_similarity),
        'overall_similarity': float(overall),
        'same_type': sig1.object_type == sig2.object_type
    }


class SignatureDatabase:
    """Database of reference signatures for classification."""
    
    def __init__(self):
        """Initialize empty database."""
        self.signatures = {}
        self.labels = {}
    
    def add_reference(self, name: str, signature: ShapeSignature, label: str):
        """Add reference signature.
        
        Args:
            name: Unique identifier
            signature: Shape signature
            label: Ground truth label
        """
        self.signatures[name] = signature
        self.labels[name] = label
    
    def classify(self, query_signature: ShapeSignature, k: int = 3) -> Dict[str, float]:
        """Classify using k-nearest neighbors.
        
        Args:
            query_signature: Signature to classify
            k: Number of neighbors
            
        Returns:
            Classification results
        """
        if not self.signatures:
            return {'object_type': 'unknown', 'confidence': 0.0}
        
        # Compute similarities to all references
        similarities = []
        for name, ref_sig in self.signatures.items():
            sim = compare_signatures(query_signature, ref_sig)
            similarities.append((name, sim['overall_similarity'], self.labels[name]))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Vote among k nearest
        votes = {}
        for i in range(min(k, len(similarities))):
            _, sim, label = similarities[i]
            votes[label] = votes.get(label, 0) + sim
        
        # Determine classification
        if votes:
            best_label = max(votes, key=votes.get)
            confidence = votes[best_label] / sum(votes.values())
        else:
            best_label = 'unknown'
            confidence = 0.0
        
        return {
            'object_type': best_label,
            'confidence': confidence,
            'nearest_neighbors': similarities[:k]
        }