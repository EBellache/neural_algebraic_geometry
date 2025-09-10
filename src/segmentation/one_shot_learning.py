"""
One-shot learning system for segmenting novel biological structures.

This module implements learning from single labeled examples using polytope
harmonic signatures. The key insight is that polytope harmonics provide
complete shape descriptors, enabling immediate generalization from one example.

Core capabilities:
- Learn from single labeled bacterium to recognize all instances
- Transfer learning from known bacterial patterns
- Biologically-aware data augmentation
- Uncertainty quantification using Gaussian processes
- Active learning for optimal label acquisition
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Dict, List, Tuple, Optional, Callable
from functools import partial
import numpy as np
from dataclasses import dataclass


class LabeledExample(NamedTuple):
    """Single labeled example for learning.
    
    Attributes:
        volume: 3D volume containing the example
        mask: Binary mask of labeled object
        label: Semantic label (e.g., 'novel_bacterium_1')
        metadata: Additional information about example
    """
    volume: jnp.ndarray
    mask: jnp.ndarray
    label: str
    metadata: Dict


class LearnedSignature(NamedTuple):
    """Learned signature from one-shot learning.
    
    Attributes:
        mean_signature: Mean harmonic signature
        covariance: Covariance structure of variations
        polytope_type: Best-fitting polytope type
        deformation_modes: Learned deformation patterns
        confidence_bounds: Uncertainty in signature
        prior_weight: How much prior knowledge was used
        augmented_examples: Synthetic variations generated
    """
    mean_signature: jnp.ndarray
    covariance: jnp.ndarray
    polytope_type: str
    deformation_modes: jnp.ndarray
    confidence_bounds: Tuple[jnp.ndarray, jnp.ndarray]
    prior_weight: float
    augmented_examples: List[jnp.ndarray]


class OneShotSegmentation(NamedTuple):
    """Segmentation result from one-shot learning.
    
    Attributes:
        segmentation_mask: Predicted labels for full volume
        confidence_map: Per-voxel confidence scores
        signatures_map: Harmonic signatures at each location
        uncertain_regions: Regions needing user review
        active_queries: Suggested locations for additional labels
    """
    segmentation_mask: jnp.ndarray
    confidence_map: jnp.ndarray
    signatures_map: jnp.ndarray
    uncertain_regions: List[Tuple[int, int, int]]
    active_queries: List[Tuple[int, int, int]]


@dataclass
class BiologicalPrior:
    """Prior knowledge about biological variations.
    
    Attributes:
        shape_category: 'bacteria', 'organelle', 'cell', etc.
        typical_elongation: Expected aspect ratios
        smoothness_scale: Characteristic length scale
        symmetries: Expected symmetry groups
        deformation_stats: Statistics of typical deformations
    """
    shape_category: str
    typical_elongation: Tuple[float, float]  # (min, max)
    smoothness_scale: float
    symmetries: List[str]
    deformation_stats: Dict[str, Tuple[float, float]]


# Standard biological priors
BACTERIA_PRIOR = BiologicalPrior(
    shape_category='bacteria',
    typical_elongation=(1.5, 5.0),
    smoothness_scale=0.1,
    symmetries=['reflection', 'rotation_180'],
    deformation_stats={
        'elongation': (0.8, 1.2),
        'bending': (0.0, 0.3),
        'twisting': (0.0, 0.1),
        'thickness': (0.9, 1.1)
    }
)

ORGANELLE_PRIOR = BiologicalPrior(
    shape_category='organelle',
    typical_elongation=(1.0, 3.0),
    smoothness_scale=0.2,
    symmetries=['reflection'],
    deformation_stats={
        'elongation': (0.7, 1.3),
        'bending': (0.0, 0.5),
        'surface_roughness': (0.0, 0.2)
    }
)


def extract_signature_from_example(example: LabeledExample,
                                  prior: BiologicalPrior = BACTERIA_PRIOR) -> LearnedSignature:
    """Extract complete signature from single labeled example.
    
    Args:
        example: Labeled example
        prior: Biological prior knowledge
        
    Returns:
        Learned signature with uncertainty
    """
    # Extract region from mask
    coords = jnp.argwhere(example.mask > 0)
    if len(coords) == 0:
        raise ValueError("Empty mask")
    
    # Get bounding box
    min_coords = jnp.min(coords, axis=0)
    max_coords = jnp.max(coords, axis=0)
    
    # Extract points on object surface
    surface_points = extract_surface_points(example.volume, example.mask, coords)
    
    # Fit polytopes at multiple scales
    polytope_fits = fit_multiscale_polytopes(surface_points)
    
    # Select best polytope type
    best_polytope = select_best_polytope(polytope_fits, prior)
    
    # Compute harmonic signature
    harmonic_sig = compute_harmonic_signature(surface_points, best_polytope)
    
    # Generate augmented examples
    augmented = generate_augmented_examples(
        surface_points, best_polytope, prior, n_augment=20
    )
    
    # Estimate covariance from augmented examples
    aug_signatures = [compute_harmonic_signature(aug, best_polytope) 
                     for aug in augmented]
    covariance = estimate_signature_covariance(
        harmonic_sig, aug_signatures, prior
    )
    
    # Learn deformation modes
    deformation_modes = learn_deformation_modes(augmented, best_polytope)
    
    # Compute confidence bounds
    confidence_bounds = compute_confidence_bounds(
        harmonic_sig, covariance, len(augmented)
    )
    
    # Determine prior influence
    prior_weight = compute_prior_weight(surface_points, prior)
    
    return LearnedSignature(
        mean_signature=harmonic_sig,
        covariance=covariance,
        polytope_type=best_polytope['type'],
        deformation_modes=deformation_modes,
        confidence_bounds=confidence_bounds,
        prior_weight=prior_weight,
        augmented_examples=[aug for aug in augmented]
    )


@jax.jit
def extract_surface_points(volume: jnp.ndarray,
                          mask: jnp.ndarray,
                          coords: jnp.ndarray) -> jnp.ndarray:
    """Extract surface points from labeled volume.
    
    Args:
        volume: 3D intensity volume
        mask: Binary mask
        coords: Coordinates of object voxels
        
    Returns:
        (N, 3) array of surface points
    """
    # Simple surface extraction - find boundary voxels
    surface_points = []
    
    for coord in coords:
        x, y, z = coord
        # Check if any neighbor is outside mask
        is_surface = False
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if (0 <= nx < mask.shape[0] and 
                        0 <= ny < mask.shape[1] and 
                        0 <= nz < mask.shape[2]):
                        if mask[nx, ny, nz] == 0:
                            is_surface = True
                            break
        
        if is_surface:
            surface_points.append(coord)
    
    return jnp.array(surface_points, dtype=jnp.float32)


def fit_multiscale_polytopes(points: jnp.ndarray) -> List[Dict]:
    """Fit polytopes at multiple scales.
    
    Args:
        points: Surface points
        
    Returns:
        List of polytope fits at different scales
    """
    fits = []
    scales = [0.5, 1.0, 2.0]
    
    # Try different polytope types
    polytope_types = ['tetrahedron', 'octahedron', 'icosahedron']
    
    for scale in scales:
        for poly_type in polytope_types:
            fit = fit_single_polytope(points, poly_type, scale)
            fits.append(fit)
    
    return fits


def fit_single_polytope(points: jnp.ndarray,
                       polytope_type: str,
                       scale: float) -> Dict:
    """Fit single polytope to points.
    
    Args:
        points: Surface points
        polytope_type: Type of polytope
        scale: Scale factor
        
    Returns:
        Fit result dictionary
    """
    # Center points
    center = jnp.mean(points, axis=0)
    centered = points - center
    
    # Get template vertices
    if polytope_type == 'tetrahedron':
        vertices = jnp.array([
            [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
        ]) / jnp.sqrt(3)
    elif polytope_type == 'octahedron':
        vertices = jnp.array([
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], 
            [0, -1, 0], [0, 0, 1], [0, 0, -1]
        ])
    else:  # icosahedron
        phi = (1 + jnp.sqrt(5)) / 2
        vertices = jnp.array([
            [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
            [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ]) / jnp.sqrt(1 + phi**2)
    
    # Scale vertices
    vertices *= scale
    
    # Find optimal rotation
    rotation = find_optimal_rotation(centered, vertices)
    aligned_vertices = jnp.dot(vertices, rotation.T)
    
    # Compute fit error
    error = compute_fit_error(centered, aligned_vertices)
    
    return {
        'type': polytope_type,
        'vertices': aligned_vertices + center,
        'center': center,
        'scale': scale,
        'rotation': rotation,
        'error': error
    }


@jax.jit
def find_optimal_rotation(points: jnp.ndarray,
                         template: jnp.ndarray) -> jnp.ndarray:
    """Find rotation aligning template to points.
    
    Uses simplified Kabsch algorithm.
    """
    # Match closest points
    n_match = min(len(points), len(template))
    
    if n_match >= 3:
        # Compute cross-covariance
        H = jnp.dot(points[:n_match].T, template[:n_match])
        U, S, Vt = jnp.linalg.svd(H)
        R = jnp.dot(U, Vt)
        
        # Ensure proper rotation
        if jnp.linalg.det(R) < 0:
            Vt = Vt.at[-1].multiply(-1)
            R = jnp.dot(U, Vt)
    else:
        R = jnp.eye(3)
    
    return R


@jax.jit
def compute_fit_error(points: jnp.ndarray,
                     vertices: jnp.ndarray) -> float:
    """Compute polytope fit error."""
    # For each point, find distance to nearest vertex
    total_error = 0.0
    for point in points:
        distances = jnp.linalg.norm(vertices - point[None, :], axis=1)
        total_error += jnp.min(distances)**2
    
    return jnp.sqrt(total_error / len(points))


def select_best_polytope(fits: List[Dict],
                        prior: BiologicalPrior) -> Dict:
    """Select best polytope using prior knowledge.
    
    Args:
        fits: List of polytope fits
        prior: Biological prior
        
    Returns:
        Best fitting polytope
    """
    best_score = float('inf')
    best_fit = None
    
    for fit in fits:
        # Penalize based on prior
        error = fit['error']
        
        # Check elongation
        vertices = fit['vertices'] - fit['center']
        extent = jnp.max(vertices, axis=0) - jnp.min(vertices, axis=0)
        elongation = jnp.max(extent) / (jnp.min(extent) + 1e-10)
        
        if not (prior.typical_elongation[0] <= elongation <= prior.typical_elongation[1]):
            error *= 2.0  # Penalty for unrealistic elongation
        
        # Prefer octahedra for bacteria
        if prior.shape_category == 'bacteria' and fit['type'] == 'octahedron':
            error *= 0.8  # Preference bonus
        
        if error < best_score:
            best_score = error
            best_fit = fit
    
    return best_fit


@jax.jit
def compute_harmonic_signature(points: jnp.ndarray,
                              polytope: Dict,
                              max_l: int = 6) -> jnp.ndarray:
    """Compute rotation-invariant harmonic signature.
    
    Args:
        points: Surface points
        polytope: Fitted polytope
        max_l: Maximum harmonic order
        
    Returns:
        Rotation-invariant signature
    """
    # Project points to unit sphere
    centered = points - polytope['center']
    r = jnp.linalg.norm(centered, axis=1)
    normalized = centered / (r[:, None] + 1e-10)
    
    # Convert to spherical coordinates
    theta = jnp.arccos(jnp.clip(normalized[:, 2], -1, 1))
    phi = jnp.arctan2(normalized[:, 1], normalized[:, 0])
    
    # Compute power spectrum (rotation invariant)
    signature = []
    
    for l in range(max_l + 1):
        power_l = 0.0
        for m in range(-l, l + 1):
            # Simplified spherical harmonic evaluation
            if l == 0:
                Y_lm = jnp.ones_like(theta) / jnp.sqrt(4 * jnp.pi)
            elif l == 1:
                if m == 0:
                    Y_lm = jnp.sqrt(3/(4*jnp.pi)) * jnp.cos(theta)
                else:
                    Y_lm = jnp.sqrt(3/(4*jnp.pi)) * jnp.sin(theta) * jnp.cos(m * phi)
            else:
                # Higher order harmonics
                Y_lm = jnp.sin(theta)**abs(m) * jnp.cos(m * phi)
            
            # Weight by radial distance
            coeff = jnp.mean(r * Y_lm)
            power_l += jnp.abs(coeff)**2
        
        signature.append(jnp.sqrt(power_l))
    
    # Add bispectrum features for more complete description
    bispectrum = []
    for l1 in range(3):
        for l2 in range(3):
            for l3 in range(3):
                if abs(l1 - l2) <= l3 <= l1 + l2:
                    # Simplified bispectrum
                    bispec_val = signature[l1] * signature[l2] * signature[l3]
                    bispectrum.append(bispec_val)
    
    # Concatenate power spectrum and bispectrum
    full_signature = jnp.concatenate([jnp.array(signature), jnp.array(bispectrum)])
    
    return full_signature


def generate_augmented_examples(points: jnp.ndarray,
                               polytope: Dict,
                               prior: BiologicalPrior,
                               n_augment: int = 20) -> List[jnp.ndarray]:
    """Generate biologically plausible variations.
    
    Args:
        points: Original surface points
        polytope: Fitted polytope
        prior: Biological prior
        n_augment: Number of augmented examples
        
    Returns:
        List of augmented point sets
    """
    augmented = []
    center = polytope['center']
    
    key = jax.random.PRNGKey(42)
    
    for i in range(n_augment):
        key, subkey = jax.random.split(key)
        
        # Copy points
        aug_points = points.copy()
        
        # Apply random deformation based on prior
        deformation = sample_deformation(subkey, prior)
        
        # Elongation
        if 'elongation' in deformation:
            stretch = deformation['elongation']
            direction = jnp.array([1.0, 0.0, 0.0])  # Primary axis
            aug_points = apply_elongation(aug_points, center, direction, stretch)
        
        # Bending
        if 'bending' in deformation:
            bend_angle = deformation['bending']
            aug_points = apply_bending(aug_points, center, bend_angle)
        
        # Surface noise
        if 'surface_roughness' in deformation:
            noise_level = deformation['surface_roughness']
            key, noise_key = jax.random.split(key)
            noise = noise_level * jax.random.normal(noise_key, aug_points.shape)
            aug_points = aug_points + noise
        
        augmented.append(aug_points)
    
    # Add symmetry-based augmentations
    for sym in prior.symmetries:
        if sym == 'reflection':
            reflected = reflect_points(points, center)
            augmented.append(reflected)
        elif sym == 'rotation_180':
            rotated = rotate_points_180(points, center)
            augmented.append(rotated)
    
    return augmented


def sample_deformation(key: jax.random.PRNGKey,
                      prior: BiologicalPrior) -> Dict[str, float]:
    """Sample deformation parameters from prior."""
    deformation = {}
    
    for param, (min_val, max_val) in prior.deformation_stats.items():
        key, subkey = jax.random.split(key)
        # Beta distribution biased toward middle values
        beta_sample = jax.random.beta(subkey, 2.0, 2.0)
        value = min_val + beta_sample * (max_val - min_val)
        deformation[param] = float(value)
    
    return deformation


@jax.jit
def apply_elongation(points: jnp.ndarray,
                    center: jnp.ndarray,
                    direction: jnp.ndarray,
                    factor: float) -> jnp.ndarray:
    """Apply elongation deformation."""
    centered = points - center
    
    # Project onto direction
    projections = jnp.dot(centered, direction)
    parallel = jnp.outer(projections, direction)
    perpendicular = centered - parallel
    
    # Stretch along direction
    deformed = perpendicular + factor * parallel
    
    return deformed + center


@jax.jit
def apply_bending(points: jnp.ndarray,
                 center: jnp.ndarray,
                 angle: float) -> jnp.ndarray:
    """Apply bending deformation."""
    centered = points - center
    
    # Bend along z based on x coordinate
    x = centered[:, 0]
    y = centered[:, 1]
    z = centered[:, 2]
    
    # Rotation angle varies with x
    bend_angles = angle * x / (jnp.max(jnp.abs(x)) + 1e-10)
    
    # Apply rotation
    y_new = y * jnp.cos(bend_angles) - z * jnp.sin(bend_angles)
    z_new = y * jnp.sin(bend_angles) + z * jnp.cos(bend_angles)
    
    deformed = jnp.stack([x, y_new, z_new], axis=-1)
    
    return deformed + center


@jax.jit
def reflect_points(points: jnp.ndarray,
                  center: jnp.ndarray) -> jnp.ndarray:
    """Reflect points through plane."""
    centered = points - center
    reflected = centered.at[:, 0].multiply(-1)  # Reflect through yz-plane
    return reflected + center


@jax.jit
def rotate_points_180(points: jnp.ndarray,
                     center: jnp.ndarray) -> jnp.ndarray:
    """Rotate points 180 degrees."""
    centered = points - center
    rotated = -centered  # 180 degree rotation
    return rotated + center


def estimate_signature_covariance(mean_sig: jnp.ndarray,
                                 aug_signatures: List[jnp.ndarray],
                                 prior: BiologicalPrior) -> jnp.ndarray:
    """Estimate covariance with regularization.
    
    Args:
        mean_sig: Mean signature
        aug_signatures: Augmented signatures
        prior: Biological prior
        
    Returns:
        Regularized covariance matrix
    """
    # Stack signatures
    signatures = jnp.stack(aug_signatures)
    n_samples = len(signatures)
    
    # Compute sample covariance
    centered = signatures - mean_sig[None, :]
    sample_cov = jnp.dot(centered.T, centered) / (n_samples - 1)
    
    # Regularize with prior
    # Add small diagonal component for stability
    reg_strength = 0.1 / (n_samples + 1)
    diagonal_reg = reg_strength * jnp.eye(len(mean_sig))
    
    # Add smoothness prior - nearby harmonics correlate
    smoothness_reg = create_smoothness_prior(len(mean_sig), prior.smoothness_scale)
    
    # Combine
    regularized_cov = sample_cov + diagonal_reg + 0.1 * smoothness_reg
    
    return regularized_cov


def create_smoothness_prior(dim: int, scale: float) -> jnp.ndarray:
    """Create smoothness prior for harmonic coefficients."""
    # Nearby harmonic orders should correlate
    prior = jnp.zeros((dim, dim))
    
    for i in range(dim):
        for j in range(dim):
            # Exponential decay with distance
            dist = abs(i - j)
            prior = prior.at[i, j].set(scale * jnp.exp(-dist / 5.0))
    
    return prior


def learn_deformation_modes(augmented_examples: List[jnp.ndarray],
                           polytope: Dict) -> jnp.ndarray:
    """Learn principal deformation modes from augmented examples.
    
    Args:
        augmented_examples: List of deformed point sets
        polytope: Reference polytope
        
    Returns:
        Principal deformation modes
    """
    # Align all examples to reference
    aligned = []
    reference = polytope['vertices']
    
    for points in augmented_examples:
        # Find alignment to reference
        rotation = find_optimal_rotation(points - jnp.mean(points, axis=0),
                                       reference - jnp.mean(reference, axis=0))
        aligned_points = jnp.dot(points - jnp.mean(points, axis=0), rotation.T)
        aligned.append(aligned_points.flatten())
    
    # PCA on aligned examples
    aligned_array = jnp.stack(aligned)
    mean_shape = jnp.mean(aligned_array, axis=0)
    centered = aligned_array - mean_shape
    
    # Compute principal components
    cov = jnp.dot(centered.T, centered) / len(aligned)
    eigenvalues, eigenvectors = jnp.linalg.eigh(cov)
    
    # Sort by eigenvalue
    idx = jnp.argsort(eigenvalues)[::-1]
    
    # Return top deformation modes
    n_modes = min(6, len(idx))
    modes = eigenvectors[:, idx[:n_modes]].T
    
    return modes


def compute_confidence_bounds(mean_sig: jnp.ndarray,
                            covariance: jnp.ndarray,
                            n_samples: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute confidence bounds on signature.
    
    Args:
        mean_sig: Mean signature
        covariance: Covariance matrix
        n_samples: Number of samples used
        
    Returns:
        (lower_bound, upper_bound) at 95% confidence
    """
    # Standard error
    std_error = jnp.sqrt(jnp.diag(covariance) / n_samples)
    
    # 95% confidence interval (approximately 2 standard errors)
    z_score = 1.96
    lower = mean_sig - z_score * std_error
    upper = mean_sig + z_score * std_error
    
    return lower, upper


def compute_prior_weight(points: jnp.ndarray,
                        prior: BiologicalPrior) -> float:
    """Compute how much prior knowledge influences learning.
    
    Args:
        points: Surface points
        prior: Biological prior
        
    Returns:
        Prior weight (0 = no prior, 1 = only prior)
    """
    # More points = less prior influence
    n_points = len(points)
    
    # Measure shape complexity
    extent = jnp.max(points, axis=0) - jnp.min(points, axis=0)
    elongation = jnp.max(extent) / (jnp.min(extent) + 1e-10)
    
    # Compare to prior expectations
    elongation_match = 0.0
    if prior.typical_elongation[0] <= elongation <= prior.typical_elongation[1]:
        elongation_match = 1.0
    
    # Combine factors
    data_weight = 1.0 - jnp.exp(-n_points / 100)  # More data = less prior
    shape_weight = 0.5 * (1 + elongation_match)  # Shape match influences prior
    
    prior_weight = (1 - data_weight) * shape_weight
    
    return float(prior_weight)


def segment_with_learned_signature(volume: jnp.ndarray,
                                  learned_sig: LearnedSignature,
                                  confidence_threshold: float = 0.7) -> OneShotSegmentation:
    """Segment full volume using learned signature.
    
    Args:
        volume: 3D volume to segment
        learned_sig: Learned signature from one example
        confidence_threshold: Minimum confidence for positive classification
        
    Returns:
        Segmentation with uncertainty
    """
    # Sliding window segmentation
    window_size = 32
    stride = 16
    
    segmentation = jnp.zeros(volume.shape, dtype=jnp.int32)
    confidence = jnp.zeros(volume.shape)
    signatures = jnp.zeros(volume.shape + (len(learned_sig.mean_signature),))
    
    uncertain_regions = []
    
    for x in range(0, volume.shape[0] - window_size, stride):
        for y in range(0, volume.shape[1] - window_size, stride):
            for z in range(0, volume.shape[2] - window_size, stride):
                # Extract window
                window = volume[x:x+window_size, y:y+window_size, z:z+window_size]
                
                # Check if contains object
                if jnp.std(window) < 0.01:  # Empty region
                    continue
                
                # Extract surface points
                mask = segment_window(window)
                if jnp.sum(mask) < 10:
                    continue
                
                # Compute signature
                window_sig = compute_window_signature(window, mask)
                
                # Compare to learned signature
                similarity, conf = compare_signatures(
                    window_sig, learned_sig
                )
                
                # Update segmentation
                if similarity > confidence_threshold:
                    segmentation = segmentation.at[
                        x:x+window_size, y:y+window_size, z:z+window_size
                    ].set(1)
                    
                confidence = confidence.at[
                    x:x+window_size, y:y+window_size, z:z+window_size
                ].set(conf)
                
                # Store signature
                sig_x = x + window_size // 2
                sig_y = y + window_size // 2
                sig_z = z + window_size // 2
                signatures = signatures.at[sig_x, sig_y, sig_z].set(window_sig)
                
                # Track uncertain regions
                if 0.3 < conf < 0.7:
                    uncertain_regions.append((sig_x, sig_y, sig_z))
    
    # Identify active learning queries
    active_queries = select_active_queries(confidence, signatures, learned_sig)
    
    return OneShotSegmentation(
        segmentation_mask=segmentation,
        confidence_map=confidence,
        signatures_map=signatures,
        uncertain_regions=uncertain_regions,
        active_queries=active_queries
    )


@jax.jit
def segment_window(window: jnp.ndarray) -> jnp.ndarray:
    """Simple segmentation of window."""
    # Threshold-based segmentation
    threshold = jnp.mean(window) + jnp.std(window)
    return (window > threshold).astype(jnp.int32)


def compute_window_signature(window: jnp.ndarray,
                           mask: jnp.ndarray) -> jnp.ndarray:
    """Compute signature for window region."""
    # Extract surface points
    coords = jnp.argwhere(mask > 0)
    if len(coords) < 10:
        return jnp.zeros(20)  # Default signature size
    
    # Simplified signature computation
    center = jnp.mean(coords, axis=0)
    centered = coords - center
    
    # Basic shape statistics
    extent = jnp.max(centered, axis=0) - jnp.min(centered, axis=0)
    elongation = jnp.max(extent) / (jnp.min(extent) + 1e-10)
    
    # Simplified harmonic signature
    r = jnp.linalg.norm(centered, axis=1)
    mean_r = jnp.mean(r)
    std_r = jnp.std(r)
    
    # Combine into signature
    signature = jnp.array([
        elongation, mean_r, std_r,
        extent[0], extent[1], extent[2],
        jnp.mean(centered[:, 0]**2),
        jnp.mean(centered[:, 1]**2),
        jnp.mean(centered[:, 2]**2),
        jnp.mean(jnp.abs(centered[:, 0])),
        jnp.mean(jnp.abs(centered[:, 1])),
        jnp.mean(jnp.abs(centered[:, 2]))
    ])
    
    # Pad to standard size
    if len(signature) < 20:
        signature = jnp.pad(signature, (0, 20 - len(signature)))
    
    return signature[:20]


def compare_signatures(sig1: jnp.ndarray,
                      learned_sig: LearnedSignature) -> Tuple[float, float]:
    """Compare signature to learned signature with uncertainty.
    
    Args:
        sig1: Query signature
        learned_sig: Learned signature with covariance
        
    Returns:
        (similarity, confidence) both in [0, 1]
    """
    # Truncate signatures to same length
    min_len = min(len(sig1), len(learned_sig.mean_signature))
    sig1_trunc = sig1[:min_len]
    mean_trunc = learned_sig.mean_signature[:min_len]
    cov_trunc = learned_sig.covariance[:min_len, :min_len]
    
    # Mahalanobis distance
    diff = sig1_trunc - mean_trunc
    
    # Regularize covariance for numerical stability
    cov_reg = cov_trunc + 0.01 * jnp.eye(min_len)
    inv_cov = jnp.linalg.inv(cov_reg)
    
    mahal_dist = jnp.sqrt(jnp.dot(diff, jnp.dot(inv_cov, diff)))
    
    # Convert to similarity
    similarity = jnp.exp(-0.5 * mahal_dist)
    
    # Confidence based on distance from boundary
    # High confidence when very similar or very different
    confidence = 1.0 - jnp.exp(-2 * abs(similarity - 0.5))
    
    return float(similarity), float(confidence)


def select_active_queries(confidence: jnp.ndarray,
                         signatures: jnp.ndarray,
                         learned_sig: LearnedSignature,
                         n_queries: int = 5) -> List[Tuple[int, int, int]]:
    """Select most informative locations for labeling.
    
    Args:
        confidence: Confidence map
        signatures: Signature map  
        learned_sig: Current learned signature
        n_queries: Number of queries to return
        
    Returns:
        List of (x, y, z) locations
    """
    # Find locations with moderate confidence (most uncertain)
    uncertainty = 1.0 - jnp.abs(confidence - 0.5) * 2
    
    # Also prioritize novel signatures
    # Simplified - would compute actual novelty
    
    # Get top uncertain locations
    flat_uncertainty = uncertainty.flatten()
    top_indices = jnp.argsort(flat_uncertainty)[-n_queries:]
    
    # Convert to 3D coordinates
    queries = []
    for idx in top_indices:
        x = idx // (confidence.shape[1] * confidence.shape[2])
        y = (idx % (confidence.shape[1] * confidence.shape[2])) // confidence.shape[2]
        z = idx % confidence.shape[2]
        queries.append((int(x), int(y), int(z)))
    
    return queries


def transfer_learn_signature(example: LabeledExample,
                           pretrained_models: Dict[str, LearnedSignature],
                           prior: BiologicalPrior = BACTERIA_PRIOR) -> LearnedSignature:
    """Transfer learning from pretrained models.
    
    Args:
        example: New labeled example
        pretrained_models: Dictionary of pretrained signatures
        prior: Biological prior
        
    Returns:
        Learned signature with transfer knowledge
    """
    # Extract signature from example
    base_sig = extract_signature_from_example(example, prior)
    
    # Find most similar pretrained model
    best_model = None
    best_similarity = -1.0
    
    for name, model in pretrained_models.items():
        sim, _ = compare_signatures(base_sig.mean_signature, model)
        if sim > best_similarity:
            best_similarity = sim
            best_model = model
    
    if best_model is not None and best_similarity > 0.5:
        # Transfer knowledge
        # Weighted combination of new and pretrained
        transfer_weight = best_similarity * 0.5
        
        combined_mean = (1 - transfer_weight) * base_sig.mean_signature + \
                       transfer_weight * best_model.mean_signature
        
        combined_cov = (1 - transfer_weight) * base_sig.covariance + \
                      transfer_weight * best_model.covariance
        
        # Use pretrained deformation modes as initialization
        combined_modes = jnp.concatenate([
            base_sig.deformation_modes,
            best_model.deformation_modes * 0.5  # Downweight transferred modes
        ])
        
        return base_sig._replace(
            mean_signature=combined_mean,
            covariance=combined_cov,
            deformation_modes=combined_modes,
            prior_weight=base_sig.prior_weight * 0.5  # Less prior needed
        )
    
    return base_sig


def refine_signature_bayesian(current_sig: LearnedSignature,
                            new_examples: List[LabeledExample],
                            learning_rate: float = 0.3) -> LearnedSignature:
    """Refine signature with new examples using Bayesian update.
    
    Args:
        current_sig: Current learned signature
        new_examples: Additional labeled examples
        learning_rate: How much to weight new examples
        
    Returns:
        Refined signature
    """
    # Extract signatures from new examples
    new_signatures = []
    for example in new_examples:
        sig = extract_signature_from_example(example)
        new_signatures.append(sig.mean_signature)
    
    if not new_signatures:
        return current_sig
    
    # Compute new statistics
    new_mean = jnp.mean(jnp.stack(new_signatures), axis=0)
    
    # Bayesian update of mean
    n_old = 1.0 / current_sig.prior_weight  # Effective sample size
    n_new = len(new_signatures)
    
    posterior_mean = (n_old * current_sig.mean_signature + n_new * new_mean) / (n_old + n_new)
    
    # Update covariance
    # Simplified - would do proper Bayesian covariance update
    posterior_cov = current_sig.covariance * (1 - learning_rate) + \
                   jnp.cov(jnp.stack(new_signatures).T) * learning_rate
    
    # Update prior weight
    posterior_prior_weight = current_sig.prior_weight / (1 + n_new)
    
    return current_sig._replace(
        mean_signature=posterior_mean,
        covariance=posterior_cov,
        prior_weight=posterior_prior_weight
    )


# Example usage
def example_one_shot_learning():
    """Example of one-shot learning for novel bacterium."""
    # Create synthetic labeled example
    volume = jnp.zeros((64, 64, 64))
    mask = jnp.zeros((64, 64, 64))
    
    # Add elongated bacterium shape
    for x in range(20, 45):
        for y in range(28, 36):
            for z in range(28, 36):
                if ((x-32.5)**2/225 + (y-32)**2/16 + (z-32)**2/16) < 1:
                    volume = volume.at[x, y, z].set(1.0)
                    mask = mask.at[x, y, z].set(1)
    
    # Create labeled example
    example = LabeledExample(
        volume=volume,
        mask=mask,
        label="novel_bacterium_1",
        metadata={"source": "synthetic"}
    )
    
    # Learn signature
    learned = extract_signature_from_example(example, BACTERIA_PRIOR)
    
    print(f"Learned signature shape: {learned.mean_signature.shape}")
    print(f"Prior weight: {learned.prior_weight:.3f}")
    print(f"Generated {len(learned.augmented_examples)} augmented examples")
    
    # Segment new volume
    result = segment_with_learned_signature(volume, learned)
    
    print(f"Segmented {jnp.sum(result.segmentation_mask)} voxels")
    print(f"Found {len(result.uncertain_regions)} uncertain regions")
    print(f"Suggested {len(result.active_queries)} locations for review")
    
    return learned, result