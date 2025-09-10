"""
Segment arbitrary smooth objects using polytope approximations and harmonics.

This module provides adaptive segmentation for smooth curves and surfaces by
fitting polytopes at multiple scales, computing harmonic decompositions, and
using curvature to guide refinement. The approach handles arbitrary smooth
objects from bacteria to organelles to tissue boundaries.

Key principles:
- Local polytope fitting captures geometry
- Harmonic analysis provides smooth interpolation
- Curvature drives adaptive refinement
- Multi-scale hierarchy enables efficiency
- Consensus from multiple approaches ensures robustness
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, List, Dict, Tuple, Optional, Callable
from functools import partial
import numpy as np


class SmoothObject(NamedTuple):
    """Representation of a smooth object to segment.
    
    Attributes:
        points: (N, 3) array of surface points
        normals: (N, 3) array of surface normals
        curvatures: (N, 2) principal curvatures at each point
        topology: 'closed', 'open', or 'branching'
        scale: Characteristic size
    """
    points: jnp.ndarray
    normals: jnp.ndarray
    curvatures: jnp.ndarray
    topology: str
    scale: float


class PolytopeFit(NamedTuple):
    """Result of fitting a polytope to a local patch.
    
    Attributes:
        polytope_type: 'tetrahedron', 'octahedron', 'icosahedron', etc.
        center: (3,) center position
        orientation: (3, 3) rotation matrix
        scale: Size of the polytope
        vertices: (M, 3) transformed polytope vertices
        fit_error: RMS distance to surface
        harmonic_coeffs: Spherical harmonic coefficients
    """
    polytope_type: str
    center: jnp.ndarray
    orientation: jnp.ndarray
    scale: float
    vertices: jnp.ndarray
    fit_error: float
    harmonic_coeffs: Dict[Tuple[int, int], complex]


class SegmentationResult(NamedTuple):
    """Complete segmentation of a smooth object.
    
    Attributes:
        polytope_patches: List of fitted polytopes covering the object
        boundaries: (B, 3) boundary points between patches
        confidence_map: (N,) confidence at each point
        harmonic_field: Smooth harmonic interpolation
        segmentation_mask: (N,) integer labels for each point
        hierarchy: Multi-scale polytope hierarchy
    """
    polytope_patches: List[PolytopeFit]
    boundaries: jnp.ndarray
    confidence_map: jnp.ndarray
    harmonic_field: jnp.ndarray
    segmentation_mask: jnp.ndarray
    hierarchy: Dict[int, List[PolytopeFit]]


# Standard polytope templates
POLYTOPE_TEMPLATES = {
    'tetrahedron': jnp.array([
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
    ]) / jnp.sqrt(3),
    
    'octahedron': jnp.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ]),
    
    'icosahedron': jnp.array([
        [0, 1, 1.618], [0, 1, -1.618], [0, -1, 1.618], [0, -1, -1.618],
        [1, 1.618, 0], [1, -1.618, 0], [-1, 1.618, 0], [-1, -1.618, 0],
        [1.618, 0, 1], [1.618, 0, -1], [-1.618, 0, 1], [-1.618, 0, -1]
    ]) / jnp.sqrt(2.618),
    
    'cube': jnp.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
    ]) / jnp.sqrt(3)
}


@jax.jit
def compute_curvature(points: jnp.ndarray, normals: jnp.ndarray,
                     neighbors: jnp.ndarray) -> jnp.ndarray:
    """Compute principal curvatures at surface points.
    
    Args:
        points: (N, 3) surface points
        normals: (N, 3) surface normals
        neighbors: (N, K) indices of K nearest neighbors
        
    Returns:
        (N, 2) principal curvatures (min, max)
    """
    n_points = points.shape[0]
    curvatures = jnp.zeros((n_points, 2))
    
    for i in range(n_points):
        # Get local neighborhood
        neighbor_idx = neighbors[i]
        local_points = points[neighbor_idx]
        local_normals = normals[neighbor_idx]
        
        # Center at current point
        centered = local_points - points[i]
        
        # Project onto tangent plane
        tangent_proj = centered - jnp.outer(jnp.dot(centered, normals[i]), normals[i])
        
        # Compute shape operator (Weingarten map)
        if len(tangent_proj) > 3:
            # Fit quadratic form
            A = jnp.column_stack([
                tangent_proj[:, 0]**2,
                tangent_proj[:, 1]**2,
                tangent_proj[:, 0] * tangent_proj[:, 1]
            ])
            b = jnp.dot(centered, normals[i])
            
            # Least squares fit
            coeffs = jnp.linalg.lstsq(A, b)[0]
            
            # Extract principal curvatures from quadratic form
            H = jnp.array([[coeffs[0], coeffs[2]/2],
                          [coeffs[2]/2, coeffs[1]]])
            eigvals = jnp.linalg.eigvalsh(H)
            curvatures = curvatures.at[i].set(eigvals)
    
    return curvatures


def fit_local_polytope(local_points: jnp.ndarray,
                      local_normals: jnp.ndarray,
                      curvature: float) -> PolytopeFit:
    """Fit best polytope to local surface patch.
    
    Args:
        local_points: (M, 3) points in local patch
        local_normals: (M, 3) normals
        curvature: Mean curvature for polytope selection
        
    Returns:
        Best fitting polytope
    """
    # Select polytope based on curvature
    if abs(curvature) < 0.1:
        # Nearly flat - use cube
        polytope_type = 'cube'
    elif curvature < 0.5:
        # Moderate curvature - use octahedron
        polytope_type = 'octahedron'
    else:
        # High curvature - use icosahedron
        polytope_type = 'icosahedron'
    
    template = POLYTOPE_TEMPLATES[polytope_type]
    
    # Find optimal transformation
    center = jnp.mean(local_points, axis=0)
    centered_points = local_points - center
    
    # Estimate scale
    scale = jnp.sqrt(jnp.mean(jnp.sum(centered_points**2, axis=1)))
    
    # Find rotation using SVD
    if len(local_points) >= 3:
        # Compute covariance between points and template
        scaled_template = template * scale
        
        # Use Kabsch algorithm for optimal rotation
        H = jnp.dot(centered_points.T, scaled_template[:len(local_points)])
        U, _, Vt = jnp.linalg.svd(H)
        rotation = jnp.dot(U, Vt)
        
        # Ensure proper rotation (det = 1)
        if jnp.linalg.det(rotation) < 0:
            Vt = Vt.at[-1].multiply(-1)
            rotation = jnp.dot(U, Vt)
    else:
        rotation = jnp.eye(3)
    
    # Transform template
    vertices = center + scale * jnp.dot(template, rotation.T)
    
    # Compute fit error
    if len(local_points) > 0:
        distances = []
        for point in local_points:
            dist_to_vertices = jnp.linalg.norm(vertices - point[None, :], axis=1)
            distances.append(jnp.min(dist_to_vertices))
        fit_error = float(jnp.sqrt(jnp.mean(jnp.array(distances)**2)))
    else:
        fit_error = 0.0
    
    # Compute harmonic coefficients for smooth interpolation
    harmonic_coeffs = compute_local_harmonics(centered_points, scale)
    
    return PolytopeFit(
        polytope_type=polytope_type,
        center=center,
        orientation=rotation,
        scale=scale,
        vertices=vertices,
        fit_error=fit_error,
        harmonic_coeffs=harmonic_coeffs
    )


def compute_local_harmonics(points: jnp.ndarray, scale: float) -> Dict[Tuple[int, int], complex]:
    """Compute spherical harmonic decomposition of local patch.
    
    Args:
        points: (N, 3) centered points
        scale: Normalization scale
        
    Returns:
        Dictionary of harmonic coefficients
    """
    # Convert to spherical coordinates
    r = jnp.linalg.norm(points, axis=1)
    theta = jnp.arccos(jnp.clip(points[:, 2] / (r + 1e-10), -1, 1))
    phi = jnp.arctan2(points[:, 1], points[:, 0])
    
    # Compute harmonics up to l=4
    coeffs = {}
    for l in range(5):
        for m in range(-l, l + 1):
            # Simplified harmonic computation
            Y_lm = spherical_harmonic_simple(l, m, theta, phi)
            coeff = jnp.mean(r / scale * Y_lm)
            coeffs[(l, m)] = complex(coeff)
    
    return coeffs


@jax.jit
def spherical_harmonic_simple(l: int, m: int, theta: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
    """Simplified spherical harmonic evaluation."""
    # Very simplified - real implementation would use proper harmonics
    if l == 0:
        return jnp.ones_like(theta) / jnp.sqrt(4 * jnp.pi)
    elif l == 1:
        if m == -1:
            return jnp.sin(theta) * jnp.sin(phi)
        elif m == 0:
            return jnp.cos(theta)
        elif m == 1:
            return jnp.sin(theta) * jnp.cos(phi)
    elif l == 2:
        if m == 0:
            return (3 * jnp.cos(theta)**2 - 1) / 2
        else:
            return jnp.sin(theta)**abs(m) * jnp.cos(m * phi)
    
    return jnp.zeros_like(theta)


def adaptive_segmentation(smooth_object: SmoothObject,
                         initial_scale: float = 1.0,
                         max_refinement: int = 3) -> SegmentationResult:
    """Adaptively segment smooth object using polytope fitting.
    
    Args:
        smooth_object: Object to segment
        initial_scale: Starting patch size
        max_refinement: Maximum refinement levels
        
    Returns:
        Complete segmentation result
    """
    points = smooth_object.points
    normals = smooth_object.normals
    curvatures = smooth_object.curvatures
    
    # Initialize with coarse segmentation
    patches = []
    hierarchy = {0: []}
    
    # Create initial patches
    n_initial = max(10, int(len(points) / 1000))
    patch_centers = points[::len(points)//n_initial]
    
    for center in patch_centers:
        # Find points in patch
        distances = jnp.linalg.norm(points - center[None, :], axis=1)
        in_patch = distances < initial_scale
        
        if jnp.sum(in_patch) > 3:
            local_points = points[in_patch]
            local_normals = normals[in_patch]
            local_curvature = jnp.mean(curvatures[in_patch])
            
            # Fit polytope
            fit = fit_local_polytope(local_points, local_normals, float(local_curvature))
            patches.append(fit)
            hierarchy[0].append(fit)
    
    # Adaptive refinement based on curvature
    for level in range(1, max_refinement + 1):
        hierarchy[level] = []
        new_patches = []
        
        for patch in patches:
            # Check if refinement needed
            if patch.fit_error > 0.1 * patch.scale:
                # High error - refine this patch
                sub_scale = patch.scale / 2
                
                # Create sub-patches around polytope vertices
                for vertex in patch.vertices:
                    distances = jnp.linalg.norm(points - vertex[None, :], axis=1)
                    in_subpatch = distances < sub_scale
                    
                    if jnp.sum(in_subpatch) > 3:
                        local_points = points[in_subpatch]
                        local_normals = normals[in_subpatch]
                        local_curvature = jnp.mean(curvatures[in_subpatch])
                        
                        sub_fit = fit_local_polytope(local_points, local_normals, float(local_curvature))
                        new_patches.append(sub_fit)
                        hierarchy[level].append(sub_fit)
            else:
                # Keep original patch
                new_patches.append(patch)
        
        patches = new_patches
    
    # Find boundaries between patches
    boundaries = find_patch_boundaries(patches, points)
    
    # Create confidence map
    confidence_map = compute_confidence_map(points, patches)
    
    # Smooth using harmonics
    harmonic_field = harmonic_smoothing(points, patches)
    
    # Assign segmentation labels
    segmentation_mask = assign_labels(points, patches)
    
    return SegmentationResult(
        polytope_patches=patches,
        boundaries=boundaries,
        confidence_map=confidence_map,
        harmonic_field=harmonic_field,
        segmentation_mask=segmentation_mask,
        hierarchy=hierarchy
    )


def find_patch_boundaries(patches: List[PolytopeFit], 
                         points: jnp.ndarray) -> jnp.ndarray:
    """Find boundary points between polytope patches.
    
    Args:
        patches: List of fitted polytopes
        points: All surface points
        
    Returns:
        (B, 3) array of boundary points
    """
    # For each point, find two nearest patches
    boundary_points = []
    
    for point in points:
        distances = []
        for patch in patches:
            # Distance to patch center
            dist = jnp.linalg.norm(point - patch.center)
            distances.append(dist)
        
        distances = jnp.array(distances)
        sorted_idx = jnp.argsort(distances)
        
        # If two nearest patches are close in distance, point is on boundary
        if len(sorted_idx) > 1:
            dist1 = distances[sorted_idx[0]]
            dist2 = distances[sorted_idx[1]]
            
            if abs(dist1 - dist2) < 0.2 * (dist1 + dist2):
                boundary_points.append(point)
    
    return jnp.array(boundary_points) if boundary_points else jnp.zeros((0, 3))


@jax.jit
def compute_confidence_map(points: jnp.ndarray,
                          patches: List[PolytopeFit]) -> jnp.ndarray:
    """Compute confidence score at each point.
    
    Args:
        points: (N, 3) surface points
        patches: Fitted polytopes
        
    Returns:
        (N,) confidence scores
    """
    confidences = jnp.zeros(len(points))
    
    for i, point in enumerate(points):
        # Find nearest patch
        min_dist = float('inf')
        best_patch = None
        
        for patch in patches:
            dist = jnp.linalg.norm(point - patch.center)
            if dist < min_dist:
                min_dist = dist
                best_patch = patch
        
        if best_patch is not None:
            # Confidence based on fit error and distance
            relative_error = best_patch.fit_error / (best_patch.scale + 1e-10)
            distance_factor = jnp.exp(-min_dist / best_patch.scale)
            
            confidence = distance_factor * (1 - relative_error)
            confidences = confidences.at[i].set(confidence)
    
    return confidences


def harmonic_smoothing(points: jnp.ndarray,
                      patches: List[PolytopeFit]) -> jnp.ndarray:
    """Create smooth field using harmonic interpolation.
    
    Args:
        points: (N, 3) surface points
        patches: Fitted polytopes with harmonics
        
    Returns:
        (N, 3) smooth vector field
    """
    field = jnp.zeros_like(points)
    
    for i, point in enumerate(points):
        # Weighted combination of nearby patch harmonics
        total_weight = 0.0
        weighted_sum = jnp.zeros(3)
        
        for patch in patches:
            # Weight by inverse distance
            dist = jnp.linalg.norm(point - patch.center)
            weight = 1.0 / (dist + 0.1 * patch.scale)
            
            # Evaluate patch harmonics at this point
            relative_pos = (point - patch.center) / patch.scale
            
            # Simplified harmonic evaluation
            harmonic_value = evaluate_harmonics_at_point(
                relative_pos, patch.harmonic_coeffs
            )
            
            weighted_sum += weight * harmonic_value * relative_pos
            total_weight += weight
        
        if total_weight > 0:
            field = field.at[i].set(weighted_sum / total_weight)
    
    return field


def evaluate_harmonics_at_point(pos: jnp.ndarray,
                               coeffs: Dict[Tuple[int, int], complex]) -> float:
    """Evaluate harmonic expansion at a point."""
    # Convert to spherical
    r = jnp.linalg.norm(pos)
    if r < 1e-10:
        return 0.0
    
    theta = jnp.arccos(jnp.clip(pos[2] / r, -1, 1))
    phi = jnp.arctan2(pos[1], pos[0])
    
    # Sum harmonic contributions
    value = 0.0
    for (l, m), coeff in coeffs.items():
        Y_lm = spherical_harmonic_simple(l, m, theta, phi)
        value += jnp.real(coeff * Y_lm)
    
    return value


def assign_labels(points: jnp.ndarray, patches: List[PolytopeFit]) -> jnp.ndarray:
    """Assign segmentation label to each point.
    
    Args:
        points: (N, 3) surface points
        patches: Fitted polytopes
        
    Returns:
        (N,) integer labels
    """
    labels = jnp.zeros(len(points), dtype=jnp.int32)
    
    for i, point in enumerate(points):
        # Find nearest patch
        min_dist = float('inf')
        best_label = 0
        
        for j, patch in enumerate(patches):
            dist = jnp.linalg.norm(point - patch.center)
            if dist < min_dist:
                min_dist = dist
                best_label = j
        
        labels = labels.at[i].set(best_label)
    
    return labels


def parallel_segmentation(smooth_object: SmoothObject,
                         n_hypotheses: int = 5) -> SegmentationResult:
    """Run multiple segmentations in parallel and combine.
    
    Args:
        smooth_object: Object to segment
        n_hypotheses: Number of parallel attempts
        
    Returns:
        Consensus segmentation
    """
    # Generate different initial conditions
    results = []
    
    for i in range(n_hypotheses):
        # Vary initial scale
        scale = 0.5 + 1.0 * i / n_hypotheses
        
        # Run segmentation
        result = adaptive_segmentation(smooth_object, initial_scale=scale)
        results.append(result)
    
    # Combine results by voting
    all_masks = jnp.stack([r.segmentation_mask for r in results])
    
    # Majority vote at each point
    consensus_mask = jnp.zeros_like(results[0].segmentation_mask)
    for i in range(len(consensus_mask)):
        votes = all_masks[:, i]
        unique_labels, counts = jnp.unique(votes, return_counts=True)
        consensus_mask = consensus_mask.at[i].set(unique_labels[jnp.argmax(counts)])
    
    # Use highest confidence result structure
    best_idx = jnp.argmax(jnp.array([jnp.mean(r.confidence_map) for r in results]))
    best_result = results[best_idx]
    
    # Update with consensus mask
    return SegmentationResult(
        polytope_patches=best_result.polytope_patches,
        boundaries=best_result.boundaries,
        confidence_map=best_result.confidence_map,
        harmonic_field=best_result.harmonic_field,
        segmentation_mask=consensus_mask,
        hierarchy=best_result.hierarchy
    )


def segment_with_topology(smooth_object: SmoothObject) -> SegmentationResult:
    """Segment considering topological constraints.
    
    Args:
        smooth_object: Object with known topology
        
    Returns:
        Topology-aware segmentation
    """
    if smooth_object.topology == 'closed':
        # Ensure patches form closed surface
        result = adaptive_segmentation(smooth_object)
        
        # Check and fix holes
        result = ensure_closed_topology(result, smooth_object.points)
        
    elif smooth_object.topology == 'branching':
        # Handle branching structures (e.g., neurons)
        result = segment_branching_structure(smooth_object)
        
    else:  # 'open'
        # Standard segmentation
        result = adaptive_segmentation(smooth_object)
    
    return result


def ensure_closed_topology(result: SegmentationResult,
                          points: jnp.ndarray) -> SegmentationResult:
    """Ensure segmentation forms closed surface."""
    # Find gaps in coverage
    covered = jnp.zeros(len(points), dtype=bool)
    
    for patch in result.polytope_patches:
        distances = jnp.linalg.norm(points - patch.center[None, :], axis=1)
        covered |= distances < patch.scale * 1.5
    
    # Fill gaps with additional patches
    gap_points = points[~covered]
    
    if len(gap_points) > 0:
        # Add patches to cover gaps
        # (Simplified - would need proper implementation)
        pass
    
    return result


def segment_branching_structure(smooth_object: SmoothObject) -> SegmentationResult:
    """Special handling for branching structures."""
    # Identify branch points using curvature
    high_curvature = jnp.max(jnp.abs(smooth_object.curvatures), axis=1) > 2.0
    branch_points = smooth_object.points[high_curvature]
    
    # Segment branches separately
    # (Simplified - would need graph-based approach)
    
    return adaptive_segmentation(smooth_object)


# Example usage
def example_smooth_segmentation():
    """Example segmenting a smooth bacterial surface."""
    # Create synthetic smooth surface (elongated shape)
    theta = jnp.linspace(0, jnp.pi, 50)
    phi = jnp.linspace(0, 2*jnp.pi, 100)
    theta_grid, phi_grid = jnp.meshgrid(theta, phi)
    
    # Elongated ellipsoid (bacterial shape)
    a, b, c = 3.0, 1.0, 1.0  # Semi-axes
    x = a * jnp.sin(theta_grid) * jnp.cos(phi_grid)
    y = b * jnp.sin(theta_grid) * jnp.sin(phi_grid)
    z = c * jnp.cos(theta_grid)
    
    points = jnp.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)
    
    # Compute normals (gradient of implicit function)
    normals = jnp.stack([
        2*x.flatten()/a**2,
        2*y.flatten()/b**2,
        2*z.flatten()/c**2
    ], axis=-1)
    normals = normals / jnp.linalg.norm(normals, axis=1, keepdims=True)
    
    # Estimate curvatures
    curvatures = jnp.zeros((len(points), 2))
    
    # Create smooth object
    smooth_obj = SmoothObject(
        points=points,
        normals=normals,
        curvatures=curvatures,
        topology='closed',
        scale=1.0
    )
    
    # Segment
    result = parallel_segmentation(smooth_obj, n_hypotheses=3)
    
    print(f"Segmented into {len(result.polytope_patches)} patches")
    print(f"Mean confidence: {jnp.mean(result.confidence_map):.3f}")
    print(f"Boundary points: {len(result.boundaries)}")
    
    return result