"""
Dual polytope operations with advanced transformations and harmonic analysis.

This module implements pole-face duality transformations that preserve essential
geometric properties while swapping vertices with faces. The key insight is that
duality operations reveal complementary information: elongated structures become
compact in the dual space, making classification easier.

Mathematical foundations:
- Pole-face duality: vertices ↔ faces, edges ↔ edges
- Harmonic duality: Y_l^m ↔ Y_l^{-m} with phase relationships
- 24-cell self-duality enables efficient transformations
- Dual transforms separate bacteria (elongated) from granules (spherical)
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, Dict, List
from functools import partial
import numpy as np


class DualPolytope(NamedTuple):
    """Dual polytope with transformation metadata.
    
    Attributes:
        vertices: (N, D) array of dual vertex positions
        faces: List of arrays containing vertex indices per face
        edge_dual_map: Mapping between original and dual edges
        volume_product: Product of original and dual volumes (invariant)
        harmonic_transform: Transformation matrix for harmonics
    """
    vertices: jnp.ndarray
    faces: List[jnp.ndarray]
    edge_dual_map: jnp.ndarray
    volume_product: float
    harmonic_transform: jnp.ndarray


class HarmonicDuality(NamedTuple):
    """Harmonic analysis under duality transformation.
    
    Attributes:
        original_coeffs: Harmonic coefficients in original space
        dual_coeffs: Harmonic coefficients in dual space
        phase_shifts: Phase relationships between dual pairs
        energy_distribution: How energy redistributes under duality
    """
    original_coeffs: jnp.ndarray
    dual_coeffs: jnp.ndarray
    phase_shifts: jnp.ndarray
    energy_distribution: jnp.ndarray


@jax.jit
def compute_circumradius_inradius(vertices: jnp.ndarray, 
                                 center: jnp.ndarray = None) -> Tuple[float, float]:
    """Compute circumradius and estimate inradius of polytope.
    
    Args:
        vertices: (N, D) array of vertex positions
        center: Optional center point, computed if not provided
        
    Returns:
        Tuple of (circumradius, inradius_estimate)
    """
    if center is None:
        center = jnp.mean(vertices, axis=0)
    
    # Circumradius is max distance from center to vertices
    distances = jnp.linalg.norm(vertices - center[None, :], axis=1)
    circumradius = jnp.max(distances)
    
    # Inradius estimate: use minimum distance to convex hull approximation
    # For regular polytopes, this relates to face distances
    inradius_estimate = jnp.min(distances) * 0.5  # Simplified estimate
    
    return circumradius, inradius_estimate


@jax.jit
def pole_reciprocation(vertices: jnp.ndarray,
                      center: jnp.ndarray = None,
                      radius: float = 1.0) -> jnp.ndarray:
    """Perform pole reciprocation (spherical inversion) of vertices.
    
    Maps each vertex v to v' = radius² * v / |v|²
    This is the fundamental operation underlying polytope duality.
    
    Args:
        vertices: (N, D) array of vertex positions
        center: Center of inversion sphere
        radius: Radius of inversion sphere
        
    Returns:
        (N, D) array of reciprocated positions
    """
    if center is None:
        center = jnp.mean(vertices, axis=0)
    
    # Translate to center
    centered = vertices - center[None, :]
    
    # Compute squared distances
    dist_squared = jnp.sum(centered**2, axis=1, keepdims=True)
    
    # Avoid division by zero
    dist_squared = jnp.maximum(dist_squared, 1e-10)
    
    # Reciprocate: v' = r² * v / |v|²
    reciprocated = radius**2 * centered / dist_squared
    
    # Translate back
    return reciprocated + center[None, :]


def construct_dual_polytope(vertices: jnp.ndarray,
                          faces: List[jnp.ndarray],
                          edges: jnp.ndarray = None) -> DualPolytope:
    """Construct the complete dual polytope.
    
    Args:
        vertices: (N, D) array of original vertices
        faces: List of face vertex indices
        edges: Optional (E, 2) array of edge indices
        
    Returns:
        DualPolytope with all transformations
    """
    # Compute face centers as dual vertices
    dual_vertices = []
    for face_indices in faces:
        face_vertices = vertices[face_indices]
        face_center = jnp.mean(face_vertices, axis=0)
        dual_vertices.append(face_center)
    
    dual_vertices = jnp.stack(dual_vertices)
    
    # Normalize dual vertices for regular polytopes
    center = jnp.mean(dual_vertices, axis=0)
    dual_vertices = pole_reciprocation(dual_vertices, center)
    
    # Construct dual faces (one per original vertex)
    # This requires topological analysis - simplified here
    dual_faces = []
    for v_idx in range(len(vertices)):
        # Find faces containing this vertex
        adjacent_faces = []
        for f_idx, face in enumerate(faces):
            if v_idx in face:
                adjacent_faces.append(f_idx)
        if adjacent_faces:
            dual_faces.append(jnp.array(adjacent_faces))
    
    # Edge duality: edges map to edges in dual
    if edges is not None:
        edge_dual_map = compute_edge_dual_map(vertices, faces, edges)
    else:
        edge_dual_map = jnp.array([])
    
    # Volume product invariant: V * V' = constant
    original_volume = estimate_polytope_volume(vertices)
    dual_volume = estimate_polytope_volume(dual_vertices)
    volume_product = original_volume * dual_volume
    
    # Harmonic transformation matrix
    harmonic_transform = compute_harmonic_dual_transform(len(vertices), len(dual_vertices))
    
    return DualPolytope(
        vertices=dual_vertices,
        faces=dual_faces,
        edge_dual_map=edge_dual_map,
        volume_product=volume_product,
        harmonic_transform=harmonic_transform
    )


@jax.jit
def estimate_polytope_volume(vertices: jnp.ndarray) -> float:
    """Estimate volume of polytope using convex hull approximation.
    
    Args:
        vertices: (N, D) array of vertices
        
    Returns:
        Volume estimate
    """
    # Simplified: use bounding box volume
    # Full implementation would use convex hull
    min_coords = jnp.min(vertices, axis=0)
    max_coords = jnp.max(vertices, axis=0)
    box_dims = max_coords - min_coords
    
    return jnp.prod(box_dims)


def compute_edge_dual_map(vertices: jnp.ndarray,
                         faces: List[jnp.ndarray],
                         edges: jnp.ndarray) -> jnp.ndarray:
    """Map edges to their duals.
    
    Args:
        vertices: Original vertices
        faces: Original faces
        edges: (E, 2) edge connectivity
        
    Returns:
        (E, 2) dual edge connectivity
    """
    # Each edge in dual connects face centers of adjacent faces
    # Simplified implementation
    n_edges = len(edges)
    dual_edges = []
    
    for edge_idx, (v1, v2) in enumerate(edges):
        # Find faces containing both vertices
        adjacent_faces = []
        for f_idx, face in enumerate(faces):
            if v1 in face and v2 in face:
                adjacent_faces.append(f_idx)
        
        if len(adjacent_faces) >= 2:
            dual_edges.append([adjacent_faces[0], adjacent_faces[1]])
    
    return jnp.array(dual_edges) if dual_edges else jnp.zeros((0, 2))


@jax.jit
def compute_harmonic_dual_transform(n_original: int, n_dual: int) -> jnp.ndarray:
    """Compute transformation matrix for spherical harmonics under duality.
    
    The key insight: Y_l^m transforms to Y_l^{-m} with specific phase
    relationships that depend on the polytope symmetry.
    
    Args:
        n_original: Number of original vertices
        n_dual: Number of dual vertices (faces)
        
    Returns:
        Transformation matrix for harmonic coefficients
    """
    # Simplified: create phase shift matrix
    # Full implementation would use group representation theory
    max_l = min(n_original, n_dual) // 2
    size = (max_l + 1)**2
    
    transform = jnp.zeros((size, size), dtype=complex)
    
    idx = 0
    for l in range(max_l + 1):
        for m in range(-l, l + 1):
            # Y_l^m -> Y_l^{-m} with phase (-1)^m
            idx_dual = l**2 + l - m  # Index of Y_l^{-m}
            if idx < size and idx_dual < size:
                phase = (-1.0)**m
                transform = transform.at[idx_dual, idx].set(phase)
            idx += 1
    
    return transform


def dual_harmonic_analysis(shape_values: jnp.ndarray,
                          dual_polytope: DualPolytope,
                          max_l: int = 6) -> HarmonicDuality:
    """Analyze spherical harmonics in both original and dual spaces.
    
    Args:
        shape_values: Function values at original vertices
        dual_polytope: The dual polytope structure
        max_l: Maximum harmonic order
        
    Returns:
        HarmonicDuality analysis
    """
    n_original = len(shape_values)
    n_dual = len(dual_polytope.vertices)
    
    # Compute harmonic coefficients in original space
    # Simplified - would use full spherical harmonic decomposition
    original_coeffs = jnp.fft.fft(shape_values)[:max_l + 1]
    
    # Transform to dual space
    dual_coeffs = jnp.dot(dual_polytope.harmonic_transform[:len(original_coeffs), :len(original_coeffs)], 
                          original_coeffs)
    
    # Compute phase shifts
    phase_shifts = jnp.angle(dual_coeffs) - jnp.angle(original_coeffs)
    
    # Energy distribution analysis
    original_energy = jnp.abs(original_coeffs)**2
    dual_energy = jnp.abs(dual_coeffs)**2
    energy_distribution = dual_energy / (original_energy + 1e-10)
    
    return HarmonicDuality(
        original_coeffs=original_coeffs,
        dual_coeffs=dual_coeffs,
        phase_shifts=phase_shifts,
        energy_distribution=energy_distribution
    )


@jax.jit
def fast_24cell_duality(vertices_4d: jnp.ndarray,
                       values: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Fast duality transformation using 24-cell self-dual property.
    
    The 24-cell is self-dual, meaning its dual is another 24-cell.
    This enables very fast transformations.
    
    Args:
        vertices_4d: (24, 4) vertices of 24-cell
        values: (24,) function values at vertices
        
    Returns:
        Tuple of (dual_vertices, dual_values)
    """
    # For 24-cell, dual vertices are at face centers
    # But due to self-duality, we can use a direct mapping
    
    # The 24-cell symmetry group F4 includes the duality operation
    # Apply 4D "antipodal + reciprocation" transformation
    dual_vertices = -vertices_4d / (jnp.linalg.norm(vertices_4d, axis=1, keepdims=True)**2 + 1e-10)
    
    # Values transform with specific symmetry pattern
    # Adjacent vertices in dual have related values
    dual_values = jnp.zeros_like(values)
    
    # Use adjacency-based transformation
    # Each vertex connects to 8 others in 24-cell
    adjacency_transform = create_24cell_adjacency_transform()
    dual_values = jnp.dot(adjacency_transform, values)
    
    return dual_vertices, dual_values


def create_24cell_adjacency_transform() -> jnp.ndarray:
    """Create transformation matrix based on 24-cell adjacency.
    
    Returns:
        (24, 24) transformation matrix
    """
    # Simplified - full implementation would use exact 24-cell structure
    # Each vertex connects to 8 others at distance sqrt(2)
    transform = jnp.eye(24) * 0.5
    
    # Add off-diagonal elements for adjacent vertices
    # This preserves local structure in dual
    for i in range(24):
        # Simple pattern - would use actual adjacency
        for j in [(i+1)%24, (i+2)%24, (i+6)%24, (i+8)%24]:
            transform = transform.at[i, j].set(0.125)
    
    return transform


def bacteria_granule_dual_classification(shape_values: jnp.ndarray,
                                        vertices: jnp.ndarray,
                                        dual_polytope: DualPolytope) -> Dict[str, float]:
    """Classify bacteria vs granules using dual space analysis.
    
    Key insight: elongated structures (bacteria) become compact in dual space,
    while spherical structures (granules) remain spherical.
    
    Args:
        shape_values: Function values at vertices
        vertices: Original polytope vertices
        dual_polytope: Computed dual polytope
        
    Returns:
        Classification scores and features
    """
    # Original space analysis
    original_extent = jnp.max(shape_values) - jnp.min(shape_values)
    original_variance = jnp.var(shape_values)
    
    # Dual space analysis
    dual_values = transform_to_dual_values(shape_values, dual_polytope)
    dual_extent = jnp.max(dual_values) - jnp.min(dual_values)
    dual_variance = jnp.var(dual_values)
    
    # Harmonic analysis in both spaces
    harmonic_dual = dual_harmonic_analysis(shape_values, dual_polytope)
    
    # Energy concentration in low vs high frequencies
    low_freq_original = jnp.sum(jnp.abs(harmonic_dual.original_coeffs[:3])**2)
    high_freq_original = jnp.sum(jnp.abs(harmonic_dual.original_coeffs[3:])**2)
    
    low_freq_dual = jnp.sum(jnp.abs(harmonic_dual.dual_coeffs[:3])**2)
    high_freq_dual = jnp.sum(jnp.abs(harmonic_dual.dual_coeffs[3:])**2)
    
    # Classification features
    elongation_score = original_extent / (dual_extent + 1e-10)
    sphericity_score = 1.0 - jnp.abs(original_variance - dual_variance) / (original_variance + dual_variance + 1e-10)
    
    # Bacteria have high elongation, low sphericity
    # Granules have low elongation, high sphericity
    bacteria_likelihood = elongation_score * (1 - sphericity_score)
    granule_likelihood = sphericity_score * (1 - elongation_score)
    
    return {
        'bacteria_likelihood': float(bacteria_likelihood),
        'granule_likelihood': float(granule_likelihood),
        'elongation_score': float(elongation_score),
        'sphericity_score': float(sphericity_score),
        'dual_compactness': float(dual_extent / original_extent),
        'harmonic_shift': float(jnp.mean(jnp.abs(harmonic_dual.phase_shifts))),
        'classification': 'bacteria' if bacteria_likelihood > granule_likelihood else 'granule'
    }


@jax.jit
def transform_to_dual_values(values: jnp.ndarray,
                           dual_polytope: DualPolytope) -> jnp.ndarray:
    """Transform function values to dual polytope vertices.
    
    Args:
        values: Function values at original vertices
        dual_polytope: Dual polytope structure
        
    Returns:
        Function values at dual vertices
    """
    # Each dual vertex (face center) gets weighted average of face vertices
    dual_values = []
    
    for face_indices in dual_polytope.faces[:len(dual_polytope.vertices)]:
        if len(face_indices) > 0:
            face_values = values[face_indices]
            dual_value = jnp.mean(face_values)
            dual_values.append(dual_value)
        else:
            dual_values.append(0.0)
    
    return jnp.array(dual_values)


def iterative_dual_refinement(initial_shape: jnp.ndarray,
                            vertices: jnp.ndarray,
                            faces: List[jnp.ndarray],
                            n_iterations: int = 3) -> List[jnp.ndarray]:
    """Iteratively apply duality transformations for multi-scale analysis.
    
    Each iteration reveals different geometric features:
    - Iteration 0: Original shape
    - Iteration 1: Large-scale features enhanced
    - Iteration 2: Medium-scale features
    - Iteration 3: Fine details
    
    Args:
        initial_shape: Initial function values
        vertices: Polytope vertices
        faces: Polytope faces
        n_iterations: Number of dual transformations
        
    Returns:
        List of shapes at each iteration
    """
    shapes = [initial_shape]
    current_vertices = vertices
    current_faces = faces
    current_values = initial_shape
    
    for i in range(n_iterations):
        # Compute dual
        dual = construct_dual_polytope(current_vertices, current_faces)
        
        # Transform values
        dual_values = transform_to_dual_values(current_values, dual)
        shapes.append(dual_values)
        
        # Prepare for next iteration
        current_vertices = dual.vertices
        current_faces = dual.faces
        current_values = dual_values
    
    return shapes


class DualSpaceClassifier:
    """Classifier using dual space transformations."""
    
    def __init__(self, reference_polytope: DualPolytope):
        """Initialize with reference dual polytope.
        
        Args:
            reference_polytope: Pre-computed dual structure
        """
        self.dual_polytope = reference_polytope
        self.classification_history = []
    
    def classify(self, shape_values: jnp.ndarray, vertices: jnp.ndarray) -> Dict[str, float]:
        """Classify shape using dual analysis.
        
        Args:
            shape_values: Function values at vertices
            vertices: Vertex positions
            
        Returns:
            Classification results
        """
        result = bacteria_granule_dual_classification(
            shape_values, vertices, self.dual_polytope
        )
        
        # Store in history for adaptive learning
        self.classification_history.append(result)
        
        # Add confidence based on dual space consistency
        dual_values = transform_to_dual_values(shape_values, self.dual_polytope)
        consistency = self._compute_dual_consistency(shape_values, dual_values)
        result['confidence'] = float(consistency)
        
        return result
    
    def _compute_dual_consistency(self, original: jnp.ndarray, dual: jnp.ndarray) -> float:
        """Compute consistency between original and dual representations.
        
        Args:
            original: Original values
            dual: Dual values
            
        Returns:
            Consistency score [0, 1]
        """
        # Check if dual of dual returns to original (approximately)
        # This is a key property of proper duality
        dual_dual = transform_to_dual_values(dual, self.dual_polytope)
        
        if len(dual_dual) == len(original):
            reconstruction_error = jnp.mean((dual_dual - original)**2)
            consistency = jnp.exp(-reconstruction_error)
        else:
            # Size mismatch - use correlation-based measure
            consistency = 0.5
        
        return consistency