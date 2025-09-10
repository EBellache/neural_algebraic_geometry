"""
Stereographic projection from 3D sphere to 2D plane and its inverse.

This module implements conformal projection used to model how 3D polytope structures
map to 2D retinal organization. The key insight is that icosahedral symmetry projects
to hexagonal patterns, explaining retinal architecture.

Mathematical properties:
- Conformal: preserves angles locally
- Circles map to circles (or lines)
- Not area-preserving: magnification near projection pole
- Smooth except at projection point
- Enables 3D reconstruction from 2D patterns
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional, NamedTuple, List
from functools import partial
import numpy as np


class ProjectionResult(NamedTuple):
    """Result of stereographic projection including metadata.
    
    Attributes:
        points_2d: (N, 2) array of projected 2D coordinates
        magnification: (N,) array of local magnification factors
        valid_mask: (N,) boolean array indicating successful projections
        projection_center: (3,) array of projection center on sphere
    """
    points_2d: jnp.ndarray
    magnification: jnp.ndarray
    valid_mask: jnp.ndarray
    projection_center: jnp.ndarray


class HexagonalPattern(NamedTuple):
    """Hexagonal tiling pattern from projected polytope.
    
    Attributes:
        centers: (N, 2) array of hexagon centers
        vertices: (N, 6, 2) array of hexagon vertices
        connectivity: (E, 2) array of connected hexagon indices
        source_3d_indices: (N,) indices of original 3D points
    """
    centers: jnp.ndarray
    vertices: jnp.ndarray
    connectivity: jnp.ndarray
    source_3d_indices: jnp.ndarray


@jax.jit
def normalize_to_sphere(points_3d: jnp.ndarray) -> jnp.ndarray:
    """Normalize 3D points to lie on unit sphere.
    
    Args:
        points_3d: (N, 3) array of 3D coordinates
        
    Returns:
        (N, 3) array of points on unit sphere
    """
    norms = jnp.linalg.norm(points_3d, axis=1, keepdims=True)
    return jnp.where(norms > 1e-10, points_3d / norms, points_3d)


@jax.jit
def stereographic_forward(points_3d: jnp.ndarray,
                         projection_center: jnp.ndarray = None) -> ProjectionResult:
    """Project points from unit sphere to 2D plane via stereographic projection.
    
    Maps point (x,y,z) on unit sphere to (X,Y) = (x/(1-z), y/(1-z)) when
    projecting from north pole (0,0,1).
    
    Args:
        points_3d: (N, 3) array of points on unit sphere
        projection_center: (3,) projection center (default: north pole)
        
    Returns:
        ProjectionResult with 2D coordinates and metadata
    """
    if projection_center is None:
        projection_center = jnp.array([0.0, 0.0, 1.0])
    
    # Normalize inputs
    points_normalized = normalize_to_sphere(points_3d)
    center_normalized = normalize_to_sphere(projection_center[None, :])[0]
    
    # Compute projections
    # For general projection center, we need to rotate coordinates
    # Here we implement the standard case from north pole
    z_coords = points_normalized[:, 2]
    
    # Check for points at projection center
    epsilon = 1e-10
    valid_mask = jnp.abs(z_coords - 1.0) > epsilon
    
    # Safe division with masking
    denominators = jnp.where(valid_mask, 1.0 - z_coords, 1.0)
    
    # Project to 2D
    X = points_normalized[:, 0] / denominators
    Y = points_normalized[:, 1] / denominators
    points_2d = jnp.stack([X, Y], axis=-1)
    
    # Compute local magnification factor
    # Magnification = 1 / (1 - z)²
    magnification = 1.0 / (denominators ** 2)
    
    return ProjectionResult(
        points_2d=points_2d,
        magnification=magnification,
        valid_mask=valid_mask,
        projection_center=center_normalized
    )


@jax.jit
def stereographic_inverse(points_2d: jnp.ndarray,
                         projection_center: jnp.ndarray = None) -> jnp.ndarray:
    """Inverse stereographic projection from 2D plane to unit sphere.
    
    Given (X,Y) on plane, recovers (x,y,z) on unit sphere:
    x = 2X / (1 + X² + Y²)
    y = 2Y / (1 + X² + Y²)
    z = (X² + Y² - 1) / (1 + X² + Y²)
    
    Args:
        points_2d: (N, 2) array of 2D coordinates
        projection_center: (3,) projection center (default: north pole)
        
    Returns:
        (N, 3) array of points on unit sphere
    """
    if projection_center is None:
        projection_center = jnp.array([0.0, 0.0, 1.0])
    
    X = points_2d[:, 0]
    Y = points_2d[:, 1]
    
    # Compute denominator
    X_sq_Y_sq = X**2 + Y**2
    denominator = 1.0 + X_sq_Y_sq
    
    # Recover 3D coordinates
    x = 2.0 * X / denominator
    y = 2.0 * Y / denominator
    z = (X_sq_Y_sq - 1.0) / denominator
    
    points_3d = jnp.stack([x, y, z], axis=-1)
    
    # For general projection center, would need to rotate back
    return points_3d


@jax.jit
def project_icosahedron_from_face(icosahedron_vertices: jnp.ndarray,
                                  face_index: int = 0) -> Tuple[ProjectionResult, jnp.ndarray]:
    """Project icosahedron vertices from one of its face centers.
    
    This reveals the hexagonal pattern: when projecting from a triangular
    face center, the 12 vertices arrange in a hexagonal pattern with
    6-fold rotational symmetry.
    
    Args:
        icosahedron_vertices: (12, 3) array of icosahedron vertices
        face_index: Which face center to project from
        
    Returns:
        Tuple of (ProjectionResult, face_center_3d)
    """
    # Define icosahedron faces (simplified - would need full connectivity)
    # Each face is a triangle of 3 vertices
    example_faces = jnp.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4]
        # ... more faces
    ])
    
    # Compute face center
    if face_index < len(example_faces):
        face_vertices = icosahedron_vertices[example_faces[face_index]]
        face_center = jnp.mean(face_vertices, axis=0)
    else:
        # Default to first three vertices
        face_center = jnp.mean(icosahedron_vertices[:3], axis=0)
    
    face_center = normalize_to_sphere(face_center[None, :])[0]
    
    # Project from face center
    result = stereographic_forward(icosahedron_vertices, face_center)
    
    return result, face_center


@jax.jit
def extract_hexagonal_pattern(projection_result: ProjectionResult,
                             neighbor_threshold: float = 1.5) -> HexagonalPattern:
    """Extract hexagonal tiling pattern from projected points.
    
    Identifies hexagonal structure by finding nearest neighbors and
    extracting 6-fold symmetric patterns.
    
    Args:
        projection_result: Result from stereographic projection
        neighbor_threshold: Distance threshold for neighbor detection
        
    Returns:
        HexagonalPattern structure
    """
    points_2d = projection_result.points_2d
    valid_points = points_2d[projection_result.valid_mask]
    
    n_points = len(valid_points)
    
    # Compute pairwise distances
    distances = jnp.linalg.norm(
        valid_points[:, None, :] - valid_points[None, :, :],
        axis=2
    )
    
    # Find hexagonal centers (points with ~6 neighbors)
    neighbor_counts = jnp.sum(
        (distances > 0.01) & (distances < neighbor_threshold),
        axis=1
    )
    
    hex_centers = valid_points[neighbor_counts >= 5]
    
    # Generate hexagon vertices around each center
    angles = jnp.linspace(0, 2*jnp.pi, 7)[:-1]  # 6 vertices
    radius = neighbor_threshold / 2
    
    hex_vertices = []
    for center in hex_centers:
        vertices = jnp.stack([
            center[0] + radius * jnp.cos(angles),
            center[1] + radius * jnp.sin(angles)
        ], axis=-1)
        hex_vertices.append(vertices)
    
    hex_vertices = jnp.stack(hex_vertices) if hex_vertices else jnp.zeros((0, 6, 2))
    
    # Build connectivity (simplified)
    connectivity = jnp.array([[i, i+1] for i in range(len(hex_centers)-1)])
    
    # Map back to original 3D indices
    valid_indices = jnp.where(projection_result.valid_mask)[0]
    source_indices = valid_indices[:len(hex_centers)]
    
    return HexagonalPattern(
        centers=hex_centers,
        vertices=hex_vertices,
        connectivity=connectivity,
        source_3d_indices=source_indices
    )


@partial(jax.jit, static_argnums=(1,))
def reconstruct_polytope_from_partial(partial_pattern: jnp.ndarray,
                                     polytope_type: str,
                                     known_vertices: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Reconstruct full 3D polytope from partial hexagonal pattern.
    
    Uses the constraint that points must lie on a known polytope type
    to complete missing information via optimization.
    
    Args:
        partial_pattern: (M, 2) partial 2D hexagonal pattern
        polytope_type: Type of polytope ('icosahedron', 'octahedron', etc.)
        known_vertices: Optional (K, 3) known 3D vertices
        
    Returns:
        (N, 3) reconstructed polytope vertices
    """
    # Inverse project partial pattern to get initial 3D points
    partial_3d = stereographic_inverse(partial_pattern)
    
    # Define target polytope structure based on type
    if polytope_type == 'icosahedron':
        n_vertices = 12
        edge_length = 2.0 / jnp.sqrt(5.0)  # For unit circumradius
    else:
        n_vertices = 6  # Default to octahedron
        edge_length = jnp.sqrt(2.0)
    
    # If we have known vertices, use them as constraints
    if known_vertices is not None:
        # Combine known and reconstructed vertices
        all_vertices = jnp.concatenate([known_vertices, partial_3d])
    else:
        all_vertices = partial_3d
    
    # Optimization: adjust vertices to satisfy polytope constraints
    # This is simplified - full implementation would use gradient descent
    
    # Normalize to unit sphere
    reconstructed = normalize_to_sphere(all_vertices)
    
    # Ensure we have the right number of vertices
    if len(reconstructed) < n_vertices:
        # Add missing vertices by symmetry operations
        # Simplified: just duplicate and rotate
        n_missing = n_vertices - len(reconstructed)
        angle = 2 * jnp.pi / n_missing
        
        new_vertices = []
        for i in range(n_missing):
            # Rotate around z-axis
            c, s = jnp.cos(i * angle), jnp.sin(i * angle)
            rotation = jnp.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            new_vertex = jnp.dot(rotation, reconstructed[0])
            new_vertices.append(new_vertex)
        
        reconstructed = jnp.concatenate([reconstructed, jnp.stack(new_vertices)])
    
    return reconstructed[:n_vertices]


def full_segmentation_pipeline(input_3d_points: jnp.ndarray,
                              polytope_type: str = 'icosahedron') -> dict:
    """Complete pipeline: 3D polytope → projection → hexagonal → reconstruction.
    
    This demonstrates the full workflow from 3D structure through retinal
    projection to reconstructed segmentation.
    
    Args:
        input_3d_points: (N, 3) input 3D points to segment
        polytope_type: Target polytope type for segmentation
        
    Returns:
        Dictionary with all intermediate results and final segmentation
    """
    # Step 1: Normalize input to sphere
    normalized_input = normalize_to_sphere(input_3d_points)
    
    # Step 2: Find optimal projection center (simplified: use first point)
    projection_center = normalized_input[0]
    
    # Step 3: Project to 2D
    projection_result = stereographic_forward(normalized_input, projection_center)
    
    # Step 4: Extract hexagonal pattern
    hex_pattern = extract_hexagonal_pattern(projection_result)
    
    # Step 5: Feature extraction from hexagonal pattern
    # Extract geometric features: areas, angles, symmetries
    hex_areas = compute_hexagon_areas(hex_pattern.vertices)
    hex_symmetry = compute_symmetry_score(hex_pattern.centers)
    
    # Step 6: Reconstruct 3D polytope
    reconstructed = reconstruct_polytope_from_partial(
        hex_pattern.centers,
        polytope_type,
        normalized_input[:3]  # Use first 3 points as known
    )
    
    # Step 7: Fit optimal polytope to input points
    fitted_polytope = fit_polytope_to_points(reconstructed, normalized_input)
    
    return {
        'input_points': input_3d_points,
        'normalized_points': normalized_input,
        'projection_2d': projection_result,
        'hexagonal_pattern': hex_pattern,
        'features': {
            'hex_areas': hex_areas,
            'symmetry_score': hex_symmetry
        },
        'reconstructed_3d': reconstructed,
        'fitted_polytope': fitted_polytope,
        'segmentation_vertices': fitted_polytope
    }


@jax.jit
def compute_hexagon_areas(hex_vertices: jnp.ndarray) -> jnp.ndarray:
    """Compute areas of hexagons using shoelace formula.
    
    Args:
        hex_vertices: (N, 6, 2) hexagon vertices
        
    Returns:
        (N,) array of hexagon areas
    """
    # Shoelace formula for polygon area
    x = hex_vertices[:, :, 0]
    y = hex_vertices[:, :, 1]
    
    # Add first vertex at the end to close polygon
    x_closed = jnp.concatenate([x, x[:, :1]], axis=1)
    y_closed = jnp.concatenate([y, y[:, :1]], axis=1)
    
    # Compute area
    areas = 0.5 * jnp.abs(
        jnp.sum(x_closed[:, :-1] * y_closed[:, 1:] - 
                x_closed[:, 1:] * y_closed[:, :-1], axis=1)
    )
    
    return areas


@jax.jit
def compute_symmetry_score(hex_centers: jnp.ndarray) -> float:
    """Compute 6-fold rotational symmetry score of hexagonal pattern.
    
    Args:
        hex_centers: (N, 2) centers of hexagons
        
    Returns:
        Symmetry score between 0 and 1
    """
    if len(hex_centers) == 0:
        return 0.0
    
    # Center the pattern
    centroid = jnp.mean(hex_centers, axis=0)
    centered = hex_centers - centroid
    
    # Test 60-degree rotations
    angle = jnp.pi / 3  # 60 degrees
    scores = []
    
    for k in range(6):
        rotation = jnp.array([
            [jnp.cos(k * angle), -jnp.sin(k * angle)],
            [jnp.sin(k * angle), jnp.cos(k * angle)]
        ])
        
        rotated = jnp.dot(centered, rotation.T)
        
        # Find nearest neighbor distances
        min_distances = []
        for point in rotated:
            distances = jnp.linalg.norm(centered - point[None, :], axis=1)
            min_dist = jnp.min(distances[distances > 1e-6])
            min_distances.append(min_dist)
        
        # Score based on how well points match
        score = jnp.exp(-jnp.mean(jnp.array(min_distances)))
        scores.append(score)
    
    return jnp.mean(jnp.array(scores))


@jax.jit
def fit_polytope_to_points(polytope_vertices: jnp.ndarray,
                          target_points: jnp.ndarray) -> jnp.ndarray:
    """Fit polytope to target points using least squares.
    
    Args:
        polytope_vertices: (M, 3) polytope template vertices
        target_points: (N, 3) target points to fit
        
    Returns:
        (M, 3) fitted polytope vertices
    """
    # Find optimal rotation and scale
    # Simplified: just use centroid alignment
    polytope_center = jnp.mean(polytope_vertices, axis=0)
    target_center = jnp.mean(target_points, axis=0)
    
    # Translate to match centroids
    fitted = polytope_vertices - polytope_center + target_center
    
    # Scale to match average radius
    polytope_radius = jnp.mean(jnp.linalg.norm(fitted - target_center, axis=1))
    target_radius = jnp.mean(jnp.linalg.norm(target_points - target_center, axis=1))
    
    scale = target_radius / (polytope_radius + 1e-10)
    fitted = target_center + scale * (fitted - target_center)
    
    return normalize_to_sphere(fitted)


# Utility functions for visualization
def create_retinal_model(n_hexagons: int = 19) -> HexagonalPattern:
    """Create a model of retinal hexagonal organization.
    
    Args:
        n_hexagons: Number of hexagons in the pattern
        
    Returns:
        HexagonalPattern representing retinal structure
    """
    # Create hexagonal lattice
    centers = []
    radius = 1.0
    
    # Central hexagon
    centers.append(jnp.array([0.0, 0.0]))
    
    # Surrounding rings
    for ring in range(1, 3):
        n_in_ring = 6 * ring
        angle_step = 2 * jnp.pi / n_in_ring
        
        for i in range(n_in_ring):
            angle = i * angle_step
            x = ring * radius * jnp.cos(angle)
            y = ring * radius * jnp.sin(angle)
            centers.append(jnp.array([x, y]))
            
            if len(centers) >= n_hexagons:
                break
        
        if len(centers) >= n_hexagons:
            break
    
    centers = jnp.stack(centers[:n_hexagons])
    
    # Generate vertices
    angles = jnp.linspace(0, 2*jnp.pi, 7)[:-1]
    hex_radius = radius / 2
    
    vertices = []
    for center in centers:
        verts = jnp.stack([
            center[0] + hex_radius * jnp.cos(angles),
            center[1] + hex_radius * jnp.sin(angles)
        ], axis=-1)
        vertices.append(verts)
    
    vertices = jnp.stack(vertices)
    
    # Simple connectivity
    connectivity = jnp.array([[i, i+1] for i in range(n_hexagons-1)])
    
    return HexagonalPattern(
        centers=centers,
        vertices=vertices,
        connectivity=connectivity,
        source_3d_indices=jnp.arange(n_hexagons)
    )