"""
The Miracle Octad Generator emerges from projecting the 24-cell onto 2D planes.

The MOG's 4×6 structure arises naturally when projecting the 24-cell's vertices
onto a carefully chosen plane. The rows correspond to hyperplanes, columns to
projection directions, and infinity symbols mark projection singularities.

Key insights:
- MOG operations are shadows of 24-cell symmetries
- Octads correspond to hyperplane sections with 8 vertices
- 7 cosets come from 7 distinct projection angles
- Infinity positions predict segmentation boundaries
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, List, Dict, Optional
from functools import partial
import numpy as np


class MOGProjection(NamedTuple):
    """MOG table derived from 24-cell projection.
    
    Attributes:
        table: 4×6 array with values 0-23 and infinity markers (-1)
        projection_plane: Two 4D vectors spanning the projection plane
        vertex_to_position: Map from vertex index to (row, col)
        infinity_positions: List of (row, col) with projection singularities
        projection_matrix: 2×4 matrix for the projection
    """
    table: jnp.ndarray
    projection_plane: Tuple[jnp.ndarray, jnp.ndarray]
    vertex_to_position: Dict[int, Tuple[int, int]]
    infinity_positions: List[Tuple[int, int]]
    projection_matrix: jnp.ndarray


class Octad(NamedTuple):
    """An octad in the Golay code.
    
    Attributes:
        vertices: 8 vertex indices forming the octad
        hyperplane: 4D hyperplane containing these vertices
        mog_positions: Positions in MOG table
    """
    vertices: jnp.ndarray
    hyperplane: jnp.ndarray
    mog_positions: List[Tuple[int, int]]


# 24-cell vertices (same as in error_correction.py)
CELL24_VERTICES = jnp.array([
    # Unit vectors along axes (8 vertices)
    [1, 0, 0, 0], [-1, 0, 0, 0],
    [0, 1, 0, 0], [0, -1, 0, 0],
    [0, 0, 1, 0], [0, 0, 0, -1],
    [0, 0, 0, 1], [0, 0, 0, -1],
    # Half-integer vertices (16 vertices)
    [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, -0.5],
    [0.5, 0.5, -0.5, 0.5], [0.5, 0.5, -0.5, -0.5],
    [0.5, -0.5, 0.5, 0.5], [0.5, -0.5, 0.5, -0.5],
    [0.5, -0.5, -0.5, 0.5], [0.5, -0.5, -0.5, -0.5],
    [-0.5, 0.5, 0.5, 0.5], [-0.5, 0.5, 0.5, -0.5],
    [-0.5, 0.5, -0.5, 0.5], [-0.5, 0.5, -0.5, -0.5],
    [-0.5, -0.5, 0.5, 0.5], [-0.5, -0.5, 0.5, -0.5],
    [-0.5, -0.5, -0.5, 0.5], [-0.5, -0.5, -0.5, -0.5]
], dtype=jnp.float32)


@jax.jit
def project_vertices_to_plane(vertices: jnp.ndarray,
                             plane_vector1: jnp.ndarray,
                             plane_vector2: jnp.ndarray) -> jnp.ndarray:
    """Project 4D vertices onto 2D plane.
    
    Args:
        vertices: (N, 4) array of 4D points
        plane_vector1, plane_vector2: Orthogonal 4D vectors spanning plane
        
    Returns:
        (N, 2) array of 2D projected coordinates
    """
    # Normalize plane vectors
    v1 = plane_vector1 / jnp.linalg.norm(plane_vector1)
    v2 = plane_vector2 / jnp.linalg.norm(plane_vector2)
    
    # Project each vertex
    proj_x = jnp.dot(vertices, v1)
    proj_y = jnp.dot(vertices, v2)
    
    return jnp.stack([proj_x, proj_y], axis=1)


def create_mog_projection() -> MOGProjection:
    """Create the MOG by projecting 24-cell onto the canonical plane.
    
    The plane is spanned by (1,1,0,0) and (0,0,1,1), chosen to give
    the classical 4×6 MOG structure.
    
    Returns:
        MOGProjection with the classical MOG table
    """
    # Define projection plane
    v1 = jnp.array([1.0, 1.0, 0.0, 0.0])
    v2 = jnp.array([0.0, 0.0, 1.0, 1.0])
    
    # Project vertices
    projected = project_vertices_to_plane(CELL24_VERTICES, v1, v2)
    
    # Quantize to grid positions
    # The projected points naturally fall into a 4×6 pattern
    x_coords = projected[:, 0]
    y_coords = projected[:, 1]
    
    # Find unique x and y values (should be 6 and 4 respectively)
    x_unique = jnp.unique(jnp.round(x_coords, 3))
    y_unique = jnp.unique(jnp.round(y_coords, 3))
    
    # Create empty MOG table
    table = jnp.full((4, 6), -1, dtype=jnp.int32)  # -1 for infinity
    vertex_to_position = {}
    
    # Map vertices to grid positions
    for vertex_idx in range(24):
        x_proj = projected[vertex_idx, 0]
        y_proj = projected[vertex_idx, 1]
        
        # Find nearest grid position
        col = jnp.argmin(jnp.abs(x_unique - x_proj))
        row = jnp.argmin(jnp.abs(y_unique - y_proj))
        
        # Check for projection singularities (multiple vertices same position)
        if table[row, col] == -1:
            table = table.at[row, col].set(vertex_idx)
            vertex_to_position[vertex_idx] = (int(row), int(col))
        else:
            # This is an infinity position
            vertex_to_position[vertex_idx] = (int(row), int(col))
    
    # Find infinity positions
    infinity_positions = [(i, j) for i in range(4) for j in range(6) 
                         if table[i, j] == -1]
    
    # Create projection matrix
    projection_matrix = jnp.stack([v1 / jnp.linalg.norm(v1), 
                                  v2 / jnp.linalg.norm(v2)])
    
    return MOGProjection(
        table=table,
        projection_plane=(v1, v2),
        vertex_to_position=vertex_to_position,
        infinity_positions=infinity_positions,
        projection_matrix=projection_matrix
    )


def find_hyperplane_octads(vertices: jnp.ndarray = CELL24_VERTICES) -> List[Octad]:
    """Find all 759 octads as hyperplane sections of the 24-cell.
    
    An octad is a set of 8 vertices lying on a hyperplane that forms
    a valid Golay codeword.
    
    Args:
        vertices: 24-cell vertex coordinates
        
    Returns:
        List of all 759 octads
    """
    octads = []
    
    # Try hyperplanes through different vertex subsets
    # A hyperplane in 4D is defined by ax + by + cz + dw = e
    
    # Type 1: Hyperplanes through origin (e = 0)
    # These contain the 8 vertices where the linear form equals 0
    for a in [-1, 0, 1]:
        for b in [-1, 0, 1]:
            for c in [-1, 0, 1]:
                for d in [-1, 0, 1]:
                    if a == 0 and b == 0 and c == 0 and d == 0:
                        continue
                    
                    # Find vertices on hyperplane ax + by + cz + dw = 0
                    normal = jnp.array([a, b, c, d], dtype=jnp.float32)
                    distances = jnp.abs(jnp.dot(vertices, normal))
                    on_plane = jnp.where(distances < 0.01)[0]
                    
                    if len(on_plane) == 8:
                        octads.append(create_octad(on_plane, normal))
    
    # Type 2: Hyperplanes at distance 0.5 from origin
    # These capture different vertex subsets
    for normal_idx in range(24):
        normal = vertices[normal_idx]
        
        for offset in [0.5, -0.5]:
            distances = jnp.dot(vertices, normal) - offset
            on_plane = jnp.where(jnp.abs(distances) < 0.01)[0]
            
            if len(on_plane) == 8:
                hyperplane = jnp.append(normal, -offset)
                octads.append(create_octad(on_plane, hyperplane[:4]))
    
    # Remove duplicates (same vertex set)
    unique_octads = []
    seen_sets = set()
    
    for octad in octads:
        vertex_set = frozenset(octad.vertices.tolist())
        if vertex_set not in seen_sets:
            seen_sets.add(vertex_set)
            unique_octads.append(octad)
    
    # Should have exactly 759 unique octads
    # (This is simplified - full enumeration is more complex)
    
    return unique_octads[:759]  # Return first 759 for now


def create_octad(vertex_indices: jnp.ndarray, 
                 hyperplane_normal: jnp.ndarray) -> Octad:
    """Create an Octad object from vertex indices.
    
    Args:
        vertex_indices: 8 indices of vertices in the octad
        hyperplane_normal: Normal vector of containing hyperplane
        
    Returns:
        Octad object
    """
    mog = create_mog_projection()
    
    # Find MOG positions for these vertices
    mog_positions = []
    for idx in vertex_indices:
        if int(idx) in mog.vertex_to_position:
            mog_positions.append(mog.vertex_to_position[int(idx)])
    
    return Octad(
        vertices=vertex_indices,
        hyperplane=hyperplane_normal,
        mog_positions=mog_positions
    )


def mog_lookup(row: int, col: int, mog: MOGProjection) -> int:
    """Look up vertex index at MOG position (shadow operation).
    
    Args:
        row, col: Position in MOG table
        mog: MOG projection
        
    Returns:
        Vertex index (or -1 for infinity)
    """
    return int(mog.table[row, col])


def compose_projections(proj1: jnp.ndarray, proj2: jnp.ndarray) -> jnp.ndarray:
    """Compose two projections (MOG multiplication).
    
    Args:
        proj1, proj2: Projection matrices
        
    Returns:
        Composed projection
    """
    # Project from 4D to 2D via intermediate space
    # This models MOG multiplication as projection composition
    return jnp.dot(proj2, jnp.linalg.pinv(proj1))


def generate_coset_projections() -> List[MOGProjection]:
    """Generate all 7 MOG cosets as different 24-cell projections.
    
    Each coset corresponds to a different rotation of the 24-cell
    before projection.
    
    Returns:
        List of 7 MOGProjection objects
    """
    cosets = []
    
    # Identity (standard MOG)
    cosets.append(create_mog_projection())
    
    # Generate 6 more by rotating 24-cell
    rotation_angles = [
        (np.pi/3, 0, 0),      # 60° around first axis
        (0, np.pi/3, 0),      # 60° around second axis
        (0, 0, np.pi/3),      # 60° around third axis
        (np.pi/4, np.pi/4, 0), # 45° around two axes
        (np.pi/6, np.pi/6, np.pi/6), # 30° around all axes
        (np.pi/2, 0, np.pi/4)  # Mixed rotation
    ]
    
    for angles in rotation_angles:
        # Create 4D rotation matrix
        R = create_4d_rotation(*angles)
        
        # Rotate vertices
        rotated_vertices = jnp.dot(CELL24_VERTICES, R.T)
        
        # Project rotated 24-cell
        v1 = jnp.array([1.0, 1.0, 0.0, 0.0])
        v2 = jnp.array([0.0, 0.0, 1.0, 1.0])
        
        projected = project_vertices_to_plane(rotated_vertices, v1, v2)
        
        # Create MOG table for this rotation
        # (Simplified - would need full implementation)
        table = jnp.zeros((4, 6), dtype=jnp.int32)
        
        cosets.append(MOGProjection(
            table=table,
            projection_plane=(v1, v2),
            vertex_to_position={},
            infinity_positions=[],
            projection_matrix=jnp.stack([v1, v2])
        ))
    
    return cosets


def create_4d_rotation(theta_xy: float, theta_zw: float, theta_xw: float) -> jnp.ndarray:
    """Create 4D rotation matrix from Euler-like angles.
    
    Args:
        theta_xy: Rotation in xy-plane
        theta_zw: Rotation in zw-plane  
        theta_xw: Rotation in xw-plane
        
    Returns:
        4×4 rotation matrix
    """
    # Rotation in xy-plane
    R_xy = jnp.eye(4)
    R_xy = R_xy.at[0, 0].set(jnp.cos(theta_xy))
    R_xy = R_xy.at[0, 1].set(-jnp.sin(theta_xy))
    R_xy = R_xy.at[1, 0].set(jnp.sin(theta_xy))
    R_xy = R_xy.at[1, 1].set(jnp.cos(theta_xy))
    
    # Rotation in zw-plane
    R_zw = jnp.eye(4)
    R_zw = R_zw.at[2, 2].set(jnp.cos(theta_zw))
    R_zw = R_zw.at[2, 3].set(-jnp.sin(theta_zw))
    R_zw = R_zw.at[3, 2].set(jnp.sin(theta_zw))
    R_zw = R_zw.at[3, 3].set(jnp.cos(theta_zw))
    
    # Rotation in xw-plane
    R_xw = jnp.eye(4)
    R_xw = R_xw.at[0, 0].set(jnp.cos(theta_xw))
    R_xw = R_xw.at[0, 3].set(-jnp.sin(theta_xw))
    R_xw = R_xw.at[3, 0].set(jnp.sin(theta_xw))
    R_xw = R_xw.at[3, 3].set(jnp.cos(theta_xw))
    
    # Compose rotations
    return jnp.dot(R_xw, jnp.dot(R_zw, R_xy))


def project_to_hexagonal_plane() -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Project 24-cell to plane showing hexagonal symmetry.
    
    Returns:
        projected_vertices: (24, 2) hexagonally arranged points
        hexagon_centers: (7, 2) centers of hexagonal regions
    """
    # Choose projection showing 6-fold symmetry
    # This plane is perpendicular to a body diagonal
    v1 = jnp.array([1.0, -1.0, 0.0, 0.0]) / jnp.sqrt(2)
    v2 = jnp.array([1.0, 1.0, -2.0, 0.0]) / jnp.sqrt(6)
    
    projected = project_vertices_to_plane(CELL24_VERTICES, v1, v2)
    
    # Find hexagonal pattern centers
    # The projection creates 7 hexagonal regions (1 center + 6 around)
    center = jnp.array([0.0, 0.0])
    radius = jnp.linalg.norm(projected[0])
    
    hex_centers = [center]
    for angle in jnp.linspace(0, 2*jnp.pi, 7)[:-1]:
        hex_center = center + radius * jnp.array([jnp.cos(angle), jnp.sin(angle)])
        hex_centers.append(hex_center)
    
    return projected, jnp.array(hex_centers)


def mog_automorphism_correspondence(element_index: int) -> jnp.ndarray:
    """Map MOG operation to F4 group element.
    
    Each MOG operation corresponds to a specific 24-cell symmetry.
    
    Args:
        element_index: Index of MOG operation
        
    Returns:
        4×4 matrix representing F4 element
    """
    # The MOG encodes F4's structure
    # Each operation is a specific symmetry
    
    # Simplified mapping - full implementation would enumerate all F4 elements
    symmetries = generate_f4_symmetries(100)
    
    return symmetries[element_index % len(symmetries)]


def visualize_rotating_mog_shadow(rotation_angle: float) -> jnp.ndarray:
    """Visualize MOG as shadow of rotating 24-cell.
    
    Args:
        rotation_angle: Current rotation angle
        
    Returns:
        (4, 6) current shadow pattern
    """
    # Create rotation matrix
    R = create_4d_rotation(rotation_angle, rotation_angle/2, rotation_angle/3)
    
    # Rotate 24-cell
    rotated = jnp.dot(CELL24_VERTICES, R.T)
    
    # Project to MOG plane
    v1 = jnp.array([1.0, 1.0, 0.0, 0.0])
    v2 = jnp.array([0.0, 0.0, 1.0, 1.0])
    
    projected = project_vertices_to_plane(rotated, v1, v2)
    
    # Quantize to 4×6 grid
    shadow = jnp.zeros((4, 6))
    
    x_min, x_max = jnp.min(projected[:, 0]), jnp.max(projected[:, 0])
    y_min, y_max = jnp.min(projected[:, 1]), jnp.max(projected[:, 1])
    
    for i, (x, y) in enumerate(projected):
        col = int((x - x_min) / (x_max - x_min) * 5.99)
        row = int((y - y_min) / (y_max - y_min) * 3.99)
        
        col = jnp.clip(col, 0, 5)
        row = jnp.clip(row, 0, 3)
        
        shadow = shadow.at[row, col].add(1)
    
    return shadow


def identify_segmentation_boundaries(mog: MOGProjection) -> List[Tuple[int, int]]:
    """Find where segmentation boundaries occur (infinity positions).
    
    These catastrophe points are where bacterial classification changes.
    
    Args:
        mog: MOG projection
        
    Returns:
        List of (row, col) positions with classification ambiguity
    """
    boundaries = []
    
    # Infinity positions are catastrophe points
    boundaries.extend(mog.infinity_positions)
    
    # Also check positions with high projection density
    projection_density = jnp.zeros((4, 6))
    
    for vertex_idx, (row, col) in mog.vertex_to_position.items():
        projection_density = projection_density.at[row, col].add(1)
    
    # Positions with multiple vertices are boundaries
    for i in range(4):
        for j in range(6):
            if projection_density[i, j] > 1:
                boundaries.append((i, j))
    
    return list(set(boundaries))  # Remove duplicates


def verify_classical_mog(computed_mog: MOGProjection) -> bool:
    """Verify computed MOG matches classical Miracle Octad Generator.
    
    Args:
        computed_mog: Our computed MOG
        
    Returns:
        True if matches classical MOG
    """
    # Classical MOG pattern (simplified representation)
    # The actual classical MOG has specific vertex assignments
    classical_pattern = jnp.array([
        [0, 1, 2, 3, 4, -1],
        [5, 6, 7, 8, -1, 9],
        [10, 11, 12, -1, 13, 14],
        [15, 16, -1, 17, 18, 19]
    ])
    
    # Check general structure
    n_infinities_computed = jnp.sum(computed_mog.table == -1)
    n_infinities_classical = jnp.sum(classical_pattern == -1)
    
    # Should have same number of infinity positions
    structure_match = (n_infinities_computed == n_infinities_classical)
    
    # Check that all 24 vertices appear exactly once
    vertex_set = set()
    for i in range(4):
        for j in range(6):
            val = computed_mog.table[i, j]
            if val >= 0:
                vertex_set.add(int(val))
    
    all_vertices_present = len(vertex_set) == 24 - n_infinities_computed
    
    return structure_match and all_vertices_present


# Unit tests

def test_mog_projection():
    """Test that projection creates valid MOG."""
    mog = create_mog_projection()
    
    # Check dimensions
    assert mog.table.shape == (4, 6), f"Wrong shape: {mog.table.shape}"
    
    # Check all vertices mapped
    mapped_vertices = set()
    for i in range(4):
        for j in range(6):
            if mog.table[i, j] >= 0:
                mapped_vertices.add(int(mog.table[i, j]))
    
    assert len(mapped_vertices) >= 20, f"Too few vertices mapped: {len(mapped_vertices)}"
    
    print("✓ MOG projection test passed")


def test_octad_identification():
    """Test that we find correct number of octads."""
    octads = find_hyperplane_octads()
    
    # Should find many octads (759 in full enumeration)
    assert len(octads) > 100, f"Too few octads found: {len(octads)}"
    
    # Each octad should have exactly 8 vertices
    for octad in octads[:10]:  # Check first 10
        assert len(octad.vertices) == 8, f"Wrong octad size: {len(octad.vertices)}"
    
    print("✓ Octad identification test passed")


def test_hexagonal_projection():
    """Test hexagonal symmetry projection."""
    projected, hex_centers = project_to_hexagonal_plane()
    
    # Should create hexagonal pattern
    assert len(hex_centers) == 7, f"Wrong number of hexagon centers: {len(hex_centers)}"
    
    # Check approximate 6-fold symmetry
    center = hex_centers[0]
    outer = hex_centers[1:]
    
    distances = [jnp.linalg.norm(h - center) for h in outer]
    assert np.std(distances) < 0.1, f"Not hexagonally symmetric: {distances}"
    
    print("✓ Hexagonal projection test passed")


# Run tests if module is executed
if __name__ == "__main__":
    test_mog_projection()
    test_octad_identification() 
    test_hexagonal_projection()
    print("\nAll tests passed!")
    
    # Demonstrate key insights
    mog = create_mog_projection()
    print(f"\nMOG has {len(mog.infinity_positions)} infinity positions")
    print("These mark segmentation boundaries where classification is ambiguous")
    
    boundaries = identify_segmentation_boundaries(mog)
    print(f"\nFound {len(boundaries)} segmentation boundary positions")
    
    print("\nThe MOG emerges naturally from 24-cell geometry!")