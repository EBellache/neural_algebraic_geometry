"""
JAX-based implementation of the 5 Platonic solids with functional programming principles.

This module provides immutable data structures and pure functions for working with
the five Platonic solids: tetrahedron, cube, octahedron, dodecahedron, and icosahedron.
Each solid is represented with its vertices, edges, faces, and symmetry group properties.

Mathematical significance:
- Platonic solids are the only convex polyhedra with congruent regular polygonal faces
- They represent fundamental symmetries in 3D space
- Their duality relationships encode deep mathematical structure
- They tile space in regular (cube) and quasicrystalline (icosahedron) patterns
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, List
from functools import partial
import numpy as np


class PlatonicSolid(NamedTuple):
    """Immutable data structure representing a Platonic solid.
    
    Attributes:
        name: Name of the solid
        vertices: (N, 3) array of vertex coordinates
        edges: (E, 2) array of vertex indices forming edges
        faces: List of arrays, each containing vertex indices for a face
        symmetry_group: Name of the symmetry group (e.g., 'A4', 'S4', 'A5')
        dual_name: Name of the dual solid
    """
    name: str
    vertices: jnp.ndarray
    edges: jnp.ndarray
    faces: List[jnp.ndarray]
    symmetry_group: str
    dual_name: str


# Golden ratio constant
PHI = (1 + jnp.sqrt(5)) / 2


@jax.jit
def normalize_to_unit_circumradius(vertices: jnp.ndarray) -> jnp.ndarray:
    """Normalize vertices to have unit circumradius (maximum distance from origin).
    
    Args:
        vertices: (N, 3) array of vertex coordinates
        
    Returns:
        Normalized vertices with unit circumradius
    """
    max_radius = jnp.max(jnp.linalg.norm(vertices, axis=1))
    return vertices / max_radius


@jax.jit
def tetrahedron_vertices() -> jnp.ndarray:
    """Generate vertices of a regular tetrahedron centered at origin.
    
    The tetrahedron has 4 vertices, 6 edges, 4 equilateral triangle faces,
    and tetrahedral symmetry group A4 (alternating group of degree 4).
    
    Returns:
        (4, 3) array of normalized vertex coordinates
    """
    # Vertices at alternating corners of a cube
    vertices = jnp.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=jnp.float32)
    
    return normalize_to_unit_circumradius(vertices)


@jax.jit
def cube_vertices() -> jnp.ndarray:
    """Generate vertices of a cube centered at origin.
    
    The cube has 8 vertices, 12 edges, 6 square faces,
    and octahedral symmetry group S4 (symmetric group of degree 4).
    
    Returns:
        (8, 3) array of normalized vertex coordinates
    """
    vertices = jnp.array([
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, 1],
        [1, -1, -1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, 1],
        [-1, -1, -1]
    ], dtype=jnp.float32)
    
    return normalize_to_unit_circumradius(vertices)


@jax.jit
def octahedron_vertices() -> jnp.ndarray:
    """Generate vertices of a regular octahedron centered at origin.
    
    The octahedron has 6 vertices, 12 edges, 8 equilateral triangle faces,
    and is the dual of the cube with the same symmetry group.
    
    Returns:
        (6, 3) array of normalized vertex coordinates
    """
    vertices = jnp.array([
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1]
    ], dtype=jnp.float32)
    
    return normalize_to_unit_circumradius(vertices)


@jax.jit
def icosahedron_vertices() -> jnp.ndarray:
    """Generate vertices of a regular icosahedron using golden ratio.
    
    The icosahedron has 12 vertices, 30 edges, 20 equilateral triangle faces,
    and icosahedral symmetry group A5 (alternating group of degree 5).
    
    Returns:
        (12, 3) array of normalized vertex coordinates
    """
    # Vertices lie on three orthogonal golden rectangles
    vertices = jnp.array([
        # Rectangle in xy-plane
        [0, 1, PHI],
        [0, 1, -PHI],
        [0, -1, PHI],
        [0, -1, -PHI],
        # Rectangle in yz-plane
        [1, PHI, 0],
        [1, -PHI, 0],
        [-1, PHI, 0],
        [-1, -PHI, 0],
        # Rectangle in xz-plane
        [PHI, 0, 1],
        [PHI, 0, -1],
        [-PHI, 0, 1],
        [-PHI, 0, -1]
    ], dtype=jnp.float32)
    
    return normalize_to_unit_circumradius(vertices)


@jax.jit
def dodecahedron_vertices() -> jnp.ndarray:
    """Generate vertices of a regular dodecahedron centered at origin.
    
    The dodecahedron has 20 vertices, 30 edges, 12 pentagonal faces,
    and is the dual of the icosahedron with the same symmetry group A5.
    
    Returns:
        (20, 3) array of normalized vertex coordinates
    """
    # Cube vertices
    cube_verts = jnp.array([
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, 1],
        [1, -1, -1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, 1],
        [-1, -1, -1]
    ], dtype=jnp.float32)
    
    # Additional vertices on coordinate axes using golden ratio
    axis_verts = jnp.array([
        [0, PHI, 1/PHI],
        [0, PHI, -1/PHI],
        [0, -PHI, 1/PHI],
        [0, -PHI, -1/PHI],
        [PHI, 1/PHI, 0],
        [PHI, -1/PHI, 0],
        [-PHI, 1/PHI, 0],
        [-PHI, -1/PHI, 0],
        [1/PHI, 0, PHI],
        [-1/PHI, 0, PHI],
        [1/PHI, 0, -PHI],
        [-1/PHI, 0, -PHI]
    ], dtype=jnp.float32)
    
    vertices = jnp.concatenate([cube_verts, axis_verts])
    return normalize_to_unit_circumradius(vertices)


def create_tetrahedron() -> PlatonicSolid:
    """Create a regular tetrahedron."""
    vertices = tetrahedron_vertices()
    
    edges = jnp.array([
        [0, 1], [0, 2], [0, 3],
        [1, 2], [1, 3], [2, 3]
    ], dtype=jnp.int32)
    
    faces = [
        jnp.array([0, 1, 2], dtype=jnp.int32),
        jnp.array([0, 1, 3], dtype=jnp.int32),
        jnp.array([0, 2, 3], dtype=jnp.int32),
        jnp.array([1, 2, 3], dtype=jnp.int32)
    ]
    
    return PlatonicSolid(
        name="tetrahedron",
        vertices=vertices,
        edges=edges,
        faces=faces,
        symmetry_group="A4",
        dual_name="tetrahedron"  # Self-dual
    )


def create_cube() -> PlatonicSolid:
    """Create a cube."""
    vertices = cube_vertices()
    
    edges = jnp.array([
        # Bottom face edges
        [0, 2], [2, 6], [6, 4], [4, 0],
        # Top face edges
        [1, 3], [3, 7], [7, 5], [5, 1],
        # Vertical edges
        [0, 1], [2, 3], [4, 5], [6, 7]
    ], dtype=jnp.int32)
    
    faces = [
        jnp.array([0, 2, 6, 4], dtype=jnp.int32),  # Bottom
        jnp.array([1, 5, 7, 3], dtype=jnp.int32),  # Top
        jnp.array([0, 1, 3, 2], dtype=jnp.int32),  # Front
        jnp.array([4, 6, 7, 5], dtype=jnp.int32),  # Back
        jnp.array([0, 4, 5, 1], dtype=jnp.int32),  # Right
        jnp.array([2, 3, 7, 6], dtype=jnp.int32)   # Left
    ]
    
    return PlatonicSolid(
        name="cube",
        vertices=vertices,
        edges=edges,
        faces=faces,
        symmetry_group="S4",
        dual_name="octahedron"
    )


def create_octahedron() -> PlatonicSolid:
    """Create a regular octahedron."""
    vertices = octahedron_vertices()
    
    edges = jnp.array([
        # Equatorial square
        [2, 0], [0, 3], [3, 1], [1, 2],
        # Top pyramid edges
        [4, 0], [4, 1], [4, 2], [4, 3],
        # Bottom pyramid edges
        [5, 0], [5, 1], [5, 2], [5, 3]
    ], dtype=jnp.int32)
    
    faces = [
        # Top pyramid faces
        jnp.array([4, 0, 2], dtype=jnp.int32),
        jnp.array([4, 2, 1], dtype=jnp.int32),
        jnp.array([4, 1, 3], dtype=jnp.int32),
        jnp.array([4, 3, 0], dtype=jnp.int32),
        # Bottom pyramid faces
        jnp.array([5, 2, 0], dtype=jnp.int32),
        jnp.array([5, 1, 2], dtype=jnp.int32),
        jnp.array([5, 3, 1], dtype=jnp.int32),
        jnp.array([5, 0, 3], dtype=jnp.int32)
    ]
    
    return PlatonicSolid(
        name="octahedron",
        vertices=vertices,
        edges=edges,
        faces=faces,
        symmetry_group="S4",
        dual_name="cube"
    )


def create_icosahedron() -> PlatonicSolid:
    """Create a regular icosahedron."""
    vertices = icosahedron_vertices()
    
    # Computing edges based on equal edge lengths
    n_verts = len(vertices)
    edges_list = []
    edge_length_squared = 4.0 / (1 + PHI**2)  # Normalized edge length squared
    
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            dist_squared = jnp.sum((vertices[i] - vertices[j])**2)
            if jnp.abs(dist_squared - edge_length_squared) < 0.01:
                edges_list.append([i, j])
    
    edges = jnp.array(edges_list, dtype=jnp.int32)
    
    # Faces are more complex - using known connectivity pattern
    # Each vertex connects to 5 others forming triangular faces
    faces = []  # Would need full connectivity analysis
    
    return PlatonicSolid(
        name="icosahedron",
        vertices=vertices,
        edges=edges,
        faces=faces,
        symmetry_group="A5",
        dual_name="dodecahedron"
    )


def create_dodecahedron() -> PlatonicSolid:
    """Create a regular dodecahedron."""
    vertices = dodecahedron_vertices()
    
    # Edges and faces would require full connectivity analysis
    edges = jnp.array([], dtype=jnp.int32)  # Placeholder
    faces = []  # Placeholder
    
    return PlatonicSolid(
        name="dodecahedron",
        vertices=vertices,
        edges=edges,
        faces=faces,
        symmetry_group="A5",
        dual_name="icosahedron"
    )


@jax.jit
def compute_face_centers(solid: PlatonicSolid) -> jnp.ndarray:
    """Compute the center points of all faces of a Platonic solid.
    
    Args:
        solid: A PlatonicSolid instance
        
    Returns:
        (F, 3) array of face center coordinates
    """
    centers = []
    for face in solid.faces:
        face_vertices = solid.vertices[face]
        center = jnp.mean(face_vertices, axis=0)
        centers.append(center)
    
    return jnp.stack(centers)


def compute_dual(solid: PlatonicSolid) -> PlatonicSolid:
    """Compute the dual of a Platonic solid by swapping vertices and face centers.
    
    The dual polytope has:
    - Vertices at the original face centers
    - Faces corresponding to original vertices
    - Same symmetry group as the original
    
    Args:
        solid: A PlatonicSolid instance
        
    Returns:
        The dual PlatonicSolid
    """
    # New vertices are the face centers of the original
    dual_vertices = compute_face_centers(solid)
    dual_vertices = normalize_to_unit_circumradius(dual_vertices)
    
    # Placeholder for edges and faces - would need full dual construction
    dual_edges = jnp.array([], dtype=jnp.int32)
    dual_faces = []
    
    return PlatonicSolid(
        name=solid.dual_name,
        vertices=dual_vertices,
        edges=dual_edges,
        faces=dual_faces,
        symmetry_group=solid.symmetry_group,
        dual_name=solid.name
    )


@partial(jax.jit, static_argnums=(1, 2, 3))
def cubic_tiling(origin: jnp.ndarray, nx: int, ny: int, nz: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate a cubic tiling of 3D space.
    
    The cube and octahedron can tile 3D space using simple translational symmetry.
    This function generates a regular grid of cube centers and orientations.
    
    Args:
        origin: (3,) array specifying the origin of the tiling
        nx, ny, nz: Number of cubes along each axis
        
    Returns:
        centers: (nx*ny*nz, 3) array of cube centers
        orientations: (nx*ny*nz, 3, 3) array of rotation matrices (all identity for cubic tiling)
    """
    # Create grid of centers
    x = jnp.arange(nx) * 2.0 + origin[0]
    y = jnp.arange(ny) * 2.0 + origin[1]
    z = jnp.arange(nz) * 2.0 + origin[2]
    
    xx, yy, zz = jnp.meshgrid(x, y, z, indexing='ij')
    centers = jnp.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
    
    # All cubes have the same orientation
    n_cubes = nx * ny * nz
    orientations = jnp.tile(jnp.eye(3)[None, :, :], (n_cubes, 1, 1))
    
    return centers, orientations


@partial(jax.jit, static_argnums=(1,))
def icosahedral_quasicrystal_tiling(origin: jnp.ndarray, n_shells: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate an icosahedral quasicrystal tiling pattern.
    
    The icosahedron and dodecahedron can form quasicrystalline tilings with
    5-fold rotational symmetry but no translational symmetry. This creates
    a Penrose-like tiling in 3D.
    
    Args:
        origin: (3,) array specifying the origin
        n_shells: Number of concentric shells to generate
        
    Returns:
        centers: (N, 3) array of icosahedron centers
        orientations: (N, 3, 3) array of rotation matrices
    """
    # Generate vertices of an icosahedron for initial directions
    ico_dirs = icosahedron_vertices()
    
    centers_list = [origin]
    orientations_list = [jnp.eye(3)]
    
    # Use golden ratio spacing for quasicrystal structure
    for shell in range(1, n_shells + 1):
        radius = shell * PHI
        
        # Place icosahedra along icosahedral directions
        for direction in ico_dirs:
            center = origin + radius * direction
            centers_list.append(center)
            
            # Random rotation for each (simplified - would use icosahedral group)
            orientations_list.append(jnp.eye(3))
    
    centers = jnp.stack(centers_list)
    orientations = jnp.stack(orientations_list)
    
    return centers, orientations


# Factory function to create all Platonic solids
def create_all_platonic_solids() -> dict:
    """Create all five Platonic solids.
    
    Returns:
        Dictionary mapping solid names to PlatonicSolid instances
    """
    return {
        "tetrahedron": create_tetrahedron(),
        "cube": create_cube(),
        "octahedron": create_octahedron(),
        "icosahedron": create_icosahedron(),
        "dodecahedron": create_dodecahedron()
    }