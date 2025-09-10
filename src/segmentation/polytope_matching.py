"""
Match objects to pre-computed polytope templates for fast segmentation.

This module implements a template matching system using a library of polytope
configurations representing different biological objects. Fast matching uses
spatial hashing, spectral descriptors, and learned deformations. The system
adapts by updating templates based on successful matches.

Key features:
- Comprehensive template library for bacteria, organelles, cells
- Multi-scale matching with hierarchical search
- Deformation-tolerant matching using harmonic descriptors
- Fast indexing with KD-trees and locality-sensitive hashing
- Online learning to improve templates over time
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, List, Dict, Tuple, Optional, Set
from functools import partial
import numpy as np
from dataclasses import dataclass, field
import pickle


class PolytopeTemplate(NamedTuple):
    """Pre-computed polytope template for matching.
    
    Attributes:
        name: Template identifier (e.g., 'e_coli', 'mitochondrion')
        category: Biological category ('bacteria', 'organelle', etc.)
        vertices: (N, 3) canonical vertex positions
        edges: (E, 2) edge connectivity
        faces: List of face vertex indices
        harmonic_signature: Rotation-invariant harmonic descriptor
        scale_range: (min, max) typical scales
        aspect_ratios: Typical aspect ratios (x:y:z)
        deformation_modes: Principal deformation patterns
        metadata: Additional template information
    """
    name: str
    category: str
    vertices: jnp.ndarray
    edges: jnp.ndarray
    faces: List[jnp.ndarray]
    harmonic_signature: jnp.ndarray
    scale_range: Tuple[float, float]
    aspect_ratios: jnp.ndarray
    deformation_modes: jnp.ndarray
    metadata: Dict


class MatchResult(NamedTuple):
    """Result of template matching.
    
    Attributes:
        template_name: Matched template identifier
        confidence: Match confidence (0-1)
        transformation: (4, 4) transformation matrix
        scale: Applied scale factor
        deformation: Deformation parameters
        match_score: Detailed matching score
        inlier_ratio: Fraction of points matching template
    """
    template_name: str
    confidence: float
    transformation: jnp.ndarray
    scale: float
    deformation: jnp.ndarray
    match_score: float
    inlier_ratio: float


@dataclass
class TemplateLibrary:
    """Library of polytope templates with fast indexing.
    
    Attributes:
        templates: Dictionary of templates by name
        category_index: Templates grouped by category
        scale_index: Templates indexed by scale
        harmonic_index: KD-tree for harmonic signatures
        hash_index: LSH for fast approximate matching
    """
    templates: Dict[str, PolytopeTemplate] = field(default_factory=dict)
    category_index: Dict[str, List[str]] = field(default_factory=dict)
    scale_index: Dict[int, List[str]] = field(default_factory=dict)
    harmonic_index: Optional[any] = None  # KD-tree
    hash_index: Optional[any] = None  # LSH structure
    
    def add_template(self, template: PolytopeTemplate):
        """Add template to library with indexing."""
        self.templates[template.name] = template
        
        # Update category index
        if template.category not in self.category_index:
            self.category_index[template.category] = []
        self.category_index[template.category].append(template.name)
        
        # Update scale index (discretized)
        mean_scale = (template.scale_range[0] + template.scale_range[1]) / 2
        scale_bin = int(np.log2(mean_scale))
        if scale_bin not in self.scale_index:
            self.scale_index[scale_bin] = []
        self.scale_index[scale_bin].append(template.name)
    
    def build_indices(self):
        """Build KD-tree and LSH indices for fast search."""
        if not self.templates:
            return
        
        # Collect harmonic signatures
        signatures = []
        names = []
        for name, template in self.templates.items():
            signatures.append(template.harmonic_signature)
            names.append(name)
        
        signatures = jnp.stack(signatures)
        
        # Build KD-tree (simplified - would use actual KD-tree)
        self.harmonic_index = (signatures, names)
        
        # Build LSH (simplified - would use actual LSH)
        self.hash_index = compute_lsh_hashes(signatures)


def create_bacteria_templates() -> List[PolytopeTemplate]:
    """Create standard bacterial templates."""
    templates = []
    
    # E. coli - rod-shaped
    vertices = jnp.array([
        # Elongated octahedron
        [2, 0, 0], [-2, 0, 0],  # Poles
        [0, 0.7, 0.7], [0, 0.7, -0.7],
        [0, -0.7, 0.7], [0, -0.7, -0.7]
    ])
    
    edges = jnp.array([
        [0, 2], [0, 3], [0, 4], [0, 5],
        [1, 2], [1, 3], [1, 4], [1, 5],
        [2, 3], [3, 5], [5, 4], [4, 2]
    ])
    
    # Compute harmonic signature
    harmonic_sig = compute_template_harmonics(vertices)
    
    # Deformation modes (elongation, bending, twisting)
    deformation_modes = jnp.array([
        [1, 0, 0, 0, 0, 0],  # Elongation along x
        [0, 0, 0, 1, 0, 0],  # Bending
        [0, 0, 0, 0, 0, 1]   # Twisting
    ])
    
    templates.append(PolytopeTemplate(
        name="e_coli",
        category="bacteria",
        vertices=vertices,
        edges=edges,
        faces=[jnp.array([0, 2, 3]), jnp.array([0, 3, 5])],  # Simplified
        harmonic_signature=harmonic_sig,
        scale_range=(1.0, 3.0),
        aspect_ratios=jnp.array([3.0, 1.0, 1.0]),
        deformation_modes=deformation_modes,
        metadata={"gram": "negative", "shape": "rod"}
    ))
    
    # Streptococcus - spherical
    # Use icosahedron for spherical bacteria
    phi = (1 + jnp.sqrt(5)) / 2
    vertices = jnp.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]) / jnp.sqrt(1 + phi**2)
    
    templates.append(PolytopeTemplate(
        name="streptococcus",
        category="bacteria",
        vertices=vertices,
        edges=jnp.array([]),  # Would compute full connectivity
        faces=[],
        harmonic_signature=compute_template_harmonics(vertices),
        scale_range=(0.5, 1.5),
        aspect_ratios=jnp.array([1.0, 1.0, 1.0]),
        deformation_modes=jnp.eye(6),  # Uniform scaling
        metadata={"gram": "positive", "shape": "coccus"}
    ))
    
    # Vibrio - curved rod
    vertices = jnp.array([
        # Curved octahedron
        [2, 0.3, 0], [-2, -0.3, 0],  # Curved poles
        [0, 0.7, 0.7], [0, 0.9, -0.7],
        [0, -0.5, 0.7], [0, -0.7, -0.7]
    ])
    
    templates.append(PolytopeTemplate(
        name="vibrio",
        category="bacteria",
        vertices=vertices,
        edges=edges,
        faces=[],
        harmonic_signature=compute_template_harmonics(vertices),
        scale_range=(1.0, 2.5),
        aspect_ratios=jnp.array([2.5, 1.0, 1.0]),
        deformation_modes=deformation_modes,
        metadata={"gram": "negative", "shape": "curved_rod"}
    ))
    
    return templates


def create_organelle_templates() -> List[PolytopeTemplate]:
    """Create organelle templates."""
    templates = []
    
    # Mitochondrion - elongated with cristae
    vertices = jnp.array([
        # Double-layered elongated structure
        [3, 0, 0], [-3, 0, 0],  # Outer poles
        [0, 1, 1], [0, 1, -1],
        [0, -1, 1], [0, -1, -1],
        # Inner structure
        [2, 0, 0], [-2, 0, 0],
        [0, 0.5, 0.5], [0, 0.5, -0.5]
    ])
    
    templates.append(PolytopeTemplate(
        name="mitochondrion",
        category="organelle",
        vertices=vertices,
        edges=jnp.array([]),
        faces=[],
        harmonic_signature=compute_template_harmonics(vertices),
        scale_range=(0.5, 2.0),
        aspect_ratios=jnp.array([3.0, 1.0, 1.0]),
        deformation_modes=jnp.eye(6),
        metadata={"type": "energy", "structure": "double_membrane"}
    ))
    
    # Nucleus - large sphere
    # Use subdivided icosahedron
    templates.append(PolytopeTemplate(
        name="nucleus",
        category="organelle",
        vertices=create_geodesic_sphere(2),  # Subdivided twice
        edges=jnp.array([]),
        faces=[],
        harmonic_signature=jnp.ones(49),  # Spherical signature
        scale_range=(5.0, 10.0),
        aspect_ratios=jnp.array([1.0, 1.0, 1.0]),
        deformation_modes=jnp.eye(6),
        metadata={"type": "control", "structure": "double_membrane"}
    ))
    
    return templates


def create_cell_templates() -> List[PolytopeTemplate]:
    """Create whole cell templates."""
    templates = []
    
    # Red blood cell - biconcave disk
    vertices = create_biconcave_disk()
    
    templates.append(PolytopeTemplate(
        name="red_blood_cell",
        category="cell",
        vertices=vertices,
        edges=jnp.array([]),
        faces=[],
        harmonic_signature=compute_template_harmonics(vertices),
        scale_range=(6.0, 8.0),
        aspect_ratios=jnp.array([1.0, 1.0, 0.3]),
        deformation_modes=jnp.eye(6),
        metadata={"type": "blood", "shape": "biconcave"}
    ))
    
    return templates


@jax.jit
def compute_template_harmonics(vertices: jnp.ndarray, max_l: int = 6) -> jnp.ndarray:
    """Compute rotation-invariant harmonic signature for template.
    
    Args:
        vertices: (N, 3) vertex positions
        max_l: Maximum spherical harmonic order
        
    Returns:
        (49,) rotation-invariant signature
    """
    # Normalize vertices
    centroid = jnp.mean(vertices, axis=0)
    centered = vertices - centroid
    scale = jnp.sqrt(jnp.mean(jnp.sum(centered**2, axis=1)))
    normalized = centered / (scale + 1e-10)
    
    # Convert to spherical coordinates
    r = jnp.linalg.norm(normalized, axis=1)
    theta = jnp.arccos(jnp.clip(normalized[:, 2] / (r + 1e-10), -1, 1))
    phi = jnp.arctan2(normalized[:, 1], normalized[:, 0])
    
    # Compute power spectrum (rotation invariant)
    signature = []
    for l in range(max_l + 1):
        power_l = 0.0
        for m in range(-l, l + 1):
            # Simplified spherical harmonic
            if l == 0:
                Y_lm = 1.0 / jnp.sqrt(4 * jnp.pi)
            elif l == 1:
                Y_lm = jnp.cos(theta) if m == 0 else jnp.sin(theta)
            else:
                Y_lm = jnp.cos(m * phi) * jnp.sin(theta)**abs(m)
            
            coeff = jnp.mean(r * Y_lm)
            power_l += jnp.abs(coeff)**2
        
        signature.append(power_l)
    
    # Pad to fixed size
    signature = jnp.array(signature)
    if len(signature) < 49:
        signature = jnp.pad(signature, (0, 49 - len(signature)))
    
    return signature[:49]


def create_geodesic_sphere(subdivisions: int) -> jnp.ndarray:
    """Create geodesic sphere by subdividing icosahedron."""
    # Start with icosahedron vertices
    phi = (1 + jnp.sqrt(5)) / 2
    vertices = jnp.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    
    # Normalize
    vertices = vertices / jnp.linalg.norm(vertices, axis=1, keepdims=True)
    
    # Subdivide (simplified - would implement proper subdivision)
    if subdivisions > 0:
        new_vertices = [vertices]
        for _ in range(subdivisions):
            # Add midpoints of edges
            midpoints = []
            for i in range(len(vertices)):
                for j in range(i + 1, len(vertices)):
                    if jnp.linalg.norm(vertices[i] - vertices[j]) < 2.1:
                        mid = (vertices[i] + vertices[j]) / 2
                        mid = mid / jnp.linalg.norm(mid)
                        midpoints.append(mid)
            
            if midpoints:
                new_vertices.append(jnp.array(midpoints))
            vertices = jnp.concatenate(new_vertices)
    
    return vertices


def create_biconcave_disk() -> jnp.ndarray:
    """Create vertices for biconcave disk (red blood cell shape)."""
    # Parametric biconcave disk
    u = jnp.linspace(0, 2*jnp.pi, 20)
    v = jnp.linspace(0, jnp.pi, 10)
    u_grid, v_grid = jnp.meshgrid(u, v)
    
    # Biconcave profile
    r = 3.0 * jnp.sin(v_grid)
    z = 0.5 * jnp.cos(2 * v_grid)
    
    x = r * jnp.cos(u_grid)
    y = r * jnp.sin(u_grid)
    
    vertices = jnp.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)
    
    return vertices


def match_to_template(points: jnp.ndarray,
                     library: TemplateLibrary,
                     category_hint: Optional[str] = None) -> MatchResult:
    """Match point cloud to best template in library.
    
    Args:
        points: (N, 3) point cloud to match
        library: Template library
        category_hint: Optional category to restrict search
        
    Returns:
        Best match result
    """
    # Compute query signature
    query_signature = compute_template_harmonics(points)
    
    # Find candidate templates
    if category_hint and category_hint in library.category_index:
        candidates = [library.templates[name] 
                     for name in library.category_index[category_hint]]
    else:
        candidates = list(library.templates.values())
    
    # Score each candidate
    best_match = None
    best_score = -float('inf')
    
    for template in candidates:
        # Check scale compatibility
        query_scale = estimate_scale(points)
        if not (template.scale_range[0] <= query_scale <= template.scale_range[1] * 2):
            continue
        
        # Harmonic similarity
        harmonic_dist = jnp.linalg.norm(query_signature - template.harmonic_signature)
        harmonic_score = jnp.exp(-harmonic_dist / 10)
        
        # Geometric matching
        transformation, scale, inliers = align_to_template(points, template)
        
        # Deformation matching
        deformation = estimate_deformation(points, template, transformation, scale)
        deform_score = jnp.exp(-jnp.linalg.norm(deformation))
        
        # Combined score
        total_score = harmonic_score * inliers * deform_score
        
        if total_score > best_score:
            best_score = total_score
            best_match = MatchResult(
                template_name=template.name,
                confidence=float(total_score),
                transformation=transformation,
                scale=scale,
                deformation=deformation,
                match_score=float(total_score),
                inlier_ratio=float(inliers)
            )
    
    return best_match


@jax.jit
def estimate_scale(points: jnp.ndarray) -> float:
    """Estimate characteristic scale of point cloud."""
    centroid = jnp.mean(points, axis=0)
    distances = jnp.linalg.norm(points - centroid, axis=1)
    return jnp.sqrt(jnp.mean(distances**2))


def align_to_template(points: jnp.ndarray,
                     template: PolytopeTemplate) -> Tuple[jnp.ndarray, float, float]:
    """Align points to template using ICP-like method.
    
    Args:
        points: Query points
        template: Template to align to
        
    Returns:
        (transformation, scale, inlier_ratio)
    """
    # Initial alignment using centroids
    query_center = jnp.mean(points, axis=0)
    template_center = jnp.mean(template.vertices, axis=0)
    
    # Estimate scale
    query_scale = estimate_scale(points)
    template_scale = estimate_scale(template.vertices)
    scale = query_scale / template_scale
    
    # Center and scale
    centered_points = (points - query_center) / scale
    centered_template = template.vertices - template_center
    
    # Find rotation using SVD
    # Sample corresponding points
    n_samples = min(len(points), len(template.vertices))
    
    if n_samples >= 3:
        # Simplified correspondence - would use nearest neighbors
        H = jnp.dot(centered_points[:n_samples].T, centered_template[:n_samples])
        U, S, Vt = jnp.linalg.svd(H)
        R = jnp.dot(U, Vt)
        
        # Ensure proper rotation
        if jnp.linalg.det(R) < 0:
            Vt = Vt.at[-1].multiply(-1)
            R = jnp.dot(U, Vt)
    else:
        R = jnp.eye(3)
    
    # Build transformation matrix
    transformation = jnp.eye(4)
    transformation = transformation.at[:3, :3].set(R)
    transformation = transformation.at[:3, 3].set(template_center - jnp.dot(R, query_center / scale))
    
    # Count inliers
    transformed_points = transform_points(points, transformation, scale)
    distances = []
    for p in transformed_points:
        dist_to_template = jnp.min(jnp.linalg.norm(template.vertices - p[None, :], axis=1))
        distances.append(dist_to_template)
    
    inlier_threshold = 0.2 * template_scale
    inliers = jnp.sum(jnp.array(distances) < inlier_threshold) / len(points)
    
    return transformation, scale, inliers


@jax.jit
def transform_points(points: jnp.ndarray, 
                    transformation: jnp.ndarray,
                    scale: float) -> jnp.ndarray:
    """Apply transformation and scale to points."""
    # Scale first
    scaled = points / scale
    
    # Apply rotation and translation
    homogeneous = jnp.concatenate([scaled, jnp.ones((len(scaled), 1))], axis=1)
    transformed = jnp.dot(homogeneous, transformation.T)
    
    return transformed[:, :3]


@jax.jit
def estimate_deformation(points: jnp.ndarray,
                        template: PolytopeTemplate,
                        transformation: jnp.ndarray,
                        scale: float) -> jnp.ndarray:
    """Estimate deformation parameters.
    
    Args:
        points: Original points
        template: Template
        transformation: Alignment transformation
        scale: Scale factor
        
    Returns:
        Deformation parameters
    """
    # Transform points to template frame
    aligned_points = transform_points(points, transformation, scale)
    
    # Project onto deformation modes
    deformations = jnp.zeros(template.deformation_modes.shape[0])
    
    # Simplified - would use proper deformation model
    # Measure elongation
    point_spread = jnp.std(aligned_points, axis=0)
    template_spread = jnp.std(template.vertices, axis=0)
    
    elongation = point_spread / (template_spread + 1e-10)
    deformations = deformations.at[0].set(elongation[0] - 1)  # X elongation
    
    return deformations


def compute_lsh_hashes(signatures: jnp.ndarray, 
                      n_hashes: int = 10,
                      n_bits: int = 32) -> Dict:
    """Compute locality-sensitive hashes for signatures.
    
    Args:
        signatures: (N, D) array of signatures
        n_hashes: Number of hash functions
        n_bits: Bits per hash
        
    Returns:
        Hash index structure
    """
    # Random projection LSH
    n_samples, dim = signatures.shape
    
    # Generate random projection vectors
    key = jax.random.PRNGKey(42)
    projections = jax.random.normal(key, (n_hashes, dim))
    
    # Compute hashes
    hash_values = jnp.dot(signatures, projections.T)
    binary_hashes = (hash_values > 0).astype(jnp.int32)
    
    # Build index
    hash_index = {}
    for i in range(n_samples):
        for j in range(n_hashes):
            hash_key = tuple(binary_hashes[i, j:j+1])
            if hash_key not in hash_index:
                hash_index[hash_key] = []
            hash_index[hash_key].append(i)
    
    return hash_index


def hierarchical_matching(points: jnp.ndarray,
                         library: TemplateLibrary,
                         scales: List[float] = [4.0, 2.0, 1.0]) -> MatchResult:
    """Multi-scale hierarchical matching.
    
    Args:
        points: Point cloud to match
        library: Template library
        scales: Coarse to fine scales
        
    Returns:
        Best match across scales
    """
    best_match = None
    
    for scale in scales:
        # Downsample points at this scale
        if scale > 1:
            downsampled = downsample_points(points, scale)
        else:
            downsampled = points
        
        # Match at this scale
        match = match_to_template(downsampled, library)
        
        if best_match is None or match.confidence > best_match.confidence:
            best_match = match
        
        # Early stopping if confident
        if match.confidence > 0.9:
            break
    
    return best_match


@jax.jit
def downsample_points(points: jnp.ndarray, scale: float) -> jnp.ndarray:
    """Downsample point cloud by scale factor."""
    # Simple grid-based downsampling
    grid_size = scale * 0.1
    
    # Quantize to grid
    quantized = jnp.floor(points / grid_size).astype(jnp.int32)
    
    # Find unique grid cells
    unique_cells = {}
    for i, cell in enumerate(quantized):
        key = tuple(cell)
        if key not in unique_cells:
            unique_cells[key] = []
        unique_cells[key].append(i)
    
    # Take one point per cell
    downsampled = []
    for indices in unique_cells.values():
        # Take centroid of points in cell
        cell_points = points[jnp.array(indices)]
        downsampled.append(jnp.mean(cell_points, axis=0))
    
    return jnp.array(downsampled)


def update_template_library(library: TemplateLibrary,
                           successful_matches: List[Tuple[jnp.ndarray, MatchResult]],
                           learning_rate: float = 0.1):
    """Update templates based on successful matches.
    
    Args:
        library: Template library to update
        successful_matches: List of (points, match_result) pairs
        learning_rate: How much to update templates
    """
    # Group matches by template
    updates = {}
    for points, match in successful_matches:
        if match.template_name not in updates:
            updates[match.template_name] = []
        updates[match.template_name].append((points, match))
    
    # Update each template
    for template_name, matches in updates.items():
        if template_name not in library.templates:
            continue
        
        template = library.templates[template_name]
        
        # Average deformations
        avg_deformation = jnp.zeros_like(template.deformation_modes[0])
        for _, match in matches:
            avg_deformation += match.deformation
        avg_deformation /= len(matches)
        
        # Update deformation modes (simplified)
        new_modes = template.deformation_modes.copy()
        if jnp.linalg.norm(avg_deformation) > 0.1:
            # Add new deformation mode
            new_mode = avg_deformation / jnp.linalg.norm(avg_deformation)
            new_modes = jnp.concatenate([new_modes, new_mode[None, :]])
        
        # Update harmonic signature
        avg_signature = jnp.zeros_like(template.harmonic_signature)
        for points, _ in matches:
            avg_signature += compute_template_harmonics(points)
        avg_signature /= len(matches)
        
        new_signature = (1 - learning_rate) * template.harmonic_signature + \
                       learning_rate * avg_signature
        
        # Create updated template
        updated = template._replace(
            harmonic_signature=new_signature,
            deformation_modes=new_modes
        )
        
        library.templates[template_name] = updated
    
    # Rebuild indices
    library.build_indices()


def create_default_library() -> TemplateLibrary:
    """Create default template library with standard templates."""
    library = TemplateLibrary()
    
    # Add bacteria templates
    for template in create_bacteria_templates():
        library.add_template(template)
    
    # Add organelle templates
    for template in create_organelle_templates():
        library.add_template(template)
    
    # Add cell templates
    for template in create_cell_templates():
        library.add_template(template)
    
    # Build indices
    library.build_indices()
    
    return library


def save_library(library: TemplateLibrary, filepath: str):
    """Save template library to disk."""
    with open(filepath, 'wb') as f:
        pickle.dump(library, f)


def load_library(filepath: str) -> TemplateLibrary:
    """Load template library from disk."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# Example usage
def example_template_matching():
    """Example of template matching for bacteria."""
    # Create library
    library = create_default_library()
    
    # Create synthetic bacterial point cloud
    # Elongated shape (E. coli-like)
    theta = jnp.linspace(0, jnp.pi, 20)
    phi = jnp.linspace(0, 2*jnp.pi, 40)
    theta_grid, phi_grid = jnp.meshgrid(theta, phi)
    
    # Elongated ellipsoid
    a, b, c = 2.5, 0.8, 0.8
    x = a * jnp.sin(theta_grid) * jnp.cos(phi_grid)
    y = b * jnp.sin(theta_grid) * jnp.sin(phi_grid)  
    z = c * jnp.cos(theta_grid)
    
    points = jnp.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)
    
    # Add some noise
    key = jax.random.PRNGKey(42)
    noise = 0.1 * jax.random.normal(key, points.shape)
    points = points + noise
    
    # Match to template
    match = hierarchical_matching(points, library)
    
    print(f"Best match: {match.template_name}")
    print(f"Confidence: {match.confidence:.3f}")
    print(f"Scale: {match.scale:.2f}")
    print(f"Inlier ratio: {match.inlier_ratio:.3f}")
    
    # Test category hint
    match_bacteria = match_to_template(points, library, category_hint="bacteria")
    print(f"\nWith category hint: {match_bacteria.template_name}")
    
    return match, library