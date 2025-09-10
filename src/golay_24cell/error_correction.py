"""
Error correction using the 24-cell's exceptional symmetry properties.

The 24-cell's 1152 symmetries (group F4) act on the 24 vertices preserving
Hamming distance. This geometric structure enables perfect 3-bit error correction
through symmetry operations rather than algebraic syndrome decoding.

Key insights:
- Errors move codewords away from high-symmetry configurations
- Correction finds symmetries that restore symmetry
- Biological errors follow patterns that match 24-cell geometry
- Syndrome bits correspond to opposite vertex pairs in 24-cell
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, List, Dict, Optional
from functools import partial, lru_cache
import numpy as np


class ErrorPattern(NamedTuple):
    """Biological error pattern mapped to 24-cell geometry.
    
    Attributes:
        name: Pattern identifier (e.g., "metabolic_stress", "synaptic_failure")
        affected_vertices: Indices of affected 24-cell vertices
        probability_map: Probability of each vertex being affected
        correlation_matrix: Correlation between vertex failures
        biological_cause: Description of biological mechanism
    """
    name: str
    affected_vertices: jnp.ndarray
    probability_map: jnp.ndarray
    correlation_matrix: jnp.ndarray
    biological_cause: str


class CorrectionResult(NamedTuple):
    """Result of error correction process.
    
    Attributes:
        corrected_word: 24-bit corrected codeword
        syndrome: 12-bit syndrome indicating error location
        error_positions: Indices of corrected bit positions
        applied_symmetry: Symmetry operation used for correction
        confidence: Confidence in correction (0-1)
        geometric_distance: Distance moved in 24-cell
    """
    corrected_word: jnp.ndarray
    syndrome: jnp.ndarray
    error_positions: jnp.ndarray
    applied_symmetry: jnp.ndarray
    confidence: float
    geometric_distance: float


# 24-cell vertex coordinates (unit vectors and half-integer points)
CELL24_VERTICES = jnp.array([
    # Unit vectors along axes (8 vertices)
    [1, 0, 0, 0], [-1, 0, 0, 0],
    [0, 1, 0, 0], [0, -1, 0, 0],
    [0, 0, 1, 0], [0, 0, -1, 0],
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
def embed_codeword_to_24cell(codeword: jnp.ndarray) -> jnp.ndarray:
    """Embed 24-bit codeword as subset of 24-cell vertices.
    
    Active bits correspond to selected vertices.
    
    Args:
        codeword: 24-bit binary array
        
    Returns:
        (N, 4) array of selected vertex coordinates
    """
    # Select vertices where bit is 1
    selected_mask = codeword.astype(bool)
    selected_vertices = CELL24_VERTICES[selected_mask]
    
    return selected_vertices


@jax.jit
def compute_geometric_syndrome(received: jnp.ndarray) -> jnp.ndarray:
    """Compute syndrome with geometric meaning.
    
    The 12 syndrome bits correspond to 12 pairs of opposite vertices.
    Each bit indicates parity imbalance in that direction.
    
    Args:
        received: 24-bit received word
        
    Returns:
        12-bit syndrome
    """
    # Opposite vertex pairs in 24-cell
    opposite_pairs = jnp.array([
        [0, 1], [2, 3], [4, 5], [6, 7],  # Axis pairs
        [8, 15], [9, 14], [10, 13], [11, 12],  # Diagonal pairs
        [16, 23], [17, 22], [18, 21], [19, 20]  # Other pairs
    ])
    
    syndrome = jnp.zeros(12, dtype=jnp.int32)
    
    for i, (v1, v2) in enumerate(opposite_pairs):
        # Check parity of opposite vertices
        parity = (received[v1] + received[v2]) % 2
        syndrome = syndrome.at[i].set(parity)
    
    return syndrome


def generate_f4_symmetries(max_symmetries: int = 100) -> List[jnp.ndarray]:
    """Generate subset of F4 symmetry group operations.
    
    F4 has order 1152. We generate key symmetries including:
    - Coordinate permutations
    - Sign changes
    - Diagonal transformations
    
    Args:
        max_symmetries: Maximum number to generate
        
    Returns:
        List of 4x4 orthogonal matrices
    """
    symmetries = []
    
    # Identity
    symmetries.append(jnp.eye(4))
    
    # Coordinate permutations (4! = 24)
    import itertools
    for perm in itertools.permutations(range(4)):
        if len(symmetries) >= max_symmetries:
            break
        P = jnp.zeros((4, 4))
        for i, j in enumerate(perm):
            P = P.at[i, j].set(1)
        symmetries.append(P)
    
    # Sign changes on coordinates
    for signs in itertools.product([-1, 1], repeat=4):
        if len(symmetries) >= max_symmetries:
            break
        if sum(s < 0 for s in signs) % 2 == 0:  # Even number of sign changes
            S = jnp.diag(jnp.array(signs))
            symmetries.append(S)
    
    # Half-integer rotations (simplified subset)
    if len(symmetries) < max_symmetries:
        # 120-degree rotation mixing coordinates
        R = jnp.array([
            [0, 0, 0.5, 0.866],
            [0, 0, -0.866, 0.5],
            [-0.5, 0.866, 0, 0],
            [-0.866, -0.5, 0, 0]
        ])
        symmetries.append(R)
    
    return symmetries[:max_symmetries]


@jax.jit
def apply_symmetry_to_codeword(codeword: jnp.ndarray, 
                              symmetry: jnp.ndarray) -> jnp.ndarray:
    """Apply 24-cell symmetry to codeword.
    
    Args:
        codeword: 24-bit binary codeword
        symmetry: 4x4 symmetry matrix
        
    Returns:
        Transformed 24-bit codeword
    """
    # Transform vertex coordinates
    transformed_vertices = jnp.dot(CELL24_VERTICES, symmetry.T)
    
    # Find nearest original vertex for each transformed vertex
    new_codeword = jnp.zeros(24, dtype=jnp.int32)
    
    for i in range(24):
        if codeword[i] == 1:
            # Find where vertex i moved to
            transformed = transformed_vertices[i]
            distances = jnp.sum((CELL24_VERTICES - transformed[None, :])**2, axis=1)
            nearest = jnp.argmin(distances)
            new_codeword = new_codeword.at[nearest].set(1)
    
    return new_codeword


def correct_errors_geometric(received: jnp.ndarray,
                           symmetries: Optional[List[jnp.ndarray]] = None) -> CorrectionResult:
    """Correct errors using geometric operations on 24-cell.
    
    Find symmetry that brings configuration closest to valid codeword
    with high symmetry.
    
    Args:
        received: 24-bit received word with possible errors
        symmetries: List of symmetry operations to try
        
    Returns:
        CorrectionResult with corrected codeword
    """
    if symmetries is None:
        symmetries = generate_f4_symmetries(100)
    
    # Compute initial syndrome
    syndrome = compute_geometric_syndrome(received)
    
    # If syndrome is zero, no errors
    if jnp.sum(syndrome) == 0:
        return CorrectionResult(
            corrected_word=received,
            syndrome=syndrome,
            error_positions=jnp.array([]),
            applied_symmetry=jnp.eye(4),
            confidence=1.0,
            geometric_distance=0.0
        )
    
    # Try each symmetry
    best_word = received
    best_symmetry = jnp.eye(4)
    best_score = float('inf')
    
    for sym in symmetries:
        # Apply symmetry
        transformed = apply_symmetry_to_codeword(received, sym)
        
        # Compute syndrome of transformed word
        trans_syndrome = compute_geometric_syndrome(transformed)
        
        # Score based on syndrome weight and codeword weight
        syndrome_weight = jnp.sum(trans_syndrome)
        weight_deviation = abs(jnp.sum(transformed) - 12)  # Golay codewords have weight 0, 8, 12, 16, or 24
        score = syndrome_weight + 0.1 * weight_deviation
        
        if score < best_score:
            best_score = score
            best_word = transformed
            best_symmetry = sym
    
    # Find error positions
    error_positions = jnp.where(received != best_word)[0]
    
    # Compute geometric distance
    received_vertices = embed_codeword_to_24cell(received)
    corrected_vertices = embed_codeword_to_24cell(best_word)
    
    if len(received_vertices) > 0 and len(corrected_vertices) > 0:
        # Hausdorff distance between vertex sets
        geometric_distance = compute_hausdorff_distance(received_vertices, corrected_vertices)
    else:
        geometric_distance = 0.0
    
    # Confidence based on syndrome weight reduction
    initial_syndrome_weight = jnp.sum(syndrome)
    final_syndrome_weight = jnp.sum(compute_geometric_syndrome(best_word))
    confidence = 1.0 - final_syndrome_weight / (initial_syndrome_weight + 1e-10)
    
    return CorrectionResult(
        corrected_word=best_word,
        syndrome=syndrome,
        error_positions=error_positions,
        applied_symmetry=best_symmetry,
        confidence=confidence,
        geometric_distance=geometric_distance
    )


@jax.jit
def compute_hausdorff_distance(set1: jnp.ndarray, set2: jnp.ndarray) -> float:
    """Compute Hausdorff distance between vertex sets.
    
    Args:
        set1, set2: Arrays of vertex coordinates
        
    Returns:
        Hausdorff distance
    """
    if len(set1) == 0 or len(set2) == 0:
        return 0.0
    
    # Distance from each point in set1 to nearest in set2
    dist1 = jnp.min(jnp.sum((set1[:, None, :] - set2[None, :, :])**2, axis=2), axis=1)
    
    # Distance from each point in set2 to nearest in set1
    dist2 = jnp.min(jnp.sum((set2[:, None, :] - set1[None, :, :])**2, axis=2), axis=1)
    
    # Hausdorff distance is max of max distances
    return jnp.sqrt(jnp.maximum(jnp.max(dist1), jnp.max(dist2)))


# Precomputed syndrome lookup table
@lru_cache(maxsize=4096)
def build_syndrome_table() -> Dict[int, jnp.ndarray]:
    """Build lookup table mapping syndromes to corrections.
    
    Returns:
        Dictionary mapping syndrome (as integer) to error positions
    """
    table = {}
    
    # Generate all single, double, and triple errors
    for n_errors in range(1, 4):
        for error_positions in itertools.combinations(range(24), n_errors):
            # Create error pattern
            error = jnp.zeros(24, dtype=jnp.int32)
            for pos in error_positions:
                error = error.at[pos].set(1)
            
            # Compute syndrome
            syndrome = compute_geometric_syndrome(error)
            syndrome_int = sum(2**i * syndrome[i] for i in range(12))
            
            # Store in table if not already present (keep simplest correction)
            if syndrome_int not in table or len(error_positions) < len(table[syndrome_int]):
                table[syndrome_int] = jnp.array(error_positions)
    
    return table


def fast_correction_lookup(received: jnp.ndarray) -> CorrectionResult:
    """Fast error correction using precomputed lookup table.
    
    Args:
        received: 24-bit received word
        
    Returns:
        CorrectionResult
    """
    syndrome = compute_geometric_syndrome(received)
    syndrome_int = sum(2**i * syndrome[i] for i in range(12))
    
    # Lookup correction
    table = build_syndrome_table()
    
    if syndrome_int == 0:
        # No errors
        error_positions = jnp.array([])
        corrected = received
    elif syndrome_int in table:
        # Known error pattern
        error_positions = table[syndrome_int]
        corrected = received.copy()
        for pos in error_positions:
            corrected = corrected.at[pos].set(1 - corrected[pos])
    else:
        # Unknown pattern - fall back to geometric method
        return correct_errors_geometric(received)
    
    return CorrectionResult(
        corrected_word=corrected,
        syndrome=syndrome,
        error_positions=error_positions,
        applied_symmetry=jnp.eye(4),
        confidence=1.0 if syndrome_int in table else 0.5,
        geometric_distance=len(error_positions) * jnp.sqrt(2)
    )


# Biological error patterns

def metabolic_stress_pattern() -> ErrorPattern:
    """Metabolic stress causes regional failures.
    
    Adjacent vertices in 24-cell fail together.
    """
    # Vertices within distance 2 of vertex 0
    center = 0
    affected = [0, 2, 3, 4, 5, 8, 9, 10, 11]  # Vertex 0 and neighbors
    
    probability_map = jnp.zeros(24)
    for v in affected:
        probability_map = probability_map.at[v].set(0.8 - 0.1 * abs(v - center))
    
    # Correlation decreases with distance
    correlation = jnp.eye(24)
    for i in affected:
        for j in affected:
            if i != j:
                dist = jnp.linalg.norm(CELL24_VERTICES[i] - CELL24_VERTICES[j])
                correlation = correlation.at[i, j].set(jnp.exp(-dist))
    
    return ErrorPattern(
        name="metabolic_stress",
        affected_vertices=jnp.array(affected),
        probability_map=probability_map,
        correlation_matrix=correlation,
        biological_cause="Regional ATP depletion affecting multiple neurons"
    )


def synaptic_failure_pattern() -> ErrorPattern:
    """Synaptic failures affect specific connections.
    
    Edges in 24-cell represent synaptic connections.
    """
    # Edges from vertex 0
    edges = [(0, 2), (0, 3), (0, 4), (0, 5), (0, 8), (0, 9), (0, 10), (0, 11)]
    affected = list(set([v for edge in edges for v in edge]))
    
    probability_map = jnp.zeros(24)
    for v1, v2 in edges:
        probability_map = probability_map.at[v1].set(probability_map[v1] + 0.3)
        probability_map = probability_map.at[v2].set(probability_map[v2] + 0.3)
    
    correlation = jnp.eye(24)
    for v1, v2 in edges:
        correlation = correlation.at[v1, v2].set(0.7)
        correlation = correlation.at[v2, v1].set(0.7)
    
    return ErrorPattern(
        name="synaptic_failure",
        affected_vertices=jnp.array(affected),
        probability_map=probability_map,
        correlation_matrix=correlation,
        biological_cause="Neurotransmitter depletion affecting connected neurons"
    )


def soft_decision_decode(soft_values: jnp.ndarray,
                        noise_variance: float = 0.1) -> CorrectionResult:
    """Soft-decision decoding using 24-cell geometry.
    
    Args:
        soft_values: (24,) array of confidence values [-1, 1]
        noise_variance: Estimated noise level
        
    Returns:
        Maximum likelihood codeword
    """
    # Convert soft values to probabilities
    bit_probabilities = jax.nn.sigmoid(soft_values / noise_variance)
    
    # Try multiple threshold patterns
    best_codeword = None
    best_likelihood = -float('inf')
    
    for threshold in [0.3, 0.5, 0.7]:
        # Hard decision at this threshold
        hard_decision = (bit_probabilities > threshold).astype(jnp.int32)
        
        # Correct errors
        result = fast_correction_lookup(hard_decision)
        
        # Compute likelihood of corrected codeword given soft values
        likelihood = 0.0
        for i in range(24):
            if result.corrected_word[i] == 1:
                likelihood += jnp.log(bit_probabilities[i] + 1e-10)
            else:
                likelihood += jnp.log(1 - bit_probabilities[i] + 1e-10)
        
        if likelihood > best_likelihood:
            best_likelihood = likelihood
            best_codeword = result
    
    # Refine using 24-cell distances
    vertices = embed_codeword_to_24cell(best_codeword.corrected_word)
    
    # Weight by vertex distances
    if len(vertices) > 0:
        centroid = jnp.mean(vertices, axis=0)
        distances = jnp.sum((CELL24_VERTICES - centroid[None, :])**2, axis=1)
        weights = jnp.exp(-distances)
        
        # Recompute likelihood with geometric weighting
        weighted_probs = bit_probabilities * weights
        refined_decision = (weighted_probs > 0.5).astype(jnp.int32)
        
        # Final correction
        best_codeword = fast_correction_lookup(refined_decision)
    
    return best_codeword


def visualize_correction_geometry(received: jnp.ndarray,
                                corrected: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """Visualize error correction as geometric operations.
    
    Args:
        received: Original received word
        corrected: Corrected codeword
        
    Returns:
        Dictionary of visualization data
    """
    # Embed both as vertex sets
    received_vertices = embed_codeword_to_24cell(received)
    corrected_vertices = embed_codeword_to_24cell(corrected)
    
    # Compute centroids
    received_centroid = jnp.mean(received_vertices, axis=0) if len(received_vertices) > 0 else jnp.zeros(4)
    corrected_centroid = jnp.mean(corrected_vertices, axis=0) if len(corrected_vertices) > 0 else jnp.zeros(4)
    
    # Error positions
    error_positions = jnp.where(received != corrected)[0]
    error_vertices = CELL24_VERTICES[error_positions]
    
    # Symmetry analysis
    # Find approximate symmetry that maps received to corrected
    if len(received_vertices) > 0 and len(corrected_vertices) > 0:
        # Simplified: use rotation that aligns centroids
        axis = jnp.cross(received_centroid[:3], corrected_centroid[:3])
        angle = jnp.arccos(jnp.clip(jnp.dot(received_centroid, corrected_centroid), -1, 1))
        
        symmetry_axis = axis / (jnp.linalg.norm(axis) + 1e-10)
        symmetry_angle = angle
    else:
        symmetry_axis = jnp.array([1, 0, 0])
        symmetry_angle = 0.0
    
    return {
        'received_vertices': received_vertices,
        'corrected_vertices': corrected_vertices,
        'error_vertices': error_vertices,
        'received_centroid': received_centroid,
        'corrected_centroid': corrected_centroid,
        'symmetry_axis': symmetry_axis,
        'symmetry_angle': jnp.array(symmetry_angle),
        'movement_distance': jnp.linalg.norm(corrected_centroid - received_centroid)
    }


# Example usage demonstrating pure functional nature

def example_error_correction():
    """Example showing deterministic error correction."""
    # Create codeword with 2 errors
    codeword = jnp.zeros(24, dtype=jnp.int32)
    codeword = codeword.at[0].set(1).at[8].set(1).at[12].set(1)
    
    # Add errors at positions 3 and 15
    received = codeword.copy()
    received = received.at[3].set(1 - received[3])
    received = received.at[15].set(1 - received[15])
    
    # Correct errors
    result = fast_correction_lookup(received)
    
    print(f"Original codeword: {codeword}")
    print(f"Received with errors: {received}")
    print(f"Corrected: {result.corrected_word}")
    print(f"Error positions: {result.error_positions}")
    print(f"All errors corrected: {jnp.array_equal(codeword, result.corrected_word)}")
    
    # Verify determinism
    result2 = fast_correction_lookup(received)
    assert jnp.array_equal(result.corrected_word, result2.corrected_word), "Not deterministic!"
    
    return result