"""
Sparse Distributed Representations (SDRs) encoding polytope-based neural states.

This module implements 2048-bit SDRs with clear geometric structure:
- 24 bits: 24-cell vertex (one-hot encoding)
- 120 bits: Icosahedral harmonic coefficients (5 bits × 24 modes)
- 384 bits: Octahedral tiling position (64 octahedra × 6 bits)
- 720 bits: Fine tetrahedral details
- 800 bits: Multi-scale integration

All operations are pure functions preserving geometric relationships.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, List, Callable, Any
from functools import partial
import numpy as np


# SDR structure constants
SDR_BITS = 2048
CELL24_BITS = 24
HARMONIC_BITS = 120
OCTAHEDRAL_BITS = 384
TETRAHEDRAL_BITS = 720
MULTISCALE_BITS = 800

# Bit range slices
CELL24_RANGE = (0, 24)
HARMONIC_RANGE = (24, 144)
OCTAHEDRAL_RANGE = (144, 528)
TETRAHEDRAL_RANGE = (528, 1248)
MULTISCALE_RANGE = (1248, 2048)


class PolytopeSDR(NamedTuple):
    """Immutable SDR representing a polytope-based neural state.
    
    Attributes:
        bits: (2048,) boolean array
        sparsity: Fraction of active bits
        metadata: Additional information about encoding
    """
    bits: jnp.ndarray
    sparsity: float
    metadata: dict


class SDRSegment(NamedTuple):
    """A segment of the full SDR with semantic meaning.
    
    Attributes:
        name: Segment identifier
        bits: Boolean array for this segment
        start_idx: Starting index in full SDR
        end_idx: Ending index in full SDR
    """
    name: str
    bits: jnp.ndarray
    start_idx: int
    end_idx: int


@jax.jit
def create_empty_sdr() -> PolytopeSDR:
    """Create an empty SDR with all bits off."""
    bits = jnp.zeros(SDR_BITS, dtype=bool)
    return PolytopeSDR(
        bits=bits,
        sparsity=0.0,
        metadata={}
    )


@jax.jit
def encode_24cell_vertex(vertex_index: int) -> jnp.ndarray:
    """Encode 24-cell vertex as one-hot vector.
    
    Args:
        vertex_index: Index of vertex (0-23)
        
    Returns:
        (24,) boolean array with single active bit
    """
    bits = jnp.zeros(CELL24_BITS, dtype=bool)
    bits = bits.at[vertex_index % CELL24_BITS].set(True)
    return bits


@jax.jit
def encode_harmonic_coefficients(coefficients: jnp.ndarray,
                                quantization_levels: int = 5) -> jnp.ndarray:
    """Encode icosahedral harmonic coefficients into SDR bits.
    
    Args:
        coefficients: (24,) array of harmonic coefficients
        quantization_levels: Number of quantization levels per coefficient
        
    Returns:
        (120,) boolean array encoding quantized coefficients
    """
    # Normalize coefficients to [0, 1]
    min_coeff = jnp.min(coefficients)
    max_coeff = jnp.max(coefficients)
    range_coeff = max_coeff - min_coeff + 1e-10
    
    normalized = (coefficients - min_coeff) / range_coeff
    
    # Quantize to levels
    quantized = jnp.floor(normalized * (quantization_levels - 1)).astype(int)
    quantized = jnp.clip(quantized, 0, quantization_levels - 1)
    
    # Encode as thermometer code (more similar values = more overlap)
    bits = jnp.zeros(HARMONIC_BITS, dtype=bool)
    
    for i in range(24):
        start_idx = i * quantization_levels
        level = quantized[i]
        # Set bits up to the quantization level
        for j in range(level + 1):
            bits = bits.at[start_idx + j].set(True)
    
    return bits


@jax.jit
def encode_octahedral_position(octahedron_index: int,
                              face_index: int) -> jnp.ndarray:
    """Encode position in octahedral tiling.
    
    Args:
        octahedron_index: Which octahedron in tiling (0-63)
        face_index: Which face of the octahedron (0-7)
        
    Returns:
        (384,) boolean array encoding position
    """
    bits = jnp.zeros(OCTAHEDRAL_BITS, dtype=bool)
    
    # Each octahedron gets 6 bits
    oct_start = (octahedron_index % 64) * 6
    
    # Encode face as distributed pattern for overlap
    # Adjacent faces share some bits
    face_patterns = jnp.array([
        [1, 1, 0, 0, 0, 0],  # Face 0
        [0, 1, 1, 0, 0, 0],  # Face 1
        [0, 0, 1, 1, 0, 0],  # Face 2
        [0, 0, 0, 1, 1, 0],  # Face 3
        [0, 0, 0, 0, 1, 1],  # Face 4
        [1, 0, 0, 0, 0, 1],  # Face 5
        [1, 0, 1, 0, 1, 0],  # Face 6
        [0, 1, 0, 1, 0, 1],  # Face 7
    ], dtype=bool)
    
    pattern = face_patterns[face_index % 8]
    
    for i in range(6):
        if oct_start + i < OCTAHEDRAL_BITS:
            bits = bits.at[oct_start + i].set(pattern[i])
    
    return bits


@jax.jit
def encode_tetrahedral_detail(position: jnp.ndarray,
                             resolution: int = 10) -> jnp.ndarray:
    """Encode fine-grained position using tetrahedral subdivision.
    
    Args:
        position: (3,) position vector
        resolution: Subdivision level
        
    Returns:
        (720,) boolean array encoding detailed position
    """
    bits = jnp.zeros(TETRAHEDRAL_BITS, dtype=bool)
    
    # Hash position to tetrahedral grid
    # This creates spatial locality: nearby positions activate similar bits
    
    # Quantize position to grid
    grid_pos = jnp.floor(position * resolution).astype(int)
    
    # Use multiple hash functions for distributed representation
    n_hashes = 10
    bits_per_hash = TETRAHEDRAL_BITS // n_hashes
    
    for h in range(n_hashes):
        # Different hash function for each segment
        hash_val = jnp.sum(grid_pos * jnp.array([73, 97, 113]) * (h + 1))
        hash_val = hash_val % bits_per_hash
        
        start_idx = h * bits_per_hash
        
        # Activate a local neighborhood of bits
        for offset in range(-2, 3):
            bit_idx = (hash_val + offset) % bits_per_hash
            bits = bits.at[start_idx + bit_idx].set(True)
    
    return bits


@jax.jit
def encode_multiscale(position: jnp.ndarray,
                     scales: jnp.ndarray = None) -> jnp.ndarray:
    """Encode multi-scale spatial information.
    
    Args:
        position: (3,) position vector
        scales: Array of scale factors
        
    Returns:
        (800,) boolean array encoding multi-scale features
    """
    if scales is None:
        scales = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0])
    
    bits = jnp.zeros(MULTISCALE_BITS, dtype=bool)
    bits_per_scale = MULTISCALE_BITS // len(scales)
    
    for i, scale in enumerate(scales):
        # Sample position at different scales
        scaled_pos = position * scale
        
        # Convert to spherical coordinates for rotation invariance
        r = jnp.linalg.norm(scaled_pos)
        theta = jnp.arccos(jnp.clip(scaled_pos[2] / (r + 1e-10), -1, 1))
        phi = jnp.arctan2(scaled_pos[1], scaled_pos[0])
        
        # Quantize spherical coordinates
        r_quant = jnp.floor(r * 10).astype(int) % 20
        theta_quant = jnp.floor(theta * 10 / jnp.pi).astype(int) % 10
        phi_quant = jnp.floor((phi + jnp.pi) * 10 / (2 * jnp.pi)).astype(int) % 10
        
        # Encode in SDR segment
        start_idx = i * bits_per_scale
        
        # Distributed encoding
        bits = bits.at[start_idx + r_quant].set(True)
        bits = bits.at[start_idx + 20 + theta_quant].set(True)
        bits = bits.at[start_idx + 30 + phi_quant].set(True)
        
        # Add some redundancy for robustness
        bits = bits.at[start_idx + 40 + (r_quant + theta_quant) % 20].set(True)
    
    return bits


def encode_polytope_state(position: jnp.ndarray,
                         harmonic_coeffs: jnp.ndarray,
                         cell24_vertex: int,
                         octahedron_idx: int,
                         face_idx: int) -> PolytopeSDR:
    """Encode complete polytope state into SDR.
    
    Args:
        position: (3,) position vector
        harmonic_coeffs: (24,) harmonic coefficients
        cell24_vertex: 24-cell vertex index
        octahedron_idx: Octahedron index in tiling
        face_idx: Face index on octahedron
        
    Returns:
        Complete PolytopeSDR
    """
    # Initialize empty SDR
    bits = jnp.zeros(SDR_BITS, dtype=bool)
    
    # Encode each segment
    cell24_bits = encode_24cell_vertex(cell24_vertex)
    harmonic_bits = encode_harmonic_coefficients(harmonic_coeffs)
    octahedral_bits = encode_octahedral_position(octahedron_idx, face_idx)
    tetrahedral_bits = encode_tetrahedral_detail(position)
    multiscale_bits = encode_multiscale(position)
    
    # Assemble full SDR
    bits = bits.at[CELL24_RANGE[0]:CELL24_RANGE[1]].set(cell24_bits)
    bits = bits.at[HARMONIC_RANGE[0]:HARMONIC_RANGE[1]].set(harmonic_bits)
    bits = bits.at[OCTAHEDRAL_RANGE[0]:OCTAHEDRAL_RANGE[1]].set(octahedral_bits)
    bits = bits.at[TETRAHEDRAL_RANGE[0]:TETRAHEDRAL_RANGE[1]].set(tetrahedral_bits)
    bits = bits.at[MULTISCALE_RANGE[0]:MULTISCALE_RANGE[1]].set(multiscale_bits)
    
    # Calculate sparsity
    sparsity = jnp.mean(bits)
    
    # Create metadata
    metadata = {
        'position': position,
        'cell24_vertex': cell24_vertex,
        'n_active_bits': jnp.sum(bits)
    }
    
    return PolytopeSDR(bits=bits, sparsity=sparsity, metadata=metadata)


# Pure functional SDR operations

@jax.jit
def sdr_and(sdr1: PolytopeSDR, sdr2: PolytopeSDR) -> PolytopeSDR:
    """Compute intersection of two SDRs.
    
    The AND operation identifies common polytope structure.
    
    Args:
        sdr1, sdr2: Input SDRs
        
    Returns:
        SDR with bits active in both inputs
    """
    bits = jnp.logical_and(sdr1.bits, sdr2.bits)
    sparsity = jnp.mean(bits)
    
    metadata = {
        'operation': 'AND',
        'overlap': jnp.sum(bits),
        'jaccard_similarity': jnp.sum(bits) / (jnp.sum(jnp.logical_or(sdr1.bits, sdr2.bits)) + 1e-10)
    }
    
    return PolytopeSDR(bits=bits, sparsity=sparsity, metadata=metadata)


@jax.jit
def sdr_or(sdr1: PolytopeSDR, sdr2: PolytopeSDR) -> PolytopeSDR:
    """Compute union of two SDRs.
    
    The OR operation combines polytope structures.
    
    Args:
        sdr1, sdr2: Input SDRs
        
    Returns:
        SDR with bits active in either input
    """
    bits = jnp.logical_or(sdr1.bits, sdr2.bits)
    sparsity = jnp.mean(bits)
    
    metadata = {
        'operation': 'OR',
        'total_active': jnp.sum(bits)
    }
    
    return PolytopeSDR(bits=bits, sparsity=sparsity, metadata=metadata)


@jax.jit
def sdr_xor(sdr1: PolytopeSDR, sdr2: PolytopeSDR) -> PolytopeSDR:
    """Compute symmetric difference of two SDRs.
    
    The XOR operation identifies differing structure.
    
    Args:
        sdr1, sdr2: Input SDRs
        
    Returns:
        SDR with bits active in exactly one input
    """
    bits = jnp.logical_xor(sdr1.bits, sdr2.bits)
    sparsity = jnp.mean(bits)
    
    metadata = {
        'operation': 'XOR',
        'difference': jnp.sum(bits)
    }
    
    return PolytopeSDR(bits=bits, sparsity=sparsity, metadata=metadata)


@jax.jit
def sdr_similarity(sdr1: PolytopeSDR, sdr2: PolytopeSDR) -> float:
    """Compute similarity between two SDRs.
    
    Args:
        sdr1, sdr2: Input SDRs
        
    Returns:
        Similarity score in [0, 1]
    """
    intersection = jnp.sum(jnp.logical_and(sdr1.bits, sdr2.bits))
    union = jnp.sum(jnp.logical_or(sdr1.bits, sdr2.bits))
    
    return intersection / (union + 1e-10)


def extract_segments(sdr: PolytopeSDR) -> List[SDRSegment]:
    """Extract semantic segments from SDR.
    
    Args:
        sdr: Input SDR
        
    Returns:
        List of SDR segments with their meanings
    """
    segments = []
    
    # Extract each segment
    segments.append(SDRSegment(
        name="24cell_vertex",
        bits=sdr.bits[CELL24_RANGE[0]:CELL24_RANGE[1]],
        start_idx=CELL24_RANGE[0],
        end_idx=CELL24_RANGE[1]
    ))
    
    segments.append(SDRSegment(
        name="harmonic_coeffs",
        bits=sdr.bits[HARMONIC_RANGE[0]:HARMONIC_RANGE[1]],
        start_idx=HARMONIC_RANGE[0],
        end_idx=HARMONIC_RANGE[1]
    ))
    
    segments.append(SDRSegment(
        name="octahedral_position",
        bits=sdr.bits[OCTAHEDRAL_RANGE[0]:OCTAHEDRAL_RANGE[1]],
        start_idx=OCTAHEDRAL_RANGE[0],
        end_idx=OCTAHEDRAL_RANGE[1]
    ))
    
    segments.append(SDRSegment(
        name="tetrahedral_detail",
        bits=sdr.bits[TETRAHEDRAL_RANGE[0]:TETRAHEDRAL_RANGE[1]],
        start_idx=TETRAHEDRAL_RANGE[0],
        end_idx=TETRAHEDRAL_RANGE[1]
    ))
    
    segments.append(SDRSegment(
        name="multiscale",
        bits=sdr.bits[MULTISCALE_RANGE[0]:MULTISCALE_RANGE[1]],
        start_idx=MULTISCALE_RANGE[0],
        end_idx=MULTISCALE_RANGE[1]
    ))
    
    return segments


# State monad for functional composition

class SDRState(NamedTuple):
    """State for monadic SDR computations.
    
    Attributes:
        sdr: Current SDR
        history: List of previous SDRs
        context: Additional computation context
    """
    sdr: PolytopeSDR
    history: List[PolytopeSDR]
    context: dict


# Type alias for state transformation
StateTransform = Callable[[SDRState], SDRState]


def pure_state(sdr: PolytopeSDR) -> SDRState:
    """Create initial state from SDR."""
    return SDRState(
        sdr=sdr,
        history=[],
        context={}
    )


def bind_state(transform1: StateTransform, 
               transform2: StateTransform) -> StateTransform:
    """Monadic bind operation for state transformations."""
    def combined(state: SDRState) -> SDRState:
        intermediate = transform1(state)
        return transform2(intermediate)
    return combined


def map_state(f: Callable[[PolytopeSDR], PolytopeSDR]) -> StateTransform:
    """Lift a pure SDR function to state transformation."""
    def transform(state: SDRState) -> SDRState:
        new_sdr = f(state.sdr)
        new_history = state.history + [state.sdr]
        return SDRState(sdr=new_sdr, history=new_history, context=state.context)
    return transform


def sequence_transforms(transforms: List[StateTransform]) -> StateTransform:
    """Sequence multiple state transformations."""
    def combined(state: SDRState) -> SDRState:
        current = state
        for transform in transforms:
            current = transform(current)
        return current
    return combined


# Example pipeline functions

def threshold_sdr(threshold: float) -> StateTransform:
    """Create transform that zeros bits below threshold."""
    def transform(state: SDRState) -> SDRState:
        # In real use, would threshold based on activation strength
        # For now, just demonstrate the pattern
        new_bits = state.sdr.bits  # Would apply threshold here
        new_sdr = PolytopeSDR(
            bits=new_bits,
            sparsity=jnp.mean(new_bits),
            metadata={'thresholded': True}
        )
        return SDRState(
            sdr=new_sdr,
            history=state.history + [state.sdr],
            context={**state.context, 'threshold': threshold}
        )
    return transform


def combine_with_sdr(other_sdr: PolytopeSDR, 
                    operation: str = 'AND') -> StateTransform:
    """Create transform that combines with another SDR."""
    def transform(state: SDRState) -> SDRState:
        if operation == 'AND':
            new_sdr = sdr_and(state.sdr, other_sdr)
        elif operation == 'OR':
            new_sdr = sdr_or(state.sdr, other_sdr)
        else:  # XOR
            new_sdr = sdr_xor(state.sdr, other_sdr)
        
        return SDRState(
            sdr=new_sdr,
            history=state.history + [state.sdr],
            context={**state.context, 'combined_with': other_sdr}
        )
    return transform


def segmentation_pipeline(position: jnp.ndarray,
                         harmonics: jnp.ndarray,
                         reference_sdrs: List[PolytopeSDR]) -> SDRState:
    """Example segmentation pipeline using monadic composition.
    
    Args:
        position: 3D position to segment
        harmonics: Harmonic coefficients
        reference_sdrs: Known SDRs for comparison
        
    Returns:
        Final SDR state after pipeline
    """
    # Encode initial state
    initial_sdr = encode_polytope_state(
        position=position,
        harmonic_coeffs=harmonics,
        cell24_vertex=0,  # Would compute from position
        octahedron_idx=0,  # Would compute from tiling
        face_idx=0
    )
    
    # Create pipeline
    pipeline = sequence_transforms([
        threshold_sdr(0.1),
        combine_with_sdr(reference_sdrs[0], 'AND'),
        map_state(lambda sdr: sdr),  # Could add more processing
    ])
    
    # Execute pipeline
    initial_state = pure_state(initial_sdr)
    final_state = pipeline(initial_state)
    
    return final_state


# Utility functions for analysis

def analyze_sdr_structure(sdr: PolytopeSDR) -> dict:
    """Analyze the structure of an SDR.
    
    Args:
        sdr: SDR to analyze
        
    Returns:
        Dictionary with structural analysis
    """
    segments = extract_segments(sdr)
    
    analysis = {
        'total_active_bits': int(jnp.sum(sdr.bits)),
        'sparsity': float(sdr.sparsity),
        'segment_activity': {}
    }
    
    for segment in segments:
        active = int(jnp.sum(segment.bits))
        total = len(segment.bits)
        analysis['segment_activity'][segment.name] = {
            'active_bits': active,
            'total_bits': total,
            'sparsity': active / total if total > 0 else 0
        }
    
    return analysis


def find_nearest_polytope(query_sdr: PolytopeSDR,
                         reference_sdrs: List[PolytopeSDR]) -> int:
    """Find the nearest polytope from a set of references.
    
    Args:
        query_sdr: Query SDR
        reference_sdrs: List of reference SDRs
        
    Returns:
        Index of nearest reference
    """
    similarities = [sdr_similarity(query_sdr, ref) for ref in reference_sdrs]
    return int(jnp.argmax(jnp.array(similarities)))