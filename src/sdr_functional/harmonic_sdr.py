"""
Encode spherical harmonic coefficients into SDR format.

This module bridges continuous harmonic representations and discrete neural codes
using place coding. Each coefficient is encoded by which thresholds it exceeds,
creating a binary representation that preserves ordering and approximate magnitude.

Key features:
- Place coding with 10 thresholds per harmonic
- 637 bits for 49 icosahedral harmonics (490 magnitude + 49 sign + 49 significance + 49 confidence)
- Reversible encoding with bounded reconstruction error
- Meaningful SDR operations (AND/OR/XOR) for shape comparison
- Dynamic evolution and compression for efficiency
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Dict, Tuple, List, Optional
from functools import partial
import numpy as np


class HarmonicSDR(NamedTuple):
    """SDR encoding of spherical harmonic coefficients.
    
    Attributes:
        bits: Binary SDR representation
        n_harmonics: Number of encoded harmonics
        threshold_levels: Threshold values used for encoding
        bit_allocation: How bits are allocated to different components
        compression_map: Optional compression information
        metadata: Additional encoding information
    """
    bits: jnp.ndarray
    n_harmonics: int
    threshold_levels: jnp.ndarray
    bit_allocation: Dict[str, Tuple[int, int]]
    compression_map: Optional[Dict]
    metadata: Dict


class HarmonicSDRConfig(NamedTuple):
    """Configuration for harmonic SDR encoding.
    
    Attributes:
        n_thresholds: Number of threshold levels per harmonic
        threshold_range: (min, max) range for thresholds
        bits_per_harmonic: Total bits allocated per harmonic
        use_compression: Whether to apply compression
        significance_threshold: Minimum coefficient magnitude to encode
    """
    n_thresholds: int = 10
    threshold_range: Tuple[float, float] = (-1.0, 1.0)
    bits_per_harmonic: int = 13  # 10 magnitude + 1 sign + 1 significance + 1 confidence
    use_compression: bool = False
    significance_threshold: float = 0.01


# Default configuration
DEFAULT_CONFIG = HarmonicSDRConfig()


@jax.jit
def create_threshold_levels(config: HarmonicSDRConfig) -> jnp.ndarray:
    """Create threshold levels for place coding.
    
    Args:
        config: SDR configuration
        
    Returns:
        Array of threshold values
    """
    min_val, max_val = config.threshold_range
    return jnp.linspace(min_val, max_val, config.n_thresholds)


@jax.jit
def encode_single_harmonic(coefficient: complex,
                          thresholds: jnp.ndarray,
                          confidence: float = 1.0) -> jnp.ndarray:
    """Encode a single harmonic coefficient using place coding.
    
    Args:
        coefficient: Complex harmonic coefficient
        thresholds: Threshold levels
        confidence: Confidence in this coefficient (0-1)
        
    Returns:
        Binary encoding (magnitude bits + sign + significance + confidence)
    """
    # Extract magnitude and phase
    magnitude = jnp.abs(coefficient)
    is_negative = jnp.real(coefficient) < 0
    
    # Place coding for magnitude
    magnitude_bits = magnitude > thresholds
    
    # Sign bit
    sign_bit = is_negative
    
    # Significance bit (is coefficient large enough to matter?)
    significance_bit = magnitude > DEFAULT_CONFIG.significance_threshold
    
    # Confidence bit (binary threshold at 0.5)
    confidence_bit = confidence > 0.5
    
    # Combine all bits
    return jnp.concatenate([
        magnitude_bits,
        jnp.array([sign_bit]),
        jnp.array([significance_bit]),
        jnp.array([confidence_bit])
    ])


def encode_harmonic_coefficients(coefficients: Dict[Tuple[int, int], complex],
                               config: HarmonicSDRConfig = DEFAULT_CONFIG,
                               confidences: Optional[Dict[Tuple[int, int], float]] = None) -> HarmonicSDR:
    """Encode harmonic coefficients into SDR format.
    
    Args:
        coefficients: Dictionary mapping (l,m) to complex coefficients
        config: Encoding configuration
        confidences: Optional confidence values per harmonic
        
    Returns:
        HarmonicSDR encoding
    """
    # Create thresholds
    thresholds = create_threshold_levels(config)
    
    # Sort harmonics for consistent ordering
    sorted_harmonics = sorted(coefficients.keys())
    n_harmonics = len(sorted_harmonics)
    
    # Allocate bits
    bits_per_harmonic = config.bits_per_harmonic
    total_bits = n_harmonics * bits_per_harmonic
    
    # Initialize bit array
    bits = jnp.zeros(total_bits, dtype=bool)
    
    # Encode each harmonic
    for idx, (l, m) in enumerate(sorted_harmonics):
        coeff = coefficients[(l, m)]
        conf = confidences.get((l, m), 1.0) if confidences else 1.0
        
        # Encode this harmonic
        harmonic_bits = encode_single_harmonic(coeff, thresholds, conf)
        
        # Place in bit array
        start_idx = idx * bits_per_harmonic
        end_idx = start_idx + bits_per_harmonic
        bits = bits.at[start_idx:end_idx].set(harmonic_bits)
    
    # Create bit allocation map
    bit_allocation = {
        'magnitude': (0, config.n_thresholds),
        'sign': (config.n_thresholds, config.n_thresholds + 1),
        'significance': (config.n_thresholds + 1, config.n_thresholds + 2),
        'confidence': (config.n_thresholds + 2, bits_per_harmonic)
    }
    
    # Apply compression if requested
    compression_map = None
    if config.use_compression:
        bits, compression_map = compress_harmonic_sdr(bits, sorted_harmonics)
    
    return HarmonicSDR(
        bits=bits,
        n_harmonics=n_harmonics,
        threshold_levels=thresholds,
        bit_allocation=bit_allocation,
        compression_map=compression_map,
        metadata={
            'harmonic_order': sorted_harmonics,
            'config': config
        }
    )


@jax.jit
def decode_single_harmonic(bits: jnp.ndarray,
                          thresholds: jnp.ndarray,
                          bit_allocation: Dict[str, Tuple[int, int]]) -> Tuple[complex, float, bool]:
    """Decode a single harmonic from its bit representation.
    
    Args:
        bits: Binary encoding for one harmonic
        thresholds: Threshold levels used in encoding
        bit_allocation: Bit range allocations
        
    Returns:
        Tuple of (coefficient, confidence, is_significant)
    """
    # Extract bit ranges
    mag_start, mag_end = bit_allocation['magnitude']
    sign_idx = bit_allocation['sign'][0]
    sig_idx = bit_allocation['significance'][0]
    conf_idx = bit_allocation['confidence'][0]
    
    # Decode magnitude using place coding
    magnitude_bits = bits[mag_start:mag_end]
    n_active = jnp.sum(magnitude_bits)
    
    if n_active == 0:
        # All thresholds exceed magnitude
        magnitude = thresholds[0] / 2  # Estimate as half of lowest threshold
    elif n_active == len(thresholds):
        # Magnitude exceeds all thresholds
        magnitude = thresholds[-1] * 1.5  # Estimate as 1.5x highest threshold
    else:
        # Magnitude between thresholds[n_active-1] and thresholds[n_active]
        if n_active > 0:
            lower = thresholds[n_active - 1]
            upper = thresholds[n_active] if n_active < len(thresholds) else thresholds[-1] * 2
            magnitude = (lower + upper) / 2
        else:
            magnitude = 0.0
    
    # Decode sign
    is_negative = bits[sign_idx]
    sign = -1.0 if is_negative else 1.0
    
    # Decode significance and confidence
    is_significant = bits[sig_idx]
    has_confidence = bits[conf_idx]
    confidence = 0.8 if has_confidence else 0.3
    
    # Construct complex coefficient (assuming real for simplicity)
    coefficient = sign * magnitude + 0j
    
    return coefficient, confidence, is_significant


def decode_harmonic_sdr(harmonic_sdr: HarmonicSDR) -> Dict[Tuple[int, int], complex]:
    """Decode SDR back to harmonic coefficients.
    
    Args:
        harmonic_sdr: Encoded harmonic SDR
        
    Returns:
        Dictionary of decoded coefficients with bounded error
    """
    # Decompress if needed
    bits = harmonic_sdr.bits
    if harmonic_sdr.compression_map:
        bits = decompress_harmonic_sdr(bits, harmonic_sdr.compression_map)
    
    # Extract metadata
    harmonic_order = harmonic_sdr.metadata['harmonic_order']
    config = harmonic_sdr.metadata['config']
    bits_per_harmonic = config.bits_per_harmonic
    
    # Decode each harmonic
    coefficients = {}
    for idx, (l, m) in enumerate(harmonic_order):
        start_idx = idx * bits_per_harmonic
        end_idx = start_idx + bits_per_harmonic
        harmonic_bits = bits[start_idx:end_idx]
        
        coeff, conf, is_sig = decode_single_harmonic(
            harmonic_bits,
            harmonic_sdr.threshold_levels,
            harmonic_sdr.bit_allocation
        )
        
        if is_sig:  # Only include significant coefficients
            coefficients[(l, m)] = coeff
    
    return coefficients


@jax.jit
def harmonic_sdr_and(sdr1: HarmonicSDR, sdr2: HarmonicSDR) -> HarmonicSDR:
    """AND operation finding common harmonic features.
    
    Args:
        sdr1, sdr2: Input harmonic SDRs
        
    Returns:
        SDR with shared features
    """
    # Ensure same structure
    assert sdr1.n_harmonics == sdr2.n_harmonics
    
    # Bitwise AND
    and_bits = jnp.logical_and(sdr1.bits, sdr2.bits)
    
    return HarmonicSDR(
        bits=and_bits,
        n_harmonics=sdr1.n_harmonics,
        threshold_levels=sdr1.threshold_levels,
        bit_allocation=sdr1.bit_allocation,
        compression_map=None,
        metadata={
            'operation': 'AND',
            'source_metadata': [sdr1.metadata, sdr2.metadata]
        }
    )


@jax.jit
def harmonic_sdr_or(sdr1: HarmonicSDR, sdr2: HarmonicSDR) -> HarmonicSDR:
    """OR operation finding union of harmonic features.
    
    Args:
        sdr1, sdr2: Input harmonic SDRs
        
    Returns:
        SDR with combined features
    """
    # Bitwise OR
    or_bits = jnp.logical_or(sdr1.bits, sdr2.bits)
    
    return HarmonicSDR(
        bits=or_bits,
        n_harmonics=sdr1.n_harmonics,
        threshold_levels=sdr1.threshold_levels,
        bit_allocation=sdr1.bit_allocation,
        compression_map=None,
        metadata={
            'operation': 'OR',
            'source_metadata': [sdr1.metadata, sdr2.metadata]
        }
    )


@jax.jit
def harmonic_sdr_xor(sdr1: HarmonicSDR, sdr2: HarmonicSDR) -> HarmonicSDR:
    """XOR operation finding differing harmonic features.
    
    Args:
        sdr1, sdr2: Input harmonic SDRs
        
    Returns:
        SDR with shape differences
    """
    # Bitwise XOR
    xor_bits = jnp.logical_xor(sdr1.bits, sdr2.bits)
    
    return HarmonicSDR(
        bits=xor_bits,
        n_harmonics=sdr1.n_harmonics,
        threshold_levels=sdr1.threshold_levels,
        bit_allocation=sdr1.bit_allocation,
        compression_map=None,
        metadata={
            'operation': 'XOR',
            'source_metadata': [sdr1.metadata, sdr2.metadata]
        }
    )


@jax.jit
def harmonic_distance(sdr1: HarmonicSDR, sdr2: HarmonicSDR) -> float:
    """Compute distance between harmonic SDRs approximating L2 distance.
    
    This critical function enables shape comparison directly in SDR space
    without full decoding, using the threshold structure.
    
    Args:
        sdr1, sdr2: Harmonic SDRs to compare
        
    Returns:
        Distance approximating continuous L2 distance
    """
    config = sdr1.metadata['config']
    bits_per_harmonic = config.bits_per_harmonic
    n_thresholds = config.n_thresholds
    
    total_distance = 0.0
    
    # Compare each harmonic
    for idx in range(sdr1.n_harmonics):
        start_idx = idx * bits_per_harmonic
        mag_end = start_idx + n_thresholds
        
        # Extract magnitude bits
        mag_bits1 = sdr1.bits[start_idx:mag_end]
        mag_bits2 = sdr2.bits[start_idx:mag_end]
        
        # Count active thresholds (approximates magnitude)
        n_active1 = jnp.sum(mag_bits1)
        n_active2 = jnp.sum(mag_bits2)
        
        # Approximate magnitudes
        thresholds = sdr1.threshold_levels
        mag1 = thresholds[n_active1] if n_active1 < len(thresholds) else thresholds[-1]
        mag2 = thresholds[n_active2] if n_active2 < len(thresholds) else thresholds[-1]
        
        # Add to distance
        total_distance += (mag1 - mag2) ** 2
    
    return jnp.sqrt(total_distance)


def evolve_harmonic_sdr(current_sdr: HarmonicSDR,
                       new_coefficients: Dict[Tuple[int, int], complex],
                       learning_rate: float = 0.1) -> HarmonicSDR:
    """Update harmonic SDR as new data arrives.
    
    Args:
        current_sdr: Current SDR state
        new_coefficients: New harmonic estimates
        learning_rate: How much to weight new data
        
    Returns:
        Updated harmonic SDR
    """
    # Decode current estimates
    current_coeffs = decode_harmonic_sdr(current_sdr)
    
    # Blend with new data
    updated_coeffs = {}
    all_harmonics = set(current_coeffs.keys()) | set(new_coefficients.keys())
    
    for (l, m) in all_harmonics:
        curr = current_coeffs.get((l, m), 0.0)
        new = new_coefficients.get((l, m), 0.0)
        updated = (1 - learning_rate) * curr + learning_rate * new
        updated_coeffs[(l, m)] = updated
    
    # Re-encode
    config = current_sdr.metadata['config']
    return encode_harmonic_coefficients(updated_coeffs, config)


def compress_harmonic_sdr(bits: jnp.ndarray,
                         harmonic_order: List[Tuple[int, int]]) -> Tuple[jnp.ndarray, Dict]:
    """Compress sparse harmonic SDR using run-length encoding.
    
    Args:
        bits: Uncompressed bit array
        harmonic_order: List of (l,m) tuples
        
    Returns:
        Tuple of (compressed_bits, compression_map)
    """
    # Find runs of zeros (insignificant harmonics)
    bits_np = np.array(bits)
    
    # Simple compression: skip all-zero harmonics
    bits_per_harmonic = len(bits) // len(harmonic_order)
    compressed_parts = []
    active_indices = []
    
    for idx in range(len(harmonic_order)):
        start = idx * bits_per_harmonic
        end = start + bits_per_harmonic
        harmonic_bits = bits_np[start:end]
        
        # Check if harmonic has any active bits
        if np.any(harmonic_bits):
            compressed_parts.append(harmonic_bits)
            active_indices.append(idx)
    
    # Concatenate active harmonics
    if compressed_parts:
        compressed_bits = jnp.concatenate([jnp.array(p) for p in compressed_parts])
    else:
        compressed_bits = jnp.array([], dtype=bool)
    
    compression_map = {
        'active_indices': active_indices,
        'original_length': len(bits),
        'n_harmonics': len(harmonic_order),
        'bits_per_harmonic': bits_per_harmonic
    }
    
    return compressed_bits, compression_map


def decompress_harmonic_sdr(compressed_bits: jnp.ndarray,
                           compression_map: Dict) -> jnp.ndarray:
    """Decompress harmonic SDR.
    
    Args:
        compressed_bits: Compressed bit array
        compression_map: Compression information
        
    Returns:
        Original uncompressed bit array
    """
    # Reconstruct full array
    original_length = compression_map['original_length']
    bits_per_harmonic = compression_map['bits_per_harmonic']
    active_indices = compression_map['active_indices']
    
    # Initialize with zeros
    decompressed = jnp.zeros(original_length, dtype=bool)
    
    # Place active harmonics
    compressed_idx = 0
    for active_idx in active_indices:
        start = active_idx * bits_per_harmonic
        end = start + bits_per_harmonic
        
        harmonic_bits = compressed_bits[compressed_idx:compressed_idx + bits_per_harmonic]
        decompressed = decompressed.at[start:end].set(harmonic_bits)
        
        compressed_idx += bits_per_harmonic
    
    return decompressed


def compute_reconstruction_error(original_coeffs: Dict[Tuple[int, int], complex],
                               harmonic_sdr: HarmonicSDR) -> Dict[str, float]:
    """Compute error bounds for SDR reconstruction.
    
    Args:
        original_coeffs: Original harmonic coefficients
        harmonic_sdr: SDR encoding
        
    Returns:
        Dictionary of error metrics
    """
    # Decode SDR
    reconstructed = decode_harmonic_sdr(harmonic_sdr)
    
    # Compute errors
    l2_error = 0.0
    linf_error = 0.0
    relative_errors = []
    
    for (l, m) in original_coeffs:
        orig = original_coeffs[(l, m)]
        recon = reconstructed.get((l, m), 0.0)
        
        error = abs(orig - recon)
        l2_error += error ** 2
        linf_error = max(linf_error, error)
        
        if abs(orig) > 1e-10:
            relative_errors.append(error / abs(orig))
    
    # Threshold quantization error bound
    n_thresholds = harmonic_sdr.metadata['config'].n_thresholds
    threshold_range = harmonic_sdr.metadata['config'].threshold_range
    max_quantization_error = (threshold_range[1] - threshold_range[0]) / (2 * n_thresholds)
    
    return {
        'l2_error': float(np.sqrt(l2_error)),
        'linf_error': float(linf_error),
        'mean_relative_error': float(np.mean(relative_errors)) if relative_errors else 0.0,
        'max_quantization_error': float(max_quantization_error),
        'n_coefficients': len(original_coeffs),
        'n_reconstructed': len(reconstructed)
    }


class HarmonicSDREncoder:
    """Stateful encoder for harmonic SDRs with evolution."""
    
    def __init__(self, config: HarmonicSDRConfig = DEFAULT_CONFIG):
        """Initialize encoder.
        
        Args:
            config: Encoding configuration
        """
        self.config = config
        self.current_sdr = None
        self.history = []
    
    def encode(self, coefficients: Dict[Tuple[int, int], complex],
              confidences: Optional[Dict[Tuple[int, int], float]] = None) -> HarmonicSDR:
        """Encode coefficients, potentially evolving from previous state.
        
        Args:
            coefficients: Harmonic coefficients
            confidences: Optional confidence values
            
        Returns:
            Encoded harmonic SDR
        """
        if self.current_sdr is None:
            # First encoding
            sdr = encode_harmonic_coefficients(coefficients, self.config, confidences)
        else:
            # Evolve from previous
            sdr = evolve_harmonic_sdr(self.current_sdr, coefficients)
        
        self.current_sdr = sdr
        self.history.append(sdr)
        
        return sdr
    
    def get_trajectory(self) -> List[HarmonicSDR]:
        """Get evolution history.
        
        Returns:
            List of SDRs over time
        """
        return self.history
    
    def reset(self):
        """Reset encoder state."""
        self.current_sdr = None
        self.history = []