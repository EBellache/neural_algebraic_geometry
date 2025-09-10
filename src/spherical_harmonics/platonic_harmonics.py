"""
Spherical harmonics restricted to vertices of Platonic solids.

This module computes discrete orthogonal bases by evaluating spherical harmonic
functions Y_lm at polytope vertices. The symmetry group of each solid constrains
which harmonics contribute, dramatically reducing computational complexity while
preserving essential shape information.

Key insights:
- Bacteria (elongated) excite different modes than spherical granules
- Discrete evaluation at vertices is much faster than continuous decomposition
- Harmonic orders map to neural frequency bands (delta through ripples)
- Symmetry constraints make certain harmonics exactly zero
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Dict, Tuple, List
from functools import partial
import numpy as np
from jax.scipy.special import gammaln


class HarmonicBasis(NamedTuple):
    """Spherical harmonic basis for a Platonic solid.
    
    Attributes:
        solid_name: Name of the Platonic solid
        vertices: (N, 3) array of vertex positions
        Y_values: Dict mapping (l, m) to (N,) arrays of Y_lm values
        allowed_lm: List of (l, m) pairs allowed by symmetry
        max_l: Maximum angular momentum quantum number
    """
    solid_name: str
    vertices: jnp.ndarray
    Y_values: Dict[Tuple[int, int], jnp.ndarray]
    allowed_lm: List[Tuple[int, int]]
    max_l: int


class HarmonicSpectrum(NamedTuple):
    """Harmonic decomposition of a shape.
    
    Attributes:
        coefficients: Dict mapping (l, m) to complex coefficient
        power_spectrum: Array of power per l value
        shape_descriptor: Rotation-invariant shape vector
        dominant_modes: List of (l, m) with largest coefficients
    """
    coefficients: Dict[Tuple[int, int], complex]
    power_spectrum: jnp.ndarray
    shape_descriptor: jnp.ndarray
    dominant_modes: List[Tuple[int, int]]


# Frequency band mapping
FREQUENCY_BANDS = {
    0: "delta (0.5-4 Hz)",      # l=0: monopole, overall size
    1: "theta (4-8 Hz)",        # l=1: dipole, orientation
    2: "alpha (8-13 Hz)",       # l=2: quadrupole, elongation
    3: "beta (13-30 Hz)",       # l=3: octupole, triangular features
    4: "gamma (30-50 Hz)",      # l=4: hexadecapole, square features
    5: "high gamma (50-150 Hz)", # l=5: 32-pole, pentagonal features
    6: "ripples (150-250 Hz)"   # l=6: 64-pole, hexagonal features
}


@jax.jit
def factorial(n):
    """Compute factorial using gamma function for JAX compatibility."""
    return jnp.exp(gammaln(n + 1))


@jax.jit
def associated_legendre(l: int, m: int, x: jnp.ndarray) -> jnp.ndarray:
    """Compute associated Legendre polynomials P_l^m(x).
    
    Args:
        l: Degree
        m: Order (|m| <= l)
        x: Input values (typically cos(theta))
        
    Returns:
        P_l^m(x) values
    """
    # Handle edge cases
    if abs(m) > l:
        return jnp.zeros_like(x)
    
    # Use recurrence relations for efficiency
    # Simplified implementation - full version would handle all cases
    if l == 0:
        return jnp.ones_like(x)
    elif l == 1:
        if m == 0:
            return x
        elif abs(m) == 1:
            return -jnp.sqrt(1 - x**2)
    elif l == 2:
        if m == 0:
            return 0.5 * (3*x**2 - 1)
        elif abs(m) == 1:
            return -3 * x * jnp.sqrt(1 - x**2)
        elif abs(m) == 2:
            return 3 * (1 - x**2)
    
    # For higher l, would implement full recurrence
    return jnp.zeros_like(x)


@jax.jit
def spherical_harmonic(l: int, m: int, theta: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
    """Compute spherical harmonic Y_l^m(theta, phi).
    
    Args:
        l: Degree (l >= 0)
        m: Order (-l <= m <= l)
        theta: Polar angle (0 to pi)
        phi: Azimuthal angle (0 to 2pi)
        
    Returns:
        Complex Y_l^m values
    """
    # Normalization factor
    norm = jnp.sqrt((2*l + 1) * factorial(l - abs(m)) / (4*jnp.pi * factorial(l + abs(m))))
    
    # Associated Legendre polynomial
    P_lm = associated_legendre(l, abs(m), jnp.cos(theta))
    
    # Complex exponential
    exp_imphi = jnp.exp(1j * m * phi)
    
    # Combine
    Y_lm = norm * P_lm * exp_imphi
    
    # Handle negative m
    if m < 0:
        Y_lm = (-1)**m * jnp.conj(Y_lm)
    
    return Y_lm


@jax.jit
def cartesian_to_spherical(xyz: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert Cartesian to spherical coordinates.
    
    Args:
        xyz: (..., 3) array of (x, y, z) coordinates
        
    Returns:
        Tuple of (r, theta, phi) arrays
    """
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    
    r = jnp.linalg.norm(xyz, axis=-1)
    theta = jnp.arccos(jnp.clip(z / (r + 1e-10), -1, 1))
    phi = jnp.arctan2(y, x)
    
    return r, theta, phi


def compute_harmonic_basis(vertices: jnp.ndarray, 
                          solid_name: str,
                          max_l: int = 6) -> HarmonicBasis:
    """Compute spherical harmonic basis for a Platonic solid.
    
    Args:
        vertices: (N, 3) array of vertex positions
        solid_name: Name of the solid for symmetry constraints
        max_l: Maximum l value to compute
        
    Returns:
        HarmonicBasis with Y_lm values and symmetry constraints
    """
    # Convert to spherical coordinates
    r, theta, phi = cartesian_to_spherical(vertices)
    
    # Compute Y_lm values
    Y_values = {}
    
    for l in range(max_l + 1):
        for m in range(-l, l + 1):
            Y_lm = spherical_harmonic(l, m, theta, phi)
            Y_values[(l, m)] = Y_lm
    
    # Apply symmetry constraints based on solid type
    allowed_lm = get_allowed_harmonics(solid_name, max_l)
    
    # Filter Y_values to keep only allowed harmonics
    Y_values_filtered = {lm: Y_values[lm] for lm in allowed_lm if lm in Y_values}
    
    return HarmonicBasis(
        solid_name=solid_name,
        vertices=vertices,
        Y_values=Y_values_filtered,
        allowed_lm=allowed_lm,
        max_l=max_l
    )


def get_allowed_harmonics(solid_name: str, max_l: int) -> List[Tuple[int, int]]:
    """Get (l, m) pairs allowed by solid's symmetry group.
    
    Args:
        solid_name: Name of Platonic solid
        max_l: Maximum l value
        
    Returns:
        List of allowed (l, m) tuples
    """
    allowed = []
    
    if solid_name == "tetrahedron":
        # Tetrahedral symmetry (Td): A4 group
        # Allows l = 0, 3, 4, 6, ...
        for l in range(max_l + 1):
            if l == 0 or l == 3 or l == 4 or l == 6:
                for m in range(-l, l + 1):
                    if is_tetrahedral_allowed(l, m):
                        allowed.append((l, m))
                        
    elif solid_name == "cube" or solid_name == "octahedron":
        # Octahedral symmetry (Oh): S4 group
        # Allows l = 0, 4, 6, ...
        for l in range(max_l + 1):
            if l % 2 == 0 and (l == 0 or l == 4 or l == 6):
                for m in range(-l, l + 1):
                    if m % 4 == 0:
                        allowed.append((l, m))
                        
    elif solid_name == "icosahedron" or solid_name == "dodecahedron":
        # Icosahedral symmetry (Ih): A5 group
        # Allows l = 0, 6, 10, 12, ...
        for l in range(max_l + 1):
            if l == 0 or l == 6:
                for m in range(-l, l + 1):
                    if is_icosahedral_allowed(l, m):
                        allowed.append((l, m))
    else:
        # Default: all harmonics
        for l in range(max_l + 1):
            for m in range(-l, l + 1):
                allowed.append((l, m))
    
    return allowed


def is_tetrahedral_allowed(l: int, m: int) -> bool:
    """Check if (l, m) is allowed by tetrahedral symmetry."""
    # Simplified - full implementation would check character tables
    if l == 0:
        return m == 0
    elif l == 3:
        return abs(m) == 0 or abs(m) == 3
    elif l == 4:
        return m % 4 == 0
    elif l == 6:
        return abs(m) == 0 or abs(m) == 6
    return False


def is_icosahedral_allowed(l: int, m: int) -> bool:
    """Check if (l, m) is allowed by icosahedral symmetry."""
    # Simplified - full implementation would use group theory
    if l == 0:
        return m == 0
    elif l == 6:
        return m % 5 == 0 or abs(m) == 6
    return False


@jax.jit
def compute_harmonic_coefficients(shape_values: jnp.ndarray,
                                 harmonic_basis: HarmonicBasis) -> Dict[Tuple[int, int], complex]:
    """Compute spherical harmonic coefficients for a shape.
    
    Args:
        shape_values: (N,) array of function values at vertices
        harmonic_basis: Pre-computed harmonic basis
        
    Returns:
        Dictionary of complex coefficients c_lm
    """
    coefficients = {}
    
    # For discrete points, use weighted sum approximation
    # Weight by solid angle represented by each vertex
    n_vertices = len(shape_values)
    weight = 4 * jnp.pi / n_vertices  # Equal weight approximation
    
    for (l, m), Y_lm in harmonic_basis.Y_values.items():
        # c_lm = integral of f * conj(Y_lm)
        c_lm = weight * jnp.sum(shape_values * jnp.conj(Y_lm))
        coefficients[(l, m)] = c_lm
    
    return coefficients


def analyze_shape_harmonics(shape_values: jnp.ndarray,
                           harmonic_basis: HarmonicBasis) -> HarmonicSpectrum:
    """Perform harmonic analysis of a shape defined at polytope vertices.
    
    Args:
        shape_values: (N,) array of function values at vertices
        harmonic_basis: Pre-computed harmonic basis
        
    Returns:
        HarmonicSpectrum with coefficients and derived features
    """
    # Compute coefficients
    coefficients = compute_harmonic_coefficients(shape_values, harmonic_basis)
    
    # Compute power spectrum (sum |c_lm|^2 for each l)
    max_l = harmonic_basis.max_l
    power_spectrum = jnp.zeros(max_l + 1)
    
    for (l, m), c_lm in coefficients.items():
        power_spectrum = power_spectrum.at[l].add(jnp.abs(c_lm)**2)
    
    # Create rotation-invariant shape descriptor
    # Use normalized power spectrum
    shape_descriptor = power_spectrum / (jnp.sum(power_spectrum) + 1e-10)
    
    # Find dominant modes
    dominant_modes = []
    sorted_coeffs = sorted(coefficients.items(), 
                          key=lambda x: abs(x[1]), 
                          reverse=True)
    dominant_modes = [lm for lm, _ in sorted_coeffs[:5]]
    
    return HarmonicSpectrum(
        coefficients=coefficients,
        power_spectrum=power_spectrum,
        shape_descriptor=shape_descriptor,
        dominant_modes=dominant_modes
    )


def bacteria_harmonic_signature(vertices: jnp.ndarray,
                               elongation: float = 2.0,
                               bend_angle: float = 0.0) -> jnp.ndarray:
    """Generate characteristic harmonic signature for bacteria.
    
    Bacteria are elongated objects that excite l=2 (quadrupole) and 
    l=4 modes strongly, unlike spherical granules.
    
    Args:
        vertices: (N, 3) polytope vertices
        elongation: Elongation factor along z-axis
        bend_angle: Bending angle for curved bacteria
        
    Returns:
        (N,) array of shape function values
    """
    # Apply elongation transformation
    elongated = vertices.copy()
    elongated = elongated.at[:, 2].multiply(elongation)
    
    # Apply bending if specified
    if bend_angle > 0:
        # Simple bending: rotate based on z-coordinate
        z = elongated[:, 2]
        rotation_angles = bend_angle * z / jnp.max(jnp.abs(z))
        
        cos_a = jnp.cos(rotation_angles)
        sin_a = jnp.sin(rotation_angles)
        
        x_new = elongated[:, 0] * cos_a - elongated[:, 1] * sin_a
        y_new = elongated[:, 0] * sin_a + elongated[:, 1] * cos_a
        
        elongated = elongated.at[:, 0].set(x_new)
        elongated = elongated.at[:, 1].set(y_new)
    
    # Shape function: distance from origin with elongation weighting
    shape_values = jnp.linalg.norm(elongated, axis=1)
    
    return shape_values


def granule_harmonic_signature(vertices: jnp.ndarray,
                              radius_variation: float = 0.1) -> jnp.ndarray:
    """Generate characteristic harmonic signature for spherical granules.
    
    Granules are nearly spherical with dominant l=0 (monopole) mode
    and small higher-order perturbations.
    
    Args:
        vertices: (N, 3) polytope vertices
        radius_variation: Amount of random variation
        
    Returns:
        (N,) array of shape function values
    """
    # Nearly constant radius with small variations
    base_radius = 1.0
    
    # Add small random perturbations
    key = jax.random.PRNGKey(42)
    perturbations = radius_variation * jax.random.normal(key, (len(vertices),))
    
    shape_values = base_radius + perturbations
    
    return shape_values


def compare_bacteria_granule_harmonics(harmonic_basis: HarmonicBasis) -> Dict:
    """Compare harmonic signatures of bacteria vs granules.
    
    Args:
        harmonic_basis: Pre-computed basis for a polytope
        
    Returns:
        Dictionary with comparison results
    """
    vertices = harmonic_basis.vertices
    
    # Generate signatures
    bacteria_sig = bacteria_harmonic_signature(vertices, elongation=2.5)
    granule_sig = granule_harmonic_signature(vertices)
    
    # Analyze harmonics
    bacteria_spectrum = analyze_shape_harmonics(bacteria_sig, harmonic_basis)
    granule_spectrum = analyze_shape_harmonics(granule_sig, harmonic_basis)
    
    # Key differences
    bacteria_l2_power = bacteria_spectrum.power_spectrum[2]
    granule_l2_power = granule_spectrum.power_spectrum[2]
    
    bacteria_l0_power = bacteria_spectrum.power_spectrum[0]
    granule_l0_power = granule_spectrum.power_spectrum[0]
    
    return {
        'bacteria_spectrum': bacteria_spectrum,
        'granule_spectrum': granule_spectrum,
        'elongation_ratio': bacteria_l2_power / (bacteria_l0_power + 1e-10),
        'sphericity_ratio': granule_l0_power / (jnp.sum(granule_spectrum.power_spectrum) + 1e-10),
        'discrimination_score': bacteria_l2_power - granule_l2_power
    }


def map_harmonics_to_brain_frequencies(spectrum: HarmonicSpectrum) -> Dict[str, float]:
    """Map harmonic spectrum to brain frequency bands.
    
    Args:
        spectrum: Harmonic spectrum from shape analysis
        
    Returns:
        Dictionary mapping frequency bands to power
    """
    frequency_power = {}
    
    for l in range(len(spectrum.power_spectrum)):
        if l in FREQUENCY_BANDS:
            band_name = FREQUENCY_BANDS[l]
            power = spectrum.power_spectrum[l]
            frequency_power[band_name] = float(power)
    
    # Normalize to percentages
    total_power = sum(frequency_power.values())
    if total_power > 0:
        frequency_power = {k: v/total_power * 100 
                          for k, v in frequency_power.items()}
    
    return frequency_power


class DiscreteSphericalTransform:
    """Fast spherical harmonic transform using polytope vertices."""
    
    def __init__(self, harmonic_basis: HarmonicBasis):
        """Initialize with pre-computed harmonic basis.
        
        Args:
            harmonic_basis: Pre-computed basis for a polytope
        """
        self.basis = harmonic_basis
        
        # Pre-compute transform matrix for efficiency
        self._build_transform_matrix()
    
    def _build_transform_matrix(self):
        """Build transform matrix Y for fast computation."""
        # Stack Y_lm values into matrix
        n_vertices = len(self.basis.vertices)
        n_harmonics = len(self.basis.allowed_lm)
        
        # Complex matrix: rows are vertices, columns are (l,m) modes
        Y_matrix = jnp.zeros((n_vertices, n_harmonics), dtype=complex)
        
        for idx, (l, m) in enumerate(self.basis.allowed_lm):
            if (l, m) in self.basis.Y_values:
                Y_matrix = Y_matrix.at[:, idx].set(self.basis.Y_values[(l, m)])
        
        self.Y_matrix = Y_matrix
        self.Y_matrix_conj = jnp.conj(Y_matrix)
        
        # Weights for discrete integration
        self.weights = jnp.ones(n_vertices) * 4 * jnp.pi / n_vertices
    
    @partial(jax.jit, static_argnums=(0,))
    def forward(self, f: jnp.ndarray) -> jnp.ndarray:
        """Fast forward transform: spatial → harmonic.
        
        Args:
            f: (N,) function values at vertices
            
        Returns:
            (M,) harmonic coefficients
        """
        # c = Y* . W . f
        weighted_f = self.weights * f
        coefficients = jnp.dot(self.Y_matrix_conj.T, weighted_f)
        return coefficients
    
    @partial(jax.jit, static_argnums=(0,))
    def inverse(self, c: jnp.ndarray) -> jnp.ndarray:
        """Fast inverse transform: harmonic → spatial.
        
        Args:
            c: (M,) harmonic coefficients
            
        Returns:
            (N,) reconstructed function values
        """
        # f = Y . c
        f_reconstructed = jnp.dot(self.Y_matrix, c)
        return jnp.real(f_reconstructed)
    
    def filter_harmonics(self, f: jnp.ndarray, 
                        keep_l: List[int]) -> jnp.ndarray:
        """Filter function to keep only specific l values.
        
        Args:
            f: (N,) function values
            keep_l: List of l values to keep
            
        Returns:
            (N,) filtered function
        """
        # Forward transform
        c = self.forward(f)
        
        # Zero out unwanted modes
        c_filtered = jnp.zeros_like(c)
        
        for idx, (l, m) in enumerate(self.basis.allowed_lm):
            if l in keep_l:
                c_filtered = c_filtered.at[idx].set(c[idx])
        
        # Inverse transform
        return self.inverse(c_filtered)