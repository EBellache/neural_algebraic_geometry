"""
State monad implementation for functional segmentation pipelines.

The State monad threads configuration and intermediate results through computations
while maintaining functional purity. State changes are explicit and composable,
enabling complex pipelines that read like imperative code but are purely functional.

Core concept: State s a = s -> (a, s)
A computation that takes a state and returns a value with a new state.
"""

from typing import TypeVar, Generic, Callable, Tuple, Optional, List, NamedTuple, Dict, Any
from functools import wraps
import jax.numpy as jnp
from dataclasses import dataclass
from abc import ABC, abstractmethod


# Type variables
S = TypeVar('S')  # State type
A = TypeVar('A')  # Result type
B = TypeVar('B')  # Another result type


class State(Generic[S, A]):
    """State monad: computation with state.
    
    Represents a stateful computation as a function s -> (a, s).
    """
    
    def __init__(self, run: Callable[[S], Tuple[A, S]]):
        """Initialize with state transformation function.
        
        Args:
            run: Function from state to (result, new_state)
        """
        self.run = run
    
    def __call__(self, state: S) -> Tuple[A, S]:
        """Execute the stateful computation."""
        return self.run(state)
    
    def bind(self, f: Callable[[A], 'State[S, B]']) -> 'State[S, B]':
        """Monadic bind (>>=): sequence stateful computations.
        
        Args:
            f: Function from result to new stateful computation
            
        Returns:
            Combined stateful computation
        """
        def run_bind(state: S) -> Tuple[B, S]:
            # Run first computation
            a, new_state = self.run(state)
            # Run second computation with result and new state
            return f(a).run(new_state)
        
        return State(run_bind)
    
    def map(self, f: Callable[[A], B]) -> 'State[S, B]':
        """Functor map: transform the result value.
        
        Args:
            f: Function to transform result
            
        Returns:
            State computation with transformed result
        """
        def run_map(state: S) -> Tuple[B, S]:
            a, new_state = self.run(state)
            return f(a), new_state
        
        return State(run_map)
    
    def then(self, next_computation: 'State[S, B]') -> 'State[S, B]':
        """Sequence computations, ignoring first result (>>).
        
        Args:
            next_computation: Computation to run after this one
            
        Returns:
            Combined computation returning second result
        """
        return self.bind(lambda _: next_computation)


# Basic State operations

def return_state(value: A) -> State[S, A]:
    """Wrap a value in State monad (return/pure).
    
    Args:
        value: Value to wrap
        
    Returns:
        State computation that returns value without changing state
    """
    return State(lambda s: (value, s))


def get_state() -> State[S, S]:
    """Get the current state.
    
    Returns:
        State computation that returns current state
    """
    return State(lambda s: (s, s))


def put_state(new_state: S) -> State[S, None]:
    """Replace the state.
    
    Args:
        new_state: New state value
        
    Returns:
        State computation that sets state
    """
    return State(lambda _: (None, new_state))


def modify_state(f: Callable[[S], S]) -> State[S, None]:
    """Modify the state using a function.
    
    Args:
        f: Function to transform state
        
    Returns:
        State computation that modifies state
    """
    return State(lambda s: (None, f(s)))


# Segmentation-specific state

@dataclass
class SegmentationState:
    """State for segmentation pipeline.
    
    Attributes:
        current_polytopes: List of active polytope tiles
        harmonic_accumulator: Running sums of harmonic coefficients
        sdr_buffer: Current 2048-bit SDR
        confidence_map: Per-voxel confidence scores
        classification_votes: Evidence for each object class
        iteration: Current iteration number
        history: List of previous states for checkpointing
    """
    current_polytopes: List[Dict[str, jnp.ndarray]]
    harmonic_accumulator: Dict[Tuple[int, int], complex]
    sdr_buffer: jnp.ndarray
    confidence_map: jnp.ndarray
    classification_votes: Dict[str, float]
    iteration: int = 0
    history: List['SegmentationState'] = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = []


# Specialized state operations

def get_polytopes() -> State[SegmentationState, List[Dict[str, jnp.ndarray]]]:
    """Get current polytope tiles."""
    return get_state().map(lambda s: s.current_polytopes)


def update_harmonics(new_harmonics: Dict[Tuple[int, int], complex]) -> State[SegmentationState, None]:
    """Update harmonic accumulator."""
    def update(state: SegmentationState) -> SegmentationState:
        # Merge harmonics
        merged = state.harmonic_accumulator.copy()
        for key, value in new_harmonics.items():
            if key in merged:
                merged[key] += value
            else:
                merged[key] = value
        
        return dataclass.replace(state, harmonic_accumulator=merged)
    
    return modify_state(update)


def update_sdr(new_sdr: jnp.ndarray) -> State[SegmentationState, None]:
    """Update SDR buffer."""
    return modify_state(lambda s: dataclass.replace(s, sdr_buffer=new_sdr))


def increment_iteration() -> State[SegmentationState, None]:
    """Increment iteration counter."""
    return modify_state(lambda s: dataclass.replace(s, iteration=s.iteration + 1))


def add_vote(class_name: str, weight: float) -> State[SegmentationState, None]:
    """Add classification vote."""
    def add(state: SegmentationState) -> SegmentationState:
        votes = state.classification_votes.copy()
        votes[class_name] = votes.get(class_name, 0.0) + weight
        return dataclass.replace(state, classification_votes=votes)
    
    return modify_state(add)


# Monad transformers

class StateT(Generic[S, A], ABC):
    """State monad transformer base class."""
    
    @abstractmethod
    def run(self, state: S) -> Any:
        """Run the transformed computation."""
        pass


class StateMaybe(StateT[S, Optional[A]]):
    """StateT with Maybe: computations that might fail.
    
    Combines State with Optional for fallible computations.
    """
    
    def __init__(self, run: Callable[[S], Optional[Tuple[A, S]]]):
        self._run = run
    
    def run(self, state: S) -> Optional[Tuple[A, S]]:
        return self._run(state)
    
    def bind(self, f: Callable[[A], 'StateMaybe[S, B]']) -> 'StateMaybe[S, B]':
        """Bind for StateMaybe."""
        def run_bind(state: S) -> Optional[Tuple[B, S]]:
            result = self._run(state)
            if result is None:
                return None
            a, new_state = result
            return f(a).run(new_state)
        
        return StateMaybe(run_bind)
    
    @staticmethod
    def lift(computation: State[S, A]) -> 'StateMaybe[S, A]':
        """Lift State computation to StateMaybe."""
        return StateMaybe(lambda s: computation.run(s))
    
    @staticmethod
    def fail() -> 'StateMaybe[S, A]':
        """Failed computation."""
        return StateMaybe(lambda s: None)


class StateList(StateT[S, List[A]]):
    """StateT with List: non-deterministic computations.
    
    Explores multiple possibilities in parallel.
    """
    
    def __init__(self, run: Callable[[S], List[Tuple[A, S]]]):
        self._run = run
    
    def run(self, state: S) -> List[Tuple[A, S]]:
        return self._run(state)
    
    def bind(self, f: Callable[[A], 'StateList[S, B]']) -> 'StateList[S, B]':
        """Bind for StateList."""
        def run_bind(state: S) -> List[Tuple[B, S]]:
            results = []
            for a, new_state in self._run(state):
                results.extend(f(a).run(new_state))
            return results
        
        return StateList(run_bind)
    
    @staticmethod
    def choices(values: List[A]) -> 'StateList[S, A]':
        """Non-deterministic choice between values."""
        return StateList(lambda s: [(v, s) for v in values])


# Combinator functions

def foreach_polytope(f: Callable[[Dict[str, jnp.ndarray]], State[SegmentationState, A]]) -> State[SegmentationState, List[A]]:
    """Map computation over polytopes, threading state.
    
    Args:
        f: Function from polytope to stateful computation
        
    Returns:
        Stateful computation returning list of results
    """
    def run_foreach(state: SegmentationState) -> Tuple[List[A], SegmentationState]:
        results = []
        current_state = state
        
        for polytope in state.current_polytopes:
            result, current_state = f(polytope).run(current_state)
            results.append(result)
        
        return results, current_state
    
    return State(run_foreach)


def until_confident(threshold: float, 
                   computation: State[SegmentationState, None],
                   max_iterations: int = 100) -> State[SegmentationState, bool]:
    """Repeat computation until confidence threshold reached.
    
    Args:
        threshold: Confidence threshold
        computation: Computation to repeat
        max_iterations: Maximum iterations
        
    Returns:
        State computation returning whether threshold was reached
    """
    def check_confidence(state: SegmentationState) -> float:
        return jnp.mean(state.confidence_map)
    
    def run_until(state: SegmentationState) -> Tuple[bool, SegmentationState]:
        current_state = state
        
        for _ in range(max_iterations):
            if check_confidence(current_state) >= threshold:
                return True, current_state
            
            _, current_state = computation.run(current_state)
        
        return False, current_state
    
    return State(run_until)


def with_checkpoint(computation: State[SegmentationState, A]) -> State[SegmentationState, Optional[A]]:
    """Save state, try computation, restore on failure.
    
    Args:
        computation: Computation to try
        
    Returns:
        Computation that returns None and restores state on failure
    """
    def run_checkpoint(state: SegmentationState) -> Tuple[Optional[A], SegmentationState]:
        # Save current state
        saved_state = dataclass.replace(
            state,
            history=state.history + [state]
        )
        
        try:
            result, new_state = computation.run(state)
            # Add checkpoint to history
            final_state = dataclass.replace(
                new_state,
                history=saved_state.history
            )
            return result, final_state
        except Exception:
            # Restore saved state on failure
            return None, saved_state
    
    return State(run_checkpoint)


def parallel_hypotheses(hypotheses: List[State[SegmentationState, A]]) -> State[SegmentationState, List[Tuple[A, float]]]:
    """Explore multiple segmentation hypotheses in parallel.
    
    Args:
        hypotheses: List of computations to try
        
    Returns:
        Results with confidence scores
    """
    def run_parallel(state: SegmentationState) -> Tuple[List[Tuple[A, float]], SegmentationState]:
        results = []
        best_state = state
        best_confidence = 0.0
        
        for hypothesis in hypotheses:
            # Try each hypothesis from original state
            result, new_state = hypothesis.run(state)
            confidence = jnp.mean(new_state.confidence_map)
            results.append((result, float(confidence)))
            
            # Keep best state
            if confidence > best_confidence:
                best_confidence = confidence
                best_state = new_state
        
        return results, best_state
    
    return State(run_parallel)


# Main segmentation pipeline

def tile_space(space_bounds: jnp.ndarray) -> State[SegmentationState, List[Dict[str, jnp.ndarray]]]:
    """Tile 3D space with polytopes."""
    def run_tiling(state: SegmentationState) -> Tuple[List[Dict[str, jnp.ndarray]], SegmentationState]:
        # Simplified tiling - would use actual polytope tiling
        tiles = []
        for i in range(8):  # Example: 8 tiles
            tiles.append({
                'center': jnp.array([i * 1.0, 0.0, 0.0]),
                'vertices': jnp.eye(3) * (i + 1)  # Placeholder
            })
        
        new_state = dataclass.replace(state, current_polytopes=tiles)
        return tiles, new_state
    
    return State(run_tiling)


def compute_harmonics(polytope: Dict[str, jnp.ndarray]) -> State[SegmentationState, Dict[Tuple[int, int], complex]]:
    """Compute spherical harmonics for a polytope."""
    def run_harmonics(state: SegmentationState) -> Tuple[Dict[Tuple[int, int], complex], SegmentationState]:
        # Simplified harmonic computation
        harmonics = {}
        for l in range(3):
            for m in range(-l, l + 1):
                # Placeholder computation
                harmonics[(l, m)] = complex(0.1 * l, 0.05 * m)
        
        return harmonics, state
    
    return State(run_harmonics)


def encode_harmonics(all_harmonics: List[Dict[Tuple[int, int], complex]]) -> State[SegmentationState, jnp.ndarray]:
    """Encode harmonics into SDR."""
    def run_encode(state: SegmentationState) -> Tuple[jnp.ndarray, SegmentationState]:
        # Merge all harmonics
        merged = {}
        for harmonics in all_harmonics:
            for key, value in harmonics.items():
                if key in merged:
                    merged[key] += value
                else:
                    merged[key] = value
        
        # Create SDR (simplified)
        sdr = jnp.zeros(2048, dtype=bool)
        # Would use actual harmonic SDR encoding
        sdr = sdr.at[:len(merged) * 10].set(True)
        
        new_state = dataclass.replace(
            state,
            harmonic_accumulator=merged,
            sdr_buffer=sdr
        )
        return sdr, new_state
    
    return State(run_encode)


def apply_error_correction(sdr: jnp.ndarray) -> State[SegmentationState, jnp.ndarray]:
    """Apply Golay error correction to SDR."""
    def run_correction(state: SegmentationState) -> Tuple[jnp.ndarray, SegmentationState]:
        # Simplified error correction
        corrected = sdr.copy()
        # Would use actual Golay correction
        
        new_state = dataclass.replace(state, sdr_buffer=corrected)
        return corrected, new_state
    
    return State(run_correction)


def classify_from_sdr(sdr: jnp.ndarray) -> State[SegmentationState, str]:
    """Classify object from SDR."""
    def run_classify(state: SegmentationState) -> Tuple[str, SegmentationState]:
        # Simplified classification
        active_bits = jnp.sum(sdr)
        
        if active_bits > 1000:
            classification = "bacteria"
            confidence = 0.8
        else:
            classification = "granule"
            confidence = 0.7
        
        # Update votes
        new_votes = state.classification_votes.copy()
        new_votes[classification] = new_votes.get(classification, 0.0) + confidence
        
        # Update confidence map
        new_confidence = jnp.ones_like(state.confidence_map) * confidence
        
        new_state = dataclass.replace(
            state,
            classification_votes=new_votes,
            confidence_map=new_confidence
        )
        
        return classification, new_state
    
    return State(run_classify)


def segment_pipeline(space_bounds: jnp.ndarray) -> State[SegmentationState, str]:
    """Main segmentation pipeline as monadic computation.
    
    This reads like imperative code but is purely functional:
    - Tile space with polytopes
    - Compute harmonics for each polytope
    - Encode harmonics into SDR
    - Apply error correction
    - Classify from corrected SDR
    
    Args:
        space_bounds: Bounds of 3D space to segment
        
    Returns:
        State computation returning classification
    """
    return (
        tile_space(space_bounds)
        .bind(lambda tiles: 
            foreach_polytope(compute_harmonics)
            .bind(lambda harmonics:
                encode_harmonics(harmonics)
                .bind(lambda sdr:
                    apply_error_correction(sdr)
                    .bind(lambda corrected:
                        classify_from_sdr(corrected)
                    )
                )
            )
        )
    )


# Alternative syntax using do-notation style
class Do:
    """Helper for do-notation style syntax."""
    
    def __init__(self):
        self.computations = []
    
    def __call__(self, computation):
        self.computations.append(computation)
        return self
    
    def build(self):
        """Build the monadic computation."""
        if not self.computations:
            return return_state(None)
        
        result = self.computations[0]
        for comp in self.computations[1:]:
            result = result.then(comp)
        
        return result


# Monad laws verification

def verify_left_identity(value: A, f: Callable[[A], State[S, B]], state: S) -> bool:
    """Verify left identity: return a >>= f ≡ f a"""
    left = return_state(value).bind(f).run(state)
    right = f(value).run(state)
    return left == right


def verify_right_identity(m: State[S, A], state: S) -> bool:
    """Verify right identity: m >>= return ≡ m"""
    left = m.bind(return_state).run(state)
    right = m.run(state)
    return left == right


def verify_associativity(m: State[S, A],
                        f: Callable[[A], State[S, B]],
                        g: Callable[[B], State[S, Any]],
                        state: S) -> bool:
    """Verify associativity: (m >>= f) >>= g ≡ m >>= (\\x -> f x >>= g)"""
    left = m.bind(f).bind(g).run(state)
    right = m.bind(lambda x: f(x).bind(g)).run(state)
    return left == right


# Example usage demonstrating functional purity

def example_segmentation():
    """Example of using the State monad for segmentation."""
    # Initial state
    initial_state = SegmentationState(
        current_polytopes=[],
        harmonic_accumulator={},
        sdr_buffer=jnp.zeros(2048, dtype=bool),
        confidence_map=jnp.zeros((64, 64, 64)),
        classification_votes={},
        iteration=0
    )
    
    # Run pipeline
    space_bounds = jnp.array([[0, 10], [0, 10], [0, 10]])
    result, final_state = segment_pipeline(space_bounds).run(initial_state)
    
    print(f"Classification: {result}")
    print(f"Final votes: {final_state.classification_votes}")
    print(f"Mean confidence: {jnp.mean(final_state.confidence_map)}")
    
    # The computation is pure - running again gives same result
    result2, _ = segment_pipeline(space_bounds).run(initial_state)
    assert result == result2, "Computation is not deterministic!"
    
    return result, final_state