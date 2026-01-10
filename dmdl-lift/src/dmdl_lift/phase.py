"""Phase computation based on k-mer context.

For ΔMDL lift detection, we use k-mer identity as the phase variable.
This gives:
- Periodic word: always 2 phases (only ABAB and BABA patterns in AB-repeat)
- Fibonacci word: exactly L+1 phases (Sturmian property)

The key is that phase count is bounded and grows linearly with L,
ensuring sufficient samples per phase for reliable estimation.
"""
import numpy as np


def kmer_id(word, t, L=8):
    """Compute k-mer hash at position t.

    Hashes the k-mer ending at position t into an integer.

    Parameters
    ----------
    word : np.ndarray
        Symbolic sequence ('A' or 'B')
    t : int
        Time index (k-mer ends at this position)
    L : int
        K-mer length

    Returns
    -------
    int or None
        Integer in [0, 2^L) representing the k-mer, or None if t < L-1
    """
    start = t - L + 1
    if start < 0:
        return None
    bits = (word[start:t+1] == "B").astype(int)
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def phase_from_drive(word, t, L=8):
    """Get phase from k-mer context only.

    Returns the k-mer hash (no temporal modulation).
    This is the phase variable used for ΔMDL model selection.

    Parameters
    ----------
    word : np.ndarray
        Symbolic sequence
    t : int
        Time index
    L : int
        K-mer length

    Returns
    -------
    int or None
        Phase identifier (k-mer hash)
    """
    return kmer_id(word, t, L=L)


def phase_state_coverage(word, L=8):
    """Count distinct phases realized by a word.

    Parameters
    ----------
    word : np.ndarray
        Symbolic sequence
    L : int
        K-mer length

    Returns
    -------
    n_states : int
        Number of distinct k-mers
    states : set
        Set of realized k-mer hashes
    """
    states = set()
    for t in range(L-1, len(word)):
        ph = kmer_id(word, t, L=L)
        if ph is not None:
            states.add(ph)
    return len(states), states


def expected_phases(word_type, L):
    """Expected number of phases for word type.

    Parameters
    ----------
    word_type : str
        'periodic' or 'fibonacci'
    L : int
        K-mer length

    Returns
    -------
    int : Expected phase count

    Notes
    -----
    - Periodic ABAB... pattern: always 2 k-mers regardless of L
      (for even L: ABAB... and BABA...; odd L depends on alignment)
    - Fibonacci word: L+1 k-mers (Sturmian property)
    """
    if word_type == 'periodic':
        return 2  # Only two k-mers possible in AB-repeat
    elif word_type == 'fibonacci':
        return L + 1  # Sturmian property
    else:
        raise ValueError(f"Unknown word type: {word_type}")
