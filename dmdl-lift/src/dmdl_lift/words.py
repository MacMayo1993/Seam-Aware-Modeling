"""Symbolic word generation and complexity analysis.

This module generates symbolic sequences (words) used as driving patterns:
- Fibonacci word: generated via substitution A→AB, B→A (Sturmian sequence)
- Periodic word: simple ABAB... pattern

The Fibonacci word is a standard example of a Sturmian sequence with the
property that for k-mer length L, it contains exactly L+1 distinct k-mers.
"""
import numpy as np


def fibonacci_word(length: int) -> np.ndarray:
    """Generate Fibonacci word via substitution system.

    Substitution rules:
    - A → AB
    - B → A

    This is a standard Sturmian sequence construction.
    For k-mer length L, contains exactly L+1 distinct k-mers.

    Parameters
    ----------
    length : int
        Desired length of output sequence

    Returns
    -------
    np.ndarray of str
        Array of 'A' and 'B' characters
    """
    w = "A"
    while len(w) < length:
        w_next = []
        for ch in w:
            w_next.append("AB" if ch == "A" else "A")
        w = "".join(w_next)
    return np.array(list(w[:length]))


def periodic_word(length: int) -> np.ndarray:
    """Generate simple periodic ABAB... word.

    Parameters
    ----------
    length : int
        Desired length of output sequence

    Returns
    -------
    np.ndarray of str
        Array of 'A' and 'B' characters alternating
    """
    return np.array(list(("AB" * (length//2 + 1))[:length]))


def kmer_coverage(word, L=8):
    """Count distinct k-mers (length-L substrings) in word.

    Parameters
    ----------
    word : array-like
        Symbolic sequence
    L : int
        K-mer length

    Returns
    -------
    int
        Number of distinct k-mers
    """
    kmers = set(tuple(word[i:i+L]) for i in range(len(word)-L+1))
    return len(kmers)


def complexity_profile(word, Ls=(2, 4, 6, 8, 10, 12)):
    """Compute k-mer coverage across multiple substring lengths.

    Parameters
    ----------
    word : array-like
        Symbolic sequence
    Ls : tuple of int
        K-mer lengths to probe

    Returns
    -------
    dict
        Maps L → number of distinct k-mers
    """
    return {L: kmer_coverage(word, L) for L in Ls}
