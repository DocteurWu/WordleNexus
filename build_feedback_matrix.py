#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build feedback matrix for exact Wordle solver.

Generates:
  - data/feedback_matrix_uint8.npy : (n_guesses, n_answers) uint8 matrix
  - data/word_index_maps.pkl         : mappings guess/answer <-> index

Each cell [guess_idx][answer_idx] contains a pattern index (0-242).
Pattern encoding: base-3 of the 5 feedback positions
  0 = BLACK (⬛), 1 = YELLOW (🟨), 2 = GREEN (🟩)
  index = p0*3^4 + p1*3^3 + p2*3^2 + p3*3^1 + p4*3^0

Run this script once on your PC. It takes ~30-60 minutes.
"""

import os
import sys
import time
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from solver import load_wordlists, feedback_optimized, config


def pattern_to_index(pattern_str: str) -> int:
    """Convert emoji pattern string to integer in [0, 242]."""
    val = 0
    for c in pattern_str:
        if c == config.black:
            digit = 0
        elif c == config.yellow:
            digit = 1
        elif c == config.green:
            digit = 2
        else:
            raise ValueError(f"Invalid pattern char: {repr(c)} in {repr(pattern_str)}")
        val = val * 3 + digit
    return val


def build():
    print("=" * 70)
    print("FEEDBACK MATRIX GENERATOR")
    print("=" * 70)

    answers, guesses = load_wordlists()
    n_answers = len(answers)
    n_guesses = len(guesses)

    print(f"\nAnswers : {n_answers}")
    print(f"Guesses : {n_guesses}")
    print(f"Matrix shape : ({n_guesses}, {n_answers})")
    print(f"Est. size    : ~{n_guesses * n_answers / 1e6:.1f} MB as uint8\n")

    # Alloc matrix
    matrix = np.empty((n_guesses, n_answers), dtype=np.uint8)

    # Build
    t0 = time.time()
    report_interval = max(1, n_guesses // 20)

    for gi, guess in enumerate(guesses):
        row = matrix[gi]
        for ai, answer in enumerate(answers):
            pat = feedback_optimized(answer, guess)
            row[ai] = pattern_to_index(pat)

        if (gi + 1) % report_interval == 0 or gi == n_guesses - 1:
            elapsed = time.time() - t0
            pct = (gi + 1) / n_guesses * 100
            rate = (gi + 1) / elapsed
            remaining = (n_guesses - (gi + 1)) / rate if rate > 0 else 0
            print(
                f"  {gi + 1:5d} / {n_guesses} ({pct:5.1f}%) | "
                f"{rate:.0f} guess/sec | ETA {remaining / 60:.1f} min"
            )

    total_time = time.time() - t0
    print(f"\nMatrix built in {total_time:.1f}s")

    # Save matrix
    os.makedirs(config.data_dir, exist_ok=True)
    mat_path = os.path.join(config.data_dir, "feedback_matrix_uint8.npy")
    np.save(mat_path, matrix)
    print(f"Saved : {mat_path} ({os.path.getsize(mat_path) / 1e6:.1f} MB)")

    # Save index maps
    guess_to_idx = {w: i for i, w in enumerate(guesses)}
    answer_to_idx = {w: i for i, w in enumerate(answers)}
    idx_to_guess = guesses
    idx_to_answer = answers

    maps = {
        "guess_to_idx": guess_to_idx,
        "answer_to_idx": answer_to_idx,
        "idx_to_guess": idx_to_guess,
        "idx_to_answer": idx_to_answer,
        "n_guesses": n_guesses,
        "n_answers": n_answers,
    }
    maps_path = os.path.join(config.data_dir, "word_index_maps.pkl")
    with open(maps_path, "wb") as f:
        pickle.dump(maps, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved : {maps_path} ({os.path.getsize(maps_path) / 1e6:.1f} MB)")

    # Sanity check
    print("\n" + "=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)
    # Check a few known pairs
    test_cases = [
        ("crane", "crane", "🟩🟩🟩🟩🟩", 242),
        ("salet", "crane", "⬛🟩⬛⬛⬛", None),
        ("adieu", "crane", "⬛⬛⬛⬛🟨", None),
    ]
    all_ok = True
    for answer, guess, expected_pat, expected_idx in test_cases:
        if expected_idx is None:
            expected_idx = pattern_to_index(expected_pat)
        gi = guess_to_idx[guess]
        ai = answer_to_idx[answer]
        actual = matrix[gi, ai]
        status = "✓" if actual == expected_idx else "✗"
        if actual != expected_idx:
            all_ok = False
        print(
            f"  {status} feedback({answer}, {guess}) = {actual} (expected {expected_idx})"
        )

    print(
        f"\n{'All checks passed!' if all_ok else 'SOME CHECKS FAILED - investigate!'}"
    )
    print("=" * 70)


if __name__ == "__main__":
    build()
