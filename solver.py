#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wordle Solver Optimized v2.0 - ENGLISH VERSION
Improvements: Performance, Logging, Advanced Scoring, Statistics
"""

import os
import pickle
import gzip
import math
import hashlib
import time
import logging
import threading
from collections import Counter, defaultdict
from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    as_completed,
    CancelledError,
)
from functools import partial, lru_cache
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


def init_worker(cache_data, mode_data=None):
    """Initializer for multiprocessing pool to set the cache and mode."""
    _player_possible_sequence_lax.cache = cache_data
    _player_possible_sequence_moderate.cache = cache_data
    _player_plausibility_score.cache = cache_data
    if mode_data is not None:
        _player_possible_sequence_moderate.mode = mode_data


# =========================
# 0. Configuration & Logging
# =========================


@dataclass
class SolverConfig:
    """Centralized solver configuration"""

    # __slots__ removed to fix conflict with class variable defaults

    word_length: int = 5
    green: str = "🟩"
    yellow: str = "🟨"
    black: str = "⬛"
    data_dir: str = "data"
    cache_path: str = "data/pattern_cache.pkl"
    log_dir: str = "logs"

    scoring_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "strict_ratio": 0.42,
            "perfect_bonus": 0.20,
            "tightness": 0.15,
            "coherence": 0.10,
            "difficulty": 0.05,
            "letter_frequency": 0.05,
            "entropy": 0.03,
        }
    )

    # ENGLISH letter frequency (normalized)
    letter_freq: Dict[str, float] = field(
        default_factory=lambda: {
            "e": 1.00,
            "a": 0.85,
            "r": 0.80,
            "i": 0.78,
            "o": 0.75,
            "t": 0.72,
            "n": 0.70,
            "s": 0.68,
            "l": 0.65,
            "c": 0.60,
            "u": 0.58,
            "d": 0.55,
            "p": 0.52,
            "m": 0.50,
            "h": 0.48,
            "g": 0.45,
            "b": 0.42,
            "f": 0.40,
            "y": 0.38,
            "w": 0.35,
            "k": 0.30,
            "v": 0.28,
            "x": 0.15,
            "z": 0.12,
            "j": 0.10,
            "q": 0.05,
        }
    )

    @property
    def answers_file(self) -> str:
        return os.path.join(self.data_dir, "answers.txt")

    @property
    def guesses_file(self) -> str:
        return os.path.join(self.data_dir, "allowed_guesses.txt")


# Global config instance
config = SolverConfig()

# Backward compatibility
WORD_LENGTH = config.word_length
GREEN, YELLOW, BLACK = config.green, config.yellow, config.black

# =========================
# 1. Logging System
# =========================


class SolverLogger:
    """Log manager with different levels"""

    def __init__(self, name: str = "WordleSolver"):
        # Force UTF-8 on stdout for emoji patterns
        import sys

        if sys.stdout.encoding != "utf-8" and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Create logs folder
        os.makedirs(config.log_dir, exist_ok=True)

        # File handler (DEBUG and above)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(
            os.path.join(config.log_dir, f"solver_{timestamp}.log"), encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)

        # Console handler (INFO and above)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Detailed format
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(funcName)-20s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def error(self, msg: str, exc_info=False):
        self.logger.error(msg, exc_info=exc_info)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def info(self, msg: str):
        self.logger.info(msg)

    def debug(self, msg: str):
        self.logger.debug(msg)


# Global instance
logger = SolverLogger()

# =========================
# 2. Statistics
# =========================


class SolverStats:
    """Performance statistics collection and analysis"""

    def __init__(self):
        self.solve_times: List[float] = []
        self.candidate_counts: List[int] = []
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.phase_times: Dict[str, List[float]] = defaultdict(list)
        self.total_solves: int = 0

    def log_solve(self, duration: float, num_candidates: int):
        self.solve_times.append(duration)
        self.candidate_counts.append(num_candidates)
        self.total_solves += 1

    def log_phase(self, phase_name: str, duration: float):
        self.phase_times[phase_name].append(duration)

    def log_cache_access(self, hit: bool):
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def get_summary(self) -> str:
        if not self.solve_times:
            return "No statistics available."

        avg_time = sum(self.solve_times) / len(self.solve_times)
        avg_candidates = sum(self.candidate_counts) / len(self.candidate_counts)
        cache_rate = (
            self.cache_hits / (self.cache_hits + self.cache_misses) * 100
            if (self.cache_hits + self.cache_misses) > 0
            else 0
        )

        lines = [
            "\n" + "=" * 60,
            "📊 PERFORMANCE STATISTICS",
            "=" * 60,
            f"Total solves            : {self.total_solves}",
            f"Average time            : {avg_time:.3f}s",
            f"Time min/max            : {min(self.solve_times):.3f}s / {max(self.solve_times):.3f}s",
            f"Average candidates      : {avg_candidates:.1f}",
            f"Cache hit rate          : {cache_rate:.1f}%",
            "",
        ]

        if self.phase_times:
            lines.append("Phase breakdown:")
            for phase, times in sorted(self.phase_times.items()):
                avg = sum(times) / len(times)
                lines.append(f"  • {phase:<20}: {avg:.3f}s")

        lines.append("=" * 60)
        return "\n".join(lines)


# Global instance
stats = SolverStats()

# =========================
# 3. Load Wordlists
# =========================


def load_wordlists():
    """Load word lists with validation"""
    logger.info("Loading wordlists...")
    start = time.time()

    try:
        with open(config.answers_file, "r", encoding="utf-8") as f:
            answers = [l.strip().lower() for l in f if l.strip()]

        with open(config.guesses_file, "r", encoding="utf-8") as f:
            guesses = [l.strip().lower() for l in f if l.strip()]

        logger.info(
            f"✓ {len(answers)} answers and {len(guesses)} guesses loaded in {time.time() - start:.3f}s"
        )

        return answers, guesses

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Loading error: {e}")
        raise


# =========================
# 4. Optimized Feedback
# =========================


def feedback_optimized(secret: str, guess: str) -> str:
    """
    Optimized feedback calculation using array operations.
    ~2-3x faster than Counter-based version.
    """
    result = [BLACK] * WORD_LENGTH
    secret_counts = [0] * 26

    # Phase 1: Greens + count remaining secret letters
    for i in range(WORD_LENGTH):
        if guess[i] == secret[i]:
            result[i] = GREEN
        else:
            secret_counts[ord(secret[i]) - 97] += 1

    # Phase 2: Yellows (only for non-green positions)
    for i in range(WORD_LENGTH):
        if result[i] == BLACK:
            idx = ord(guess[i]) - 97
            if secret_counts[idx] > 0:
                result[i] = YELLOW
                secret_counts[idx] -= 1

    return "".join(result)


# Alias for compatibility
feedback = feedback_optimized

# =========================
# 5. Global Cache with Compression
# =========================


def build_pattern_map_for_secret(
    secret: str, allowed_guesses: List[str]
) -> Tuple[str, Dict]:
    """Build pattern map for a given secret"""
    m = defaultdict(list)
    for g in allowed_guesses:
        p = feedback(secret, g)
        m[p].append(g)
    return secret, dict(m)


def compute_stable_hash(word_list: List[str]) -> str:
    """Compute stable hash to detect changes"""
    content = "\n".join(sorted(word_list)).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def build_global_pattern_cache(
    answers: List[str], allowed_guesses: List[str], status_callback=None
) -> Dict:
    """
    Build or load global pattern cache
    Uses parallelization and compression
    """
    logger.info("Initializing pattern cache...")

    cache_meta = {
        "answers_hash": compute_stable_hash(answers),
        "guesses_hash": compute_stable_hash(allowed_guesses),
        "version": "3.0",
    }

    # Try loading existing cache
    if os.path.exists(config.cache_path):
        logger.info(f"Cache found: {config.cache_path}")
        try:
            with open(config.cache_path, "rb") as f:
                cached_data = pickle.load(f)

            if isinstance(cached_data, dict) and "meta" in cached_data:
                cached_meta = cached_data.get("meta")
                is_valid = True
                for key, value in cache_meta.items():
                    if cached_meta.get(key) != value:
                        logger.warning(
                            f"Cache mismatch on key '{key}'. Expected: {value}, Found: {cached_meta.get(key)}"
                        )
                        is_valid = False

                if is_valid:
                    logger.info("✓ Valid cache loaded from disk.")
                    stats.log_cache_access(True)
                    return cached_data["cache"]

            logger.warning("Cache metadata mismatch or invalid format. Rebuilding...")
            stats.log_cache_access(False)

        except Exception as e:
            logger.error(f"Cache loading error: {e}")
            stats.log_cache_access(False)

    # Build cache (sequential — faster than ThreadPool for tiny tasks due to GIL)
    logger.info("Building cache (may take 1-2 minutes)...")
    start_time = time.time()

    cache = {}
    for i, secret in enumerate(answers):
        _, patterns = build_pattern_map_for_secret(secret, allowed_guesses)
        cache[secret] = patterns
        if (i + 1) % 500 == 0:
            pct = (i + 1) / len(answers) * 100
            logger.debug(f"Progress: {i + 1}/{len(answers)} ({pct:.1f}%)")

    # Save cache atomically to avoid corruption
    cache_data = {"cache": cache, "meta": cache_meta}
    os.makedirs(config.data_dir, exist_ok=True)

    temp_path = config.cache_path + ".tmp"
    with open(temp_path, "wb") as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp_path, config.cache_path)

    elapsed = time.time() - start_time
    logger.info(f"✓ Cache built and saved in {elapsed:.2f}s ({len(cache)} entries)")

    return cache


# =========================
# 6. Human Logic
# =========================


def info_letters(pattern: str, guess: str) -> Set[str]:
    """Letters that provide information (green or yellow)"""
    return {g for g, p in zip(guess, pattern) if p in (GREEN, YELLOW)}


def black_letters(pattern: str, guess: str) -> Set[str]:
    """Letters completely absent (all black)"""
    letter_patterns = defaultdict(list)
    for g, p in zip(guess, pattern):
        letter_patterns[g].append(p)

    return {
        letter
        for letter, patterns in letter_patterns.items()
        if all(p == BLACK for p in patterns)
    }


def green_positions(pattern: str, guess: str) -> Dict[int, str]:
    """Positions where green is observed: {index: letter}"""
    return {i: guess[i] for i in range(len(pattern)) if pattern[i] == GREEN}


def guesses_respect_greens(prev_guess: str, prev_pat: str, next_guess: str) -> bool:
    """Check that next guess respects green positions from previous attempt"""
    for i, c in enumerate(prev_pat):
        if c == GREEN and prev_guess[i] != next_guess[i]:
            return False
    return True


def compute_plausibility(prev_guess: str, prev_pat: str, next_guess: str) -> float:
    """
    Compute plausibility score that a human player would make next_guess
    after seeing prev_pat for prev_guess.

    Rules (tuned for "human enough but not perfect"):
    - Greens are HARD (score=0 if violated).
    - Yellows: bonus for keeping, penalty for dropping.
    - Blacks: mild penalty for reusing (Wordle allows it, but humans avoid it).
    - Yellow positions: bonus for NOT putting yellow back in same wrong position.
    """
    score = 1.0

    prev_greens = green_positions(prev_pat, prev_guess)
    prev_info = info_letters(prev_pat, prev_guess)
    prev_blacks = black_letters(prev_pat, prev_guess)

    for pos, letter in prev_greens.items():
        if next_guess[pos] != letter:
            return 0.0

    # Yellow logic
    for letter in prev_info:
        if letter in next_guess:
            score *= 1.0
        else:
            score *= 0.55  # Drop yellow = significant penalty

    # Black logic (mild: game allows reusing blacks, humans usually don't)
    for letter in prev_blacks:
        if letter in next_guess:
            score *= 0.70  # Reuse black = noticeable penalty

    # Yellow position logic: avoid putting yellow back at same position
    for i, p in enumerate(prev_pat):
        if p == YELLOW and next_guess[i] == prev_guess[i]:
            score *= 0.80  # Same wrong position again

    return score


# =========================
# 7. Verification Functions (with LRU cache)
# =========================


@lru_cache(maxsize=50000)
def _player_possible_sequence_lax(secret: str, pats_tuple: Tuple[str, ...]) -> bool:
    """Lax verification: all patterns exist"""
    m = _player_possible_sequence_lax.cache.get(secret)

    if not m:
        return False

    return all(p in m for p in pats_tuple)


def _player_possible_sequence_moderate(
    secret: str, pats_tuple: Tuple[str, ...]
) -> bool:
    """
    Moderate verification: accumulated greens must be respected.
    O(num_patterns × num_guesses) — no combinatorial explosion.
    """
    m = _player_possible_sequence_moderate.cache.get(secret)

    if not m:
        return False

    acc_greens: Dict[int, str] = {}

    for pat in pats_tuple:
        candidates = m.get(pat, [])
        if not candidates:
            return False

        found = False
        for guess in candidates:
            # Must respect all accumulated greens
            if any(guess[pos] != letter for pos, letter in acc_greens.items()):
                continue
            # Add this guess's greens to accumulated constraints
            for i, p in enumerate(pat):
                if p == GREEN:
                    acc_greens[i] = guess[i]
            found = True
            break

        if not found:
            return False

    return True


def _player_plausibility_score(secret: str, pats_tuple: Tuple[str, ...]) -> float:
    """
    Probabilistic verification with beam search.
    Keeps top 200 paths to avoid combinatorial explosion.
    Returns the best plausibility score [0, 1].
    """
    m = _player_plausibility_score.cache.get(secret)

    if not m:
        return 0.0

    candidates = m.get(pats_tuple[0], [])
    if not candidates:
        return 0.0

    # Each beam item: (accumulated_greens_dict, last_guess, last_pat, cumul_score)
    beams = [
        (
            {i: g[i] for i, p in enumerate(pats_tuple[0]) if p == GREEN},
            g,
            pats_tuple[0],
            1.0,
        )
        for g in candidates
    ]

    for pat in pats_tuple[1:]:
        candidates = m.get(pat, [])
        if not candidates:
            return 0.0

        new_beams = []
        for acc_greens, prev_guess, prev_pat, cumul_score in beams:
            for guess in candidates:
                # Hard constraint: greens must match
                if any(guess[pos] != letter for pos, letter in acc_greens.items()):
                    continue

                p = compute_plausibility(prev_guess, prev_pat, guess)
                combined = cumul_score * p
                if combined > 0.001:
                    new_greens = dict(acc_greens)
                    for i, p_char in enumerate(pat):
                        if p_char == GREEN:
                            new_greens[i] = guess[i]
                    new_beams.append((new_greens, guess, pat, combined))

        if not new_beams:
            return 0.0

        # Beam pruning: keep best 100
        if len(new_beams) > 100:
            new_beams.sort(key=lambda x: x[3], reverse=True)
            new_beams = new_beams[:100]

        beams = new_beams

    if beams:
        return max(s for _, _, _, s in beams)

    return 0.0


def check_player_coherence_loose(patterns: List[str]) -> bool:
    """Check temporal coherence of patterns"""
    greens = [p.count(GREEN) for p in patterns]
    return all(greens[i] >= greens[i - 1] - 1 for i in range(1, len(patterns)))


_player_possible_sequence_lax.cache = {}
_player_possible_sequence_moderate.cache = {}
_player_plausibility_score.cache = {}

# =========================
# 8. Advanced Scoring
# =========================


class AdvancedScorer:
    """Multi-criteria advanced scoring system"""

    def __init__(self, cache: Dict, answers: List[str]):
        self.cache = cache
        self.answers = set(answers)
        logger.debug("AdvancedScorer initialized")

    def calculate_letter_frequency_score(self, word: str) -> float:
        """Score based on letter frequency"""
        return sum(config.letter_freq.get(c, 0.1) for c in word) / WORD_LENGTH

    def calculate_entropy(
        self, word: str, candidates: Set[str], sample_size: int = 300
    ) -> float:
        """Calculate information entropy of a word with sampling for large sets"""
        if len(candidates) <= 1:
            return 0.0

        if len(candidates) > sample_size:
            import random

            sampled = set(random.sample(sorted(candidates), sample_size))
        else:
            sampled = candidates

        pattern_dist = defaultdict(int)
        for candidate in sampled:
            pattern = feedback(candidate, word)
            pattern_dist[pattern] += 1

        total = len(sampled)
        entropy = 0.0
        for count in pattern_dist.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        return entropy

    def calculate_tightness(self, word: str) -> float:
        """Pattern rarity score (rarer = better)"""
        num_patterns = len(self.cache[word])
        total_patterns = len(self.cache)
        return -math.log((num_patterns / total_patterns) + 1e-9)

    def score_candidates(
        self,
        candidates_info: List[Tuple[str, int]],
        players_grids: List[List[str]],
        plausibility_scores: Optional[Dict[str, float]] = None,
        mode: str = "moderate",
    ) -> List[Tuple]:
        """
        Main scoring with all criteria.
        Adapts weights based on validation mode.

        Returns: [(word, score, ratio, perfect, tight, avg_tries, freq, entropy), ...]
        If plausibility_scores provided, appends plausibility score as 9th element.
        """
        n_players = len(players_grids) if players_grids else 1
        avg_tries = (
            sum(len(p) for p in players_grids) / n_players if players_grids else 0
        )
        coherence = (
            sum(check_player_coherence_loose(p) for p in players_grids) / n_players
            if players_grids
            else 1.0
        )
        difficulty = 1 / (avg_tries + 1)

        candidate_words = {w for w, _ in candidates_info}

        results = []
        w = config.scoring_weights

        # Dynamic weight adjustment based on mode
        if mode == "probabilistic" and plausibility_scores:
            # Probabilistic: plausibility is the discriminant
            weights = {
                "plausibility": 0.50,
                "tightness": 0.20,
                "coherence": 0.10,
                "difficulty": 0.05,
                "letter_frequency": 0.05,
                "entropy": 0.10,
            }
        elif mode == "moderate":
            # Moderate: strict_ratio is constant (all passed), redistribute
            weights = {
                "tightness": 0.30,
                "coherence": 0.15,
                "difficulty": 0.10,
                "letter_frequency": 0.10,
                "entropy": 0.35,
            }
        else:
            # Lax: original weights (strict_ratio still matters here)
            weights = {
                "strict_ratio": w["strict_ratio"],
                "perfect_bonus": w["perfect_bonus"],
                "tightness": w["tightness"],
                "coherence": w["coherence"],
                "difficulty": w["difficulty"],
                "letter_frequency": w["letter_frequency"],
                "entropy": w["entropy"],
            }

        for word, strict_count in candidates_info:
            ratio = strict_count / n_players if n_players > 0 else 1.0
            perfect = 1 if strict_count == n_players else 0
            tight = self.calculate_tightness(word)
            freq = self.calculate_letter_frequency_score(word)
            entropy = self.calculate_entropy(word, candidate_words)

            plaus = plausibility_scores.get(word, 1.0) if plausibility_scores else 1.0

            if mode == "probabilistic" and plausibility_scores:
                score = (
                    weights["plausibility"] * plaus
                    + weights["tightness"] * tight
                    + weights["coherence"] * coherence
                    + weights["difficulty"] * difficulty
                    + weights["letter_frequency"] * freq
                    + weights["entropy"] * entropy
                )
                ratio = plaus  # for display
                perfect = 1 if plaus > 0.5 else 0
            elif mode == "moderate":
                score = (
                    weights["tightness"] * tight
                    + weights["coherence"] * coherence
                    + weights["difficulty"] * difficulty
                    + weights["letter_frequency"] * freq
                    + weights["entropy"] * entropy
                )
                # In moderate, ratio/perfect are display-only (all passed)
                ratio = 1.0
                perfect = 1
            else:
                score = (
                    weights["strict_ratio"] * ratio
                    + weights["perfect_bonus"] * perfect
                    + weights["tightness"] * tight
                    + weights["coherence"] * coherence
                    + weights["difficulty"] * difficulty
                    + weights["letter_frequency"] * freq
                    + weights["entropy"] * entropy
                )

            if plausibility_scores is not None:
                results.append(
                    (
                        word,
                        score,
                        ratio,
                        perfect,
                        tight,
                        avg_tries,
                        freq,
                        entropy,
                        plaus,
                    )
                )
            else:
                results.append(
                    (word, score, ratio, perfect, tight, avg_tries, freq, entropy)
                )

        return sorted(results, key=lambda x: x[1], reverse=True)


# =========================
# 9. Personal Solver Mode
# =========================


class PersonalSolver:
    """Interactive personal solver with step-by-step guidance"""

    def __init__(self, solver):
        self.solver = solver
        self.attempts = []
        self.remaining_candidates = set(solver.answers)
        self.exact_solver = ExactSolver(solver.guesses, solver.answers)

    def parse_pattern_input(self, pattern_str: str) -> str:
        """Parse user pattern input (flexible formats)"""
        pattern_str = pattern_str.strip().upper()

        # Replace common formats
        replacements = {
            "V": GREEN,  # French Vert
            "J": YELLOW,  # French Jaune
            "G": BLACK,  # French Gris
            "Y": YELLOW,  # English Yellow
            "B": BLACK,  # English Black
            "🟩": GREEN,
            "🟨": YELLOW,
            "⬛": BLACK,
        }

        result = pattern_str
        for old, new in replacements.items():
            result = result.replace(old, new)

        return result

    def add_attempt(self, guess: str, pattern: str):
        """Add an attempt and filter candidates"""
        guess = guess.lower()
        self.attempts.append({"guess": guess, "pattern": pattern})

        # Filter candidates
        self.remaining_candidates = {
            secret
            for secret in self.remaining_candidates
            if feedback(secret, guess) == pattern
        }

        print(
            f"Attempt added: {guess.upper()} -> {len(self.remaining_candidates)} candidates remain"
        )

    def get_best_next_guess(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """Get best next guesses based on remaining candidates"""
        if len(self.remaining_candidates) <= 2:
            # If few candidates remain, suggest them directly
            return [(w, 1.0) for w in sorted(self.remaining_candidates)]

        if self.exact_solver.is_ready():
            return self.exact_solver.get_best_guesses(self.remaining_candidates, top_n)

        # Fallback to old heuristic if matrix not loaded
        scores = []
        for word in list(self.solver.guesses)[:500]:
            entropy = self.solver.scorer.calculate_entropy(
                word, self.remaining_candidates
            )
            letter_variety = len(set(word))
            freq = self.solver.scorer.calculate_letter_frequency_score(word)
            score = entropy * 0.60 + letter_variety * 0.25 + freq * 0.15
            scores.append((word, score))

        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]

    def get_remaining_words(self, max_words: int = 50) -> List[str]:
        """Get list of remaining possible words"""
        return sorted(self.remaining_candidates)[:max_words]

    def reset(self):
        """Reset solver state"""
        self.attempts = []
        self.remaining_candidates = set(self.solver.answers)


# =========================
# 9b. Exact Solver (Matrix-based)
# =========================


class ExactSolver:
    """
    Exact Expected Remaining Size (ERS) solver using precomputed feedback matrix.
    Loads data/feedback_matrix_uint8.npy for instant O(1) partitions.
    """

    def __init__(self, guesses: List[str], answers: List[str]):
        self.guesses = guesses
        self.answers = answers
        self.matrix: Optional[np.ndarray] = None
        self.guess_to_idx: Dict[str, int] = {}
        self.answer_to_idx: Dict[str, int] = {}
        self.idx_to_guess: List[str] = []
        self.idx_to_answer: List[str] = []
        self._loaded = False
        self._load_matrix()

    def _load_matrix(self):
        mat_path = os.path.join(config.data_dir, "feedback_matrix_uint8.npy")
        maps_path = os.path.join(config.data_dir, "word_index_maps.pkl")

        if not os.path.exists(mat_path) or not os.path.exists(maps_path):
            logger.warning(
                "ExactSolver: Matrix files not found. Run build_feedback_matrix.py first."
            )
            return

        try:
            import numpy as np

            self.matrix = np.load(mat_path)
            with open(maps_path, "rb") as f:
                maps = pickle.load(f)

            self.guess_to_idx = maps["guess_to_idx"]
            self.answer_to_idx = maps["answer_to_idx"]
            self.idx_to_guess = maps["idx_to_guess"]
            self.idx_to_answer = maps["idx_to_answer"]
            self._loaded = True
            logger.info(
                f"ExactSolver: Loaded matrix {self.matrix.shape} ({self.matrix.nbytes / 1e6:.1f} MB)"
            )
        except Exception as e:
            logger.error(f"ExactSolver: Failed to load matrix: {e}")

    def is_ready(self) -> bool:
        return self._loaded

    def ers_for_guess(self, guess: str, candidate_indices: np.ndarray) -> float:
        """
        Compute Expected Remaining Size for a guess against candidates.
        ERS = sum(count(pattern)^2) / n_candidates
        Lower is better.
        """
        if not self._loaded:
            return float("inf")

        gi = self.guess_to_idx.get(guess)
        if gi is None:
            return float("inf")

        # Extract pattern values for this guess against all candidates
        patterns = self.matrix[gi, candidate_indices]

        # Count occurrences of each pattern (0-242)
        counts = np.bincount(patterns, minlength=243)
        total = candidate_indices.shape[0]

        # ERS = sum(counts^2) / total
        ers = float(np.dot(counts, counts)) / total
        return ers

    def get_all_ers(self, candidates: Set[str]) -> List[Tuple[str, float]]:
        """Return ERS for each word in candidates against all candidates, sorted."""
        if not self._loaded or not candidates:
            return []

        candidate_indices_list = []
        valid_candidates = []
        for w in candidates:
            ai = self.answer_to_idx.get(w)
            if ai is not None:
                candidate_indices_list.append(ai)
                valid_candidates.append(w)

        if not candidate_indices_list:
            return []

        candidate_indices = np.array(candidate_indices_list, dtype=np.int32)

        results = []
        for word in valid_candidates:
            ers = self.ers_for_guess(word, candidate_indices)
            results.append((word, ers))

        results.sort(key=lambda x: x[1])
        return results

    def get_best_guesses(
        self, candidates: Set[str], top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find best guesses that minimize Expected Remaining Size.
        Evaluates all 14,854 guesses exactly (no sampling).
        Returns: [(word, ers), ...] sorted by ERS ascending.
        """
        if not self._loaded:
            return []

        if len(candidates) <= 1:
            return [(w, 0.0) for w in sorted(candidates)[:top_n]]

        # Build numpy array of candidate answer indices
        candidate_indices_list = []
        for w in candidates:
            ai = self.answer_to_idx.get(w)
            if ai is not None:
                candidate_indices_list.append(ai)

        if not candidate_indices_list:
            return []

        candidate_indices = np.array(candidate_indices_list, dtype=np.int32)

        # Evaluate every guess
        results = []
        for guess in self.guesses:
            ers = self.ers_for_guess(guess, candidate_indices)
            results.append((guess, ers))

        # Sort by ERS ascending (lower = better split)
        results.sort(key=lambda x: x[1])
        return results[:top_n]


# =========================
# 10. Main Solver
# =========================


class WordleSolver:
    """Optimized Wordle Solver with logging and statistics"""

    VALIDATION_MODES = ("lax", "probabilistic", "exact")

    def __init__(self, status_callback=None):
        logger.info("=" * 60)
        logger.info("Initializing WordleSolver v3.0")
        logger.info("=" * 60)

        start_time = time.time()

        self.answers, self.guesses = load_wordlists()

        self.cache = build_global_pattern_cache(
            self.answers, self.guesses, status_callback
        )

        self.scorer = AdvancedScorer(self.cache, self.answers)

        self._set_current_cache_for_lru()

        # Persistent thread pool (zero spawn overhead vs ProcessPool)
        self._executor = ThreadPoolExecutor(
            max_workers=min(8, (os.cpu_count() or 1) * 2)
        )
        logger.debug(
            f"Persistent ThreadPool initialized ({min(8, (os.cpu_count() or 1) * 2)} workers)"
        )

        # Pre-compute best opening words once
        self._opening_words = None
        self._opening_candidates = None
        self.exact_solver = ExactSolver(self.guesses, self.answers)

        elapsed = time.time() - start_time
        logger.info(f"✓ WordleSolver initialized in {elapsed:.2f}s")
        logger.info("=" * 60 + "\n")

    def _set_current_cache_for_lru(self):
        """Attach cache to verification functions"""
        _player_possible_sequence_lax.cache_clear()
        _player_possible_sequence_lax.cache = self.cache
        _player_possible_sequence_moderate.cache = self.cache
        _player_plausibility_score.cache = self.cache
        logger.debug("LRU cache reset")

    def solve(
        self,
        players_grids: List[List[str]],
        personal_attempts: List[Dict],
        validation_mode: str = "moderate",
        candidates_to_check: Optional[set] = None,
        progress_callback=None,
        stop_event: Optional[threading.Event] = None,
    ) -> Tuple[str, set]:
        """
        Solve Wordle with detailed logging.

        Args:
            players_grids: List of player grids (patterns).
            personal_attempts: List of personal attempts {guess, pattern}.
            validation_mode: 'lax', 'moderate', or 'probabilistic'.
            candidates_to_check: Optional set of words to start with.
            progress_callback: Optional callback for progress updates.
            stop_event: Optional threading.Event to gracefully stop the analysis.

        Returns:
            A tuple containing:
            - Formatted string with results.
            - A set of candidate words.
        """
        if validation_mode not in self.VALIDATION_MODES:
            raise ValueError(
                f"Invalid mode: {validation_mode}. Use {self.VALIDATION_MODES}"
            )

        logger.info("\n" + "=" * 60)
        logger.info(f"NEW SOLVE (Mode: {validation_mode.upper()})")
        logger.info("=" * 60)

        solve_start = time.time()

        if not players_grids and not personal_attempts:
            logger.warning("No data provided")
            return "No data entered.", set()

        logger.info(
            f"Players: {len(players_grids)}, Personal attempts: {len(personal_attempts)}"
        )

        # --- Phase 1: Lax Filtering ---
        logger.info("\n--- Phase 1: Lax Filtering ---")
        phase_start = time.time()

        if candidates_to_check is None:
            common_candidates = set(self.answers)
        else:
            common_candidates = candidates_to_check

        logger.debug(f"Initial candidates: {len(common_candidates)}")

        if players_grids:
            sorted_grids = sorted(
                players_grids,
                key=lambda g: sum(p.count(GREEN) for p in g),
                reverse=True,
            )
            for idx, player_grid in enumerate(sorted_grids, 1):
                before = len(common_candidates)
                common_candidates = {
                    secret
                    for secret in common_candidates
                    if _player_possible_sequence_lax(secret, tuple(player_grid))
                }
                after = len(common_candidates)
                logger.debug(f"Player {idx}: {before} -> {after} candidates")
                if not common_candidates:
                    logger.warning(f"No candidates after player {idx}")
                    break
                if len(common_candidates) == 1:
                    logger.info(f"✓ Unique solution found after player {idx}")
                    break

        phase1_time = time.time() - phase_start
        stats.log_phase("Phase1_Lax", phase1_time)
        logger.info(
            f"Phase 1 completed in {phase1_time:.3f}s: {len(common_candidates)} candidates"
        )

        # --- Phase 1.5: Personal Attempts ---
        if personal_attempts and common_candidates:
            logger.info("\n--- Phase 1.5: Personal Attempts ---")
            phase_start = time.time()

            before = len(common_candidates)
            common_candidates = {
                secret
                for secret in common_candidates
                if all(
                    feedback(secret, att["guess"]) == att["pattern"]
                    for att in personal_attempts
                )
            }
            after = len(common_candidates)

            phase15_time = time.time() - phase_start
            stats.log_phase("Phase1.5_Personal", phase15_time)
            logger.info(
                f"Phase 1.5 completed in {phase15_time:.3f}s: {before} -> {after} candidates"
            )

        if not common_candidates:
            logger.error("No candidates found")
            return "No possible words found.", set()

        # --- Lax mode: return immediately ---
        if validation_mode == "lax":
            total_time = time.time() - solve_start
            logger.info(f"\n✓ Lax solve in {total_time:.3f}s")
            return self._format_lax_results(
                common_candidates, total_time
            ), common_candidates

        # --- Phase 2: Moderate / Probabilistic / Exact Filtering ---
        logger.info(f"\n--- Phase 2: {validation_mode.capitalize()} Validation ---")
        phase_start = time.time()

        validated_candidates = set(common_candidates)
        plausibility_scores: Dict[str, float] = {}
        exact_guesses: List[Tuple[str, float]] = []

        if validation_mode == "exact":
            # Exact ERS mode: skip moderate/probabilistic, use ExactSolver
            logger.info("Exact ERS mode: computing best guesses with matrix...")
            exact_start = time.time()
            exact_guesses = self.exact_solver.get_best_guesses(
                common_candidates, top_n=20
            )
            exact_time = time.time() - exact_start
            logger.info(
                f"Exact ERS computed in {exact_time:.3f}s: {len(exact_guesses)} top guesses"
            )
            # Skip to Phase 3 (formatting)
            validated_candidates = common_candidates
        elif players_grids:
            num_players = len(players_grids)
            sorted_grids = sorted(
                players_grids,
                key=lambda g: (
                    sum(p.count(GREEN) for p in g),
                    sum(p.count(YELLOW) for p in g),
                ),
                reverse=True,
            )

            for idx, player_grid in enumerate(sorted_grids, 1):
                if not validated_candidates:
                    break

                before = len(validated_candidates)
                if progress_callback:
                    progress_callback(
                        {
                            "type": "player_start",
                            "player_idx": idx,
                            "num_players": num_players,
                            "num_candidates": before,
                        }
                    )

                logger.info(
                    f"{validation_mode.capitalize()} validation for Player {idx}/{num_players} on {before} candidates..."
                )

                player_start_time = time.time()

                if validation_mode == "moderate":
                    validated_this_round = self._run_moderate_validation(
                        validated_candidates,
                        player_grid,
                        idx,
                        num_players,
                        progress_callback,
                        stop_event,
                        player_start_time,
                        before,
                    )
                    if validated_this_round is None:
                        break
                    validated_candidates = validated_this_round
                else:
                    candidate_scores = self._run_probabilistic_validation(
                        validated_candidates,
                        player_grid,
                        idx,
                        num_players,
                        progress_callback,
                        stop_event,
                        player_start_time,
                        before,
                    )
                    if candidate_scores is None:
                        break
                    # Probabilistic: keep ALL words, score = min plausibility across players
                    for word, score in candidate_scores.items():
                        if word not in plausibility_scores:
                            plausibility_scores[word] = score
                        else:
                            # Take worst (minimum) plausibility across players
                            plausibility_scores[word] = min(
                                plausibility_scores[word], score
                            )
                    # In probabilistic mode, keep all candidates (they're ranked later)
                    pass

        num_players_for_score = len(players_grids) if players_grids else 1
        candidates_info = [
            (word, num_players_for_score) for word in validated_candidates
        ]

        phase2_time = time.time() - phase_start
        stats.log_phase(f"Phase2_{validation_mode.capitalize()}", phase2_time)
        logger.info(
            f"Phase 2 completed in {phase2_time:.3f}s: {len(candidates_info)} validated candidates"
        )

        if not candidates_info:
            logger.error("No candidates after validation")
            return "No valid words found after validation.", set()

        # --- Phase 3: Scoring ---
        if validation_mode == "exact":
            # Exact ERS: skip old scoring, format exact results directly
            total_time = time.time() - solve_start
            return self._format_exact_results(
                exact_guesses, validated_candidates, total_time
            ), validated_candidates

        logger.info("\n--- Phase 3: Scoring ---")
        phase_start = time.time()

        ranked = self.scorer.score_candidates(
            candidates_info, players_grids, plausibility_scores, mode=validation_mode
        )

        phase3_time = time.time() - phase_start
        stats.log_phase("Phase3_Scoring", phase3_time)
        logger.info(f"Phase 3 completed in {phase3_time:.3f}s")

        total_time = time.time() - solve_start
        stats.log_solve(total_time, len(candidates_info))

        logger.info(f"\n✓ Complete solve in {total_time:.3f}s")
        logger.info(f"Top 3: {', '.join(w.upper() for w, *_ in ranked[:3])}")
        logger.info("=" * 60 + "\n")

        return self._format_results(
            ranked, players_grids, total_time, validation_mode
        ), set(w for w, *_ in ranked)

    def _run_moderate_validation(
        self,
        candidates,
        player_grid,
        idx,
        num_players,
        progress_callback,
        stop_event,
        player_start_time,
        before,
    ):
        validated_this_round = set()

        use_mp = len(candidates) > 50
        if use_mp:
            futures = {
                self._executor.submit(
                    _player_possible_sequence_moderate, word, tuple(player_grid)
                ): word
                for word in candidates
            }
            processed_count = 0
            for future in as_completed(futures):
                if stop_event and stop_event.is_set():
                    for f in futures:
                        f.cancel()
                    return None

                processed_count += 1
                word = futures[future]
                try:
                    is_valid = future.result()
                except CancelledError:
                    continue

                if progress_callback:
                    progress_callback(
                        {
                            "type": "word_validated",
                            "word": word,
                            "is_valid": is_valid,
                        }
                    )
                if is_valid:
                    validated_this_round.add(word)

                self._report_progress(
                    progress_callback,
                    idx,
                    num_players,
                    processed_count,
                    before,
                    player_start_time,
                )
        else:
            for word in candidates:
                if stop_event and stop_event.is_set():
                    return None
                is_valid = _player_possible_sequence_moderate(word, tuple(player_grid))
                if progress_callback:
                    progress_callback(
                        {"type": "word_validated", "word": word, "is_valid": is_valid}
                    )
                if is_valid:
                    validated_this_round.add(word)

        return validated_this_round

    def _run_probabilistic_validation(
        self,
        candidates,
        player_grid,
        idx,
        num_players,
        progress_callback,
        stop_event,
        player_start_time,
        before,
    ):
        candidate_scores: Dict[str, float] = {}

        # --- PRE-FILTRAGE MODERATE (évite le beam search sur les impossibles) ---
        moderate_valid = set()
        for word in candidates:
            if stop_event and stop_event.is_set():
                return None
            if _player_possible_sequence_moderate(word, tuple(player_grid)):
                moderate_valid.add(word)
        candidates = moderate_valid

        if not candidates:
            return {}

        processed_count = 0
        use_mp = len(candidates) > 10

        if use_mp:
            import multiprocessing as mp
            from concurrent.futures import (
                ProcessPoolExecutor,
                TimeoutError as FutureTimeoutError,
            )

            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=min(2, (os.cpu_count() or 2)),
                initializer=init_worker,
                initargs=(self.cache, None),
                mp_context=ctx,
            ) as executor:
                futures = {
                    executor.submit(
                        _player_plausibility_score, word, tuple(player_grid)
                    ): word
                    for word in candidates
                }
                for future in as_completed(futures):
                    if stop_event and stop_event.is_set():
                        for f in futures:
                            f.cancel()
                        return None

                    processed_count += 1
                    word = futures[future]
                    try:
                        score = future.result(timeout=2.0)
                    except (FutureTimeoutError, CancelledError):
                        score = 0.0
                    except Exception:
                        score = 0.0

                    candidate_scores[word] = score
                    is_valid = score > 0.01
                    if progress_callback:
                        progress_callback(
                            {
                                "type": "word_validated",
                                "word": word,
                                "is_valid": is_valid,
                            }
                        )

                    self._report_progress(
                        progress_callback,
                        idx,
                        num_players,
                        processed_count,
                        before,
                        player_start_time,
                    )
        else:
            for word in candidates:
                if stop_event and stop_event.is_set():
                    return None

                processed_count += 1
                try:
                    score = _player_plausibility_score(word, tuple(player_grid))
                except Exception:
                    score = 0.0

                candidate_scores[word] = score
                is_valid = score > 0.01
                if progress_callback:
                    progress_callback(
                        {"type": "word_validated", "word": word, "is_valid": is_valid}
                    )

                self._report_progress(
                    progress_callback,
                    idx,
                    num_players,
                    processed_count,
                    before,
                    player_start_time,
                )

        return candidate_scores

    @staticmethod
    def _report_progress(
        progress_callback, player_idx, num_players, processed_count, total, start_time
    ):
        if (
            progress_callback
            and processed_count % 5 == 0
            and processed_count < total
            and total > 20
        ):
            elapsed = time.time() - start_time
            if elapsed > 0.1:
                words_per_sec = processed_count / elapsed
                remaining = total - processed_count
                eta = remaining / words_per_sec
                progress_callback(
                    {
                        "type": "player_progress",
                        "player_idx": player_idx,
                        "current": processed_count,
                        "total": total,
                        "pct": (processed_count / total) * 100,
                        "eta_m": int(eta // 60),
                        "eta_s": int(eta % 60),
                    }
                )

    def _format_results(
        self,
        ranked: List[Tuple],
        players_grids: List[List[str]],
        solve_time: float,
        validation_mode: str = "moderate",
    ) -> str:
        """Format results readably"""
        mode_label = validation_mode.upper()
        lines = [
            "\n" + "=" * 90,
            f"ANALYSIS RESULTS ({mode_label} MODE)",
            "=" * 90,
            f"Solve time: {solve_time:.3f}s | Candidates analyzed: {len(ranked)}",
            "",
        ]
        if validation_mode == "probabilistic" and ranked and len(ranked[0]) > 8:
            lines.append(
                f"{'Rank':<6}{'Word':<10}{'Score':<9}{'Plaus%':<9}{'Perfect':<9}{'Freq':<8}{'Entropy':<9}{'Tries'}"
            )
        else:
            lines.append(
                f"{'Rank':<6}{'Word':<10}{'Score':<9}{'Valid%':<10}{'Perfect':<9}{'Freq':<8}{'Entropy':<9}{'Tries'}"
            )
        lines.append("-" * 90)

        for i, row in enumerate(ranked[:20], 1):
            if len(row) > 8:
                w, sc, ra, pe, ti, av, freq, entropy, plaus = row
                lines.append(
                    f"{i:<6}{w.upper():<10}{sc:.4f}   {plaus * 100:5.1f}%    "
                    f"{'Y' if pe else 'N':<9}{freq:.3f}   {entropy:.3f}    {av:.1f}"
                )
            elif len(row) >= 8:
                w, sc, ra, pe, ti, av, freq, entropy = row[:8]
                lines.append(
                    f"{i:<6}{w.upper():<10}{sc:.4f}   {ra * 100:5.1f}%    "
                    f"{'Y' if pe else 'N':<9}{freq:.3f}   {entropy:.3f}    {av:.1f}"
                )
            else:
                # Fallback for short rows
                w = row[0]
                sc = row[1] if len(row) > 1 else 0.0
                lines.append(f"{i:<6}{w.upper():<10}{sc:.4f}")

        if ranked and len(ranked[0]) >= 2:
            row = ranked[0]
            w = row[0]
            sc = row[1]
            lines.extend(
                [
                    "",
                    "=" * 90,
                    f"MOST PROBABLE WORD: {w.upper()}",
                    f"   Global score         : {sc:.4f}",
                    "=" * 90,
                ]
            )

        return "\n".join(lines)

    def _format_lax_results(self, candidates: set, solve_time: float) -> str:
        """Format lax results readably"""
        lines = [
            "\n" + "=" * 85,
            "LAX FILTERING RESULTS",
            "=" * 85,
            f"Solve time: {solve_time:.3f}s | Lax candidates found: {len(candidates)}",
            "",
            "Some candidates:",
            ", ".join(sorted(list(candidates))[:50]),
        ]
        if len(candidates) > 50:
            lines[-1] += "..."
        return "\n".join(lines)

    def _format_exact_results(
        self,
        exact_guesses: List[Tuple[str, float]],
        candidates: Set[str],
        solve_time: float,
    ) -> str:
        """Format exact ERS results readably"""
        lines = [
            "\n" + "=" * 90,
            "EXACT ERS ANALYSIS",
            "=" * 90,
            f"Solve time: {solve_time:.3f}s | Candidates remaining: {len(candidates)}",
            "",
            f"{'Rank':<6}{'Guess':<10}{'ERS':<12}{'Type'}",
            "-" * 90,
        ]
        for i, (word, ers) in enumerate(exact_guesses, 1):
            is_candidate = "*CANDIDATE" if word in candidates else "guess"
            lines.append(f"{i:<6}{word.upper():<10}{ers:.2f}      {is_candidate}")
        lines.extend(
            [
                "",
                "=" * 90,
                f"BEST GUESS: {exact_guesses[0][0].upper()}",
                f"   Expected Remaining Size: {exact_guesses[0][1]:.2f}",
                "   Lower ERS = faster solve",
                "=" * 90,
            ]
        )
        return "\n".join(lines)

    def suggest_opening_words(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """Suggest best opening words using exact ERS (Expected Remaining Size)"""
        if self._opening_words is not None:
            return self._opening_words[:top_n]

        logger.info(f"Precomputing {top_n} best opening words using exact ERS...")
        start = time.time()

        candidates = set(self.answers)
        best = self.exact_solver.get_best_guesses(candidates, top_n)

        elapsed = time.time() - start
        logger.info(
            f"✓ Opening words precomputed in {elapsed:.2f}s (top: {best[0][0].upper()} ERS={best[0][1]:.2f})"
        )

        self._opening_words = best
        return best

    def get_statistics(self) -> str:
        """Return performance statistics"""
        return stats.get_summary()

    def create_personal_solver(self) -> PersonalSolver:
        """Create a personal solver instance"""
        return PersonalSolver(self)


# =========================
# 11. Interactive Mode
# =========================


class InteractiveSolver:
    """Interactive interface for the solver"""

    def __init__(self):
        self.solver = WordleSolver()
        self.personal_solver = None
        logger.info("Interactive mode activated")

    def parse_grid_input(self, grid_str: str) -> List[str]:
        """Parse grid entered by user"""
        patterns = []
        for line in grid_str.strip().split("\n"):
            line = line.strip()
            if len(line) == WORD_LENGTH and all(
                c in (GREEN, YELLOW, BLACK) for c in line
            ):
                patterns.append(line)
        return patterns

    def run(self):
        """Launch interactive mode"""
        print("\n" + "=" * 70)
        print("🎮 WORDLE SOLVER INTERACTIVE v2.0 - ENGLISH VERSION")
        print("=" * 70)
        print("\nAvailable commands:")
        print("  • 'community'  : Analyze community grids")
        print("  • 'personal'   : Personal solver with step-by-step guidance")
        print("  • 'opening'    : Suggest opening words")
        print("  • 'stats'      : Display statistics")
        print("  • 'quit'       : Exit")
        print("=" * 70 + "\n")

        while True:
            command = input("Command > ").strip().lower()

            if command == "quit":
                print("\n👋 Goodbye!")
                break

            elif command == "community":
                self._run_community_solve()

            elif command == "personal":
                self._run_personal_solver()

            elif command == "opening":
                self._suggest_opening()

            elif command == "stats":
                print(self.solver.get_statistics())

            else:
                print(
                    "❌ Unknown command. Use: community, personal, opening, stats, quit"
                )

    def _run_community_solve(self):
        """Community resolution"""
        print("\n--- Community Grid Entry ---")
        print("Enter player grids (one per line, empty to finish)")
        print(f"Format: {GREEN * 5} or {YELLOW * 3}{BLACK * 2}, etc.")

        players_grids = []
        player_num = 1

        while True:
            print(f"\nPlayer {player_num} (empty to finish):")
            grid_input = []

            while True:
                line = input("  Pattern > ").strip()
                if not line:
                    break
                if len(line) == WORD_LENGTH and all(
                    c in (GREEN, YELLOW, BLACK) for c in line
                ):
                    grid_input.append(line)
                else:
                    print("  ❌ Invalid format")

            if not grid_input:
                break

            players_grids.append(grid_input)
            player_num += 1

        if not players_grids:
            print("❌ No grids entered")
            return

        # Resolution
        result = self.solver.solve(players_grids, [])
        print(result)

    def _run_personal_solver(self):
        """Interactive personal solver with step-by-step guidance"""
        print("\n" + "=" * 70)
        print("🎯 PERSONAL SOLVER - STEP BY STEP")
        print("=" * 70)

        # Step 1: Analyze community grids (optional)
        print("\n📊 STEP 1: Community Analysis (optional)")
        print("Do you want to analyze other players' grids first? (y/n)")

        community_candidates = None
        if input("> ").strip().lower() == "y":
            print("\nEnter player grids:")
            players_grids = []
            player_num = 1

            while True:
                print(f"\nPlayer {player_num} (empty to finish):")
                grid_input = []

                while True:
                    line = input("  Pattern > ").strip()
                    if not line:
                        break
                    if len(line) == WORD_LENGTH and all(
                        c in (GREEN, YELLOW, BLACK) for c in line
                    ):
                        grid_input.append(line)
                    else:
                        print("  ❌ Invalid format")

                if not grid_input:
                    break

                players_grids.append(grid_input)
                player_num += 1

            if players_grids:
                # Quick analysis to narrow down candidates
                print("\n⏳ Analyzing community grids...")
                community_candidates = set(self.solver.answers)

                for player_grids in players_grids:
                    community_candidates = {
                        secret
                        for secret in community_candidates
                        if _player_possible_sequence_lax(secret, tuple(player_grids))
                    }

                print(
                    f"✓ Community analysis: {len(community_candidates)} possible words"
                )

        # Step 2: Best opening word
        print("\n" + "=" * 70)
        print("🚀 STEP 2: Best Opening Word")
        print("=" * 70)

        # If community analysis done, calculate best word from those candidates
        if community_candidates:
            print(
                f"\nCalculating best opening word from {len(community_candidates)} candidates..."
            )
            if len(community_candidates) <= 10:
                print(f"\n💡 Possible words: {', '.join(sorted(community_candidates))}")
                print("You can try any of these!")
                best_opening = [(w, 1.0) for w in sorted(community_candidates)[:5]]
            else:
                # Calculate entropy for remaining candidates
                scores = []
                for word in list(self.solver.guesses)[:300]:
                    entropy = self.solver.scorer.calculate_entropy(
                        word, community_candidates
                    )
                    letter_variety = len(set(word))
                    freq = self.solver.scorer.calculate_letter_frequency_score(word)
                    score = entropy * 0.60 + letter_variety * 0.25 + freq * 0.15
                    scores.append((word, score))
                best_opening = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
        else:
            print("\nCalculating best opening words from all possibilities...")
            best_opening = self.solver.suggest_opening_words(5)

        print(f"\n{'Rank':<6}{'Word':<12}{'Score':<10}")
        print("-" * 28)
        for i, (word, score) in enumerate(best_opening, 1):
            print(f"{i:<6}{word.upper():<12}{score:.4f}")

        print("\n💡 Recommended: " + best_opening[0][0].upper())

        # Step 3: Personal attempts
        print("\n" + "=" * 70)
        print("📝 STEP 3: Your Attempts")
        print("=" * 70)
        print("\nEnter your guesses and their patterns.")
        print("Pattern format: G=Green, Y=Yellow, B=Black")
        print("Example: GYBBB or 🟩🟨⬛⬛⬛")
        print("Type 'done' when finished.\n")

        # Initialize personal solver
        if community_candidates:
            self.personal_solver = PersonalSolver(self.solver)
            self.personal_solver.remaining_candidates = community_candidates
        else:
            self.personal_solver = PersonalSolver(self.solver)

        attempt_num = 1

        while attempt_num <= 6:
            print(f"\n--- Attempt {attempt_num}/6 ---")

            # Get guess
            guess = input(f"Your guess > ").strip().lower()

            if guess == "done":
                break

            if len(guess) != WORD_LENGTH or not guess.isalpha():
                print("❌ Invalid word (must be 5 letters)")
                continue

            # Get pattern
            pattern_input = input(f"Pattern for {guess.upper()} > ").strip()

            if pattern_input.lower() == "done":
                break

            # Parse pattern
            pattern = self.personal_solver.parse_pattern_input(pattern_input)

            if len(pattern) != WORD_LENGTH or not all(
                c in (GREEN, YELLOW, BLACK) for c in pattern
            ):
                print("❌ Invalid pattern")
                continue

            # Check if solved
            if pattern == GREEN * WORD_LENGTH:
                print(f"\n🎉 CONGRATULATIONS! You found it: {guess.upper()}")
                break

            # Add attempt and filter
            self.personal_solver.add_attempt(guess, pattern)

            remaining = len(self.personal_solver.remaining_candidates)
            print(f"\n📊 Remaining candidates: {remaining}")

            if remaining == 0:
                print("❌ No possible words remaining. Check your patterns!")
                break

            # Show possible words
            if remaining <= 20:
                words = self.personal_solver.get_remaining_words(20)
                print(f"\n💡 Possible words: {', '.join(w.upper() for w in words)}")
            else:
                words = self.personal_solver.get_remaining_words(10)
                print(
                    f"\n💡 Some possible words: {', '.join(w.upper() for w in words)}..."
                )

            # Suggest next guess
            if remaining > 1 and attempt_num < 6:
                print("\n🎯 Suggested next guesses:")
                suggestions = self.personal_solver.get_best_next_guess(5)
                for i, (word, score) in enumerate(suggestions[:5], 1):
                    in_candidates = (
                        "✓"
                        if word in self.personal_solver.remaining_candidates
                        else " "
                    )
                    print(
                        f"  {i}. {word.upper():<10} {in_candidates}  (score: {score:.3f})"
                    )

            attempt_num += 1

        print("\n" + "=" * 70)
        print("Session complete!")
        print("=" * 70)

    def _suggest_opening(self):
        """Opening word suggestion"""
        print("\n--- Best Opening Words ---")
        suggestions = self.solver.suggest_opening_words(15)

        print(f"\n{'Rank':<6}{'Word':<12}{'Score':<10}")
        print("-" * 28)

        for i, (word, score) in enumerate(suggestions, 1):
            print(f"{i:<6}{word.upper():<12}{score:.4f}")

        print("\n💡 These words maximize information on the first guess")


# =========================
# 12. Main Entry Point
# =========================


def main():
    """Main function with argument handling"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Wordle Solver v3.0 - ENGLISH VERSION",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python solver.py --interactive          # Interactive mode
  python solver.py --opening              # Suggest opening words
  python solver.py --personal             # Personal solver
  python solver.py --stats                # Show statistics
  
  New in v3.0: 3 validation modes
    - lax: patterns must be possible for the secret word
    - moderate: greens must be respected between guesses (realistic)
    - probabilistic: score plausibility instead of eliminating
        """,
    )

    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Launch in interactive mode"
    )
    parser.add_argument(
        "--personal",
        "-p",
        action="store_true",
        help="Personal solver with step-by-step guidance",
    )
    parser.add_argument(
        "--opening", "-o", action="store_true", help="Show best opening words"
    )
    parser.add_argument("--stats", "-s", action="store_true", help="Display statistics")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # Adjust log level
    logger.logger.setLevel(getattr(logging, args.log_level))

    # Interactive mode
    if args.interactive:
        interactive = InteractiveSolver()
        interactive.run()
        return

    # Initialize solver
    solver = WordleSolver()

    # Personal solver mode
    if args.personal:
        interactive = InteractiveSolver()
        interactive._run_personal_solver()
        return

    # Opening words
    if args.opening:
        print("\n🎯 BEST OPENING WORDS")
        print("=" * 50)
        suggestions = solver.suggest_opening_words(15)
        print(f"\n{'Rank':<6}{'Word':<12}{'Score':<10}")
        print("-" * 28)
        for i, (word, score) in enumerate(suggestions, 1):
            print(f"{i:<6}{word.upper():<12}{score:.4f}")
        print("\n💡 These words maximize information on the first guess")
        return

    # Statistics
    if args.stats:
        print(solver.get_statistics())
        return

    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  User interruption")
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
    finally:
        # Save stats on error
        if stats.total_solves > 0:
            print(f"\n📊 Statistics saved in {config.log_dir}/")
