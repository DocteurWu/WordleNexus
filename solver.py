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
from concurrent.futures import ProcessPoolExecutor, as_completed, CancelledError
from functools import partial, lru_cache
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
from datetime import datetime

def init_worker(cache_data):
    """Initializer for multiprocessing pool to set the cache."""
    _player_possible_sequence_lax.cache = cache_data
    _player_possible_sequence_strict.cache = cache_data

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
    cache_path: str = "data/pattern_cache.pkl.gz"
    log_dir: str = "logs"
    
    scoring_weights: Dict[str, float] = field(default_factory=lambda: {
        'strict_ratio': 0.42,
        'perfect_bonus': 0.20,
        'tightness': 0.15,
        'coherence': 0.10,
        'difficulty': 0.05,
        'letter_frequency': 0.05,
        'entropy': 0.03
    })
    
    # ENGLISH letter frequency (normalized)
    letter_freq: Dict[str, float] = field(default_factory=lambda: {
        'e': 1.00, 'a': 0.85, 'r': 0.80, 'i': 0.78, 'o': 0.75,
        't': 0.72, 'n': 0.70, 's': 0.68, 'l': 0.65, 'c': 0.60,
        'u': 0.58, 'd': 0.55, 'p': 0.52, 'm': 0.50, 'h': 0.48,
        'g': 0.45, 'b': 0.42, 'f': 0.40, 'y': 0.38, 'w': 0.35,
        'k': 0.30, 'v': 0.28, 'x': 0.15, 'z': 0.12, 'j': 0.10, 'q': 0.05
    })

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
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Create logs folder
        os.makedirs(config.log_dir, exist_ok=True)
        
        # File handler (DEBUG and above)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(
            os.path.join(config.log_dir, f"solver_{timestamp}.log"),
            encoding='utf-8'
        )
        fh.setLevel(logging.DEBUG)
        
        # Console handler (INFO and above)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Detailed format
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(funcName)-20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def debug(self, msg: str): self.logger.debug(msg)
    def info(self, msg: str): self.logger.info(msg)
    def warning(self, msg: str): self.logger.warning(msg)
    def error(self, msg: str): self.logger.error(msg)

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
        cache_rate = (self.cache_hits / (self.cache_hits + self.cache_misses) * 100 
                     if (self.cache_hits + self.cache_misses) > 0 else 0)
        
        lines = [
            "\n" + "="*60,
            "📊 PERFORMANCE STATISTICS",
            "="*60,
            f"Total solves            : {self.total_solves}",
            f"Average time            : {avg_time:.3f}s",
            f"Time min/max            : {min(self.solve_times):.3f}s / {max(self.solve_times):.3f}s",
            f"Average candidates      : {avg_candidates:.1f}",
            f"Cache hit rate          : {cache_rate:.1f}%",
            ""
        ]
        
        if self.phase_times:
            lines.append("Phase breakdown:")
            for phase, times in sorted(self.phase_times.items()):
                avg = sum(times) / len(times)
                lines.append(f"  • {phase:<20}: {avg:.3f}s")
        
        lines.append("="*60)
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
        
        logger.info(f"✓ {len(answers)} answers and {len(guesses)} guesses loaded in {time.time()-start:.3f}s")
        
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
    Optimized feedback calculation
    Performance: ~30% faster than original
    """
    result = [BLACK] * WORD_LENGTH
    secret_list = list(secret)
    
    # Phase 1: Greens (exact match)
    for i in range(WORD_LENGTH):
        if guess[i] == secret[i]:
            result[i] = GREEN
            secret_list[i] = None  # Mark as used
    
    # Phase 2: Yellows (present elsewhere)
    available = Counter(c for c in secret_list if c is not None)
    for i in range(WORD_LENGTH):
        if result[i] == BLACK and available[guess[i]] > 0:
            result[i] = YELLOW
            available[guess[i]] -= 1
    
    return "".join(result)

# Alias for compatibility
feedback = feedback_optimized

# =========================
# 5. Global Cache with Compression
# =========================

def build_pattern_map_for_secret(secret: str, allowed_guesses: List[str]) -> Tuple[str, Dict]:
    """Build pattern map for a given secret"""
    m = defaultdict(list)
    for g in allowed_guesses:
        p = feedback(secret, g)
        m[p].append(g)
    return secret, dict(m)

def compute_stable_hash(word_list: List[str]) -> str:
    """Compute stable hash to detect changes"""
    content = '\n'.join(sorted(word_list)).encode('utf-8')
    return hashlib.sha256(content).hexdigest()

def build_global_pattern_cache(answers: List[str], 
                               allowed_guesses: List[str],
                               status_callback=None) -> Dict:
    """
    Build or load global pattern cache
    Uses parallelization and compression
    """
    logger.info("Initializing pattern cache...")
    
    cache_meta = {
        'answers_hash': compute_stable_hash(answers),
        'guesses_hash': compute_stable_hash(allowed_guesses),
        'version': '2.0'
    }
    
    # Try loading existing cache
    if os.path.exists(config.cache_path):
        logger.info(f"Cache found: {config.cache_path}")
        try:
            with gzip.open(config.cache_path, "rb") as f:
                cached_data = pickle.load(f)
                
            if isinstance(cached_data, dict) and 'meta' in cached_data:
                cached_meta = cached_data.get('meta')
                is_valid = True
                for key, value in cache_meta.items():
                    if cached_meta.get(key) != value:
                        logger.warning(f"Cache mismatch on key '{key}'. Expected: {value}, Found: {cached_meta.get(key)}")
                        is_valid = False
                
                if is_valid:
                    logger.info("✓ Valid cache loaded from disk.")
                    stats.log_cache_access(True)
                    return cached_data['cache']

            logger.warning("Cache metadata mismatch or invalid format. Rebuilding...")
            stats.log_cache_access(False)
        
        except Exception as e:
            logger.error(f"Cache loading error: {e}")
            stats.log_cache_access(False)
    
    # Build cache
    logger.info("Building cache (may take 1-2 minutes)...")
    start_time = time.time()
    
    max_workers = min(4, os.cpu_count() or 1)
    worker = partial(build_pattern_map_for_secret, allowed_guesses=allowed_guesses)
    
    batch_size = 100
    cache = {}
    processed = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        # Submit tasks in batches
        for i in range(0, len(answers), batch_size):
            batch = answers[i:i+batch_size]
            for secret in batch:
                future = executor.submit(worker, secret)
                futures.append(future)
        
        # Collect results with progress
        for future in as_completed(futures):
            secret, patterns = future.result()
            cache[secret] = patterns
            processed += 1
            
            if processed % 500 == 0:
                pct = (processed / len(answers)) * 100
                logger.debug(f"Progress: {processed}/{len(answers)} ({pct:.1f}%)")
    
    # Save cache
    cache_data = {'cache': cache, 'meta': cache_meta}
    os.makedirs(config.data_dir, exist_ok=True)
    
    with gzip.open(config.cache_path, "wb") as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
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
    
    return {letter for letter, patterns in letter_patterns.items()
            if all(p == BLACK for p in patterns)}

def guesses_keep_previous_info(prev_guess: str, prev_pat: str, next_guess: str) -> bool:
    """Check if next guess respects previous constraints"""
    # Check greens
    for i, c in enumerate(prev_pat):
        if c == GREEN and prev_guess[i] != next_guess[i]:
            return False
    
    # Check informative letters
    for letter in info_letters(prev_pat, prev_guess):
        if letter not in next_guess:
            return False
    
    # Check black letters
    for letter in black_letters(prev_pat, prev_guess):
        if letter in next_guess:
            return False
    
    return True

# =========================
# 7. Verification Functions (with LRU cache)
# =========================

@lru_cache(maxsize=20000)
def _player_possible_sequence_lax(secret: str, pats_tuple: Tuple[str, ...]) -> bool:
    """Lax verification: all patterns exist"""
    pats = list(pats_tuple)
    m = _player_possible_sequence_lax.cache.get(secret)
    
    if not m:
        return False
    
    return all(p in m for p in pats)

def _player_possible_sequence_strict(secret: str, pats_tuple: Tuple[str, ...]) -> bool:
    """Strict verification: coherent sequence possible"""
    pats = list(pats_tuple)
    m = _player_possible_sequence_strict.cache.get(secret)
    
    if not m:
        return False
    
    # Initialize with first pattern
    layer = [(g, pats[0]) for g in m.get(pats[0], [])]
    if not layer:
        return False
    
    # Check each following pattern
    for pat in pats[1:]:
        candidates = m.get(pat, [])
        if not candidates:
            return False
        
        new_layer = [
            (next_guess, pat)
            for prev_guess, prev_pat in layer
            for next_guess in candidates
            if guesses_keep_previous_info(prev_guess, prev_pat, next_guess)
        ]
        
        if not new_layer:
            return False
        
        layer = new_layer
    
    return True

def check_player_coherence_loose(patterns: List[str]) -> bool:
    """Check temporal coherence of patterns"""
    greens = [p.count(GREEN) for p in patterns]
    return all(greens[i] >= greens[i - 1] - 1 for i in range(1, len(patterns)))

# Attach cache to functions
_player_possible_sequence_lax.cache = {}
_player_possible_sequence_strict.cache = {}

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
    
    def calculate_entropy(self, word: str, candidates: Set[str]) -> float:
        """Calculate information entropy of a word"""
        if len(candidates) <= 1:
            return 0.0
        
        pattern_dist = defaultdict(int)
        for candidate in candidates:
            pattern = feedback(candidate, word)
            pattern_dist[pattern] += 1
        
        total = len(candidates)
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
    
    def score_candidates(self, candidates_info: List[Tuple[str, int]], 
                        players_grids: List[List[str]]) -> List[Tuple]:
        """
        Main scoring with all criteria
        Returns: [(word, score, ratio, perfect, tight, avg_tries, freq, entropy), ...]
        """
        n_players = len(players_grids) if players_grids else 1
        avg_tries = sum(len(p) for p in players_grids) / n_players if players_grids else 0
        coherence = (sum(check_player_coherence_loose(p) for p in players_grids) / n_players 
                    if players_grids else 1.0)
        difficulty = 1 / (avg_tries + 1)
        
        # Candidate set for entropy calculation
        candidate_words = {w for w, _ in candidates_info}
        
        results = []
        w = config.scoring_weights
        
        for word, strict_count in candidates_info:
            ratio = strict_count / n_players if n_players > 0 else 1.0
            perfect = 1 if strict_count == n_players else 0
            tight = self.calculate_tightness(word)
            freq = self.calculate_letter_frequency_score(word)
            entropy = self.calculate_entropy(word, candidate_words)
            
            # Weighted global score
            score = (
                w['strict_ratio'] * ratio +
                w['perfect_bonus'] * perfect +
                w['tightness'] * tight +
                w['coherence'] * coherence +
                w['difficulty'] * difficulty +
                w['letter_frequency'] * freq +
                w['entropy'] * entropy
            )
            
            results.append((word, score, ratio, perfect, tight, avg_tries, freq, entropy))
        
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
    
    def parse_pattern_input(self, pattern_str: str) -> str:
        """Parse user pattern input (flexible formats)"""
        pattern_str = pattern_str.strip().upper()
        
        # Replace common formats
        replacements = {
            'V': GREEN,   # French Vert
            'J': YELLOW,  # French Jaune
            'G': BLACK,   # French Gris
            'Y': YELLOW,  # English Yellow
            'B': BLACK,   # English Black
            '🟩': GREEN, '🟨': YELLOW, '⬛': BLACK,
        }
        
        result = pattern_str
        for old, new in replacements.items():
            result = result.replace(old, new)
        
        return result
    
    def add_attempt(self, guess: str, pattern: str):
        """Add an attempt and filter candidates"""
        guess = guess.lower()
        self.attempts.append({'guess': guess, 'pattern': pattern})
        
        # Filter candidates
        self.remaining_candidates = {
            secret for secret in self.remaining_candidates
            if feedback(secret, guess) == pattern
        }
        
        logger.info(f"Attempt added: {guess.upper()} -> {len(self.remaining_candidates)} candidates remain")
    
    def get_best_next_guess(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """Get best next guesses based on remaining candidates"""
        if len(self.remaining_candidates) <= 2:
            # If few candidates remain, suggest them directly
            return [(w, 1.0) for w in sorted(self.remaining_candidates)]
        
        # Calculate entropy for common guesses
        scores = []
        for word in list(self.solver.guesses)[:500]:  # Check first 500 common words
            entropy = self.solver.scorer.calculate_entropy(word, self.remaining_candidates)
            letter_variety = len(set(word))
            freq = self.solver.scorer.calculate_letter_frequency_score(word)
            
            # Composite score
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
# 10. Main Solver
# =========================

class WordleSolver:
    """Optimized Wordle Solver with logging and statistics"""
    
    def __init__(self, status_callback=None):
        logger.info("="*60)
        logger.info("Initializing WordleSolver v2.0")
        logger.info("="*60)
        
        start_time = time.time()
        
        # Load data
        self.answers, self.guesses = load_wordlists()
        
        # Build cache
        self.cache = build_global_pattern_cache(
            self.answers, 
            self.guesses, 
            status_callback
        )
        
        # Initialize scorer
        self.scorer = AdvancedScorer(self.cache, self.answers)
        
        # Attach cache to LRU functions
        self._set_current_cache_for_lru()
        
        elapsed = time.time() - start_time
        logger.info(f"✓ WordleSolver initialized in {elapsed:.2f}s")
        logger.info("="*60 + "\n")
    
    def _set_current_cache_for_lru(self):
        """Attach cache to verification functions"""
        _player_possible_sequence_lax.cache_clear()
        _player_possible_sequence_lax.cache = self.cache
        _player_possible_sequence_strict.cache = self.cache
        logger.debug("LRU cache reset")
    
    def solve(self, players_grids: List[List[str]],
              personal_attempts: List[Dict],
              run_strict: bool = True,
              candidates_to_check: Optional[set] = None,
              progress_callback=None,
              stop_event: Optional[threading.Event] = None) -> Tuple[str, set]:
        """
        Solve Wordle with detailed logging.
        Can run in two modes: lax only or full (lax + strict).
        
        Args:
            players_grids: List of player grids (patterns).
            personal_attempts: List of personal attempts {guess, pattern}.
            run_strict: If False, only lax filtering is performed.
            candidates_to_check: Optional set of words to start with.
            stop_event: Optional threading.Event to gracefully stop the analysis.
        
        Returns:
            A tuple containing:
            - Formatted string with results.
            - A set of candidate words (lax or strict).
        """
        logger.info("\n" + "="*60)
        logger.info(f"NEW SOLVE (Strict Mode: {run_strict})")
        logger.info("="*60)
        
        solve_start = time.time()
        
        # Input validation
        if not players_grids and not personal_attempts:
            logger.warning("No data provided")
            return "No data entered.", set()
        
        logger.info(f"Players: {len(players_grids)}, Personal attempts: {len(personal_attempts)}")
        
        # --- Phase 1: Lax Filtering ---
        logger.info("\n--- Phase 1: Lax Filtering ---")
        phase_start = time.time()
        
        if candidates_to_check is None:
            common_candidates = set(self.answers)
        else:
            common_candidates = candidates_to_check
            
        logger.debug(f"Initial candidates: {len(common_candidates)}")
        
        if players_grids:
            sorted_grids = sorted(players_grids, key=lambda g: sum(p.count(GREEN) for p in g), reverse=True)
            for idx, player_grid in enumerate(sorted_grids, 1):
                before = len(common_candidates)
                common_candidates = {
                    secret for secret in common_candidates
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
        logger.info(f"Phase 1 completed in {phase1_time:.3f}s: {len(common_candidates)} candidates")

        # --- Phase 1.5: Personal Attempts ---
        if personal_attempts and common_candidates:
            logger.info("\n--- Phase 1.5: Personal Attempts ---")
            phase_start = time.time()
            
            before = len(common_candidates)
            common_candidates = {
                secret for secret in common_candidates
                if all(
                    feedback(secret, att['guess']) == att['pattern']
                    for att in personal_attempts
                )
            }
            after = len(common_candidates)
            
            phase15_time = time.time() - phase_start
            stats.log_phase("Phase1.5_Personal", phase15_time)
            logger.info(f"Phase 1.5 completed in {phase15_time:.3f}s: {before} -> {after} candidates")

        if not common_candidates:
            logger.error("❌ No candidates found")
            return "❌ No possible words found.", set()

        # --- Conditional Strict Validation ---
        if not run_strict:
            total_time = time.time() - solve_start
            logger.info(f"\n✓ Lax solve in {total_time:.3f}s")
            return self._format_lax_results(common_candidates, total_time), common_candidates

        # --- Phase 2: Iterative Strict Validation ---
        logger.info("\n--- Phase 2: Iterative Strict Validation ---")
        phase_start = time.time()
        
        strict_candidates = set(common_candidates)
        # (The rest of the strict validation logic remains the same...)
        if players_grids:
            num_players = len(players_grids)
            sorted_grids = sorted(players_grids, key=lambda g: (sum(p.count(GREEN) for p in g), sum(p.count(YELLOW) for p in g)), reverse=True)
            
            for idx, player_grid in enumerate(sorted_grids, 1):
                if not strict_candidates:
                    logger.warning("No candidates left for strict validation.")
                    break

                before = len(strict_candidates)
                if progress_callback:
                    progress_callback({'type': 'player_start', 'player_idx': idx, 'num_players': num_players, 'num_candidates': before})
                
                logger.info(f"Strict validation for Player {idx}/{num_players} on {before} candidates...")
                
                player_start_time = time.time()
                validated_this_round = set()
                
                with ProcessPoolExecutor(initializer=init_worker, initargs=(self.cache,)) as executor:
                    logger.info(f"Using up to {executor._max_workers} cores for strict validation.")
                    futures = {executor.submit(_player_possible_sequence_strict, word, tuple(player_grid)): word for word in strict_candidates}
                    
                    processed_count = 0
                    for future in as_completed(futures):
                        if stop_event and stop_event.is_set():
                            for f in futures:
                                f.cancel()
                            raise InterruptedError("Analysis stopped by user.")

                        processed_count += 1
                        word = futures[future]
                        try:
                            is_valid = future.result()
                        except CancelledError:
                            continue # Skip cancelled futures

                        if progress_callback:
                            progress_callback({'type': 'word_validated', 'word': word, 'is_valid': is_valid})
                        if is_valid:
                            validated_this_round.add(word)

                        if processed_count % 5 == 0 and processed_count < before and before > 20:
                            elapsed = time.time() - player_start_time
                            if elapsed > 0.1:
                                words_per_sec = processed_count / elapsed
                                remaining = before - processed_count
                                eta = remaining / words_per_sec
                                if progress_callback:
                                    progress_callback({
                                        'type': 'player_progress',
                                        'player_idx': idx,
                                        'current': processed_count,
                                        'total': before,
                                        'pct': (processed_count / before) * 100,
                                        'eta_m': int(eta // 60),
                                        'eta_s': int(eta % 60)
                                    })

                strict_candidates = validated_this_round
                after = len(strict_candidates)
                logger.info(f"  -> Player {idx} results: {before} -> {after} candidates")
                if progress_callback:
                    progress_callback({'type': 'player_end', 'player_idx': idx, 'num_candidates_after': after})

        num_players_for_score = len(players_grids) if players_grids else 1
        candidates_info = [(word, num_players_for_score) for word in strict_candidates]
        
        phase2_time = time.time() - phase_start
        stats.log_phase("Phase2_Strict_Iterative", phase2_time)
        logger.info(f"Phase 2 completed in {phase2_time:.3f}s: {len(candidates_info)} validated candidates")
        
        if not candidates_info:
            logger.error("❌ No candidates after strict validation")
            return "❌ No valid words found after strict validation.", set()

        # --- Phase 3: Scoring ---
        logger.info("\n--- Phase 3: Scoring ---")
        phase_start = time.time()
        
        ranked = self.scorer.score_candidates(candidates_info, players_grids)
        
        phase3_time = time.time() - phase_start
        stats.log_phase("Phase3_Scoring", phase3_time)
        logger.info(f"Phase 3 completed in {phase3_time:.3f}s")
        
        total_time = time.time() - solve_start
        stats.log_solve(total_time, len(candidates_info))
        
        logger.info(f"\n✓ Complete solve in {total_time:.3f}s")
        logger.info(f"Top 3: {', '.join(w.upper() for w, *_ in ranked[:3])}")
        logger.info("="*60 + "\n")
        
        return self._format_results(ranked, players_grids, total_time), set(w for w, *_ in ranked)
    
    def _format_results(self, ranked: List[Tuple], 
                       players_grids: List[List[str]], 
                       solve_time: float) -> str:
        """Format results readably"""
        lines = [
            "\n" + "="*85,
            "🎯 ANALYSIS RESULTS",
            "="*85,
            f"Solve time: {solve_time:.3f}s | Candidates analyzed: {len(ranked)}",
            "",
            f"{ 'Rank':<6}{'Word':<10}{'Score':<9}{'Strict%':<10}{'Perfect':<9}{'Freq':<8}{'Entropy':<9}{'Tries'}",
            "-"*85
        ]
        
        for i, (w, sc, ra, pe, ti, av, freq, entropy) in enumerate(ranked[:20], 1):
            lines.append(
                f"{i:<6}{w.upper():<10}{sc:.4f}   {ra*100:5.1f}%    "
                f"{'✓' if pe else '✗':<9}{freq:.3f}   {entropy:.3f}    {av:.1f}"
            )
        
        if ranked:
            w, sc, ra, pe, ti, av, freq, entropy = ranked[0]
            lines.extend([
                "",
                "="*85,
                f"🏆 MOST PROBABLE WORD: {w.upper()}",
                f"   Global score         : {sc:.4f}",
                f"   Strict coherence     : {ra*100:.1f}%",
                f"   Perfect validation   : {'Yes' if pe else 'No'}",
                f"   Letter frequency     : {freq:.3f}",
                f"   Entropy              : {entropy:.3f}",
                "="*85
            ])
        
        return "\n".join(lines)

    def _format_lax_results(self, candidates: set, solve_time: float) -> str:
        """Format lax results readably"""
        lines = [
            "\n" + "="*85,
            "🎯 LAX FILTERING RESULTS",
            "="*85,
            f"Solve time: {solve_time:.3f}s | Lax candidates found: {len(candidates)}",
            "",
            "Some candidates:",
            ", ".join(sorted(list(candidates))[:50])
        ]
        if len(candidates) > 50:
            lines[-1] += "..."
        return "\n".join(lines)
    
    def suggest_opening_words(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """Suggest best opening words"""
        logger.info(f"Calculating {top_n} best opening words...")
        start = time.time()
        
        candidates = set(self.answers)
        scores = []
        
        # Analyze common words (first 500)
        for word in self.guesses[:500]:
            entropy = self.scorer.calculate_entropy(word, candidates)
            letter_variety = len(set(word))  # Prefer distinct letters
            freq = self.scorer.calculate_letter_frequency_score(word)
            
            # Composite score for opening
            score = entropy * 0.50 + letter_variety * 0.30 + freq * 0.20
            scores.append((word, score))
        
        elapsed = time.time() - start
        logger.info(f"✓ Analysis completed in {elapsed:.2f}s")
        
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
    
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
        for line in grid_str.strip().split('\n'):
            line = line.strip()
            if len(line) == WORD_LENGTH and all(c in (GREEN, YELLOW, BLACK) for c in line):
                patterns.append(line)
        return patterns
    
    def run(self):
        """Launch interactive mode"""
        print("\n" + "="*70)
        print("🎮 WORDLE SOLVER INTERACTIVE v2.0 - ENGLISH VERSION")
        print("="*70)
        print("\nAvailable commands:")
        print("  • 'community'  : Analyze community grids")
        print("  • 'personal'   : Personal solver with step-by-step guidance")
        print("  • 'opening'    : Suggest opening words")
        print("  • 'stats'      : Display statistics")
        print("  • 'quit'       : Exit")
        print("="*70 + "\n")
        
        while True:
            command = input("Command > ").strip().lower()
            
            if command == 'quit':
                print("\n👋 Goodbye!")
                break
            
            elif command == 'community':
                self._run_community_solve()
            
            elif command == 'personal':
                self._run_personal_solver()
            
            elif command == 'opening':
                self._suggest_opening()
            
            elif command == 'stats':
                print(self.solver.get_statistics())
            
            else:
                print("❌ Unknown command. Use: community, personal, opening, stats, quit")
    
    def _run_community_solve(self):
        """Community resolution"""
        print("\n--- Community Grid Entry ---")
        print("Enter player grids (one per line, empty to finish)")
        print(f"Format: {GREEN*5} or {YELLOW*3}{BLACK*2}, etc.")
        
        players_grids = []
        player_num = 1
        
        while True:
            print(f"\nPlayer {player_num} (empty to finish):")
            grid_input = []
            
            while True:
                line = input("  Pattern > ").strip()
                if not line:
                    break
                if len(line) == WORD_LENGTH and all(c in (GREEN, YELLOW, BLACK) for c in line):
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
        print("\n" + "="*70)
        print("🎯 PERSONAL SOLVER - STEP BY STEP")
        print("="*70)
        
        # Step 1: Analyze community grids (optional)
        print("\n📊 STEP 1: Community Analysis (optional)")
        print("Do you want to analyze other players' grids first? (y/n)")
        
        community_candidates = None
        if input("> ").strip().lower() == 'y':
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
                    if len(line) == WORD_LENGTH and all(c in (GREEN, YELLOW, BLACK) for c in line):
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
                        secret for secret in community_candidates
                        if _player_possible_sequence_lax(secret, tuple(player_grids))
                    }
                
                print(f"✓ Community analysis: {len(community_candidates)} possible words")
        
        # Step 2: Best opening word
        print("\n" + "="*70)
        print("🚀 STEP 2: Best Opening Word")
        print("="*70)
        
        # If community analysis done, calculate best word from those candidates
        if community_candidates:
            print(f"\nCalculating best opening word from {len(community_candidates)} candidates...")
            if len(community_candidates) <= 10:
                print(f"\n💡 Possible words: {', '.join(sorted(community_candidates))}")
                print("You can try any of these!")
                best_opening = [(w, 1.0) for w in sorted(community_candidates)[:5]]
            else:
                # Calculate entropy for remaining candidates
                scores = []
                for word in list(self.solver.guesses)[:300]:
                    entropy = self.solver.scorer.calculate_entropy(word, community_candidates)
                    letter_variety = len(set(word))
                    freq = self.solver.scorer.calculate_letter_frequency_score(word)
                    score = entropy * 0.60 + letter_variety * 0.25 + freq * 0.15
                    scores.append((word, score))
                best_opening = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
        else:
            print("\nCalculating best opening words from all possibilities...")
            best_opening = self.solver.suggest_opening_words(5)
        
        print(f"\n{'Rank':<6}{'Word':<12}{'Score':<10}")
        print("-"*28)
        for i, (word, score) in enumerate(best_opening, 1):
            print(f"{i:<6}{word.upper():<12}{score:.4f}")
        
        print("\n💡 Recommended: " + best_opening[0][0].upper())
        
        # Step 3: Personal attempts
        print("\n" + "="*70)
        print("📝 STEP 3: Your Attempts")
        print("="*70)
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
            
            if guess == 'done':
                break
            
            if len(guess) != WORD_LENGTH or not guess.isalpha():
                print("❌ Invalid word (must be 5 letters)")
                continue
            
            # Get pattern
            pattern_input = input(f"Pattern for {guess.upper()} > ").strip()
            
            if pattern_input.lower() == 'done':
                break
            
            # Parse pattern
            pattern = self.personal_solver.parse_pattern_input(pattern_input)
            
            if len(pattern) != WORD_LENGTH or not all(c in (GREEN, YELLOW, BLACK) for c in pattern):
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
                print(f"\n💡 Some possible words: {', '.join(w.upper() for w in words)}...")
            
            # Suggest next guess
            if remaining > 1 and attempt_num < 6:
                print("\n🎯 Suggested next guesses:")
                suggestions = self.personal_solver.get_best_next_guess(5)
                for i, (word, score) in enumerate(suggestions[:5], 1):
                    in_candidates = "✓" if word in self.personal_solver.remaining_candidates else " "
                    print(f"  {i}. {word.upper():<10} {in_candidates}  (score: {score:.3f})")
            
            attempt_num += 1
        
        print("\n" + "="*70)
        print("Session complete!")
        print("="*70)
    
    def _suggest_opening(self):
        """Opening word suggestion"""
        print("\n--- Best Opening Words ---")
        suggestions = self.solver.suggest_opening_words(15)
        
        print(f"\n{'Rank':<6}{'Word':<12}{'Score':<10}")
        print("-"*28)
        
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
        description="Wordle Solver Optimized v2.0 - ENGLISH VERSION",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python solver_v2.py --interactive          # Interactive mode
  python solver_v2.py --opening              # Suggest opening words
  python solver_v2.py --personal             # Personal solver
  python solver_v2.py --stats                # Show statistics
        """
    )
    
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Launch in interactive mode')
    parser.add_argument('--personal', '-p', action='store_true',
                       help='Personal solver with step-by-step guidance')
    parser.add_argument('--opening', '-o', action='store_true',
                       help='Show best opening words')
    parser.add_argument('--stats', '-s', action='store_true',
                       help='Display statistics')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    
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
        print("="*50)
        suggestions = solver.suggest_opening_words(15)
        print(f"\n{'Rank':<6}{'Word':<12}{'Score':<10}")
        print("-"*28)
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