#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sys
import os
import threading
from typing import List

# Add current directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from solver import WordleSolver, PersonalSolver, logger, _player_possible_sequence_lax, GREEN
from gemini_analyzer import analyze_wordle_screenshots

class WordleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Wordle Solver Optimized v2.0")
        self.root.geometry("1100x800")

        self.solver: WordleSolver | None = None
        self.personal_solver: PersonalSolver | None = None

        self.COLORS = {'green': '#6aaa64', 'yellow': '#c9b458', 'gray': '#787c7e'}
        self.players_data = []
        self.lax_candidates = set()
        self.strict_candidates = set()
        self.live_remaining_candidates = []
        self.last_players_grids = []
        self.is_strict_analysis_running = False
        self.stop_strict_analysis_event = threading.Event()

        self.setup_styles()
        self.setup_ui()
        
        threading.Thread(target=self._load_solver_async, daemon=True).start()

    def setup_styles(self):
        s = ttk.Style()
        s.configure('TButton', padding=5, font=('Segoe UI', 9))
        s.configure('Title.TLabel', font=('Segoe UI', 16, 'bold'))
        s.configure('Header.TLabel', font=('Segoe UI', 12, 'bold'))
        s.configure('Results.TLabel', font=('Courier', 10))

    def _load_solver_async(self):
        def update_status(message):
            self.root.after(0, lambda: self.status_var.set(f"⏳ {message}"))
        try:
            self.solver = WordleSolver(status_callback=update_status)
            self.personal_solver = self.solver.create_personal_solver()
            self.root.after(0, lambda: self.status_var.set("✅ Solver Ready"))
            self.root.after(0, self._enable_buttons)
        except Exception as e:
            logger.error(f"Critical loading error: {e}", exc_info=True)
            messagebox.showerror("Loading Error", f"Could not initialize solver: {e}")

    def _enable_buttons(self):
        self.solve_button.config(state=tk.NORMAL)
        self.openings_button.config(state=tk.NORMAL)
        self.stats_button.config(state=tk.NORMAL)
        self.personal_submit_button.config(state=tk.NORMAL)

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(top_frame, text="Wordle Solver Optimized", style='Title.TLabel').pack(side=tk.LEFT)
        
        action_buttons_frame = ttk.Frame(top_frame)
        action_buttons_frame.pack(side=tk.RIGHT)
        self.openings_button = ttk.Button(action_buttons_frame, text="Suggest Openers", command=self.suggest_openers, state=tk.DISABLED)
        self.openings_button.pack(side=tk.LEFT, padx=5)
        self.stats_button = ttk.Button(action_buttons_frame, text="Show Stats", command=self.show_stats, state=tk.DISABLED)
        self.stats_button.pack(side=tk.LEFT, padx=5)

        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, pady=5)

        personal_pane = ttk.Frame(paned_window, padding=5)
        self.setup_personal_solver_ui(personal_pane)
        paned_window.add(personal_pane, weight=2)

        community_pane = ttk.Frame(paned_window, padding=5)
        self.setup_community_solver_ui(community_pane)
        paned_window.add(community_pane, weight=3)

        self.status_var = tk.StringVar(value="⏳ Initializing solver...")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=5)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))

    def setup_personal_solver_ui(self, parent):
        ttk.Label(parent, text="Personal Solver", style='Header.TLabel').pack(anchor=tk.W, pady=(0,10))

        # Input frame
        input_frame = ttk.Frame(parent)
        input_frame.pack(fill=tk.X)
        ttk.Label(input_frame, text="Guess:").pack(side=tk.LEFT, padx=(0,5))
        self.personal_word_var = tk.StringVar()
        word_entry = ttk.Entry(input_frame, textvariable=self.personal_word_var, width=10, font=("Courier", 10))
        word_entry.pack(side=tk.LEFT, padx=5)

        self.personal_color_buttons = []
        for i in range(5):
            btn = tk.Button(input_frame, text=str(i+1), width=3, bg=self.COLORS['gray'], command=lambda i=i: self._cycle_personal_color(i))
            btn.pack(side=tk.LEFT, padx=1)
            self.personal_color_buttons.append(btn)

        # Action buttons
        action_frame = ttk.Frame(parent)
        action_frame.pack(fill=tk.X, pady=5)
        self.personal_submit_button = ttk.Button(action_frame, text="Submit Attempt", command=self.submit_personal_attempt, state=tk.DISABLED)
        self.personal_submit_button.pack(side=tk.LEFT)
        ttk.Button(action_frame, text="Reset", command=self.reset_personal_solver).pack(side=tk.LEFT, padx=5)
        self.prefilter_community_button = ttk.Button(action_frame, text="Pre-filter with Community", command=self.prefilter_with_community)
        self.prefilter_community_button.pack(side=tk.LEFT, padx=5)
        self.prefilter_live_button = ttk.Button(action_frame, text="Pre-filter with Live Results", command=self.prefilter_with_live_results, state=tk.DISABLED)
        self.prefilter_live_button.pack(side=tk.LEFT, padx=5)

        # Info display
        info_frame = ttk.Frame(parent)
        info_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.personal_info_var = tk.StringVar(value="Enter your first guess (e.g., SALET) and the resulting pattern.")
        ttk.Label(info_frame, textvariable=self.personal_info_var, wraplength=350, justify=tk.LEFT).pack(anchor=tk.NW)

        ttk.Label(info_frame, text="Suggestions:", style='Header.TLabel').pack(anchor=tk.W, pady=(10,0))
        self.personal_suggestions_text = tk.Text(info_frame, height=5, font=('Courier', 10), state=tk.DISABLED, wrap=tk.WORD)
        self.personal_suggestions_text.pack(fill=tk.X, pady=5)

        ttk.Label(info_frame, text="Remaining Words:", style='Header.TLabel').pack(anchor=tk.W, pady=(10,0))
        self.personal_remaining_text = tk.Text(info_frame, height=10, font=('Courier', 9), state=tk.DISABLED, wrap=tk.WORD)
        self.personal_remaining_text.pack(fill=tk.BOTH, expand=True, pady=5)

    def _cycle_personal_color(self, index):
        btn = self.personal_color_buttons[index]
        current_bg = btn.cget('bg')
        new_color_name = 'yellow' if current_bg == self.COLORS['gray'] else 'green' if current_bg == self.COLORS['yellow'] else 'gray'
        btn.config(bg=self.COLORS[new_color_name])

    def submit_personal_attempt(self):
        if not self.personal_solver: return

        if self.is_strict_analysis_running:
            messagebox.showinfo("Analysis Running", "A strict analysis is currently running. Please stop it before submitting a new attempt.")
            return

        guess = self.personal_word_var.get().strip().lower()
        if len(guess) != 5 or not guess.isalpha():
            messagebox.showerror("Invalid Guess", "Guess must be 5 alphabetic characters.")
            return

        pattern = "".join([{"#6aaa64": '🟩', "#c9b458": '🟨', "#787c7e": '⬛'}[btn.cget('bg')] for btn in self.personal_color_buttons])
        
        self.personal_solver.add_attempt(guess, pattern)
        
        remaining_count = len(self.personal_solver.remaining_candidates)
        self.personal_info_var.set(f"Attempt '{guess.upper()}' processed. {remaining_count} candidates remaining.")

        # Update suggestions
        suggestions = self.personal_solver.get_best_next_guess(5)
        sugg_text = "\n".join([f"{i}. {word.upper():<8} (Score: {score:.3f})" for i, (word, score) in enumerate(suggestions, 1)])
        self._update_text_widget(self.personal_suggestions_text, sugg_text)

        # Update remaining words
        remaining_words = self.personal_solver.get_remaining_words(100)
        rem_text = ", ".join(w.upper() for w in remaining_words)
        if remaining_count > 100:
            rem_text += "..."
        self._update_text_widget(self.personal_remaining_text, rem_text)

        if pattern == '🟩'*5:
            messagebox.showinfo("Congratulations!", f"You solved it with '{guess.upper()}'!")
            self.reset_personal_solver()

    def reset_personal_solver(self):
        if not self.personal_solver: return
        self.personal_solver.reset()
        self.personal_word_var.set("")
        for btn in self.personal_color_buttons:
            btn.config(bg=self.COLORS['gray'])
        self.personal_info_var.set("Personal solver has been reset.")
        self._update_text_widget(self.personal_suggestions_text, "")
        self._update_text_widget(self.personal_remaining_text, "")

    def _update_text_widget(self, widget, content):
        widget.config(state=tk.NORMAL)
        widget.delete(1.0, tk.END)
        widget.insert(tk.END, content)
        widget.config(state=tk.DISABLED)

    def setup_community_solver_ui(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        ttk.Label(parent, text="Community Solver", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, pady=(0,10))

        # Scrolled frame for players
        canvas = tk.Canvas(parent, borderwidth=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        self.players_frame = ttk.Frame(canvas)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=1, column=0, sticky='nsew')
        scrollbar.grid(row=1, column=1, sticky='ns')
        canvas.create_window((0, 0), window=self.players_frame, anchor="nw")
        self.players_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        self.root.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        # Controls
        controls_frame = ttk.Frame(parent)
        controls_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky=tk.EW)
        ttk.Button(controls_frame, text="Add Player", command=self.add_player).pack(side=tk.LEFT, padx=5)
        self.solve_button = ttk.Button(controls_frame, text="Lax Analysis", command=self.solve_lax_community, state=tk.DISABLED)
        self.solve_button.pack(side=tk.LEFT, padx=5)
        self.strict_solve_button = ttk.Button(controls_frame, text="Strict Analysis", command=self.solve_strict_community, state=tk.DISABLED)
        self.strict_solve_button.pack(side=tk.LEFT, padx=5)
        self.stop_analysis_button = ttk.Button(controls_frame, text="Stop Analysis", command=self.stop_strict_analysis, state=tk.DISABLED)
        self.stop_analysis_button.pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Import Screenshots", command=self.import_screenshots).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Clear All", command=self.clear_community).pack(side=tk.LEFT, padx=5)

        # Live Results
        self.live_results_frame = ttk.LabelFrame(parent, text="Live Remaining Candidates")
        self.live_results_frame.columnconfigure(0, weight=1)
        self.community_live_results_text = tk.Text(self.live_results_frame, height=5, font=('Courier', 9), state=tk.DISABLED, wrap=tk.WORD)
        self.community_live_results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Live Words
        self.live_words_frame = ttk.LabelFrame(parent, text="Live Processed Words")
        self.live_words_frame.columnconfigure(0, weight=1)
        self.community_live_words_text_widget = tk.Text(self.live_words_frame, height=5, font=('Courier', 9), state=tk.DISABLED, wrap=tk.WORD)
        self.community_live_words_text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Results
        self.results_frame = ttk.LabelFrame(parent, text="Final Analysis")
        self.results_frame.columnconfigure(0, weight=1)
        self.community_results_text = tk.Text(self.results_frame, height=12, font=('Courier', 9), state=tk.DISABLED, wrap=tk.NONE)
        self.community_results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.live_results_frame.grid_remove()
        self.live_words_frame.grid_remove()
        self.results_frame.grid(row=3, column=0, columnspan=2, sticky=tk.EW)

        self.add_player()

    def add_player(self):
        player_idx = len(self.players_data)
        player_frame = ttk.LabelFrame(self.players_frame, text=f"Player {player_idx + 1}")
        player_frame.pack(fill=tk.X, expand=True, padx=5, pady=5)

        attempts_frame = ttk.Frame(player_frame)
        attempts_frame.pack(fill=tk.X, expand=True, pady=5)

        player_data = {'frame': player_frame, 'attempts_frame': attempts_frame, 'attempts': []}
        self.players_data.append(player_data)

        controls_frame = ttk.Frame(player_frame)
        controls_frame.pack(fill=tk.X, pady=5)
        ttk.Button(controls_frame, text="Add Attempt", command=lambda p_idx=player_idx: self.add_community_attempt(p_idx)).pack(side=tk.LEFT)
        ttk.Button(controls_frame, text="Remove Player", command=lambda p_idx=player_idx: self.remove_player(p_idx)).pack(side=tk.LEFT, padx=5)

        self.add_community_attempt(player_idx)

    def add_community_attempt(self, player_idx, initial_pattern=None):
        player_data = self.players_data[player_idx]
        attempt_idx = len(player_data['attempts'])
        attempt_frame = ttk.Frame(player_data['attempts_frame'])
        attempt_frame.pack(fill=tk.X, expand=True)

        ttk.Label(attempt_frame, text=f"#{attempt_idx+1}").pack(side=tk.LEFT, padx=5)
        pattern_var = tk.StringVar()
        if initial_pattern:
            pattern_var.set(initial_pattern)
        pattern_entry = ttk.Entry(attempt_frame, textvariable=pattern_var, width=10, font=("Courier", 10))
        pattern_entry.pack(side=tk.LEFT, padx=5)
        pattern_entry.bind("<Return>", lambda event, p_idx=player_idx: self._handle_community_enter(p_idx))

        # Add remove button for the attempt
        remove_button = ttk.Button(attempt_frame, text="×", width=2, command=lambda p_idx=player_idx, a_idx=attempt_idx: self.remove_community_attempt(p_idx, a_idx))
        remove_button.pack(side=tk.LEFT, padx=5)

        player_data['attempts'].append({'frame': attempt_frame, 'pattern_var': pattern_var, 'remove_button': remove_button})

    def remove_community_attempt(self, player_idx, attempt_idx):
        player_data = self.players_data[player_idx]
        if len(player_data['attempts']) > 1: # Always keep at least one attempt row
            attempt_data = player_data['attempts'].pop(attempt_idx)
            attempt_data['frame'].destroy()
            # Re-number remaining attempts
            for i, att in enumerate(player_data['attempts']):
                att['frame'].winfo_children()[0].config(text=f"#{i+1}")
                # Update the command for the remove button to reflect new index
                att['remove_button'].config(command=lambda p_idx=player_idx, a_idx=i: self.remove_community_attempt(p_idx, a_idx))

    def _handle_community_enter(self, player_idx):
        if not self.personal_solver: return
        player_data = self.players_data[player_idx]
        if player_data['attempts']:
            last_pattern = player_data['attempts'][-1]['pattern_var'].get().strip().upper()
            parsed_pattern = self.personal_solver.parse_pattern_input(last_pattern)
            if parsed_pattern == GREEN * 5:
                if player_idx == len(self.players_data) - 1:
                    self.add_player()
                # Focus next player
                self.root.after(100, lambda: self.players_data[player_idx + 1]['attempts'][0]['frame'].winfo_children()[1].focus_set())
                return
        # Add new attempt for current player
        self.add_community_attempt(player_idx)
        # Focus new entry
        self.root.after(100, lambda: player_data['attempts'][-1]['frame'].winfo_children()[1].focus_set())

    def remove_player(self, player_idx):
        self.players_data[player_idx]['frame'].destroy()
        self.players_data.pop(player_idx)
        for i, p_data in enumerate(self.players_data):
            p_data['frame'].config(text=f"Player {i + 1}")

    def solve_lax_community(self):
        if not self.solver: return
        self.live_results_frame.grid_remove()
        self.live_words_frame.grid_remove()
        self.results_frame.grid(row=3, column=0, columnspan=2, sticky=tk.EW)
        self.last_players_grids = self._get_players_grids()
        if not self.last_players_grids:
            messagebox.showinfo("No Data", "No community grids were entered.")
            return
        self.status_var.set("🔄 Running Lax Analysis...")
        self.solve_button.config(state=tk.DISABLED)
        self.strict_solve_button.config(state=tk.DISABLED)
        self.prefilter_live_button.config(state=tk.DISABLED)
        self.stop_analysis_button.config(state=tk.DISABLED)
        threading.Thread(target=self._run_solve_thread, args=(self.last_players_grids, [], False), daemon=True).start()

    def solve_strict_community(self):
        if not self.solver or not self.lax_candidates:
            messagebox.showwarning("No Lax Candidates", "Please run Lax Analysis first.")
            return
        self.live_results_frame.grid(row=3, column=0, columnspan=2, sticky=tk.EW, pady=5)
        self.live_words_frame.grid(row=4, column=0, columnspan=2, sticky=tk.EW, pady=5)
        self.results_frame.grid(row=5, column=0, columnspan=2, sticky=tk.EW)
        self.status_var.set("🔄 Running Strict Analysis...")
        self.solve_button.config(state=tk.DISABLED)
        self.strict_solve_button.config(state=tk.DISABLED)
        self.prefilter_live_button.config(state=tk.NORMAL)
        self.stop_analysis_button.config(state=tk.NORMAL)
        threading.Thread(target=self._run_solve_thread, args=(self.last_players_grids, self.personal_solver.attempts, True, self.personal_solver.remaining_candidates), daemon=True).start()

    def stop_strict_analysis(self):
        if self.is_strict_analysis_running:
            self.stop_strict_analysis_event.set()

    def _get_players_grids(self):
        players_grids = []
        for p_data in self.players_data:
            grid = [att['pattern_var'].get().strip().upper() for att in p_data['attempts'] if att['pattern_var'].get().strip()]
            if grid:
                if any(len(p) != 5 for p in grid):
                    messagebox.showerror("Invalid Pattern", f"A pattern for Player {self.players_data.index(p_data)+1} is not 5 characters long.")
                    return None
                players_grids.append(self.personal_solver.parse_pattern_input("\n".join(grid)).split('\n'))
        return players_grids

    def _run_solve_thread(self, players_grids, personal_attempts, run_strict, candidates_to_check=None):
        self.live_remaining_candidates = sorted(list(candidates_to_check)) if candidates_to_check else []
        if run_strict:
            self.is_strict_analysis_running = True
            self.stop_strict_analysis_event.clear()
        try:
            self.root.after(0, lambda: self._update_text_widget(self.community_live_results_text, ""))
            self.root.after(0, lambda: self._update_text_widget(self.community_results_text, ""))
            self.root.after(0, lambda: self._update_text_widget(self.community_live_words_text_widget, ""))
            def progress_handler(update):
                if self.stop_strict_analysis_event.is_set():
                    raise InterruptedError("Analysis stopped by user.")

                if update['type'] == 'player_start':
                    status_msg = f"🔄 Strict (Player {update['player_idx']}/{update['num_players']}): {update['num_candidates']} words..."
                    self.root.after(0, lambda: self.status_var.set(status_msg))
                    self.root.after(0, lambda: self._update_text_widget(self.community_live_results_text, ", ".join(self.live_remaining_candidates)))
                elif update['type'] == 'player_progress':
                    eta_str = f"{update['eta_m']}m {update['eta_s']}s"
                    status_msg = f"🔄 Strict (Player {update['player_idx']}): {update['current']}/{update['total']} ({update['pct']:.1f}%) | ETA: {eta_str}"
                    self.root.after(0, lambda: self.status_var.set(status_msg))
                elif update['type'] == 'player_end':
                    live_text = f"Player {update['player_idx']} done. {update['num_candidates_after']} words remaining."
                    self.root.after(0, lambda: self._update_text_widget(self.community_live_results_text, live_text))
                elif update['type'] == 'word_validated':
                    self.root.after(0, lambda: self._update_text_widget(self.community_live_words_text_widget, f"Checking: {update['word'].upper()}"))
                    if not update['is_valid']:
                        if update['word'] in self.live_remaining_candidates:
                            self.live_remaining_candidates.remove(update['word'])
                            self.root.after(0, lambda: self._update_text_widget(self.community_live_results_text, ", ".join(self.live_remaining_candidates)))


            results, candidates = self.solver.solve(players_grids, personal_attempts, run_strict=run_strict, candidates_to_check=candidates_to_check, progress_callback=progress_handler, stop_event=self.stop_strict_analysis_event)
            
            if not run_strict:
                self.lax_candidates = candidates
                self.personal_solver.remaining_candidates = candidates
                self.root.after(0, lambda: self.strict_solve_button.config(state=tk.NORMAL))
            else:
                self.strict_candidates = candidates

            self.root.after(0, lambda: self._update_text_widget(self.community_results_text, results))
            self.root.after(0, lambda: self.status_var.set("✅ Analysis complete."))
            if run_strict:
                 self.root.after(0, lambda: self._update_text_widget(self.community_live_results_text, "All strict validations passed."))

        except InterruptedError:
            self.root.after(0, lambda: self.status_var.set("🛑 Analysis stopped."))
        except Exception as e:
            logger.error(f"Error in solve thread: {e}", exc_info=True)
            self.root.after(0, lambda: messagebox.showerror("Error", f"An error occurred during analysis:\n{e}"))
            self.root.after(0, lambda: self.status_var.set("❌ Analysis failed."))
        finally:
            if run_strict:
                self.is_strict_analysis_running = False
            self.root.after(0, lambda: self.solve_button.config(state=tk.NORMAL))
            if run_strict:
                self.root.after(0, lambda: self.strict_solve_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.prefilter_live_button.config(state=tk.DISABLED))
            self.root.after(0, lambda: self.stop_analysis_button.config(state=tk.DISABLED))

    def suggest_openers(self):
        if not self.solver: return
        openers = self.solver.suggest_opening_words(15)
        result_str = "\n".join([f"{i:<2}. {word.upper():<8} (Score: {score:.4f})" for i, (word, score) in enumerate(openers, 1)])
        messagebox.showinfo("Best Opening Words", result_str)

    def show_stats(self):
        if not self.solver: return
        messagebox.showinfo("Performance Statistics", self.solver.get_statistics())

    def clear_community(self):
        self.live_results_frame.grid_remove()
        self.live_words_frame.grid_remove()
        self.results_frame.grid(row=3, column=0, columnspan=2, sticky=tk.EW)
        for i in range(len(self.players_data) - 1, -1, -1):
            self.remove_player(i)
        self.add_player()
        self._update_text_widget(self.community_results_text, "")

    def import_screenshots(self):
        if not self.solver: return
        filepaths = filedialog.askopenfilenames(
            title="Select Wordle Screenshot(s)",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
        )
        if not filepaths:
            return

        self.status_var.set(f"🔄 Analyzing {len(filepaths)} screenshot(s)...")
        self.root.update_idletasks()

        try:
            all_grids = analyze_wordle_screenshots(list(filepaths))
            self.clear_community()
            for i, grid in enumerate(all_grids):
                if i > 0: self.add_player()
                for j, pattern in enumerate(grid):
                    if j > 0: self.add_community_attempt(i)
                    self.players_data[i]['attempts'][j]['pattern_var'].set(pattern)
            self.status_var.set(f"✅ Imported {len(all_grids)} grids from {len(filepaths)} image(s).")
        except Exception as e:
            logger.error(f"Screenshot analysis failed: {e}", exc_info=True)
            messagebox.showerror("Analysis Error", f"Could not analyze screenshots: {e}")
            self.status_var.set("❌ Screenshot analysis failed.")

    def prefilter_with_community(self):
        if not self.personal_solver: return

        candidates_to_use = None
        if self.strict_candidates:
            candidates_to_use = self.strict_candidates
        elif self.lax_candidates:
            candidates_to_use = self.lax_candidates
        else:
            messagebox.showwarning("No Community Data", "Please run a 'Lax Analysis' or 'Strict Analysis' on the community data first to generate a candidate list.")
            return
        
        self.personal_solver.remaining_candidates = candidates_to_use
        remaining_count = len(self.personal_solver.remaining_candidates)
        
        self.personal_info_var.set(f"Prefiltered with community results. {remaining_count} candidates remaining.")
        
        # Update remaining words display
        remaining_words = self.personal_solver.get_remaining_words(100)
        rem_text = ", ".join(w.upper() for w in remaining_words)
        if remaining_count > 100:
            rem_text += "..."
        self._update_text_widget(self.personal_remaining_text, rem_text)

        # Update suggestions
        suggestions = self.personal_solver.get_best_next_guess(5)
        sugg_text = "\n".join([f"{i}. {word.upper():<8} (Score: {score:.3f})" for i, (word, score) in enumerate(suggestions, 1)])
        self._update_text_widget(self.personal_suggestions_text, sugg_text)
        
        messagebox.showinfo("Prefilter Complete", f"Personal solver's word list has been updated with the {remaining_count} candidates from the community's analysis.")

    def prefilter_with_live_results(self):
        if not self.personal_solver or not self.live_remaining_candidates:
            messagebox.showwarning("No Live Data", "No live results available to prefilter.")
            return
        
        self.personal_solver.remaining_candidates = set(self.live_remaining_candidates)
        remaining_count = len(self.personal_solver.remaining_candidates)
        
        self.personal_info_var.set(f"Prefiltered with live results. {remaining_count} candidates remaining.")
        
        # Update remaining words display
        remaining_words = self.personal_solver.get_remaining_words(100)
        rem_text = ", ".join(w.upper() for w in remaining_words)
        if remaining_count > 100:
            rem_text += "..."
        self._update_text_widget(self.personal_remaining_text, rem_text)

        # Update suggestions
        suggestions = self.personal_solver.get_best_next_guess(5)
        sugg_text = "\n".join([f"{i}. {word.upper():<8} (Score: {score:.3f})" for i, (word, score) in enumerate(suggestions, 1)])
        self._update_text_widget(self.personal_suggestions_text, sugg_text)
        
        messagebox.showinfo("Prefilter Complete", f"Personal solver's word list has been updated with the {remaining_count} live candidates.")

def main():
    root = tk.Tk()
    app = WordleGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
