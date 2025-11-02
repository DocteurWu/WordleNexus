# Wordle Solver Optimized v2.0

## Description

This project is an advanced Wordle solver featuring a powerful analysis engine and a comprehensive graphical user interface (GUI). It offers two main modes: a **Personal Solver** for step-by-step guidance on your own game, and a **Community Solver** to find the most probable word based on the grids of multiple players.

The solver engine uses advanced concepts like information entropy to provide the best possible suggestions.

## Key Features

- **Dual Solver Modes**: Interactive personal guidance and multi-grid community analysis.
- **Advanced Scoring**: Uses entropy, letter frequency, and pattern tightness for highly accurate suggestions.
- **AI-Powered Import**: Uses the Gemini API to import game grids directly from screenshots.
- **Performance Statistics**: Track solver performance, cache hit rate, and phase timings.
- **Opening Word Suggester**: Get the best statistical opening words to start your game.
- **Robust Caching**: The pattern cache is automatically built and validated to ensure fast subsequent runs.

## Files

- `solver.py`: The core analysis engine with all the logic.
- `wordle_gui.py`: The main Tkinter GUI application.
- `run_gui.py`: The script to launch the GUI.
- `gemini_analyzer.py`: Module for screenshot analysis via Gemini API.
- `config.ini`: Configuration file for API keys and solver settings.
- `data/`: Directory for word lists.
- `logs/`: Directory where detailed log files are stored.

## How to Use the GUI

### Launching the Application

```bash
python run_gui.py
```

### Main Window

The interface is split into two main parts: **Personal Solver** on the left and **Community Solver** on the right.

### Personal Solver (Step-by-Step Guidance)

This mode helps you solve your own game.

1.  **Enter Your Guess**: Type the word you played (e.g., `AUDIO`) in the "Guess" field.
2.  **Set the Pattern**: Click the 5 buttons next to the guess field to set the color pattern you received in the game (Gray -> Yellow -> Green).
3.  **Submit**: Click **"Submit Attempt"**. The solver will process the information and update the displays:
    - **Suggestions**: Shows the best words to play next, based on information entropy.
    - **Remaining Words**: Shows the list of all possible answers.
4.  Repeat for each attempt.
5.  **Pre-filtering (Optional)**: Before starting your personal solve, you can enter community grids on the right and click **"Pre-filter with Community"**. This will use the community data to narrow down the initial list of words for the personal solver.

### Community Solver (Find the Common Word)

This mode is for finding the single word that fits multiple players' game grids.

1.  **Add Players**: Use the **"Add Player"** button to create a section for each player.
2.  **Enter Patterns**: For each player, enter the sequence of patterns they received. You can use `VJG` format (Vert, Jaune, Gris) or emojis. Pressing **Enter** automatically moves to the next attempt.
3.  **Manage Attempts**: Use the **"×"** button to remove an incorrect attempt.
4.  **Solve**: Once all grids are entered, click **"Solve Community"**. The results will appear in the text box below, ranked by score.
5.  **Import**: Use **"Import Screenshots"** to automatically populate the grids using the Gemini AI.

### Global Actions

- **Suggest Openers**: Shows a list of the statistically best opening words.
- **Show Stats**: Displays performance statistics for the solver session.

## Configuration

Before using the screenshot import, you must edit `config.ini` and add your Gemini API key:

```ini
[API]
GEMINI_API_KEY = "YOUR_API_KEY_HERE"
```