# Wordle Solver Optimized v2.0 - Complete Guide

## Overview

This is a sophisticated Wordle solving application that provides multiple solving modes with advanced algorithms, AI-powered image analysis, and comprehensive user interfaces. It supports community solving (analyzing multiple players' patterns), personal solving (step-by-step individual guidance), and AI-powered screenshot analysis.

## Features

### 🔍 **Multi-Mode Solving**
- **Community Solving**: Analyze multiple players' Wordle patterns simultaneously
- **Personal Solving**: Step-by-step guidance for individual Wordle games
- **Pre-filtering**: Use community data to narrow down personal solver candidates
- **Opening Word Suggestions**: Optimized first guess recommendations

### 🤖 **AI Integration**
- **Screenshot Analysis**: Import Wordle screenshots using Google Gemini AI
- **Automatic Pattern Extraction**: AI analyzes images and extracts color patterns
- **Multi-image Processing**: Batch processing of multiple screenshots

### 📊 **Advanced Algorithms**
- **Multi-Criteria Scoring**: Comprehensive scoring system with weighted factors
- **Pattern Cache**: Pre-computed feedback patterns for optimal performance
- **Lax & Strict Validation**: Two-phase constraint verification
- **Entropy-Based Suggestions**: Information gain optimization

### 🎨 **Modern GUI**
- **Dual-Mode Interface**: Personal + Community solvers in one window
- **Visual Color Selection**: Clickable color buttons for pattern entry
- **Real-time Statistics**: Live status updates and performance metrics
- **Asynchronous Loading**: Non-blocking solver initialization

### 📈 **Performance & Analytics**
- **Comprehensive Logging**: Detailed performance tracking
- **Statistics Collection**: Solve times, cache performance, success rates
- **Parallel Processing**: Multi-threaded pattern cache generation
- **Compression**: Efficient storage of precomputed data

## System Architecture

### Core Components
- **WordleSolver**: Main solver engine with 3-phase processing (lax filtering → strict validation → advanced scoring)
- **PersonalSolver**: Interactive mode for individual Wordle guidance
- **AdvancedScorer**: Multi-criteria scoring system with entropy calculations
- **SolverLogger**: Comprehensive logging and statistics

### Data
- **2,309** valid Wordle answers
- **14,855** allowed guess words  
- **Pre-computed pattern cache**: Over 2GB of optimized pattern data

## Installation & Setup

### Prerequisites
- Python 3.6+
- Required packages: `google-generativeai`, `Pillow`, `tkinter`

### Installation
```bash
pip install google-generativeai Pillow
```

### API Configuration
1. Get a Google Gemini API key from [Google AI Studio](https://aistudio.google.com/)
2. Create a file named `config.ini` in the root directory of the project.
3. Add the following content to `config.ini`, replacing `"your_api_key_here"` with your actual API key:
```ini
[API]
GEMINI_API_KEY = "your_api_key_here"
```

## How to Run

To get started quickly, follow these steps:

1.  **Install dependencies**: Ensure you have Python 3.6+ and the required packages installed.
2.  **Configure API Key**: Set up your Google Gemini API key in `config.ini` as described above.
3.  **Launch GUI**: Run the graphical interface using `python run_gui.py`.
4.  **Explore Modes**: Choose between Community Solver, Personal Solver, or use AI-powered screenshot analysis.



## Usage

### GUI Mode (Recommended)
```bash
python run_gui.py
```

#### Community Solver
1. **Add players**: Click "Add Player" for each Wordle player
2. **Enter patterns**: Input each player's pattern sequence (e.g., 🟩🟨⬛🟨⬛)
3. **Add attempts**: Use "Add Attempt" button for multiple attempts per player
4. **Solve**: Click "Solve Community" for analysis
5. **Import screenshots**: Use "Import Screenshots" for AI-powered pattern extraction

#### Personal Solver
1. **Enter your guess**: Type 5-letter word in the guess field
2. **Set pattern**: Click color buttons (Gray→Yellow→Green→Gray cycle)
3. **Submit**: Click "Submit Attempt" 
4. **Get suggestions**: View best next guesses and remaining candidates
5. **Pre-filter**: Use community data to narrow your options

### Command Line Mode
```bash
# Interactive mode
python solver.py --interactive

# Personal solver
python solver.py --personal

# Opening words
python solver.py --opening

# Statistics
python solver.py --stats
```

## Development

To set up a development environment, consider the following:

-   **Code Structure**: The project is organized into `run_gui.py` (GUI launcher), `wordle_gui.py` (graphical interface), `solver.py` (core logic), and `gemini_analyzer.py` (AI integration).
-   **Logging**: Detailed logs are generated in the `logs/` directory, which can be helpful for debugging and performance analysis.
-   **Testing**: (Add details about testing if available, otherwise mention future plans or how to manually test)



## Technical Details

### Scoring Algorithm
The system uses weighted scoring across multiple factors:
- **strict_ratio** (42%): How many players the word works for
- **perfect_bonus** (20%): Bonus for working for all players  
- **tightness** (15%): Pattern rarity scoring
- **coherence** (10%): Temporal consistency
- **difficulty** (5%): Based on average attempts
- **letter_frequency** (5%): Common letters preference
- **entropy** (3%): Information gain

### Validation Process
1. **Lax Filtering**: Quick elimination using pattern existence
2. **Strict Validation**: Sequence coherence verification  
3. **Advanced Scoring**: Multi-criteria ranking

### Performance Optimizations
- Parallel pattern cache building
- LRU caching for repeated calculations
- Compressed cache storage
- Asynchronous UI operations
- Early exit optimizations

## File Structure
```
wordle_solver/
├── run_gui.py              # GUI launcher
├── wordle_gui.py           # Graphical interface
├── solver.py              # Core solver engine
├── gemini_analyzer.py     # AI image analysis
├── config.ini            # Configuration settings
├── README.md             # This file
└── data/
    ├── answers.txt       # Valid solution words
    ├── allowed_guesses.txt # All allowed guesses
    └── pattern_cache.pkl.gz # Precomputed patterns
└── logs/
    └── solver_*.log      # Performance logs
```

## Algorithms Explained

### Pattern Feedback
- **Phase 1**: Green (exact match) processing
- **Phase 2**: Yellow (present elsewhere) processing
- **Counter-based**: Handles duplicate letters correctly

### Constraint Verification
- **Lax**: Checks if patterns exist for each attempt
- **Strict**: Ensures temporal sequence validity
- **Coherence**: Validates progression rules (greens can't become blanks, etc.)

### Multi-Criteria Scoring
Combined score = Σ(weight × factor) for all criteria

## Use Cases

### Community Analysis
- Multiple players sharing their Wordle results
- Finding the most probable shared solution
- Handling inconsistent patterns or cheating attempts

### Personal Play
- Step-by-step guidance for your own Wordle
- Optimal next guess suggestions
- Remaining candidate tracking

### Competitive Play
- Analyzing other players' strategies
- Understanding solution difficulty
- Opening word optimization

## Performance & Statistics

The system tracks:
- Solve times and efficiency
- Cache hit/miss rates
- Phase-by-phase performance
- Overall usage statistics

## Troubleshooting

### Common Issues
- **API Key**: Ensure Gemini API key is properly configured
- **Cache**: First run may take 1-2 minutes to build pattern cache
- **Memory**: Large word lists require ~2GB RAM for optimal performance

### Screenshot Analysis
- Use clear, high-contrast Wordle images
- Avoid screenshots with additional UI elements
- Support for various Wordle-style games

## Contributing

The project is designed with extensibility in mind. Feel free to contribute via pull requests or issue reports.

## License

This project is open source and available under the [MIT License](LICENSE).