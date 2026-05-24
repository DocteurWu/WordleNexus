#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import google.generativeai as genai
from PIL import Image
import json
import re
import configparser

# --- Configuration ---

def load_api_key():
    """Charge la clé API depuis config.ini."""
    config = configparser.ConfigParser()
    try:
        config.read('config.ini')
        api_key = config.get('API', 'GEMINI_API_KEY', fallback=None)
        if not api_key or api_key == "YOUR_API_KEY_HERE":
            print("⚠️ Clé API non configurée dans config.ini. Veuillez la remplacer.")
            return None
        return api_key
    except configparser.Error as e:
        print(f"Erreur de lecture de config.ini : {e}")
        return None

API_KEY = load_api_key()
MODEL_NAME = "gemini-1.5-flash"

# --- Prompt for the AI ---
# This prompt is carefully designed to get a structured JSON output.
PROMPT = """
Analyze the provided screenshot of a Wordle game.
Identify all player grids visible in the image.

For each player, extract only the color patterns for each attempt.
The colors are represented by:
- 'V' for Green (correct letter in the correct position)
- 'J' for Yellow (correct letter in the wrong position)
- 'G' for Gray/Black (incorrect letter)

Return the data as a single JSON object. The JSON object should be a list of players.
Each player object in the list should have a key "patterns", which is a list of strings.
Each string in the "patterns" list represents a single color pattern, consisting of 5 characters (V, J, or G).

Example of a valid JSON output for two players:
[
  {
    "patterns": [
      "GVJGG",
      "VVVGV"
    ]
  },
  {
    "patterns": [
      "JGGJV",
      "GVVGG",
      "VVVVV"
    ]
  }
]

Do not include any explanations or text outside of the JSON object.
"""

def analyze_wordle_screenshots(image_paths: list):
    """
    Analyzes one or more Wordle screenshots using the Gemini API.

    Args:
        image_paths: A list of file paths to the screenshot images.

    Returns:
        A list of player data dictionaries extracted from the images, or None if an error occurs.
        Example: [{"patterns": ["GVJGG", ...]}, ...]
    """
    if not API_KEY:
        raise ValueError("Clé API non configurée. Veuillez la définir dans config.ini.")

    genai.configure(api_key=API_KEY)

    model = genai.GenerativeModel(MODEL_NAME)

    all_players_data = []

    try:
        for image_path in image_paths:
            print(f"Analyzing image: {image_path}...")
            img = Image.open(image_path)

            # Prepare the request for the model
            response = model.generate_content([PROMPT, img])

            # Clean and parse the response
            raw_text = response.text
            
            # Remove markdown backticks and clean up the string
            cleaned_json_str = re.sub(r'```json\s*|\s*```', '', raw_text).strip()

            # Parse the JSON string into a Python list
            players_data = json.loads(cleaned_json_str)

            if isinstance(players_data, list):
                all_players_data.extend(players_data)
            else:
                print(f"AI response was not a list: {players_data}")
                return None

    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON from AI response: {e}")
        print(f"Raw AI response: {raw_text}")
        import traceback
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"An error occurred during AI analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

    return all_players_data

def list_available_models():
    """Lists all available Gemini models and their supported methods."""
    if not API_KEY:
        raise ValueError("API key is not configured.")
    genai.configure(api_key=API_KEY)
    print("\n--- Listing available Gemini models ---")
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            print(f"Model: {m.name}, Supported methods: {m.supported_generation_methods}")
    print("-------------------------------------\n")

if __name__ == '__main__':
    list_available_models()
    # This is for testing the module directly.
    # You would need to provide a path to a real screenshot.
    # Example:
    # test_image_path = 'path/to/your/screenshot.png'
    # result = analyze_wordle_screenshots([test_image_path])
    # if result:
    #     print("Analysis successful:")
    #     print(json.dumps(result, indent=2))
    # else:
    #     print("Analysis failed.")
    pass
