#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lanceur pour l'interface graphique du Solveur Wordle Communautaire.
Utilisez ce script pour démarrer l'interface graphique.
"""

import multiprocessing
from wordle_gui import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
