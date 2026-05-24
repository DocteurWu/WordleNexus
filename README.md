# Solveur Wordle Optimisé v3.0 - Guide Complet

## Aperçu

Cette application est un solveur Wordle hautement sophistiqué doté d'algorithmes avancés de théorie de l'information et d'interfaces utilisateur complètes (graphique et console). Il propose une résolution communautaire (analyse simultanée des grilles de plusieurs joueurs) et une résolution personnelle (guidage pas à pas).

La version 3.0 introduit le **Solveur Exact basé sur une matrice de feedback précalculée**, réduisant drastiquement le temps d'évaluation des partitions à un temps constant $O(1)$, ainsi que des stratégies décisionnelles optimisées comme **Proba Crisp** et **Une Pierre Deux Coups**.

---

## Fonctionnalités Clés

### 🔍 Multi-Mode de Résolution
- **Solveur Communautaire** : Analyse les grilles de plusieurs joueurs pour en extraire le mot solution commun le plus probable.
- **Solveur Personnel** : Fournit une assistance pas à pas et en temps réel pour vos parties individuelles.
- **Pré-filtrage Communautaire** : Permet d'injecter des données communautaires pour restreindre instantanément l'espace initial de recherche personnelle.
- **Suggestions d'Ouverture** : Recommande les meilleurs premiers mots d'après des critères statistiques stricts.

### 📊 Algorithmes Mathématiques et Théorie de l'Information
- **Solveur ERS Exact (Expected Remaining Size)** : Évalue instantanément le pouvoir de partitionnement de l'intégralité des 14 855 essais autorisés contre les candidats restants grâce à une matrice numpy stockée en mémoire.
- **Optimisation "Une Pierre Deux Coups"** : Détecte s'il existe un mot candidat valide dont le pouvoir de partition (ERS) est supérieur ou égal au meilleur mot d'essai global. Jouer ce candidat permet d'avoir une chance de trouver le mot secret en un coup, tout en conservant la même efficacité moyenne de réduction.
- **Entropie de Shannon** : Maximisation de l'information théorique moyenne pour diviser au mieux l'espace de recherche restant :
  $$H(X) = - \sum_{i} p(x_i) \log_2 p(x_i)$$
- **Scoring Multicritères** : Système de pondération dynamique combinant le ratio de validation strict, le bonus parfait, la rareté du pattern ("tightness"), la cohérence temporelle, la difficulté, la fréquence de lettre et l'entropie.

### 🎨 Interface Graphique Moderne (GUI)
- **Interface à Double Paneau** : Résolution personnelle à gauche, résolution communautaire à droite.
- **Indicateurs Visuels Interactifs** : Boutons de couleur cliquables (Gris ⬛ → Jaune 🟨 → Vert 🟩) pour saisir les motifs.
- **Affichage en Temps Réel** : Suivi asynchrone des analyses, des mots traités et des statistiques de performance.
- **Défilement et Transitions Fluides** : Raccourcis clavier pour naviguer d'un champ à l'autre et gestion de la molette de souris.

---

## Mathématiques & Optimisation

### 1. Expected Remaining Size (ERS)
L'ERS mesure la taille moyenne attendue de l'espace de recherche restant après avoir joué un mot d'essai $g$. La formule de calcul est :

$$ERS(g) = \frac{\sum_{p} |S_{g, p}|^2}{|C|}$$

Où :
- $|C|$ est le nombre total de candidats actuellement possibles.
- $p$ représente l'un des 243 motifs de feedback possibles ($3^5$).
- $|S_{g, p}|$ désigne le sous-ensemble de candidats dans $C$ qui renverraient le feedback $p$ si le mot secret était évalué avec l'essai $g$.

Le solveur cherche à **minimiser** l'ERS pour réduire l'espace des solutions le plus rapidement possible.

### 2. Matrice de Feedback Précalculée
Au lieu de recalculer le feedback pour chaque couple (essai, candidat) en $O(5)$ à la volée, le module `ExactSolver` charge en mémoire :
- `feedback_matrix_uint8.npy` : Une matrice bidimensionnelle globale de taille $(14855, 2309)$ stockée sous forme d'entiers non-signés de 8 bits.
- `word_index_maps.pkl` : Les dictionnaires d'indexation bijectifs entre les chaînes de caractères et les lignes/colonnes de la matrice.

Grâce à cela, le partitionnement et le calcul de l'ERS d'un mot s'effectuent via des opérations vectorisées extrêmement rapides sous **NumPy**.

### 3. Mode "Proba Crisp"
Par défaut, le solveur considère que tous les candidats restants ont une probabilité uniforme d'être la solution. L'activation du mode **Proba Crisp** applique une pondération probabiliste non-uniforme sur les candidats restants en se basant sur la fréquence théorique des lettres dans la langue :
- Les mots contenant des lettres plus courantes reçoivent une probabilité d'occurrence plus élevée.
- Cela affine les calculs statistiques et priorise les suggestions les plus naturelles pour un joueur humain.

---

## Installation et Configuration

### Prérequis
- Python 3.8 ou supérieur
- Bibliothèque graphique Tkinter (généralement incluse avec Python)
- Packages requis : `numpy`

### Installation des Dépendances
Installez les packages nécessaires via `pip` :
```bash
pip install numpy
```

### Génération de la Matrice Exacte
Pour bénéficier de la puissance et de la rapidité du solveur exact $O(1)$, vous devez générer la matrice de feedback. Exécutez la commande suivante (cette opération dure entre 30 et 60 secondes et ne doit être lancée **qu'une seule fois**) :
```bash
python build_feedback_matrix.py
```
Ce script créera les fichiers `feedback_matrix_uint8.npy` et `word_index_maps.pkl` dans le dossier `data/`.

---

## Utilisation

### Mode Graphique (Recommandé)
Pour démarrer l'application avec l'interface graphique Tkinter :
```bash
python run_gui.py
```

### Mode Console (CLI)
Le solveur propose également plusieurs outils interactifs en ligne de commande :
```bash
# Lancer le solveur communautaire interactif
python solver.py --interactive

# Lancer le solveur personnel
python solver.py --personal

# Afficher les meilleurs mots d'ouverture statistiques
python solver.py --opening

# Afficher les statistiques de performance de la session
python solver.py --stats
```

---

## Structure des Fichiers

```
Wordle/
├── run_gui.py                # Lanceur principal de la GUI
├── wordle_gui.py             # Code de l'interface graphique Tkinter
├── solver.py                 # Moteur de résolution principal & ExactSolver
├── build_feedback_matrix.py  # Script de génération de la matrice numpy
├── RAPPORT_SCIENTIFIQUE.md   # Documentation scientifique approfondie
├── README.md                 # Ce fichier
├── data/
│   ├── answers.txt           # Liste des 2 309 mots solutions valides
│   ├── allowed_guesses.txt   # Liste des 14 855 tentatives autorisées
│   ├── pattern_cache.pkl     # Cache de patterns standardisé
│   ├── feedback_matrix_uint8.npy # Matrice numpy précalculée pour le solveur exact
│   └── word_index_maps.pkl   # Index de correspondance des mots pour NumPy
└── logs/
    └── solver_*.log          # Fichiers de log détaillés
```

---

## Contribution & Licence

Ce projet est distribué sous licence open-source. Les contributions sous forme de pull requests, d'optimisations mathématiques ou de rapports d'anomalies sont les bienvenues.