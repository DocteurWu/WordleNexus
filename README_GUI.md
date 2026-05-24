# Guide de l'Interface Graphique - Wordle Solver v3.0

## Description

Ce guide détaille l'utilisation de l'interface graphique (GUI) de l'application **Wordle Solver v3.0**. L'interface est conçue pour être extrêmement réactive (asynchrone) et propose deux modes d'exploitation principaux :
1. **Personal Solver** (Solveur Personnel) : Pour vous guider pas à pas dans votre propre partie de Wordle.
2. **Community Solver** (Solveur Communautaire) : Pour analyser simultanément les essais de plusieurs joueurs afin d'en déduire le mot mystère commun.

L'application intègre des algorithmes mathématiques avancés basés sur la **théorie de l'information** et supporte la résolution exacte en $O(1)$ grâce à une matrice précalculée de feedback en NumPy.

---

## Lancement de la GUI

Pour démarrer l'interface graphique, exécutez la commande suivante depuis le dossier racine du projet :
```bash
python run_gui.py
```

---

## Ergonomie Générale & Raccourcis

- **Disposition** : Divisée en deux colonnes principales : le Solveur Personnel à gauche et le Solveur Communautaire à droite.
- **Défilement à la souris** : Vous pouvez utiliser la molette de votre souris sur le panneau contenant les joueurs pour naviguer de haut en bas sans cliquer sur la barre de défilement.
- **Saisie Rapide** : Lors de la saisie des motifs dans le Solveur Communautaire, appuyer sur la touche `Entrée` ajoute automatiquement une tentative ou passe au joueur suivant.

---

## 1. Solveur Personnel (Panneau de Gauche)

Ce mode vous donne la meilleure stratégie de résolution pour votre partie individuelle en temps réel.

### Mode d'emploi pas à pas :
1. **Saisie de l'essai** : Entrez votre mot de 5 lettres dans le champ **"Guess"** (ex: `SALET` ou `AUDIO`).
2. **Configuration du motif coloré** : Cliquez sur les 5 boutons à côté du champ pour configurer les couleurs reçues en jeu :
   - Chaque clic fait cycler le bouton : **Gris (Absent)** ⬛ → **Jaune (Mal placé)** 🟨 → **Vert (Bien placé)** 🟩.
3. **Validation** : Cliquez sur **"Submit Attempt"** pour soumettre la ligne.
4. **Suggestions et candidats** :
   - **Suggestions** : Le système affiche les 5 meilleurs mots à jouer ensuite, classés par pouvoir de réduction (score ERS décroissant).
   - **Remaining Words** : Affiche la liste alphabétique des candidats valides encore possibles.
5. **Réinitialisation** : Utilisez le bouton **"Reset"** pour démarrer une nouvelle partie.

### Pré-filtrage :
- **Pre-filter with Community** : Avant de démarrer, si des grilles sont entrées sur le panneau communautaire, cliquez ici pour restreindre instantanément les mots de départ de votre solveur personnel aux mots qui satisfont déjà les motifs communautaires.
- **Pre-filter with Live Results** : Permet d'injecter dynamiquement les candidats trouvés en temps réel lors d'une analyse communautaire active.

---

## 2. Solveur Communautaire (Panneau de Gauche & Droite)

Ce mode permet de trouver la solution partagée par un groupe d'amis ou de joueurs ayant la même grille mystère journalière.

### Saisie des données :
1. **Ajout de joueurs** : Cliquez sur **"Add Player"** pour insérer un nouveau joueur.
2. **Saisie des motifs** : Pour chaque joueur, entrez les motifs colorés de chaque ligne d'essai. Les formats acceptés sont extrêmement souples :
   - Les emojis officiels : `🟩`, `🟨`, `⬛` (ou `⬜`)
   - Les initiales en français : `V` (Vert), `J` (Jaune), `G` (Gris)
   - Les initiales en anglais : `G` (Green), `Y` (Yellow), `B` (Black/Gray)
   *(Exemple : Saisir `VJGJV` sera automatiquement interprété comme 🟩🟨⬛🟨🟩)*.
3. **Retrait** : Utilisez le bouton **"×"** pour retirer une tentative erronée et **"Remove Player"** pour supprimer entièrement un joueur.

### Commandes d'Analyse :
- **Lax Analysis** : Lance un filtrage logique rapide sur les candidats. C'est l'étape recommandée à exécuter en premier pour générer les candidats initiaux.
- **Exact ERS Analysis** (Recommandé) : Exploite le moteur exact NumPy pour calculer les valeurs réelles d'Expected Remaining Size (ERS) de tous les mots. Il offre les meilleures suggestions mathématiques.
- **Proba Crisp** :
  - `[OFF]` (Par défaut) : L'analyse considère que tous les candidats restants ont la même probabilité d'être le mot solution.
  - `[ON]` : Utilise la fréquence linguistique des lettres pour attribuer une probabilité d'apparition pondérée aux candidats.
- **Stop Analysis** : Permet d'interrompre instantanément une recherche longue.
- **Import Screenshots** : Utilise l'intelligence artificielle Google Gemini pour analyser des captures d'écran de Wordle et préremplir automatiquement les grilles des joueurs en quelques secondes.

---

## Fonctionnalités Avancées de la GUI

### HUD "Une Pierre Deux Coups"
Dans l'affichage de l'**Exact ERS Analysis**, le solveur calcule séparément le meilleur coup théorique global (qui peut être un mot d'essai n'étant pas dans les candidats) et les meilleurs candidats. 
Si un mot candidat valide possède un pouvoir de partition (ERS) équivalent ou meilleur que le meilleur mot d'essai global, le système affiche une bannière spéciale dans le panneau d'analyse :
```
================================================================================
MEILLEUR COUP: CRANE (ERS=1.42) [CANDIDAT]  [UNE PIERRE DEUX COUPS: jouer ce mot trouve LE mot en 1 coup si c'est lui]
================================================================================
```
Jouer ce mot vous permet de tenter votre chance pour gagner immédiatement tout en maintenant la meilleure réduction moyenne statistique possible.

### Suggestion d'Ouverture (Suggest Openers)
Cliquez sur ce bouton en haut à droite pour obtenir instantanément la liste des meilleurs premiers mots statistiquement optimaux pour entamer n'importe quelle partie de Wordle.

### Affichage des Statistiques (Show Stats)
Affiche le temps de calcul des phases, le taux de réussite du cache LRU et d'autres indicateurs système pour analyser les performances de votre machine.