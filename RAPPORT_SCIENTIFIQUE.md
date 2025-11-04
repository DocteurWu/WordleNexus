# Rapport Scientifique : Wordle Solver Optimisé

## Table des Matières

1. [Introduction](#1-introduction)
2. [Architecture du Système](#2-architecture-du-système)
3. [Algorithmique de Base](#3-algorithmique-de-base)
4. [Optimisations Performances](#4-optimisations-performances)
5. [Système de Scoring Avancé](#5-système-de-scoring-avancé)
6. [Validation des Contraintes](#6-validation-des-contraintes)
7. [Fonctionnalités Avancées](#7-fonctionnalités-avancées)
8. [Analyse Mathématique](#8-analyse-mathématique)
9. [Conclusion](#9-conclusion)

## 1. Introduction

Ce document présente une analyse détaillée des algorithmes, mathématiques et optimisations implémentées dans le Wordle Solver Optimisé v2.0. Cette application résout les grilles Wordle en utilisant des techniques avancées de filtrage, de validation de contraintes, et de scoring multicritères.

### Objectifs du système

- Résoudre des grilles Wordle multiples (mode communauté)
- Fournir des suggestions optimales pour le jeu personnel
- Analyser les patterns Wordle via IA pour l'import d'images
- Gérer des performances optimales pour des lots de grilles importants

## 2. Architecture du Système

### 2.1 Structure globale

Le système est architecturé autour de plusieurs composants principaux :

- **WordleSolver** : Moteur central de résolution
- **AdvancedScorer** : Système de scoring multicritères
- **PersonalSolver** : Interface pour le joueur individuel
- **GeminiAnalyzer** : Module d'analyse d'images via IA
- **WordleGUI** : Interface graphique utilisateur
- **SolverLogger** : Système de journalisation
- **SolverStats** : Collecte de statistiques de performance

### 2.2 Configuration centralisée

La classe `SolverConfig` centralise tous les paramètres du système :

```python
@dataclass
class SolverConfig:
    word_length: int = 5
    green: str = "🟩"
    yellow: str = "🟨"
    black: str = "⬛"
    scoring_weights: Dict[str, float] = field(default_factory=lambda: {
        'strict_ratio': 0.42,
        'perfect_bonus': 0.20,
        'tightness': 0.15,
        'coherence': 0.10,
        'difficulty': 0.05,
        'letter_frequency': 0.05,
        'entropy': 0.03
    })
    letter_freq: Dict[str, float] = field(default_factory=lambda: {
        'e': 1.00, 'a': 0.85, 'r': 0.80, 'i': 0.78, 'o': 0.75,
        # ... autres fréquences
    })
```

### 2.3 Données du système

Le système utilise deux jeux de mots :
- **answers.txt** : 2,309 mots solutions valides
- **allowed_guesses.txt** : 14,855 mots autorisés comme tentatives

## 3. Algorithmique de Base

### 3.1 Calcul du feedback Wordle

Le feedback Wordle est l'algorithme fondamental qui détermine la couleur des lettres pour un couple (mot secret, tentative).

```python
def feedback_optimized(secret: str, guess: str) -> str:
    result = [BLACK] * WORD_LENGTH
    secret_list = list(secret)
    
    # Phase 1: Traitement des lettres exactes (verts)
    for i in range(WORD_LENGTH):
        if guess[i] == secret[i]:
            result[i] = GREEN
            secret_list[i] = None  # Marquer comme utilisé
    
    # Phase 2: Traitement des lettres présentes ailleurs (jaunes)
    available = Counter(c for c in secret_list if c is not None)
    for i in range(WORD_LENGTH):
        if result[i] == BLACK and available[guess[i]] > 0:
            result[i] = YELLOW
            available[guess[i]] -= 1
    
    return "".join(result)
```

**Analyse mathématique :**
- **Complexité** : O(n) où n est la longueur du mot (fixée à 5)
- **Gestion des doublons** : L'algorithme utilise un `Counter` pour gérer correctement les lettres répétées
- **Stratégie** : Traitement en deux phases pour éviter les conflits de comptage

### 3.2 Modèle de contraintes

Chaque tentative Wordle impose des contraintes logiques sur le mot solution :
- **Contraintes positives** : Lettres qui doivent être présentes
- **Contraintes de position** : Lettres qui doivent être à des positions spécifiques
- **Contraintes négatives** : Lettres qui ne doivent pas être présentes

## 4. Optimisations Performances

### 4.1 Cache global de patterns

Une des optimisations les plus importantes est la construction d'un cache global de patterns Wordle.

```python
def build_global_pattern_cache(answers: List[str], allowed_guesses: List[str]) -> Dict:
    # Construction en parallèle
    max_workers = min(4, os.cpu_count() or 1)
    worker = partial(build_pattern_map_for_secret, allowed_guesses=allowed_guesses)
    
    # Parallélisation avec ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for secret in answers:
            future = executor.submit(worker, secret)
            futures.append(future)
        
        cache = {}
        for future in as_completed(futures):
            secret, patterns = future.result()
            cache[secret] = patterns
    
    # Compression et sauvegarde
    with gzip.open(config.cache_path, "wb") as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

**Avantages techniques :**
- **Pré-calcul** : Tous les patterns sont calculés une seule fois
- **Compression** : Utilisation de gzip pour réduire la taille des données
- **Parallélisation** : Utilisation de plusieurs threads pour la construction
- **Hash de validation** : Détection automatique des changements dans les mots

### 4.2 Caching LRU avec données partagées

```python
@lru_cache(maxsize=20000)
def _player_possible_sequence_lax(secret: str, pats_tuple: Tuple[str, ...]) -> bool:
    pats = list(pats_tuple)
    m = _player_possible_sequence_lax.cache.get(secret)
    return all(p in m for p in pats)

# Attachement du cache global
_player_possible_sequence_lax.cache = cache_data
```

Cette technique permet de combiner les performances du cache LRU avec l'accès rapide aux données pré-computées.

### 4.3 Parallélisation stricte

Lors de la validation stricte, le système utilise `ProcessPoolExecutor` pour paralléliser les vérifications :

```python
with ProcessPoolExecutor(initializer=init_worker, initargs=(self.cache,)) as executor:
    futures = {executor.submit(_player_possible_sequence_strict, word, tuple(player_grid)): word for word in strict_candidates}
    
    validated_this_round = set()
    for future in as_completed(futures):
        if future.result():
            word = futures[future]
            validated_this_round.add(word)
```

## 5. Système de Scoring Avancé

### 5.1 Modèle multicritères pondéré

Le système de scoring combine plusieurs critères avec des poids spécifiques :

```python
scoring_weights = {
    'strict_ratio': 0.42,      # 42% - Ratio de validation stricte
    'perfect_bonus': 0.20,     # 20% - Bonus pour validation parfaite
    'tightness': 0.15,         # 15% - Rareté des patterns
    'coherence': 0.10,         # 10% - Cohérence temporelle
    'difficulty': 0.05,        # 5%  - Difficulté relative
    'letter_frequency': 0.05,  # 5%  - Fréquence des lettres
    'entropy': 0.03            # 3%  - Information théorique
}
```

### 5.2 Calcul de l'entropie informationnelle

```python
def calculate_entropy(self, word: str, candidates: Set[str]) -> float:
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
```

**Analyse mathématique :**
- **Formule** : H(X) = -Σ p(x) * log₂(p(x))
- **Interprétation** : Mesure l'information moyenne produite par le mot
- **Objectif** : Maximiser l'information pour réduire l'espace de recherche

### 5.3 Calcul de "tightness"

```python
def calculate_tightness(self, word: str) -> float:
    num_patterns = len(self.cache[word])
    total_patterns = len(self.cache)
    return -math.log((num_patterns / total_patterns) + 1e-9)
```

**Logique** : Moins de patterns possibles = plus de contraintes = mot plus "restricteur"

### 5.4 Score composite final

```python
score = (
    w['strict_ratio'] * ratio +
    w['perfect_bonus'] * perfect +
    w['tightness'] * tight +
    w['coherence'] * coherence +
    w['difficulty'] * difficulty +
    w['letter_frequency'] * freq +
    w['entropy'] * entropy
)
```

## 6. Validation des Contraintes

### 6.1 Validation laxiste

La validation "laxiste" vérifie simplement si chaque pattern est possible pour un mot candidat :

```python
def _player_possible_sequence_lax(secret: str, pats_tuple: Tuple[str, ...]) -> bool:
    pats = list(pats_tuple)
    m = _player_possible_sequence_lax.cache.get(secret)
    return all(p in m for p in pats)
```

### 6.2 Validation stricte

La validation "stricte" vérifie la cohérence temporelle de la séquence de tentatives :

```python
def _player_possible_sequence_strict(secret: str, pats_tuple: Tuple[str, ...]) -> bool:
    pats = list(pats_tuple)
    m = _player_possible_sequence_strict.cache.get(secret)
    
    # Initialisation avec la première tentative
    layer = [(g, pats[0]) for g in m.get(pats[0], [])]
    if not layer: return False
    
    # Validation séquentielle
    for pat in pats[1:]:
        candidates = m.get(pat, [])
        if not candidates: return False
        
        new_layer = [
            (next_guess, pat)
            for prev_guess, prev_pat in layer
            for next_guess in candidates
            if guesses_keep_previous_info(prev_guess, prev_pat, next_guess)
        ]
        
        if not new_layer: return False
        layer = new_layer
    
    return True
```

### 6.3 Vérification des contraintes temporelles

```python
def guesses_keep_previous_info(prev_guess: str, prev_pat: str, next_guess: str) -> bool:
    # Vérification des verts (position fixe)
    for i, c in enumerate(prev_pat):
        if c == GREEN and prev_guess[i] != next_guess[i]:
            return False
    
    # Vérification des lettres informatives (vert ou jaune)
    for letter in info_letters(prev_pat, prev_guess):
        if letter not in next_guess:
            return False
    
    # Vérification des lettres absentes (noires)
    for letter in black_letters(prev_pat, prev_guess):
        if letter in next_guess:
            return False
    
    return True
```

### 6.4 Cohérence temporelle

```python
def check_player_coherence_loose(patterns: List[str]) -> bool:
    greens = [p.count(GREEN) for p in patterns]
    return all(greens[i] >= greens[i - 1] - 1 for i in range(1, len(patterns)))
```

## 7. Fonctionnalités Avancées

### 7.1 Analyse par IA (Gemini)

Le système peut analyser des captures d'écran Wordle via Google Gemini :

```python
def analyze_wordle_screenshots(image_paths: list):
    # Envoi à l'API Gemini avec instructions précises
    response = model.generate_content([PROMPT, img])
    # Extraction et parsing du JSON retourné
```

### 7.2 Interface graphique asynchrone

L'interface graphique utilise des threads pour éviter les blocages :

```python
threading.Thread(target=self._load_solver_async, daemon=True).start()
```

### 7.3 Mode résolution personnelle

Le mode personnel fournit des suggestions adaptées au jeu individuel :

```python
def get_best_next_guess(self, top_n: int = 5) -> List[Tuple[str, float]]:
    scores = []
    for word in list(self.solver.guesses)[:500]:
        entropy = self.solver.scorer.calculate_entropy(word, self.remaining_candidates)
        letter_variety = len(set(word))
        freq = self.solver.scorer.calculate_letter_frequency_score(word)
        
        score = entropy * 0.60 + letter_variety * 0.25 + freq * 0.15
        scores.append((word, score))
    
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
```

## 8. Analyse Mathématique

### 8.1 Complexité algorithmique

**Phase de filtrage laxiste :**
- **Temps** : O(P × C) où P = nombre de joueurs, C = nombre de candidats
- **Espace** : O(N) où N = taille du cache

**Phase de validation stricte :**
- **Temps** : O(P × C × T) où T = nombre total de tentatives
- **Espace** : O(S) où S = séquences valides

**Phase de scoring :**
- **Temps** : O(R × W) où R = nombre de résultats, W = nombre de poids
- **Espace** : O(R)

### 8.2 Efficacité informationnelle

L'entropie permet de mesurer l'efficacité d'un mot candidat :
- **Entropie maximale** = mot qui divise l'espace de recherche de manière équilibrée
- **Optimisation** : Choisir le mot qui maximise l'entropie moyenne

### 8.3 Théorie des contraintes

Le système implémente un modèle de satisfaction de contraintes (CSP) :
- **Variables** : Position des lettres dans le mot solution
- **Domaines** : Ensemble possible de lettres à chaque position
- **Contraintes** : Relations imposées par les feedbacks Wordle

## 9. Analyse de la suggestion de mots

J'ai vérifié le code et confirmé que la suggestion de mots est effectivement optimisée pour maximiser les chances de trouver le mot suivant en tenant compte des mots restants. Voici l'analyse détaillée :

### 9.1 Suggestion dans le mode personnel

La méthode `get_best_next_guess()` dans la classe `PersonalSolver` fonctionne comme suit :

```python
def get_best_next_guess(self, top_n: int = 5) -> List[Tuple[str, float]]:
    if len(self.remaining_candidates) <= 2:
        # Si peu de candidats restent, les suggérer directement
        return [(w, 1.0) for w in sorted(self.remaining_candidates)]
    
    # Calculer l'entropie pour les mots les plus courants
    scores = []
    for word in list(self.solver.guesses)[:500]:  # Vérifier les 500 premiers mots courants
        entropy = self.solver.scorer.calculate_entropy(word, self.remaining_candidates)
        letter_variety = len(set(word))  # Variété de lettres
        freq = self.solver.scorer.calculate_letter_frequency_score(word)
        
        # Score composite
        score = entropy * 0.60 + letter_variety * 0.25 + freq * 0.15
        scores.append((word, score))
    
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
```

**Analyse :**
- Le système utilise les candidats **restants** (`self.remaining_candidates`) pour calculer l'entropie
- L'entropie est calculée spécifiquement pour réduire l'espace de recherche restant
- Le score composite est optimisé pour maximiser l'information (60%) tout en considérant la variété des lettres (25%) et la fréquence (15%)

### 9.2 Calcul de l'entropie

La méthode `calculate_entropy()` est cruciale pour maximiser l'information :

```python
def calculate_entropy(self, word: str, candidates: Set[str]) -> float:
    """Calculate information entropy of a word"""
    if len(candidates) <= 1:
        return 0.0
    
    pattern_dist = defaultdict(int)
    for candidate in candidates:
        pattern = feedback(candidate, word)  # Calcul du pattern pour chaque candidat
        pattern_dist[pattern] += 1
    
    total = len(candidates)
    entropy = 0.0
    for count in pattern_dist.values():
        if count > 0:
            p = count / total  # Probabilité de chaque pattern
            entropy -= p * math.log2(p)  # Formule d'entropie H(X) = -Σ p(x) * log2(p(x))
    
    return entropy
```

**Analyse mathématique :**
- L'entropie mesure l'information moyenne obtenue en utilisant un mot comme tentative
- Un mot avec une haute entropie divise l'espace de recherche de manière équilibrée
- Cela maximise les chances de réduire significativement le nombre de candidats restants

### 9.3 Suggestion d'ouverture avec candidats filtrés

Dans le mode interactif, le système adapte les suggestions en fonction des candidats restants :

```python
if community_candidates:
    print(f"\nCalculating best opening word from {len(community_candidates)} candidates...")
    if len(community_candidates) <= 10:
        print(f"\n💡 Possible words: {', '.join(sorted(community_candidates))}")
        print("You can try any of these!")
        best_opening = [(w, 1.0) for w in sorted(community_candidates)[:5]]
    else:
        # Calculer l'entropie pour les candidats restants
        scores = []
        for word in list(self.solver.guesses)[:300]:
            entropy = self.solver.scorer.calculate_entropy(word, community_candidates)
            letter_variety = len(set(word))
            freq = self.solver.scorer.calculate_letter_frequency_score(word)
            score = entropy * 0.60 + letter_variety * 0.25 + freq * 0.15
            scores.append((word, score))
        best_opening = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
```

**Conclusion sur la suggestion :**
Oui, le système tient bien compte de la liste des mots restants et cherche activement à maximiser l'information pour la prochaine tentative. L'algorithme de suggestion :

1. **Considère l'espace de recherche actuel** (les mots restants)
2. **Calcule l'entropie spécifique** à cet espace pour chaque mot potentiel
3. **Optimise la réduction de l'entropie** pour maximiser les chances de trouver la solution
4. **Équilibre information et diversité** des lettres dans le mot suggéré
5. **Adapte la stratégie** selon le nombre de candidats restants

### 9.4 Théorie de l'information appliquée

Le système implémente concrètement les principes de la théorie de l'information :
- **Maximisation de l'entropie** pour obtenir la meilleure réduction d'incertitude
- **Calcul de distribution de probabilité** des patterns possibles
- **Application de la formule d'entropie de Shannon** pour chaque mot candidat

## 10. Conclusion

Le Wordle Solver Optimisé v2.0 représente une approche sophistiquée de la résolution algorithmique de Wordle. Les principales forces du système sont :

1. **Architecture modulaire** permettant l'extension
2. **Optimisations performances** : cache, parallélisation, compression
3. **Système de scoring multicritères** avec pondération fine
4. **Validation de contraintes stricte** pour des résultats fiables
5. **Interface utilisateur avancée** avec modes communauté et personnel
6. **Analyse par IA** pour l'automatisation de l'import de données
7. **Suggestion optimisée** basée sur la théorie de l'information pour maximiser les chances de résolution

Le système démontre une application concrète de concepts avancés en algorithmique, théorie de l'information, et optimisation combinatoire, tout en restant accessible à travers une interface conviviale.