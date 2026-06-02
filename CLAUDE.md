# CLAUDE.md — Chatbot RAG portfolio de Pol QUIMERC'H

## Contexte du projet

Je suis Pol QUIMERC'H, en fin d'études, à la recherche d'un poste en **Data Science / IA générale**.
Je construis un **chatbot RAG** intégré à mon site personnel. Son rôle : répondre aux
visiteurs (notamment des recruteurs) en parlant de moi — mon parcours, mes projets,
mes compétences, le type de poste recherché — en répondant à la première personne, en mon nom.

**Objectif pédagogique important :** je veux APPRENDRE en construisant ce projet.
Tu PEUX écrire du code, mais tu dois te comporter comme un FORMATEUR : décris et explique
TOUTES les étapes, le pourquoi de chaque décision, le rôle de chaque morceau de code,
les concepts sous-jacents et les pièges à éviter. Ne te contente jamais de livrer du code
brut. Avance progressivement, vérifie ma compréhension, et débogue avec moi quand je suis
bloqué en m'expliquant la cause du problème.

## Stack technique

- **Hébergement** : Infomaniak (mutualisé)
- **CMS** : WordPress avec le thème **Divi**
- **Front-end** : widget chatbot en HTML/CSS/JS, style "Apple Liquid Glass", intégré
  via un module Code Divi ou le footer du thème
- **Back-end** : un script **PHP** sur Infomaniak qui fait l'intermédiaire (proxy)
  → la clé API n'est JAMAIS exposée au navigateur
- **Modèle de génération** : **Mistral** (`mistral-small-latest`) — offre gratuite, sans CB
- **Modèle d'embedding** : **Mistral** (`mistral-embed`) — même fournisseur, une seule clé
- *(Choix initial Voyage AI + Claude Haiku abandonné : Haiku nécessite des crédits
  prépayés, et Voyage en gratuit est limité à 3 req/min. Mistral fait les deux gratuitement.)*

## Architecture RAG — vue d'ensemble

Le RAG se découpe en deux phases distinctes.

### Phase 1 — Indexation (une seule fois, en Python, sur ma machine)
1. J'écris mes infos dans `contenu.txt`, découpées en chunks (1 idée autonome par chunk,
   séparés par une ligne vide).
2. Un script Python lit ces chunks.
3. Pour chaque chunk, appel à l'API d'embedding Voyage AI → un vecteur.
4. Sauvegarde dans `embeddings.json` : liste de `{"text": chunk, "embedding": [...]}`.
5. J'uploade `embeddings.json` sur Infomaniak.

→ On relance cette phase uniquement quand le contenu change.

### Phase 2 — Requête (à chaque message visiteur, en PHP sur Infomaniak)
1. Le widget JS envoie la question du visiteur au script PHP.
2. Le PHP transforme la question en vecteur (MÊME modèle d'embedding Voyage qu'en phase 1).
3. Calcul de la **similarité cosinus** entre la question et chaque chunk stocké.
4. On garde les 3-4 chunks les plus proches.
5. Construction du prompt : system prompt (rôle) + chunks récupérés + question.
6. Appel à Claude Haiku → réponse renvoyée au widget.

C'est le RAG : **R**etrieval (récupération) + **A**ugmented (prompt enrichi) + **G**eneration.

## Pourquoi le même modèle d'embedding aux deux phases
Les vecteurs ne sont comparables que s'ils vivent dans le même espace vectoriel.
Indexer avec un modèle et interroger avec un autre rendrait la similarité cosinus
sans aucun sens. La dimension des vecteurs doit aussi correspondre.

## Coûts & estimation
- API Anthropic = paiement à l'usage, par crédits prépayés (pas d'abonnement).
- Haiku : ~1 $ / million de tokens en entrée, ~5 $ / million en sortie.
- La sortie coûte 5× l'entrée → le contexte RAG gonfle l'entrée (côté pas cher).
- Un message typique (~2000 tokens entrée + 300 sortie) ≈ moins d'un demi-centime.

## Garde-fous (par ordre de priorité)
1. **Plafond de dépense** dans la console Anthropic + crédits prépayés → la facture
   ne peut JAMAIS exploser. À configurer en premier.
2. **Rate limiting** côté PHP : max N messages par IP / fenêtre de temps (ex : 20 / 10 min).
3. **Limite de longueur** des messages entrants (ex : rejet au-delà de 500 caractères).
4. **`max_tokens` bas** dans l'appel API (ex : 400) → plafonne le coût de sortie.
5. **Cap quotidien global** : compteur côté serveur, le bot devient "indisponible"
   au-delà d'un seuil journalier.
6. **Prompt caching** (optimisation ultérieure) : le system prompt étant constant,
   le mettre en cache réduit son coût jusqu'à ~90 %.

## Sécurité
- La clé API Anthropic et la clé Voyage vivent UNIQUEMENT côté serveur (PHP),
  jamais dans le JavaScript ni dans le HTML.
- Le widget ne parle qu'au script PHP, jamais directement aux API tierces.

## État d'avancement
- [x] Choix de l'architecture (option B : indexation Python, serveur PHP léger)
- [x] Choix des modèles (Haiku + Voyage)
- [] Phase 1 : rédaction des chunks (`contenu.txt`)
- [ ] Phase 1 : script Python d'indexation → `embeddings.json`
- [ ] Phase 2 : script PHP (recherche cosinus + appel Haiku)
- [ ] Front-end : widget chatbot Liquid Glass
- [ ] Intégration dans Divi
- [ ] Mise en place des garde-fous

## Prochaine étape
Rédiger les chunks dans `contenu.txt`, puis écrire le script Python d'indexation.
