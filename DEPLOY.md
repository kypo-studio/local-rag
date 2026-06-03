# Déploiement sur Infomaniak

Guide pas à pas pour mettre le chatbot en ligne. On suppose que WordPress + Divi
tournent déjà sur ton hébergement Infomaniak.

## Vue d'ensemble

| Fichier | Va où | Rôle |
|---|---|---|
| `chat.php` | serveur (dossier `/chatbot/`) | le proxy (Phase 2) |
| `embeddings.json` | serveur (`/chatbot/`) | l'index vectoriel |
| `data/.htaccess` | serveur (`/chatbot/data/`) | protège les compteurs |
| `secrets.php` | serveur (`/chatbot/`) | **ta clé Mistral** (à créer sur place) |
| `widget.html` | **pas uploadé tel quel** | son contenu est collé dans Divi |

> Les phases 1 (Python, `index.py`) restent **sur ta machine**. Sur le serveur,
> on ne met que ce qui sert à répondre aux visiteurs.

---

## Étape 1 — Préparer la clé Mistral côté serveur

Sur un hébergement mutualisé, le plus fiable est le fichier `secrets.php` (le repli
prévu par `chat.php`). **Ne l'uploade pas depuis le repo** (il est gitignoré) :
crée-le directement sur le serveur.

1. Copie `secrets.example.php` en `secrets.php`.
2. Remplace la valeur par ta vraie clé :
   ```php
   <?php
   return "ta_vraie_cle_mistral";
   ```
3. Place-le dans `/chatbot/` à côté de `chat.php`.

> Apache **n'expose jamais** le contenu d'un `.php` (il l'exécute), donc la clé
> n'est pas lisible depuis le web. Elle n'est pas non plus dans Git.

---

## Étape 2 — Uploader les fichiers

Via le **Gestionnaire de fichiers** Infomaniak ou en **FTP/SFTP** :

1. Crée un dossier `chatbot/` à la racine web de ton site.
2. Uploade dedans : `chat.php`, `embeddings.json`, et le dossier `data/` (avec
   son `.htaccess`).
3. Ajoute-y ton `secrets.php` (créé à l'étape 1).
4. Vérifie dans le manager Infomaniak que la version **PHP est >= 8.1** et que
   l'extension **cURL** est activée (c'est le cas par défaut).
5. Le dossier `data/` doit être **inscriptible** par PHP (chmod 755, normalement
   déjà bon — `chat.php` le crée tout seul au besoin).

---

## Étape 3 — Tester le serveur en production

Depuis ton terminal (remplace le domaine) :

```bash
curl -s -X POST https://TON-DOMAINE/chatbot/chat.php \
  -H "Content-Type: application/json" \
  -d '{"question":"Quels sont tes projets ?"}'
```

Tu dois recevoir un JSON `{"answer":"..."}`. Si erreur :
- `{"error":"Cle API non configuree..."}` → `secrets.php` mal placé ou clé vide.
- page blanche / 500 → regarde les logs PHP dans le manager Infomaniak.

---

## Étape 4 — Intégrer le widget dans Divi

Le widget doit apparaître sur **toutes les pages** → on le met dans le pied de page
du thème (et non dans un module d'une seule page).

1. Ouvre [widget.html](widget.html) et **change l'endpoint** en haut du `<script>` :
   ```js
   const CONFIG = {
     endpoint: "https://TON-DOMAINE/chatbot/chat.php",
     ...
   };
   ```
2. Dans WordPress : **Divi → Options du thème → Intégration → Code du corps (body)**.
3. Colle **uniquement** : le bloc `<style>...</style>`, les deux `<div>`
   (`#chat-bubble` et `#chat-panel`), et le bloc `<script>...</script>`.
   **N'inclus pas** `<!DOCTYPE>`, `<html>`, `<head>`, `<body>` ni le `background`
   de démo (c'est juste pour le test local).
4. Enregistre, ouvre ton site : la bulle apparaît en bas à droite. 🎉

> **CORS** : comme le widget (sur ton site) et `chat.php` (sur le même domaine)
> partagent la même origine, aucun réglage CORS n'est nécessaire. Si un jour tu
> héberges `chat.php` sur un autre domaine, il faudra ajouter un en-tête
> `Access-Control-Allow-Origin` dans `chat.php`.

---

## Étape 5 — Plafonner les coûts côté Mistral

Même si l'offre est gratuite, va dans la console **console.mistral.ai** vérifier/
poser une limite d'usage sur ton espace de travail (Workspace → Limits/Billing).
C'est le garde-fou n°1 « ceinture + bretelles » avec ceux déjà codés dans `chat.php`.

---

## Mettre à jour le contenu plus tard

Quand tu modifies `contenu.txt` :

```bash
uv run index.py            # régénère embeddings.json (sur ta machine)
```

Puis ré-uploade **uniquement** `embeddings.json` dans `/chatbot/` sur le serveur.
Rien d'autre à toucher.

---

## Checklist sécurité finale

- [ ] `secrets.php` présent sur le serveur, **absent** du dépôt Git
- [ ] La clé Mistral n'apparaît nulle part dans le JS / HTML du widget
- [ ] `data/.htaccess` en place (compteurs non accessibles par URL)
- [ ] Test `curl` en prod : réponse JSON correcte
- [ ] Limite d'usage posée dans la console Mistral
