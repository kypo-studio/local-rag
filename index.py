"""
Script d'indexation (Phase 1 du RAG).

Role : lire contenu.txt -> transformer chaque chunk en vecteur via Mistral
       -> sauvegarder le tout dans embeddings.json.

On relance ce script UNIQUEMENT quand le contenu change.
"""

import json
import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv

# --- Constantes -----------------------------------------------------------

# IMPORTANT : ce meme modele DOIT etre utilise cote PHP (Phase 2).
# Indexer avec un modele et interroger avec un autre rendrait la
# similarite cosinus sans aucun sens (vecteurs d'espaces differents).
MODELE_EMBEDDING = "mistral-embed"
URL_EMBEDDINGS = "https://api.mistral.ai/v1/embeddings"

FICHIER_CONTENU = Path("contenu.txt")
FICHIER_SORTIE = Path("embeddings.json")


def lire_chunks(chemin: Path) -> list[dict]:
    """Lit le fichier et renvoie la liste des chunks avec leur categorie.

    Regles :
      - les lignes commencant par # sont des commentaires -> ignorees
      - SAUF "#@cat: Label" qui definit la categorie des chunks suivants
      - les chunks sont separes par une ou plusieurs lignes vides

    Retourne une liste de dicts : {"text": ..., "category": ...}.
    """
    chunks: list[dict] = []
    categorie_courante = "Général"  # valeur par defaut si aucune directive
    buffer: list[str] = []

    def flush() -> None:
        """Transforme le buffer accumule en un chunk, puis le vide."""
        texte = "\n".join(buffer).strip()
        if texte:
            chunks.append({"text": texte, "category": categorie_courante})
        buffer.clear()

    for ligne in chemin.read_text(encoding="utf-8").splitlines():
        depouillee = ligne.strip()

        # Directive de categorie : "#@cat: Projets"
        match = re.match(r"#@cat:\s*(.+)", depouillee)
        if match:
            categorie_courante = match.group(1).strip()
            continue

        # Autre commentaire -> ignore.
        if depouillee.startswith("#"):
            continue

        # Ligne vide -> fin du chunk en cours.
        if depouillee == "":
            flush()
            continue

        # Ligne de contenu -> on l'accumule.
        buffer.append(ligne)

    flush()  # ne pas oublier le dernier chunk
    return chunks


def main() -> None:
    # On charge le .env -> la cle devient accessible via os.environ.
    load_dotenv()
    cle = os.environ.get("MISTRAL_API_KEY")
    if not cle:
        raise SystemExit("Cle MISTRAL_API_KEY introuvable. Verifie ton .env.")

    chunks = lire_chunks(FICHIER_CONTENU)
    print(f"{len(chunks)} chunks lus depuis {FICHIER_CONTENU}.")

    # On embedde uniquement le TEXTE (la categorie est une metadonnee, pas
    # du contenu a vectoriser). On garde l'ordre pour reassocier ensuite.
    textes = [c["text"] for c in chunks]

    # On envoie TOUS les textes en un seul appel (batch) : plus rapide et
    # moins d'appels reseau. Mistral ne distingue pas document/query
    # (contrairement a Voyage) : on envoie juste le texte.
    reponse = requests.post(
        URL_EMBEDDINGS,
        headers={
            "Authorization": f"Bearer {cle}",
            "Content-Type": "application/json",
        },
        json={"model": MODELE_EMBEDDING, "input": textes},
    )
    reponse.raise_for_status()  # leve une erreur claire si l'API repond mal
    data = reponse.json()

    vecteurs = [item["embedding"] for item in data["data"]]

    # On assemble la structure finale : une liste de {text, category, embedding}.
    donnees = [
        {"text": chunk["text"], "category": chunk["category"], "embedding": vecteur}
        for chunk, vecteur in zip(chunks, vecteurs)
    ]

    FICHIER_SORTIE.write_text(
        json.dumps(donnees, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    dimension = len(donnees[0]["embedding"]) if donnees else 0
    print(f"{len(donnees)} vecteurs ecrits dans {FICHIER_SORTIE}.")
    print(f"Dimension de chaque vecteur : {dimension}")
    if "usage" in data:
        print(f"Tokens factures : {data['usage'].get('total_tokens', '?')}")


if __name__ == "__main__":
    main()
