"""
Script d'indexation (Phase 1 du RAG).

Role : lire contenu.txt -> transformer chaque chunk en vecteur via Voyage AI
       -> sauvegarder le tout dans embeddings.json.

On relance ce script UNIQUEMENT quand le contenu change.
"""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
import voyageai

# --- Constantes -----------------------------------------------------------

# IMPORTANT : ce meme modele DOIT etre utilise cote PHP (Phase 2).
# Indexer avec un modele et interroger avec un autre rendrait la
# similarite cosinus sans aucun sens (vecteurs d'espaces differents).
MODELE_EMBEDDING = "voyage-3.5-lite"

FICHIER_CONTENU = Path("contenu.txt")
FICHIER_SORTIE = Path("embeddings.json")


def lire_chunks(chemin: Path) -> list[str]:
    """Lit le fichier et renvoie la liste des chunks.

    Regles :
      - les lignes commencant par # sont des commentaires -> ignorees
      - les chunks sont separes par une ou plusieurs lignes vides
    """
    texte_brut = chemin.read_text(encoding="utf-8")

    # 1) On retire les lignes de commentaire.
    lignes = [
        ligne for ligne in texte_brut.splitlines()
        if not ligne.strip().startswith("#")
    ]
    texte = "\n".join(lignes)

    # 2) On decoupe sur les lignes vides, on nettoie, on jette les vides.
    blocs = re.split(r"\n\s*\n", texte)
    chunks = [bloc.strip() for bloc in blocs if bloc.strip()]
    return chunks


def main() -> None:
    # On charge le .env -> la cle devient accessible via os.environ.
    load_dotenv()
    cle = os.environ.get("VOYAGE_API_KEY")
    if not cle:
        raise SystemExit("Cle VOYAGE_API_KEY introuvable. Verifie ton .env.")

    chunks = lire_chunks(FICHIER_CONTENU)
    print(f"{len(chunks)} chunks lus depuis {FICHIER_CONTENU}.")

    # Le client lit automatiquement la cle dans la variable d'environnement.
    client = voyageai.Client()

    # On envoie TOUS les chunks en un seul appel (batch) : plus rapide et
    # moins d'appels reseau. input_type="document" indique a Voyage qu'on
    # indexe un corpus (cote PHP, pour la question, ce sera "query").
    resultat = client.embed(
        chunks,
        model=MODELE_EMBEDDING,
        input_type="document",
    )

    # On assemble la structure finale : une liste de {text, embedding}.
    donnees = [
        {"text": chunk, "embedding": vecteur}
        for chunk, vecteur in zip(chunks, resultat.embeddings)
    ]

    FICHIER_SORTIE.write_text(
        json.dumps(donnees, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    dimension = len(donnees[0]["embedding"]) if donnees else 0
    print(f"{len(donnees)} vecteurs ecrits dans {FICHIER_SORTIE}.")
    print(f"Dimension de chaque vecteur : {dimension}")
    print(f"Tokens factures : {resultat.total_tokens}")


if __name__ == "__main__":
    main()
