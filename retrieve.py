"""
Script de validation du retrieval (le "R" de RAG).

Role : prendre une question, la transformer en vecteur, et afficher les
       chunks de embeddings.json les plus proches (similarite cosinus).

Ce script ne sert PAS en production (ce sera le role du PHP). Il sert a
verifier que notre index repond bien, et a comprendre le calcul cosinus.

Usage :
    uv run retrieve.py "Pol sait-il faire du deep learning ?"
"""

import json
import math
import sys
from pathlib import Path

from dotenv import load_dotenv
import voyageai

# Le MEME modele qu'a l'indexation (regle d'or du RAG).
MODELE_EMBEDDING = "voyage-3.5-lite"
NB_RESULTATS = 4  # combien de chunks on garde (le "top-k")


def cosinus(a: list[float], b: list[float]) -> float:
    """Similarite cosinus entre deux vecteurs.

    cos = (a . b) / (||a|| * ||b||)
      - a . b      : produit scalaire (somme des produits terme a terme)
      - ||a||      : norme = longueur du vecteur = sqrt(somme des carres)
    Resultat dans [-1, 1] : 1 = sens identique, 0 = aucun rapport.
    """
    produit_scalaire = sum(x * y for x, y in zip(a, b))
    norme_a = math.sqrt(sum(x * x for x in a))
    norme_b = math.sqrt(sum(y * y for y in b))
    return produit_scalaire / (norme_a * norme_b)


def main() -> None:
    load_dotenv()

    # La question vient de la ligne de commande, sinon une question par defaut.
    question = " ".join(sys.argv[1:]) or "Pol sait-il faire du deep learning ?"

    # On recharge l'index qu'on a genere a l'etape precedente.
    donnees = json.loads(Path("embeddings.json").read_text(encoding="utf-8"))

    # On vectorise la QUESTION. Note : input_type="query" (et non "document").
    client = voyageai.Client()
    vecteur_question = client.embed(
        [question], model=MODELE_EMBEDDING, input_type="query"
    ).embeddings[0]

    # On calcule la proximite de la question avec CHAQUE chunk.
    scores = [
        (cosinus(vecteur_question, entree["embedding"]), entree["text"])
        for entree in donnees
    ]

    # On trie du plus proche au plus lointain, on garde le top-k.
    scores.sort(key=lambda paire: paire[0], reverse=True)

    print(f"\nQuestion : {question}\n")
    print(f"Top {NB_RESULTATS} chunks les plus proches :\n")
    for rang, (score, texte) in enumerate(scores[:NB_RESULTATS], start=1):
        apercu = texte[:90].replace("\n", " ")
        print(f"  {rang}. [score {score:.3f}] {apercu}...")


if __name__ == "__main__":
    main()
