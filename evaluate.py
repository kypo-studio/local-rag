"""
evaluate.py — Evaluation du RAG (qualite du Retrieval).

But : mesurer OBJECTIVEMENT si le pipeline de recherche recupere les bons chunks,
au lieu de juger "au feeling". On rejoue ici EXACTEMENT le retrieval de chat.php
(hybride cosinus + BM25, fusion RRF, reranking MMR) avec les memes parametres.

Metriques :
  - Recall@k : pour quelle proportion de questions le bon chunk est-il dans le top-k ?
  - MRR (Mean Reciprocal Rank) : a quel rang moyen (inverse) apparait le bon chunk ?

"Verite terrain" : chaque question de test cible une RUBRIQUE attendue (categorie
du chunk, cf. Module 5). Un chunk est "pertinent" si sa categorie == rubrique visee.

Usage : uv run evaluate.py
"""

import json
import math
import os
import re
import unicodedata
from pathlib import Path

import requests
from dotenv import load_dotenv

# --- Parametres : DOIVENT correspondre a ceux de chat.php ------------------
MODELE_EMBEDDING = "mistral-embed"
URL_EMBEDDINGS = "https://api.mistral.ai/v1/embeddings"
NB_CHUNKS = 4        # top-k final (= NB_CHUNKS de chat.php)
NB_CANDIDATS = 10    # large filet avant reranking
LAMBDA_MMR = 0.8
BM25_K1, BM25_B = 1.5, 0.75
RRF_K = 60

FICHIER_INDEX = Path("embeddings.json")

# --- Jeu de test : question -> rubrique attendue (verite terrain) ----------
JEU_DE_TEST = [
    ("Quel age as-tu ?",                              "Présentation"),
    ("Quel type de poste recherches-tu ?",           "Poste recherché"),
    ("Quel est ton parcours scolaire ?",             "Formation"),
    ("Quelles ecoles as-tu faites ?",                "Formation"),
    ("Tu sais utiliser Docker et MLflow ?",          "Compétences"),
    ("Tu fais du traitement d'images satellites ?",  "Compétences"),
    ("Parle-moi de ton projet Sentinel-2",           "Projets"),
    ("Ton projet de consommation electrique ?",      "Projets"),
    ("Ou fais-tu ton alternance ?",                  "Expérience"),
    ("Quels sont tes points forts ?",                "Soft skills"),
    ("Tu parles anglais ?",                          "Langues"),
    ("Quels sont tes loisirs ?",                     "Centres d'intérêt"),
    ("Comment te contacter ?",                        "Contact"),
]


# === Replique du retrieval de chat.php ====================================

def cosinus(a, b):
    produit = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return produit / (na * nb)


def tokeniser(texte):
    """Doit matcher tokeniser() de chat.php : minuscules, sans accents, stopwords."""
    texte = texte.lower()
    texte = unicodedata.normalize("NFKD", texte).encode("ascii", "ignore").decode()
    mots = re.split(r"[^a-z0-9]+", texte)
    stop = {
        "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "a", "au",
        "aux", "en", "dans", "sur", "pour", "par", "avec", "sans", "que", "qui",
        "quoi", "dont", "est", "es", "suis", "sont", "mon", "ma", "mes", "ton",
        "ta", "tes", "son", "sa", "ses", "ce", "cet", "cette", "ces", "tu", "je",
        "il", "elle", "on", "nous", "vous", "se", "ne", "pas", "plus", "tres",
        "quel", "quels", "quelle", "quelles", "comment",
    }
    return [m for m in mots if len(m) > 2 and m not in stop]


def bm25_scores(tokens_q, index):
    N = len(index)
    docs = [({}, 0) for _ in index]
    somme = 0
    for i, e in enumerate(index):
        toks = tokeniser(e["text"])
        compte = {}
        for t in toks:
            compte[t] = compte.get(t, 0) + 1
        docs[i] = (compte, len(toks))
        somme += len(toks)
    avg = somme / N if N else 0
    uniques = set(tokens_q)
    idf = {}
    for m in uniques:
        n = sum(1 for compte, _ in docs if m in compte)
        idf[m] = math.log(1 + (N - n + 0.5) / (n + 0.5))
    scores = {}
    for i, (compte, longueur) in enumerate(docs):
        s = 0.0
        for m in uniques:
            f = compte.get(m, 0)
            if not f:
                continue
            norm = 1 - BM25_B + BM25_B * (longueur / max(avg, 1))
            s += idf[m] * (f * (BM25_K1 + 1)) / (f + BM25_K1 * norm)
        scores[i] = s
    return scores


def rrf_fusion(scores_cos, scores_bm25, k=RRF_K):
    rang_cos = {i: r for r, (i, _) in enumerate(
        sorted(scores_cos.items(), key=lambda kv: kv[1], reverse=True), start=1)}
    rang_bm = {i: r for r, (i, _) in enumerate(
        sorted(scores_bm25.items(), key=lambda kv: kv[1], reverse=True), start=1)}
    rrf = {}
    for i in scores_cos:
        rrf[i] = 1 / (k + rang_cos[i]) + 1 / (k + rang_bm.get(i, 10**9))
    return rrf


def mmr_rerank(candidats, pertinence, index, k, lam):
    selection, restants = [], list(candidats)
    while len(selection) < k and restants:
        meilleur, meilleur_score = None, -math.inf
        for i in restants:
            redondance = max(
                (cosinus(index[i]["embedding"], index[j]["embedding"]) for j in selection),
                default=0.0)
            score = lam * pertinence[i] - (1 - lam) * redondance
            if score > meilleur_score:
                meilleur_score, meilleur = score, i
        selection.append(meilleur)
        restants.remove(meilleur)
    return selection


def retrieve(vecteur_q, tokens_q, index):
    """Rejoue le pipeline complet et renvoie la liste ordonnee des index retenus."""
    scores_cos = {i: cosinus(vecteur_q, e["embedding"]) for i, e in enumerate(index)}
    scores_bm = bm25_scores(tokens_q, index)
    rrf = rrf_fusion(scores_cos, scores_bm)
    candidats = [i for i, _ in sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:NB_CANDIDATS]]
    return mmr_rerank(candidats, scores_cos, index, NB_CHUNKS, LAMBDA_MMR)


# === Programme principal ==================================================

def main():
    load_dotenv()
    cle = os.environ.get("MISTRAL_API_KEY")
    if not cle:
        raise SystemExit("Cle MISTRAL_API_KEY introuvable (verifie .env).")

    index = json.loads(FICHIER_INDEX.read_text(encoding="utf-8"))

    # On embedde toutes les questions de test en un seul appel (batch).
    questions = [q for q, _ in JEU_DE_TEST]
    rep = requests.post(
        URL_EMBEDDINGS,
        headers={"Authorization": f"Bearer {cle}", "Content-Type": "application/json"},
        json={"model": MODELE_EMBEDDING, "input": questions},
    )
    rep.raise_for_status()
    vecteurs = [d["embedding"] for d in rep.json()["data"]]

    recalls, reciprocal_ranks = [], []
    print(f"\n{'Question':<42} {'Attendu':<18} {'Trouve@rang':<12} OK")
    print("-" * 82)

    for (question, attendu), vec in zip(JEU_DE_TEST, vecteurs):
        retenus = retrieve(vec, tokeniser(question), index)
        cats = [index[i]["category"] for i in retenus]
        # Rang (1-based) du premier chunk de la bonne rubrique, sinon None.
        rang = next((r for r, c in enumerate(cats, start=1) if c == attendu), None)

        hit = rang is not None
        recalls.append(1 if hit else 0)
        reciprocal_ranks.append(1 / rang if rang else 0)

        rang_txt = f"#{rang}" if rang else "absent"
        print(f"{question[:40]:<42} {attendu:<18} {rang_txt:<12} {'✅' if hit else '❌'}")

    n = len(JEU_DE_TEST)
    print("-" * 82)
    print(f"\nRecall@{NB_CHUNKS} : {sum(recalls)}/{n} = {sum(recalls)/n:.0%}")
    print(f"MRR        : {sum(reciprocal_ranks)/n:.3f}  (1.0 = bon chunk toujours en 1er)")


if __name__ == "__main__":
    main()
