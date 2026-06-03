<?php
/**
 * chat.php — Proxy RAG (Phase 2), a deposer sur Infomaniak.
 *
 * Role : reçoit la question d'un visiteur (envoyee par le widget JS),
 *        fait le Retrieval (embedding Mistral + cosinus sur embeddings.json),
 *        construit le prompt et appelle un modele de chat Mistral, puis
 *        renvoie la reponse.
 *
 * SECURITE : la cle API vit UNIQUEMENT ici, cote serveur, jamais dans le
 * JavaScript. Elle est lue depuis la variable d'environnement du serveur.
 */

// =====================================================================
// 0. CONFIGURATION
// =====================================================================

// MEME modele d'embedding qu'a l'indexation Python (regle d'or du RAG).
const MODELE_EMBEDDING = "mistral-embed";
// Modele de generation (gratuit sur l'offre Mistral).
const MODELE_GENERATION = "mistral-small-latest";

const NB_CHUNKS = 4;        // top-k final : nb de chunks injectes dans le prompt
const NB_CANDIDATS = 10;    // reranking : nb de candidats recuperes avant re-tri fin
const LAMBDA_MMR = 0.8;     // reranking : equilibre pertinence (1) <-> diversite (0)
                            // 0.8 = on favorise la pertinence (corpus petit/focalise),
                            // tout en diversifiant les questions larges.
const MAX_TOKENS = 400;     // garde-fou : plafonne la longueur (et le cout) de la reponse
const MAX_LONGUEUR_QUESTION = 500; // garde-fou : rejette les messages trop longs
const MAX_HISTORIQUE = 6;   // memoire : nb max de messages d'historique gardes (fenetre glissante)

const RATE_LIMIT_MAX = 20;        // garde-fou : messages max par IP...
const RATE_LIMIT_FENETRE = 600;   // ...sur cette fenetre (600 s = 10 min)
const CAP_QUOTIDIEN = 500;        // garde-fou : requetes max par jour (tous visiteurs)
const DOSSIER_DATA = __DIR__ . "/data"; // compteurs des garde-fous (fichiers JSON)

// La cle est lue dans l'environnement du serveur (jamais en dur ici).
// Repli pour l'hebergement mutualise : un fichier secrets.php (non versionne)
// qui fait simplement `return "ta_cle";`.
$cle_mistral = getenv("MISTRAL_API_KEY");
if (!$cle_mistral && file_exists(__DIR__ . "/secrets.php")) {
    $cle_mistral = require __DIR__ . "/secrets.php";
}

// Le system prompt : il definit le ROLE du bot. Defensif : interdit d'inventer.
const SYSTEM_PROMPT = <<<PROMPT
Tu es l'assistant personnel de Pol Quimerc'h, sur son site portfolio.
Tu reponds aux visiteurs (souvent des recruteurs) a la PREMIERE PERSONNE,
au nom de Pol ("je", "mon parcours"...), de façon chaleureuse et concise.

Tu peux t'appuyer sur deux sources : (1) le CONTEXTE fourni a chaque message et
(2) l'historique de la conversation en cours. Tu comprends ainsi les questions de
suivi (ex. "et celui-la ?", "resume ca").

N'invente JAMAIS de faits sur Pol : un fait (formation, projet, competence, date,
coordonnee) doit provenir du CONTEXTE ou d'un message precedent de la conversation.
Si une information factuelle demandee ne figure dans aucune de ces sources, dis
simplement que tu ne l'as pas et invite a contacter Pol directement.

Chaque element du CONTEXTE est precede de sa rubrique entre parentheses, ex.
"(Projets)", "(Formation)". Quand c'est pertinent, ancre ta reponse en evoquant
NATURELLEMENT la rubrique concernee ("Cote projets, ...", "Au niveau de ma
formation, ..."). N'ecris JAMAIS la rubrique sous forme d'etiquette brute entre
parentheses ni de balise ; reste fluide et a la premiere personne.
PROMPT;


// =====================================================================
// 1. ENTREE & GARDE-FOUS
// =====================================================================

header("Content-Type: application/json; charset=utf-8");

// On n'accepte que des requetes POST.
if ($_SERVER["REQUEST_METHOD"] !== "POST") {
    http_response_code(405);
    echo json_encode(["error" => "Methode non autorisee."]);
    exit;
}

if (!$cle_mistral) {
    http_response_code(500);
    echo json_encode(["error" => "Cle API non configuree sur le serveur."]);
    exit;
}

// Le widget envoie du JSON : { "question": "..." }
$entree = json_decode(file_get_contents("php://input"), true);
$question = trim($entree["question"] ?? "");

if ($question === "") {
    http_response_code(400);
    echo json_encode(["error" => "Question vide."]);
    exit;
}
if (mb_strlen($question) > MAX_LONGUEUR_QUESTION) {
    http_response_code(400);
    echo json_encode(["error" => "Question trop longue."]);
    exit;
}

// --- Memoire de conversation : on recupere l'historique envoye par le widget.
// Le navigateur envoie { "history": [ {role, content}, ... ] }. On ne lui fait
// JAMAIS confiance : on assainit tout (roles autorises, longueur, nb de messages).
function nettoyer_historique($brut): array {
    if (!is_array($brut)) return [];
    $propre = [];
    foreach ($brut as $msg) {
        $role = $msg["role"] ?? "";
        $contenu = trim((string)($msg["content"] ?? ""));
        // On n'accepte que les deux roles legitimes d'une conversation.
        if (($role === "user" || $role === "assistant") && $contenu !== "") {
            // On tronque chaque message au cas ou (anti-abus de longueur).
            $propre[] = [
                "role" => $role,
                "content" => mb_substr($contenu, 0, MAX_LONGUEUR_QUESTION),
            ];
        }
    }
    // Fenetre glissante : on ne garde que les derniers messages.
    if (count($propre) > MAX_HISTORIQUE) {
        $propre = array_slice($propre, -MAX_HISTORIQUE);
    }
    return $propre;
}
$historique = nettoyer_historique($entree["history"] ?? []);


// =====================================================================
// 1bis. GARDE-FOUS — rate limiting par IP + cap quotidien global
// Verifies AVANT tout appel API (donc avant tout cout). L'etat est stocke
// dans des fichiers JSON, adapte a un hebergement mutualise sans base de donnees.
// =====================================================================

if (!is_dir(DOSSIER_DATA)) {
    @mkdir(DOSSIER_DATA, 0755, true);
}

/** IP du visiteur (hashee ensuite, pour ne pas stocker d'IP en clair — RGPD). */
function client_ip(): string {
    return $_SERVER["REMOTE_ADDR"] ?? "inconnu";
}

function lire_json(string $fichier): array {
    if (!file_exists($fichier)) return [];
    return json_decode(file_get_contents($fichier), true) ?: [];
}

function ecrire_json(string $fichier, array $donnees): void {
    file_put_contents($fichier, json_encode($donnees), LOCK_EX);
}

/** Limite le nombre de messages par IP sur une fenetre glissante. */
function rate_limit_ok(): bool {
    $fichier = DOSSIER_DATA . "/ratelimit.json";
    $maintenant = time();
    $hash_ip = hash("sha256", client_ip());
    $data = lire_json($fichier);

    // Purge des horodatages expires (toutes IP) -> le fichier reste compact.
    foreach ($data as $h => $stamps) {
        $stamps = array_filter($stamps, fn($t) => $t > $maintenant - RATE_LIMIT_FENETRE);
        if ($stamps) { $data[$h] = array_values($stamps); } else { unset($data[$h]); }
    }

    $stamps_ip = $data[$hash_ip] ?? [];
    if (count($stamps_ip) >= RATE_LIMIT_MAX) {
        return false;
    }
    $stamps_ip[] = $maintenant;
    $data[$hash_ip] = $stamps_ip;
    ecrire_json($fichier, $data);
    return true;
}

/** Compteur global de requetes, remis a zero chaque jour. */
function cap_quotidien_ok(): bool {
    $fichier = DOSSIER_DATA . "/daily.json";
    $aujourdhui = date("Y-m-d");
    $data = lire_json($fichier);
    if (($data["date"] ?? "") !== $aujourdhui) {
        $data = ["date" => $aujourdhui, "count" => 0];
    }
    if ($data["count"] >= CAP_QUOTIDIEN) {
        return false;
    }
    $data["count"]++;
    ecrire_json($fichier, $data);
    return true;
}

if (!rate_limit_ok()) {
    http_response_code(429);
    echo json_encode([
        "answer" => "Tu m'as envoye beaucoup de messages d'un coup ! Patiente quelques minutes avant de reessayer."
    ], JSON_UNESCAPED_UNICODE);
    exit;
}

if (!cap_quotidien_ok()) {
    http_response_code(503);
    echo json_encode([
        "answer" => "L'assistant a atteint sa limite de messages pour aujourd'hui. Reviens demain, ou ecris directement a Pol : contact@kypolab.com."
    ], JSON_UNESCAPED_UNICODE);
    exit;
}


// =====================================================================
// 2. RETRIEVAL — embedding de la question + cosinus
// =====================================================================

/** Similarite cosinus entre deux vecteurs (meme calcul qu'en Python). */
function cosinus(array $a, array $b): float {
    $produit = 0.0; $norme_a = 0.0; $norme_b = 0.0;
    foreach ($a as $i => $va) {
        $produit += $va * $b[$i];
        $norme_a += $va * $va;
        $norme_b += $b[$i] * $b[$i];
    }
    return $produit / (sqrt($norme_a) * sqrt($norme_b));
}

/**
 * Decoupe un texte en "tokens" (mots normalises) pour la recherche par mots-cles.
 * On passe en minuscules, on retire les accents et la ponctuation, et on jette
 * les mots-outils (le, la, de...) et les mots trop courts, peu discriminants.
 */
function tokeniser(string $texte): array {
    $texte = mb_strtolower($texte, "UTF-8");
    // Translitteration des accents (e accent -> e) pour matcher "experience" et "expérience".
    $texte = iconv("UTF-8", "ASCII//TRANSLIT//IGNORE", $texte) ?: $texte;
    // Tout ce qui n'est pas lettre/chiffre devient un separateur.
    $mots = preg_split('/[^a-z0-9]+/', $texte, -1, PREG_SPLIT_NO_EMPTY);
    $stopwords = [
        "le","la","les","un","une","des","de","du","et","ou","a","au","aux","en",
        "dans","sur","pour","par","avec","sans","que","qui","quoi","dont","est",
        "es","suis","sont","mon","ma","mes","ton","ta","tes","son","sa","ses",
        "ce","cet","cette","ces","tu","je","il","elle","on","nous","vous","se",
        "ne","pas","plus","tres","tu","quel","quels","quelle","quelles","comment",
    ];
    return array_values(array_filter($mots, function ($m) use ($stopwords) {
        return mb_strlen($m) > 2 && !in_array($m, $stopwords, true);
    }));
}

/**
 * Score BM25 de chaque chunk pour les tokens de la question.
 * Retourne un tableau [index_chunk => score]. Plus le score est haut, plus le
 * chunk contient les mots (rares) de la question.
 */
function bm25_scores(array $tokens_question, array $index): array {
    $k1 = 1.5; $b = 0.75; // parametres classiques de BM25
    $N = count($index);

    // Pre-tokenisation des chunks + longueurs.
    $docs = [];
    $longueurs = [];
    $somme_long = 0;
    foreach ($index as $i => $e) {
        $toks = tokeniser($e["text"]);
        $docs[$i] = array_count_values($toks); // {mot => nb d'occurrences}
        $longueurs[$i] = count($toks);
        $somme_long += $longueurs[$i];
    }
    $long_moyenne = $N > 0 ? $somme_long / $N : 0;

    // IDF : combien de chunks contiennent chaque mot de la question ?
    $idf = [];
    foreach (array_unique($tokens_question) as $mot) {
        $nb_docs_avec_mot = 0;
        foreach ($docs as $compte) {
            if (isset($compte[$mot])) $nb_docs_avec_mot++;
        }
        // Formule IDF de BM25 : un mot rare pese plus lourd.
        $idf[$mot] = log(1 + ($N - $nb_docs_avec_mot + 0.5) / ($nb_docs_avec_mot + 0.5));
    }

    $scores = [];
    foreach ($docs as $i => $compte) {
        $score = 0.0;
        foreach (array_unique($tokens_question) as $mot) {
            $f = $compte[$mot] ?? 0;            // frequence du mot dans ce chunk
            if ($f === 0) continue;
            $norm = 1 - $b + $b * ($longueurs[$i] / max($long_moyenne, 1));
            // Saturation (k1) + normalisation par longueur (b) + ponderation par raretE (idf).
            $score += $idf[$mot] * ($f * ($k1 + 1)) / ($f + $k1 * $norm);
        }
        $scores[$i] = $score;
    }
    return $scores;
}

/**
 * Fusion des deux classements (cosinus + BM25) par Reciprocal Rank Fusion.
 * On combine les RANGS, pas les scores (echelles incomparables). Un chunk bien
 * classe par l'une OU l'autre methode remonte. Retourne [index => score_rrf].
 */
function rrf_fusion(array $scores_cos, array $scores_bm25, int $k = 60): array {
    // Classement par cosinus (rang 1 = meilleur).
    arsort($scores_cos);
    $rang_cos = [];
    $r = 1; foreach ($scores_cos as $i => $_) { $rang_cos[$i] = $r++; }

    // Classement par BM25.
    arsort($scores_bm25);
    $rang_bm25 = [];
    $r = 1; foreach ($scores_bm25 as $i => $_) { $rang_bm25[$i] = $r++; }

    $rrf = [];
    foreach (array_keys($scores_cos) as $i) {
        $rrf[$i] = 1 / ($k + $rang_cos[$i]) + 1 / ($k + ($rang_bm25[$i] ?? PHP_INT_MAX));
    }
    arsort($rrf);
    return $rrf;
}

/**
 * Reranking par MMR (Maximal Marginal Relevance).
 * Parmi une liste de candidats, selectionne $k chunks en equilibrant :
 *   - la PERTINENCE par rapport a la question (cosinus question<->chunk)
 *   - la DIVERSITE : on penalise un chunk trop similaire a ceux deja choisis.
 * Evite de renvoyer plusieurs chunks quasi identiques. Retourne une liste d'index.
 *
 * @param int[]   $candidats      index des chunks candidats (deja pre-tries)
 * @param float[] $pertinence     [index => score de pertinence question<->chunk]
 * @param array   $index          l'index complet (pour acceder aux embeddings)
 */
function mmr_rerank(array $candidats, array $pertinence, array $index, int $k, float $lambda): array {
    $selection = [];
    $restants = $candidats;

    while (count($selection) < $k && $restants) {
        $meilleur = null;
        $meilleur_score = -INF;

        foreach ($restants as $i) {
            // Redondance = similarite max avec un chunk DEJA selectionne.
            $redondance = 0.0;
            foreach ($selection as $j) {
                $sim = cosinus($index[$i]["embedding"], $index[$j]["embedding"]);
                if ($sim > $redondance) $redondance = $sim;
            }
            // Score MMR : pertinence ponderee - redondance ponderee.
            $score = $lambda * $pertinence[$i] - (1 - $lambda) * $redondance;
            if ($score > $meilleur_score) {
                $meilleur_score = $score;
                $meilleur = $i;
            }
        }

        $selection[] = $meilleur;
        // On retire le chunk choisi des restants.
        $restants = array_values(array_filter($restants, fn($x) => $x !== $meilleur));
    }
    return $selection;
}

/** Appelle l'API Mistral pour vectoriser la question. */
function embed_question(string $question, string $cle): array {
    $ch = curl_init("https://api.mistral.ai/v1/embeddings");
    curl_setopt_array($ch, [
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_HTTPHEADER => [
            "Authorization: Bearer $cle",
            "Content-Type: application/json",
        ],
        CURLOPT_POSTFIELDS => json_encode([
            "model" => MODELE_EMBEDDING,
            "input" => [$question],
        ]),
    ]);
    $reponse = curl_exec($ch);
    return json_decode($reponse, true)["data"][0]["embedding"];
}

/**
 * Reformule une question de suivi en question AUTONOME, a l'aide de l'historique.
 * Ex : "Quelles technos dessus ?" + historique Sentinel-2 -> "Quelles technologies
 * as-tu utilisees sur le projet Sentinel-2 ?". Cette version sert UNIQUEMENT a la
 * recherche (embedding + BM25), pas a la generation. En cas d'echec, on renvoie la
 * question d'origine (jamais de plantage).
 */
function reformuler_question(string $question, array $historique, string $cle): string {
    // Resume compact de l'historique pour le prompt de reformulation.
    $fil = "";
    foreach ($historique as $msg) {
        $qui = $msg["role"] === "user" ? "Visiteur" : "Assistant";
        $fil .= "$qui : " . $msg["content"] . "\n";
    }

    $prompt = "Voici une conversation, puis une nouvelle question du visiteur.\n"
        . "Reecris cette derniere question en une SEULE phrase autonome et complete,\n"
        . "en remplaçant les pronoms/references (ça, dessus, celui-la...) par leur sens\n"
        . "explicite d'apres la conversation. Reponds UNIQUEMENT par la question reecrite.\n\n"
        . "CONVERSATION :\n$fil\nNOUVELLE QUESTION : $question";

    $ch = curl_init("https://api.mistral.ai/v1/chat/completions");
    curl_setopt_array($ch, [
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_HTTPHEADER => [
            "Authorization: Bearer $cle",
            "Content-Type: application/json",
        ],
        CURLOPT_POSTFIELDS => json_encode([
            "model" => MODELE_GENERATION,
            "max_tokens" => 60, // une phrase courte suffit -> cout minimal
            "temperature" => 0, // deterministe : on veut une reformulation fidele
            "messages" => [["role" => "user", "content" => $prompt]],
        ]),
    ]);
    $reponse = json_decode(curl_exec($ch), true);
    $reecrite = trim($reponse["choices"][0]["message"]["content"] ?? "");
    // Repli de securite : si vide ou anormalement long, on garde l'original.
    return ($reecrite !== "" && mb_strlen($reecrite) <= MAX_LONGUEUR_QUESTION)
        ? $reecrite : $question;
}

// --- Query rewriting : si la conversation a un historique, la question peut etre
// une question de suivi ("et dessus ?"). On la reformule en question autonome
// AVANT de chercher, pour que le retrieval dispose des bons mots-cles.
// Sans historique, la question est deja autonome -> on saute l'etape (economie).
$question_recherche = $historique
    ? reformuler_question($question, $historique, $cle_mistral)
    : $question;

// On vectorise la question de RECHERCHE (reformulee le cas echeant).
$vecteur_question = embed_question($question_recherche, $cle_mistral);

// On charge l'index pre-calcule.
$index = json_decode(file_get_contents(__DIR__ . "/embeddings.json"), true);

// --- Recherche HYBRIDE : on score chaque chunk de deux façons...
// 1) Semantique : similarite cosinus question <-> chunk (capte le SENS).
$scores_cos = [];
foreach ($index as $i => $entree_index) {
    $scores_cos[$i] = cosinus($vecteur_question, $entree_index["embedding"]);
}
// 2) Mots-cles : BM25 question <-> chunk (capte les TERMES EXACTS, noms propres).
$scores_bm25 = bm25_scores(tokeniser($question_recherche), $index);

// ...puis on FUSIONNE les deux classements (Reciprocal Rank Fusion).
$scores = rrf_fusion($scores_cos, $scores_bm25);

// --- RERANKING en deux temps :
// 1) On recupere LARGE : les NB_CANDIDATS meilleurs de la fusion (filet large).
$candidats = array_keys(array_slice($scores, 0, NB_CANDIDATS, true));
// 2) On re-trie FIN avec MMR : garde NB_CHUNKS chunks pertinents ET diversifies
//    (pertinence = cosinus question<->chunk, deja calcule dans $scores_cos).
$meilleurs = mmr_rerank($candidats, $scores_cos, $index, NB_CHUNKS, LAMBDA_MMR);

// Chaque chunk est prefixe de sa CATEGORIE (metadonnee) : le modele sait ainsi
// de quelle rubrique vient chaque info et peut la citer naturellement.
$contexte = "";
foreach ($meilleurs as $i) {
    $cat = $index[$i]["category"] ?? "Général";
    $contexte .= "- (" . $cat . ") " . $index[$i]["text"] . "\n";
}


// =====================================================================
// 3. GENERATION — appel au modele de chat Mistral
// =====================================================================

$message_utilisateur = "CONTEXTE :\n$contexte\n\nQUESTION DU VISITEUR : $question";

// On assemble la liste de messages envoyee au modele :
//   [ system ] + [ historique des echanges precedents ] + [ question courante ]
// L'historique donne au modele le fil de la conversation (il comprend ainsi les
// questions de suivi type "et celui-la ?"). Seule la question COURANTE est
// enrichie du contexte RAG ; les anciens messages restent tels quels.
$messages = [["role" => "system", "content" => SYSTEM_PROMPT]];
foreach ($historique as $msg) {
    $messages[] = $msg;
}
$messages[] = ["role" => "user", "content" => $message_utilisateur];

// --- STREAMING : on bascule la reponse HTTP en flux texte (plus de JSON ici).
// On desactive tout buffering pour que les fragments partent immediatement.
header("Content-Type: text/plain; charset=utf-8");
header("X-Accel-Buffering: no"); // demande a nginx de ne pas bufferiser
header("Cache-Control: no-cache");
while (ob_get_level() > 0) { ob_end_flush(); }

$buffer = "";          // accumule les fragments SSE (gere les lignes coupees)
$texte_reponse = "";   // reponse complete (pour log/diagnostic eventuel)

$ch = curl_init("https://api.mistral.ai/v1/chat/completions");
curl_setopt_array($ch, [
    CURLOPT_HTTPHEADER => [
        "Authorization: Bearer $cle_mistral",
        "Content-Type: application/json",
        "Accept: text/event-stream",
    ],
    CURLOPT_POSTFIELDS => json_encode([
        "model" => MODELE_GENERATION,
        "max_tokens" => MAX_TOKENS,
        "messages" => $messages,
        "stream" => true, // <-- Mistral renvoie un flux SSE token par token
    ]),
    // Appelee a CHAQUE fragment reseau recu depuis Mistral.
    CURLOPT_WRITEFUNCTION => function ($ch, $fragment) use (&$buffer, &$texte_reponse) {
        $buffer .= $fragment;
        // On traite ligne complete par ligne complete (les fragments peuvent
        // couper une ligne SSE en deux -> on garde le reste dans $buffer).
        while (($pos = strpos($buffer, "\n")) !== false) {
            $ligne = trim(substr($buffer, 0, $pos));
            $buffer = substr($buffer, $pos + 1);
            if ($ligne === "" || strncmp($ligne, "data:", 5) !== 0) continue;
            $charge = trim(substr($ligne, 5));
            if ($charge === "[DONE]") continue;
            $j = json_decode($charge, true);
            $delta = $j["choices"][0]["delta"]["content"] ?? "";
            if ($delta !== "") {
                echo $delta;
                @ob_flush(); @flush(); // pousse le texte vers le navigateur tout de suite
                $texte_reponse .= $delta;
            }
        }
        return strlen($fragment); // obligatoire : indique a cURL qu'on a tout consomme
    },
]);
curl_exec($ch);

// Repli : si rien n'a ete streame (erreur amont), on envoie un message minimal.
if ($texte_reponse === "") {
    echo "Desole, une erreur est survenue.";
}
