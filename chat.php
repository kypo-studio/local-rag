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

const NB_CHUNKS = 4;        // top-k : nb de chunks injectes dans le prompt
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

// On vectorise la question.
$vecteur_question = embed_question($question, $cle_mistral);

// On charge l'index pre-calcule et on score chaque chunk.
$index = json_decode(file_get_contents(__DIR__ . "/embeddings.json"), true);
$scores = [];
foreach ($index as $i => $entree_index) {
    $scores[$i] = cosinus($vecteur_question, $entree_index["embedding"]);
}
arsort($scores); // tri decroissant en gardant les cles (index)

// On recupere le texte des NB_CHUNKS meilleurs.
$meilleurs = array_slice($scores, 0, NB_CHUNKS, true);
$contexte = "";
foreach ($meilleurs as $i => $score) {
    $contexte .= "- " . $index[$i]["text"] . "\n";
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

$ch = curl_init("https://api.mistral.ai/v1/chat/completions");
curl_setopt_array($ch, [
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_HTTPHEADER => [
        "Authorization: Bearer $cle_mistral",
        "Content-Type: application/json",
    ],
    CURLOPT_POSTFIELDS => json_encode([
        "model" => MODELE_GENERATION,
        "max_tokens" => MAX_TOKENS,
        "messages" => $messages,
    ]),
]);
$reponse_brute = curl_exec($ch);

$reponse = json_decode($reponse_brute, true);
$texte_reponse = $reponse["choices"][0]["message"]["content"]
    ?? "Desole, une erreur est survenue.";


// =====================================================================
// 4. SORTIE — on renvoie la reponse au widget
// =====================================================================

echo json_encode(["answer" => $texte_reponse], JSON_UNESCAPED_UNICODE);
