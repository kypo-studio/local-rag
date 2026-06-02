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

// La cle est lue dans l'environnement du serveur (jamais en dur ici).
$cle_mistral = getenv("MISTRAL_API_KEY");

// Le system prompt : il definit le ROLE du bot. Defensif : interdit d'inventer.
const SYSTEM_PROMPT = <<<PROMPT
Tu es l'assistant personnel de Pol Quimerc'h, sur son site portfolio.
Tu reponds aux visiteurs (souvent des recruteurs) a la PREMIERE PERSONNE,
au nom de Pol ("je", "mon parcours"...), de façon chaleureuse et concise.

Tu ne dois utiliser QUE les informations fournies dans le CONTEXTE ci-dessous.
Si l'information demandee n'y figure pas, dis simplement que tu n'as pas cette
information et invite a contacter Pol directement. N'invente JAMAIS de faits.
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
    curl_close($ch);
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
        "messages" => [
            ["role" => "system", "content" => SYSTEM_PROMPT],
            ["role" => "user", "content" => $message_utilisateur],
        ],
    ]),
]);
$reponse_brute = curl_exec($ch);
curl_close($ch);

$reponse = json_decode($reponse_brute, true);
$texte_reponse = $reponse["choices"][0]["message"]["content"]
    ?? "Desole, une erreur est survenue.";


// =====================================================================
// 4. SORTIE — on renvoie la reponse au widget
// =====================================================================

echo json_encode(["answer" => $texte_reponse], JSON_UNESCAPED_UNICODE);
