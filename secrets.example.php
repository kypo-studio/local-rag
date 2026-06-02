<?php
/**
 * MODELE — a copier en "secrets.php" SUR LE SERVEUR uniquement.
 *
 * Sert a fournir la cle Mistral a chat.php quand l'hebergeur ne permet pas
 * de definir une variable d'environnement. chat.php lit d'abord
 * getenv("MISTRAL_API_KEY"), puis se rabat sur ce fichier.
 *
 * secrets.php est ignore par Git (.gitignore) : la cle ne part jamais en ligne
 * via le depot. Apache ne sert jamais les fichiers .php en clair (il les execute),
 * donc la cle n'est pas exposee.
 */

return "ta_cle_mistral_ici";
