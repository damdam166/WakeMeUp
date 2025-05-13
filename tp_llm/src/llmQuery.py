import requests
import urllib3
import json
import argparse


#############
#BASE_URL = "http://localhost:1234/v1"  # URL de l'API LMStudio
BASE_URL = "https://51.91.251.201:8234/v1"
SECRET = ""  # Token d'authentification (A demander à votre chargé de TD)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
#############

assert SECRET != "", "Le token d'authentification (SECRET) doit être renseigné."  # Vérifie que le token est renseigné

# Configuration des paramètres de la requête (Vous pouvez les modifier)
MAX_TOKENS = 1500 
TEMPERATURE = 0.7  # Température pour la génération de texte

def get_models():
    # Envoie une requête GET à l'API pour obtenir la liste des modèles disponibles
    response = requests.get(
        BASE_URL + "/models",
        headers={
            "Content-Type": "application/json",  # Type de contenu de la requête
            "Authorization": "Bearer " + SECRET
        },
        verify=False  # Ignore les erreurs de certificat SSL
    )
    return response.json()  # Retourne la réponse JSON

# Affiche les modèles disponibles

# Classe à utiliser pour le modele BASE (Non INSTRUCT)
# Ce modèle n'a pas été fine-tuné pour le dialogue
class LLMClientCompletions:
    def __init__(self, url=BASE_URL + "/completions"):
        # Initialisation du client avec l'URL de l'API
        self.url = url

    def stream_response(self, prompt, max_tokens=MAX_TOKENS, temperature=TEMPERATURE):
        # Envoie une requête POST à l'API pour obtenir une réponse en streaming
        response = requests.post(
            self.url,
            headers={
                "Content-Type": "application/json",  # Type de contenu de la requête
                "Authorization": "Bearer " + SECRET
            },
            json={
                "prompt": prompt,               # Le texte de départ pour le modèle
                "max_tokens": max_tokens,       # Nombre maximum de tokens à générer
                "temperature": temperature,     # Niveau de créativité du modèle
                "stream": True                  # Active le mode streaming
            },
            verify=False,                       # Ignore les erreurs de certificat SSL
            stream=True                         # Permet de recevoir les données en continu
        )
        return response.iter_lines()  # Retourne un itérateur sur les lignes de la réponse

    def stream_text_tokens(self, prompt, max_tokens, temperature):
        # Traite les réponses en streaming et extrait les tokens de texte
        for line in self.stream_response(prompt, max_tokens, temperature):
            if line:
                decoded = line.decode("utf-8").strip()  # Décodage de la ligne en UTF-8
                if decoded == "data: [DONE]":
                    break  # Arrête le traitement si la réponse est terminée
                if decoded.startswith("data: "):
                    try:
                        # Extrait le texte du token depuis la réponse JSON
                        token = json.loads(decoded[6:])["choices"][0]["text"]
                        yield token  # Retourne le token
                    except json.JSONDecodeError:
                        continue  # Ignore les erreurs de décodage JSON

    def stream_direct(self, prompt, max_tokens=150, temperature=0.4):
        # Gère le streaming direct et affiche les tokens au fur et à mesure
        output = ""
        for token in self.stream_text_tokens(prompt, max_tokens, temperature):
            print(token, end='', flush=True)  # Affiche le token sans saut de ligne
            output += token  # Ajoute le token à la sortie finale
        return output  # Retourne la sortie complète


# classe à utiliser pour le modèle INSTRUCT 
# 
class LLMClientChat(LLMClientCompletions):
    def __init__(self, url=BASE_URL + "/chat/completions"):
        # Initialisation du client avec l'URL de l'API de chat
        super().__init__(url)  # Appelle le constructeur de la classe parente

    def stream_response(self, messages = [], max_tokens=MAX_TOKENS, temperature=TEMPERATURE):
        # messages : liste de messages au format [{"role": "user", "content": "message"}, ..]
        # Envoie une requête POST à l'API de chat pour obtenir une réponse en streaming
        response = requests.post(
            self.url,
            headers={
                "Content-Type": "application/json",  # Type de contenu de la requête
                "Authorization": "Bearer " + SECRET
            },
            json={
                "messages": messages,
                "max_tokens": max_tokens,       # Nombre maximum de tokens à générer
                "temperature": temperature,     # Niveau de randomisation du modèle
                "stream": True                  # Active le mode streaming
            },
            verify=False,                       # Ignore les erreurs de certificat SSL
            stream=True                         # Permet de recevoir les données en continu
        )
        return response.iter_lines()  # Retourne un itérateur sur les lignes de la réponse
    
    def stream_text_tokens(self, messages, max_tokens, temperature):
        for line in self.stream_response(messages, max_tokens, temperature):
            if line:
                decoded = line.decode("utf-8").strip()
                if decoded == "data: [DONE]":
                    break
                if decoded.startswith("data: "):
                    json_str = decoded[len("data: "):]  # Enlève le préfixe une seule fois
                    try:
                        data = json.loads(json_str)
                        token = data["choices"][0]["delta"].get("content", "")
                        if token:
                            yield token
                    except json.JSONDecodeError as e:
                        print(f"Erreur JSON : {e}")
                        continue 

    # Gère le streaming direct et affiche les tokens au fur et à mesure
    # de la réponse du modèle
    # Si vous souhaitez que le modèle n'affiche rien et ne garder que la réponse
    # c'est ici qu'il faut le faire
    def stream_direct(self, messages, max_tokens=150, temperature=0.4):
        output = ""
        for token in self.stream_text_tokens(messages, max_tokens, temperature):
            print(token, end='', flush=True)
            output += token
        return output

# Exemple d'utilisation 
# Ensuite, vous pourrez importer ce fichier et faire la suite du TP
# en important les bonnes classes, puis en ajoutant vos propres classes
# héritées de LLMClientCompletions ou LLMClientChat pour intercepter les tokens et interagir plus
# finement avec le modèle.
if __name__ == "__main__":
    # Configuration de l'analyseur d'arguments pour la ligne de commande
    parser = argparse.ArgumentParser(description="Stream text completions from LMStudio.")
    parser.add_argument("--list", action="store_true", help="List available models.")  # Argument pour lister les modèles
    parser.add_argument("prompt", type=str, help="The prompt to send to the language model.")  # Argument pour le prompt
    args = parser.parse_args()  # Analyse les arguments fournis

    if args.list:
        print("Modèles disponibles :")
        models = get_models()["data"]
        for model in models:
            print(f" - {model['id']}")  # Affiche les modèles disponibles
        print("Mais attention, dans ce TP, un seul modèle est disponible par route (/completions ou /chat/completions)")
        print("Ne précisez pas le nom du modèle, il sera pris par défaut suivant la route.")

    # Création d'une instance du client et exécution du streaming
    client = LLMClientCompletions()
    sortie = client.stream_direct(args.prompt)
    
    print("\n\n=== Résultat final  (reprend le prompt et la réponse) ===\n")
    print(args.prompt + sortie)
