import re
import wikipedia
from llmQuery import *

### Classe spécifique pour ici, avec une option permettant de stopper la réponse dès qu'il y a "Observation:"
### dans la sortie
class LLMClientChatEarlyStopping(LLMClientChat):
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
                "stream": True,                  # Active le mode streaming
                "stop": ["Observation:"]       # permet de stopper la réponse 
            },
            verify=False,                       # Ignore les erreurs de certificat SSL
            stream=True                         # Permet de recevoir les données en continu
        )
        return response.iter_lines()  # Retourne un itérateur sur les lignes de la réponse

class WikipediaAPIWrapper:
    def __init__(self, lang="en"):
        self.lang = lang
        wikipedia.set_lang(lang)

    def run(self, query: str) -> str:
        try:
            # Récupère les 1ers résultats liés à la requête
            search_results = wikipedia.search(query)
            if not search_results:
                return "No results found on Wikipédia."

            # Prend le 1er résultat, récupère la page
            page = wikipedia.page(search_results[0])
            summary = page.summary
            return summary

        except Exception as e:
            return f"Erreur lors de la recherche Wikipédia : {e}"
      
# Initialiser l'outil Wikipedia
wikiwrapper = WikipediaAPIWrapper(lang="en")

# Prompt de base
def build_base_prompt(question: str):
    return f"""Answer the following questions as best you can.
You have access to the following tools:

Wikipedia(query: str) -> str - Useful for obtaining encyclopaedic information on well-known people, places or concepts. This tool needs precise queries, like names of people, places or concepts. It will return a summary of the Wikipedia page.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [Wikipedia]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {question}
"""


# Fonction principale
def run_agent(question: str):
    client = LLMClientChatEarlyStopping()
    prompt = build_base_prompt(question)

    iterations = 0
    while True:
        iterations += 1
        response = client.stream_direct([{"role": "user", "content": prompt}], max_tokens=500)
        print(f"\nRéponse du LLM:\n{response}")

        # Ajoute la réponse dans le prompt
        prompt += response + "\n"
        print(f"\n  {iterations} / prompt:\n{prompt}")

        # Cherche s’il a proposé une action
        action_match = re.search(r"Action:\s*(.*)\nAction Input:\s*(.*)", response)
        if action_match:
            action = action_match.group(1).strip()
            action_input = action_match.group(2).strip()
            print(f"\nAction détectée : {action}('{action_input}')")

            if action == "Wikipedia":
                obs = wikiwrapper.run(action_input)
            else:
                obs = f"Outil inconnu : {action}"

            print(f"\n Observation : {obs[:300]}...")  # tronque pour pas tout afficher

            # Injecte l'observation et relance
            prompt += f"Observation: {obs}\nThought: "
        elif "Final Answer:" in response:
            print("L'agent a donné une réponse finale.")
            break
        else:
            print("️Pas d'action détectée, boucle arrêtée.")
            break

run_agent("What was the name of the scientist who discovered the tuberculosis vaccine?")