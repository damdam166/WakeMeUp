from llmQuery import *


questions = [
  {
    "question": "Jean a 24 billes. Il les partage également entre 4 amis. Combien chaque ami reçoit-il de billes ?",
    "reponse": "6",
    "niveau": "primaire"
  },
  {
    "question": "Combien font 3/4 de 20 ?",
    "reponse": "15",
    "niveau": "primaire"
  },
  {
    "question": "Si 5x = 45, quelle est la valeur de x ?",
    "reponse": "9",
    "niveau": "collège"
  },
  {
    "question": "Quel est le périmètre d’un rectangle de 8 cm de long et 3 cm de large ?",
    "reponse": "22 cm",
    "niveau": "collège"
  },
  {
    "question": "Une voiture parcourt 120 km en 2 heures. Quelle est sa vitesse moyenne ?",
    "reponse": "60 km/h",
    "niveau": "collège"
  },
  {
    "question": "Combien de combinaisons de 2 lettres peut-on faire avec A, B et C (sans répétition) ?",
    "reponse": "6",
    "niveau": "lycée"
  },
  {
    "question": "Quelle est la dérivée de f(x) = 3x² ?",
    "reponse": "f'(x) = 6x",
    "niveau": "lycée"
  },
  {
    "question": "Quelle est la solution de l’équation quadratique x² - 5x + 6 = 0 ?",
    "reponse": "x = 2 ou x = 3",
    "niveau": "lycée"
  },
  {
    "question": "Une boîte contient 5 boules rouges et 3 boules bleues. Quelle est la probabilité de tirer une boule bleue ?",
    "reponse": "3/8",
    "niveau": "lycée"
  },
  {
    "question": "Simplifie l’expression : (2x + 3x) - (x - 4)",
    "reponse": "4x + 4",
    "niveau": "collège"
  },

  {
    "question": "Si j'ai 3 paquets de 5 bonbons chacun et que je donne 4 bonbons à mon ami, combien me reste-t-il de bonbons ?",
    "reponse": "11",
    "niveau": "primaire"
  },
  {
    "question": "Résoudre l'équation suivante : 2x + 3 = 11",
    "reponse": "x = 4",
    "niveau": "collège"
  },
  {
    "question": "Un triangle a des côtés de longueurs 5 cm, 12 cm et 13 cm. Ce triangle est-il rectangle ?",
    "reponse": "Oui, car 5² + 12² = 13²",
    "niveau": "lycée"
  },
  {
    "question": "Une pièce équilibrée est lancée deux fois. Quelle est la probabilité d'obtenir exactement un pile ?",
    "reponse": "0.5",
    "niveau": "lycée"
  },
  {
    "question": "Tous les chats sont des animaux. Certains animaux sont des chiens. Peut-on en déduire que certains chats sont des chiens ?",
    "reponse": "Non",
    "niveau": "collège"
  },
  {
    "question": "La moyenne des nombres 4, 8, 6, 10 et 12 est :",
    "reponse": "8",
    "niveau": "lycée"
  },
  {
    "question": "Si aujourd'hui est mercredi, quel jour sera-t-il dans 10 jours ?",
    "reponse": "Samedi",
    "niveau": "primaire"
  },
  {
    "question": "Si tous les roses sont des fleurs et certaines fleurs fanent rapidement, peut-on en déduire que certains roses fanent rapidement ?",
    "reponse": "Non",
    "niveau": "collège"
  },
  {
    "question": "Combien de mètres y a-t-il dans 3 kilomètres ?",
    "reponse": "3000",
    "niveau": "primaire"
  },
  {
    "question": "Quel est le résultat de 7 × 8 ?",
    "reponse": "56",
    "niveau": "primaire"
  }
]

client = LLMClientChat()

def build_prompt(question):
    return [{"role" : "assistant", 
            "content" : '''Tu es un assistant qui doit répondre à des questions . 
            Tu dois répondre directement un nombre sans autre explications.'''},
            {"role" : "user", 
             "content" : question}]

sorties = []
for question in questions:
  print(question, end=' :')
  sorties.append(client.stream_direct(build_prompt(question["question"]), max_tokens=3000, temperature=0.2))
  print()

print("*"*20)

for i in range(len(sorties)):
  print(f"Question {i+1} : {questions[i]['question']}")
  print(f"Réponse LLM : {sorties[i]}")
  print(f"Réponse attendue : {questions[i]['reponse']}")
  print(f"Niveau : {questions[i]['niveau']}")
  print("-"*20)
print()
