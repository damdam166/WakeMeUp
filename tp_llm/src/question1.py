from llmQuery import *

# idée : on utilise un modele non fine tuné mais dans un contexte de dialogue
# ou le reste de l'histoire est une conversation entre le professeur et l'élève
# la logique veut que le dialogue se poursuive suivant les mêmes règles.

prompt = '''Pourquoi le ciel est-il bleu ?'''

client = LLMClientCompletions()
reponse = client.stream_direct(prompt, temperature=0.5, max_tokens=50)
print()
print("*"*20)
print(reponse)

