import json
import re
import time
from llmQuery import LLMClientChat

class PromptProcessor:
    def __init__(self, doubt_phrase=" Wait, wait... "):
        self.doubt_phrase = doubt_phrase
        self.conclusion_patterns = [r"\b(en conclusion|donc|ainsi|la réponse est|il est clair que)\b",
                                    r"\b(on peut en déduire|en résumé|cela signifie que)\b"]    

    def inject_doubt(self, text):
        return text + self.doubt_phrase + "\n"

    def count_sentences(self, text):
        return len(re.findall(r"[.!?]", text))

    def is_conclusion_detected(self, text):
        return any(re.search(pat, text, re.IGNORECASE) for pat in self.conclusion_patterns)

def prompt_to_messages(prompt):
    messages = [{"role": "user", "content": prompt}]
    print(messages)
    return messages

class ChainOfThoughtRunner:
    def __init__(self, client, processor, max_rounds=5):
        self.client = client
        self.processor = processor
        self.max_rounds = max_rounds

    def initial_direct_response(self, prompt, max_tokens=300):
        print("\n=== Réponse directe initiale (sans doute) ===\n")
        tokens = []
        for token in self.client.stream_text_tokens(prompt_to_messages(prompt), max_tokens, temperature=0.7):
            print(token, end='', flush=True)
            tokens.append(token)
        return ''.join(tokens).strip()

    def generate_internal_thoughts(self, initial_prompt):
        full_context = ""
        for i in range(self.max_rounds):
            print(f"\n--- Génération round {i + 1} ---\n")
            if full_context == "":
                chunk = self._stream_until_doubt(prompt_to_messages(initial_prompt)) 
            else:
                chunk = self._stream_until_doubt(prompt_to_messages(initial_prompt) +
                                                  [{"role": "assistant", "content": full_context}])
            full_context += chunk
            print(full_context)
            full_context = self.processor.inject_doubt(full_context)
            time.sleep(0.3)
        return full_context

    def _stream_direct(self, context, max_tokens=150, temperature=0.4):
        output = ""
        for token in self.client.stream_text_tokens(context, max_tokens, temperature):
            print(token, end='', flush=True)
            output += token
        return output

    def _stream_until_doubt(self, context, max_tokens=150):
        output = ""
        sentence_count = 0

        for token in self.client.stream_text_tokens(context, max_tokens, temperature=0.4):
            print(token, end='', flush=True)
            output += token

            sentence_count += self.processor.count_sentences(token)

            # Conclusion anticipée (si implémentée)
            if self.processor.is_conclusion_detected(token):
                print("\n[Conclusion détectée, arrêt anticipé.]\n")
                break

            # Injection de doute après 2 phrases
            if sentence_count >= 2:
                print("\n[Injection d’un doute...]\n")
                break

        return output.strip()

    def run(self, prompt):
        initial = self.initial_direct_response(prompt)
        context = self.generate_internal_thoughts(prompt)
        print("\n-------\nVoici la réflexion totale:\n")
        print("\n-------\nÀ comparer avec la version sans réflexion:\n")
        print(initial)

if __name__ == "__main__":
    client = LLMClientChat()
    processor = PromptProcessor()
    runner = ChainOfThoughtRunner(client, processor)

    # Ici le prompt n'est pas la liste des messages
    # les messages sont construits dans les différentes fonctions
    prompt = "What was the name of the scientist who discovered the tuberculosis vaccine? "
    runner.run(prompt)