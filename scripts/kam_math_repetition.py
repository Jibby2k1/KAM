"""Fixed diagnostic for repeated text at a preregistered generation prefix."""
import re

WINDOW = 144
NGRAM = 6
RATE = 0.75


def repetitive(text):
    if text is None:
        return False
    tokens = re.findall(r"\w+|[^\w\s]", text.casefold())[-WINDOW:]
    if len(tokens) < WINDOW:
        return False
    grams = {tuple(tokens[i:i+NGRAM]) for i in range(len(tokens) - NGRAM + 1)}
    return 1 - len(grams) / (len(tokens) - NGRAM + 1) >= RATE
