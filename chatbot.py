import json
import random
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

# Load intents
with open("intents.json") as file:
    data = json.load(file)

# Prepare training data
corpus = []
tags = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        corpus.append(pattern.lower())
        tags.append(intent["tag"])

# Vectorize the patterns
vectorizer = CountVectorizer().fit_transform(corpus)

def chatbot_response(user_input):
    user_input = user_input.lower()
    input_vec = CountVectorizer().fit(corpus + [user_input])
    corpus_with_input = corpus + [user_input]
    vectors = input_vec.transform(corpus_with_input)

    cosine_sim = cosine_similarity(vectors[-1], vectors[:-1])
    index = cosine_sim.argmax()

    if cosine_sim[0][index] < 0.3:
        return "Sorry, I don't understand."

    response_tag = tags[index]
    for intent in data["intents"]:
        if intent["tag"] == response_tag:
            return random.choice(intent["responses"])

# Chat loop
print("ChatBot: Hello! Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("ChatBot: Goodbye!")
        break
    response = chatbot_response(user_input)
    print("ChatBot:", response)
