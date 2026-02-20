import warnings
warnings.filterwarnings("ignore")

import nltk
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class FAQChatbot:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

        # FAQ dataset
        self.faq_questions = [
            "Hello",
            "What is your name?",
            "What is artificial intelligence?",
            "What is AI?",
            "What is machine learning?",
            "What is ML?",
            "What are your working hours?",
            "What payment methods do you accept?",
            "How do I reset my password?",
            "Bye"
        ]

        self.faq_answers = [
            "Hi there! How can I help you?",
            "I'm your FAQ Chatbot.",
            "Artificial Intelligence is the simulation of human intelligence in machines.",
            "AI stands for Artificial Intelligence, which enables machines to think and learn.",
            "Machine Learning is a subset of AI that allows systems to learn from data.",
            "ML stands for Machine Learning, a field of AI focused on learning from data.",
            "Our working hours are 9 AM to 6 PM, Monday to Friday.",
            "We accept credit cards, debit cards, and PayPal.",
            "To reset your password, go to settings and click 'Reset Password'.",
            "Goodbye! Have a great day."
        ]

        # TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(tokenizer=self.preprocess_text)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.faq_questions)

        # Short keyword responses
        self.short_forms = {
            "ai": "AI stands for Artificial Intelligence, which enables machines to think and learn.",
            "ml": "ML stands for Machine Learning, a subset of AI that learns from data.",
            "dl": "DL stands for Deep Learning, which uses neural networks."
        }

    def preprocess_text(self, text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words]
        return tokens

    def get_response(self, user_query):
        cleaned_query = user_query.lower().strip()

        # Handle short forms
        if cleaned_query in self.short_forms:
            return self.short_forms[cleaned_query]

        processed_query = " ".join(self.preprocess_text(user_query))
        user_vector = self.vectorizer.transform([processed_query])

        similarities = cosine_similarity(user_vector, self.tfidf_matrix)
        best_match_index = np.argmax(similarities)

        if similarities[0][best_match_index] < 0.15:
            return "Sorry, I couldn't understand your question. Please try again."

        return self.faq_answers[best_match_index]

    def start_chat(self):
        print("🤖 FAQ Chatbot is running (type 'exit' to stop)")
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                print("Chatbot: Goodbye! 👋")
                break

            response = self.get_response(user_input)
            print("Chatbot:", response)


if __name__ == "__main__":
    chatbot = FAQChatbot()
    chatbot.start_chat()
