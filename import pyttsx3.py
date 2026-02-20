import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

# ---------------- NLTK downloads ----------------
nltk.download('punkt')
nltk.download('stopwords')

class FAQChatbot:
    def __init__(self):
        # ---------------- FAQ Database ----------------
        self.faqs = [
            {"question": "What is your product warranty?", "answer": "Our product has a 1-year warranty from the date of purchase."},
            {"question": "How can I reset my password?", "answer": "You can reset your password by clicking 'Forgot Password' on the login page."},
            {"question": "Do you provide international shipping?", "answer": "Yes, we ship to most countries worldwide with additional shipping charges."},
            {"question": "What are the payment options?", "answer": "We accept credit cards, debit cards, and PayPal payments."},
            {"question": "How do I contact customer support?", "answer": "You can contact our support team via email or the contact form on our website."},
            {"question": "Can I track my order?", "answer": "Yes, you can track your order using the tracking link sent to your email after shipment."},
            {"question": "Is there a return policy?", "answer": "Yes, you can return products within 30 days of purchase in original packaging."}
        ]

        # ---------------- Text Preprocessing ----------------
        self.stop_words = set(stopwords.words('english'))
        self.questions = [self.preprocess(faq["question"]) for faq in self.faqs]

        # ---------------- GUI Setup ----------------
        self.root = tk.Tk()
        self.root.title("💬 FAQ Chatbot")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        self.root.configure(bg="#f5f6fa")

        self.ui()
        self.root.mainloop()

    # ---------------- Preprocessing ----------------
    def preprocess(self, text):
        text = text.lower()  # lowercase
        text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
        tokens = word_tokenize(text)
        filtered = [word for word in tokens if word not in self.stop_words]  # remove stopwords
        return ' '.join(filtered)

    # ---------------- Get Best Matching Answer ----------------
    def get_answer(self, user_input):
        user_input_processed = self.preprocess(user_input)
        corpus = self.questions + [user_input_processed]  # FAQs + user input

        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(corpus)

        similarity = cosine_similarity(tfidf[-1], tfidf[:-1])
        best_idx = similarity.argmax()

        if similarity[0, best_idx] < 0.1:  # threshold for unmatched questions
            return "Sorry, I don't have an answer to that question."
        else:
            return self.faqs[best_idx]["answer"]

    # ---------------- GUI ----------------
    def ui(self):
        tk.Label(self.root, text="FAQ Chatbot", font=("Helvetica", 16, "bold"), bg="#f5f6fa").pack(pady=10)

        # Chat display area
        self.chat_area = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, state='disabled', width=70, height=20,
            font=("Helvetica", 11), bg="#ffffff", padx=10, pady=10
        )
        self.chat_area.pack(pady=10)

        # Entry and send button
        frame = tk.Frame(self.root, bg="#f5f6fa")
        frame.pack(pady=5)

        self.entry = tk.Entry(frame, width=50, font=("Helvetica", 12))
        self.entry.grid(row=0, column=0, padx=5)
        self.entry.bind("<Return>", self.send_message)

        self.send_btn = tk.Button(
            frame, text="Send", width=12, bg="#4cd137", fg="white",
            font=("Helvetica", 11, "bold"), command=self.send_message, cursor="hand2"
        )
        self.send_btn.grid(row=0, column=1, padx=5)

    # ---------------- Send Message ----------------
    def send_message(self, event=None):
        user_msg = self.entry.get().strip()
        if not user_msg:
            return

        # Display user message
        self.chat_area.configure(state='normal')
        self.chat_area.insert(tk.END, f"You: {user_msg}\n")
        self.entry.delete(0, tk.END)

        # Get chatbot answer
        answer = self.get_answer(user_msg)
        self.chat_area.insert(tk.END, f"Bot: {answer}\n\n")
        self.chat_area.configure(state='disabled')
        self.chat_area.yview(tk.END)  # auto-scroll


# ---------------- Run Chatbot ----------------
if __name__ == "__main__":
    FAQChatbot()
