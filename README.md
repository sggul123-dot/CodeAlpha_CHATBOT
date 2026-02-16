# CodeAlpha — FAQ Chatbot 🤖

**Internship Project:** Artificial Intelligence Tasks & Instructions — CodeAlpha
## Internship Overview

This project is part of the **CodeAlpha AI Internship**, which provides hands-on experience in **Artificial Intelligence** and **Machine Learning**. Interns work on real-world AI projects with mentorship and contribute to innovative solutions.

This repository contains the implementation for **Task 2: Chatbot for FAQs**.
# Project Description

The **FAQ Chatbot** answers common questions related to a topic or product. It uses **Natural Language Processing (NLP)** techniques to understand user queries and provide accurate answers.

**Key highlights:**

* Answers predefined FAQ questions.
* Recognizes short forms such as `AI`, `ML`, and `DL`.
* Provides a default response when the question is not recognized.

---

## Features

* Responds to FAQs with accurate answers.
* Handles abbreviations and short forms.
* Uses **TF-IDF** vectorization and **cosine similarity** for question matching.
* Preprocesses text (lowercase, punctuation removal, stopwords) using NLTK.
* Terminal-based chatbot for easy testing.

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/SGgul123_git/CodeAlpha_FAQChatbot.git
```

2. **Navigate to the project folder:**

```bash
cd CodeAlpha_FAQChatbot
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

> Or manually:

```bash
pip install nltk scikit-learn numpy
```

4. **Download NLTK data:**

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## Usage

Run the chatbot in the terminal:

```bash
python chatbot.py
```

**Example Interaction:**

```
🤖 FAQ Chatbot is running (type 'exit' to stop)
You: Hello
Chatbot: Hi there! How can I help you?
You: What is ML?
Chatbot: ML stands for Machine Learning, a subset of AI that learns from data.
You: exit
Chatbot: Goodbye! 👋
```

---

## How It Works

1. **Preprocessing:** Converts text to lowercase, removes punctuation, tokenizes, and removes stopwords.
2. **TF-IDF Vectorization:** Converts all FAQ questions into numerical vectors.
3. **Cosine Similarity:** Compares user input with FAQ vectors to find the best matching response.
4. **Short Form Handling:** Responds to abbreviations like AI, ML, DL quickly.
5. **Fallback Response:** If similarity is below 0.15, responds with “Sorry, I couldn’t understand your question.”
---

 Future Improvements

* Add a GUI or web-based interface for better interaction.
* Support voice input and text-to-speech output.
* Expand FAQ dataset for more comprehensive answers.
*  Contact & References

* **Website:** [www.codealpha.tech](https://www.codealpha.tech)
* **WhatsApp:** +91 8052293611
* **Email:** [services@codealpha.tech](mailto:services@codealpha.tech)


Do you want me to do that next?
