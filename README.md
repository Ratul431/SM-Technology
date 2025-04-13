# 🔍 Offensive Language Classification (Multi-label)

This project aims to build and evaluate multiple machine learning models to detect offensive content in online feedback/comments. Each comment can exhibit none, one, or multiple forms of offensiveness across six categories: `toxic`, `abusive`, `vulgar`, `menace`, `offense`, and `bigotry`.

---

## 📌 Project Overview

The goal of this task is to classify feedback text into multiple offensive categories. The dataset contains labeled samples with six offensive tags. Since a single comment can belong to more than one category, this is a **multi-label classification** problem.

### Key Features:
- **Multi-label output** — one comment may be toxic, vulgar, and offensive at once.
- **Custom word embeddings** using Word2Vec.
- **Sequential model (LSTM)** to leverage the temporal nature of text.
- **Transformer-based fine-tuned BERT model** for contextual understanding and multilingual handling.

---
## 📌 Dataset description
train.csv (Labeled Training Data):
id: Unique identifier for each comment
feedback_text: The feedback to be classified
toxic: 1 if the comment is toxic
abusive: 1 if the comment contains severe toxicity
vulgar: 1 if the comment contains obscene language
menace: 1 if the comment contains threats
offense: 1 if the comment contains insults
bigotry: 1 if the comment contains identity-based hate
test.csv (Unlabeled data for prediction)
Note: Each label is binary (0 = offensive content not present, 1 = offensive content present), and multiple labels can be active for a single comment.


## ⚙️ Model Implementation Details

### ✅ Traditional + Embedding-Based Model
- Preprocessed the text (lowercase, punctuation removal, stopword removal, tokenization).
- Trained a **Word2Vec** model using Gensim.
- Combined Word2Vec sentence embeddings with engineered features (`text_len`, `char_count`, `word_count`).
- Trained **Logistic Regression** and **Random Forest** classifiers using scikit-learn.

### ✅ LSTM-Based Model
- Tokenized and padded input sequences.
- Built a **Bidirectional LSTM** using TensorFlow and Keras with an embedding layer.
- Trained the model using binary cross-entropy loss with sigmoid activation for multi-label classification.

### ✅ Transformer-Based Model (BERT)
- Used HuggingFace’s `bert-base-uncased` pretrained model.
- Tokenized inputs using `BertTokenizer`.
- Built a custom Keras model using the **BERT base model** and a custom classification head.
- Trained using `binary_crossentropy` loss with `Adam` optimizer and monitored AUC and accuracy.
- Evaluated using classification report and AUC-ROC curves.

---

## 🚀 Steps to Run the Code
Download the three notebooks in your local machine. Import them in colab or kaggle and trun on GPU.
When the session in on click run all to run all the cells in a notebook to see the output or outcome. 
Same thing applicable for all the three notebooks. While running code kindly change the path of dataset 
based on your given name. Ensure that it is the trtain.csv file.

