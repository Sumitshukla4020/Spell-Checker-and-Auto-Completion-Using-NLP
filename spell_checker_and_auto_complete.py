import streamlit as st
import re
from collections import defaultdict, Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np

# Download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
# Function to preprocess the text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s\-\']', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Function to generate n-grams
def generate_ngrams(tokenized_sentences, n):
    ngrams = []
    for sentence in tokenized_sentences:
        for i in range(len(sentence) - n + 1):
            ngram = tuple(sentence[i:i + n])
            ngrams.append(ngram)
    return ngrams

# Function to build the bigram model
def build_bigram_model(tokenized_sentences):
    bigrams = generate_ngrams(tokenized_sentences, 2)
    bigram_freq = Counter(bigrams)
    bigram_model = defaultdict(dict)
    for (w1, w2), freq in bigram_freq.items():
        bigram_model[w1][w2] = freq
    for w1 in bigram_model:
        total_count = sum(bigram_model[w1].values())
        for w2 in bigram_model[w1]:
            bigram_model[w1][w2] /= total_count
    return bigram_model

# Function to predict the next word
def predict_next_word(word, bigram_model, top_k=3):
    if word in bigram_model:
        next_word_probs = bigram_model[word]
        sorted_predictions = sorted(next_word_probs.items(), key=lambda x: x[1], reverse=True)
        return [word for word, prob in sorted_predictions[:top_k]]
    else:
        return []

# Function to generate possible misspellings
def generate_possible_misspellings(word):
    def generate_deletions(word):
        return [word[:i] + word[i+1:] for i in range(len(word))]

    def generate_insertions(word):
        letters = 'abcdefghijklmnopqrstuvwxyz'
        return [word[:i] + c + word[i:] for i in range(len(word) + 1) for c in letters]

    def generate_substitutions(word):
        letters = 'abcdefghijklmnopqrstuvwxyz'
        return [word[:i] + c + word[i+1:] for i in range(len(word)) for c in letters if c != word[i]]

    def generate_transpositions(word):
        return [word[:i] + word[i+1] + word[i] + word[i+2:] for i in range(len(word) - 1)]

    return set(generate_deletions(word) + generate_insertions(word) +
               generate_substitutions(word) + generate_transpositions(word))

# Function to build emission probabilities
def build_emission_probabilities(tokenized_sentences):
    emission_probs = defaultdict(lambda: defaultdict(float))
    for sentence in tokenized_sentences:
        for word in sentence:
            misspellings = generate_possible_misspellings(word)
            for misspelled_word in misspellings:
                emission_probs[word][misspelled_word] += 1
    for true_word in emission_probs:
        total_misspellings = sum(emission_probs[true_word].values())
        for misspelled_word in emission_probs[true_word]:
            emission_probs[true_word][misspelled_word] /= total_misspellings
    return emission_probs

# Function to correct spelling using Viterbi
def viterbi(observed_sequence, states, transition_probs, emission_probs, start_prob=1.0):
    n = len(observed_sequence)
    dp = np.zeros((len(states), n))
    path = np.zeros((len(states), n), dtype=int)
    state_index = {state: i for i, state in enumerate(states)}
    for s in states:
        if observed_sequence[0] in emission_probs[s]:
            dp[state_index[s], 0] = start_prob * emission_probs[s][observed_sequence[0]]
    for t in range(1, n):
        for s in states:
            max_prob, max_state = 0, 0
            for prev_s in states:
                prob = dp[state_index[prev_s], t-1] * transition_probs[prev_s].get(s, 0) * emission_probs[s].get(observed_sequence[t], 0)
                if prob > max_prob:
                    max_prob, max_state = prob, state_index[prev_s]
            dp[state_index[s], t] = max_prob
            path[state_index[s], t] = max_state
    best_path = []
    max_prob, last_state = max((dp[i, n-1], i) for i in range(len(states)))
    best_path.append(states[last_state])
    for t in range(n - 1, 0, -1):
        last_state = path[last_state, t]
        best_path.append(states[last_state])
    best_path.reverse()
    return best_path
# Streamlit app
st.title("Spell Checker and Auto-Complete System")
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
    st.text_area("Uploaded Text", text, height=200)
    cleaned_text = preprocess_text(text)
    sentences = sent_tokenize(cleaned_text)
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
    bigram_model = build_bigram_model(tokenized_sentences)
    emission_probs = build_emission_probabilities(tokenized_sentences)
    
    st.header("Auto-Complete")
    input_phrase = st.text_input("Enter a phrase for auto-complete:")
    if input_phrase:
        input_tokens = word_tokenize(input_phrase.lower())
        if input_tokens:
            last_word = input_tokens[-1]
            predictions = predict_next_word(last_word, bigram_model)
            st.write(f"Predictions: {predictions}")
        else:
            st.write("Enter a valid phrase.")

    st.header("Spell Checker")
    misspelled_phrase = st.text_input("Enter a misspelled word for correction - please wait for 30 secs:")
    if misspelled_phrase:
        misspelled_tokens = word_tokenize(misspelled_phrase.lower())
        if misspelled_tokens:
            states = list(emission_probs.keys())
            corrected_sequence = viterbi(misspelled_tokens, states, bigram_model, emission_probs)
            st.write(f"Corrected Phrase: {' '.join(corrected_sequence)}")
        else:
            st.write("Enter a valid phrase.")
