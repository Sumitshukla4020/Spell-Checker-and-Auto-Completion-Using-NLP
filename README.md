# Spell Checker and Auto-Complete System

This project implements a **Spell Checker** and an **Auto-Complete System** using Python and Streamlit. It combines **Hidden Markov Models (HMMs)** and **N-gram Language Models** to enhance text processing capabilities such as autocorrect and text prediction.

---

## Features
1. **Spell Checker**  
   - Detects and corrects misspelled words using a probabilistic approach.
   - Leverages the Viterbi algorithm with HMMs to identify the most likely sequence of correct words.

2. **Auto-Complete**  
   - Predicts the next possible words based on user input using a bigram language model.

3. **Interactive UI**  
   - Built with Streamlit for a user-friendly, interactive interface.
   ---

## Live Demo
Try the application live at: [Link](https://spellcheckandautocomplete.streamlit.app/)
  ---

---

## How It Works
- **Preprocessing**: Uploaded text files are cleaned and tokenized into sentences and words.
- **N-grams**: A bigram model is built from the tokenized text to predict word probabilities.
- **Misspelling Handling**: The app generates possible misspellings and computes emission probabilities for spell checking.
- **HMM and Viterbi Algorithm**: These are used to find the most probable correction for a sequence of misspelled words.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/spellchecker-autocomplete.git
   cd spellchecker-autocomplete
   ```
2. Install the required Python packages:
   ```bash
    pip install -r requirements.txt
    ```
3. Run the application:
   ```bash
    streamlit run app.py
    ```
