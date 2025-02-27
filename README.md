%%markdown
# Improving Hate Speech Detection with Data Augmentation Techniques

## Objective
This project investigates how various data augmentation techniques can improve the performance of NLP models for hate speech detection.

## Motivation
Hate speech detection models often struggle with limited or imbalanced datasets. By employing augmentation techniques, we aim to:

- Enhance model performance
- Improve classification on minority classes
- Provide a robust pipeline for hate speech detection

## Features

### Data Augmentation Techniques
- **Synonym Replacement**: Replace words with their synonyms from a pre-defined thesaurus (using WordNet).
- **Back Translation**: Translate text into another language and back to the original.
- **Contextual Word Embeddings**: Mask words and predict replacements using models like BERT.
- **TextAttack Augmentation**: Utilize TextAttack framework for adversarial text augmentations.
- **WordNet-based Augmentation**: Leverage WordNet to find and replace words with appropriate synonyms.

## Paraphrasing Model
After augmenting the dataset, we developed a **finetuned** (T5, GPT-2, DHATEBERT for reference hate speech detection) paraphrasing model that generates diverse reworded versions of input prompts, further improving robustness in hate speech detection.

**For further improvement of the model, anyone can use this repository and train the model with additional samples.**

---

## Installation & Usage

### Requirements
Ensure you have Python installed. Then, install the necessary dependencies by running:
```bash
pip install -r requirements.txt
```

### Running the API (FastAPI)
To start the FastAPI server, run:
```bash
python api.py
```
This will launch the API on `http://127.0.0.1:8000`.

### Running the Streamlit App
After running the API, start the Streamlit web app by executing:
```bash
streamlit run app.py
```
This will launch the Streamlit interface for interacting with the model.

## Usage Example
Once both FastAPI and Streamlit are running, open the Streamlit UI in your browser, enter a text prompt, and explore augmented/paraphrased outputs.

---

## Contributors
- **Anmol Shantharam Jadhav**
- **Edwin Victor Justin**
- **SravanKumar Mudireddy**

### Guidance Provided by:
**Prof. Alaa Bhakthi**
