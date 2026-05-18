"""
API de clasificación de currículos — FastAPI
Basada exactamente en el código de modelos_nlp.ipynb
"""

import os, json, pickle, re
import joblib
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model

nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

# ─── Rutas ───────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, 'saved_models')

# ─── Cargar artefactos al iniciar la API ─────────────────────────────────────
bundle = json.load(open(os.path.join(MODELS_DIR, 'production_bundle.json')))
le     = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.joblib'))

# Mismo preprocesamiento que en el notebook
EXTRA_SW = set(bundle['preprocessing']['extra_stopwords'])
STOP_EN  = set(stopwords.words('english')) | EXTRA_SW


# ─── Funciones de preprocesamiento (idénticas al notebook) ───────────────────

def clean_text(text: str) -> str:
    """Limpia texto crudo de PDF: quita HTML, números, puntuación y stopwords."""
    text = re.sub(r'[\n\r\t]+', ' ', text)
    text = re.sub(r' {2,}',     ' ', text)
    text = re.sub(r'/',         ' ', text)
    text = text.lower().strip()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_EN and len(t) > 2]
    return ' '.join(tokens)

def build_sequences(token_lists, word2idx, max_len):
    """Convierte listas de tokens a secuencias numéricas paddeadas."""
    result = []
    for tokens in token_lists:
        idx  = [word2idx.get(t, 1) for t in tokens][:max_len]
        idx += [0] * (max_len - len(idx))
        result.append(idx)
    return np.array(result)

def create_char_ngrams(text, min_n, max_n):
    """Genera n-gramas de caracteres (usado por FastText)."""
    grams = []
    for n in range(min_n, max_n + 1):
        grams += [text[i:i+n] for i in range(len(text) - n + 1)]
    return grams

def prepare_ft_seqs(texts, vocab, min_n, max_n, max_len):
    """Prepara secuencias de n-gramas para FastText."""
    result = []
    for text in texts:
        grams = create_char_ngrams(text, min_n, max_n)
        idx   = [vocab.get(g, 1) for g in grams][:max_len]
        idx  += [0] * (max_len - len(idx))
        result.append(idx)
    return np.array(result)


# ─── Función de predicción (igual que predict_resume del notebook) ────────────

def predict_resume(raw_text: str) -> dict:
    model_name = bundle['best_model_name']
    text_clean = clean_text(raw_text)

    if model_name == 'TF-IDF+XGBoost':
        tfidf = joblib.load(os.path.join(MODELS_DIR, 'tfidf_best.joblib'))
        model = joblib.load(os.path.join(MODELS_DIR, 'xgboost_best.joblib'))
        X     = tfidf.transform([text_clean])
        proba = model.predict_proba(X)[0]
        pred  = int(np.argmax(proba))

    elif model_name == 'Word2Vec+BiLSTM':
        with open(os.path.join(MODELS_DIR, 'bilstm_word2idx.pkl'), 'rb') as f:
            word2idx = pickle.load(f)
        model  = load_model(os.path.join(MODELS_DIR, 'bilstm_best.keras'))
        tokens = text_clean.split()
        seq    = build_sequences([tokens], word2idx, bundle['seq_len'])
        proba  = model.predict(seq, verbose=0)[0]
        pred   = int(np.argmax(proba))

    elif model_name == 'CNN-1D':
        with open(os.path.join(MODELS_DIR, 'cnn1d_vocab.pkl'), 'rb') as f:
            vocab = pickle.load(f)
        cfg   = json.load(open(os.path.join(MODELS_DIR, 'cnn1d_config.json')))
        model = load_model(os.path.join(MODELS_DIR, 'cnn1d_best.keras'))
        seq   = build_sequences([text_clean.split()], vocab, cfg['max_len'])
        proba = model.predict(seq, verbose=0)[0]
        pred  = int(np.argmax(proba))

    elif model_name == 'FastText':
        with open(os.path.join(MODELS_DIR, 'fasttext_vocab.pkl'), 'rb') as f:
            vocab = pickle.load(f)
        cfg   = json.load(open(os.path.join(MODELS_DIR, 'fasttext_config.json')))
        model = load_model(os.path.join(MODELS_DIR, 'fasttext_best.keras'))
        seq   = prepare_ft_seqs(
            [text_clean], vocab,
            cfg['ngram_min'], cfg['ngram_max'], cfg['max_len']
        )
        proba = model.predict(seq, verbose=0)[0]
        pred  = int(np.argmax(proba))

    elif model_name == 'RoBERTa':                                        # ← NUEVO
        from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
        import tensorflow as tf

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        n_classes = len(bundle['classes'])
        max_len   = bundle['max_len']

        model = TFRobertaForSequenceClassification.from_pretrained(
            'roberta-base',
            num_labels=n_classes,
            from_pt=True
        )
        model.load_weights(os.path.join(MODELS_DIR, 'roberta_best.weights.h5'))

        enc    = tokenizer(
            [raw_text],                      # texto original sin limpiar
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )
        logits = model(enc, training=False).logits
        proba  = tf.nn.softmax(logits).numpy()[0]
        pred   = int(np.argmax(proba))

    else:
        raise HTTPException(status_code=500,
                            detail=f"Modelo '{model_name}' no soportado en API.")

    return {
        'categoria': le.inverse_transform([pred])[0],
        'confianza': round(float(np.max(proba)), 4)
    }


# ─── FastAPI ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Resume Classifier API",
    description="Clasifica currículos en categorías laborales usando Deep Learning.",
    version="1.0.0"
)

class ResumeInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    categoria_predicha: str
    confianza: float
    modelo_usado: str


@app.get("/")
def root():
    """Info general de la API y del mejor modelo."""
    return {
        "mensaje":      "API de clasificación de CVs — funcionando ✓",
        "mejor_modelo": bundle["best_model_name"],
        "f1_test":      bundle["best_f1_test"],
        "clases":       bundle["classes"],
        "docs":         "/docs"
    }


@app.post("/predict_text", response_model=PredictionOutput)
def predict(resume: ResumeInput):
    """
    Recibe el texto de un CV y devuelve:
    - categoria_predicha: categoría laboral
    - confianza: probabilidad del modelo (0-1)
    - modelo_usado: nombre del mejor modelo entrenado
    """
    if not resume.text.strip():
        raise HTTPException(status_code=400,
                            detail="El texto del CV no puede estar vacío.")
    resultado = predict_resume(resume.text)
    return PredictionOutput(
        categoria_predicha = resultado['categoria'],
        confianza          = resultado['confianza'],
        modelo_usado       = bundle['best_model_name']
    )


@app.get("/models/info")
def model_info():
    """Devuelve las métricas de todos los modelos entrenados."""
    return {
        "mejor_modelo": bundle["best_model_name"],
        "f1_test":      bundle["best_f1_test"],
        "metricas":     bundle["metrics_all"]
    }
