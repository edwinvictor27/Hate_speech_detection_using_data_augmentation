from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import JWTError, jwt
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoModelForCausalLM,
    pipeline
)
import torch
import PyPDF2
from docx import Document
import io
from typing import Optional, List

from hatesonar import Sonar

import os
import base64
import uuid


#                  FASTAPI APP & DATABASE CONNECTION


app = FastAPI()

def get_db_connection():
    """
    Creates a new PostgreSQL database connection
    using RealDictCursor for dict-like row access.
    """
    return psycopg2.connect(
        dbname="user_auth",
        user="postgres",
        password="root",
        host="localhost",
        cursor_factory=RealDictCursor,
    )


#                  PASSWORD & JWT CONFIG


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30  # 30-minute token expiration

def create_access_token(data: dict):
    """
    Create JWT access token with expiration.
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_access_token(token: str):
    """
    Verify a given JWT token. Raises 401 if invalid or expired.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

#####################################################################
#                        MODEL PATHS
#####################################################################

MBERT_PATH = "C:/Users/mudir/Desktop/HATESPEECH-Data Aug/mbert_hate_speech"
SCRATCH_PATH = "C:/Users/mudir/Desktop/HATESPEECH-Data Aug/mbert_hate_speech_noise"
BERT_PATH = "C:/Users/mudir/Desktop/HATESPEECH-Data Aug/mbert_hate_speech_synonym_fine tuned"
ROBERTA_PATH = "C:/Users/mudir/Desktop/HATESPEECH-Data Aug/saved_robert_model"

# Fine-tuned T5 for Rephrasing
T5_REPHRASE_PATH = "C:/Users/mudir/Desktop/HATESPEECH-Data Aug/fine_tunedd_t5_best"


# T5 used for paragraph analysis
T5_PATH = "C:/Users/mudir/Desktop/HATESPEECH-Data Aug/t5_rephrase_model"

# French Model Path
FRENCH_MODEL_PATH = "Hate-speech-CNERG/dehatebert-mono-french"


#                      LOAD MODELS


# 1) MBERT
mbert_tokenizer = AutoTokenizer.from_pretrained(MBERT_PATH)
mbert_model = AutoModelForSequenceClassification.from_pretrained(MBERT_PATH)

# 2) Scratch Model
scratch_tokenizer = AutoTokenizer.from_pretrained(SCRATCH_PATH)
scratch_model = AutoModelForSequenceClassification.from_pretrained(SCRATCH_PATH)

# 3) BERT
bert_tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_PATH)

# 4) RoBERTa
roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_PATH)
roberta_model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_PATH)

# 5) T5 Fine-tuned for Rephrasing
t5_rephrase_tokenizer = T5Tokenizer.from_pretrained(T5_REPHRASE_PATH)
t5_rephrase_model = T5ForConditionalGeneration.from_pretrained(T5_REPHRASE_PATH)


#  `text2text-generation` pipeline for T5 rephrasing
t5_rephrase_pipeline = pipeline("text2text-generation", model=t5_rephrase_model, tokenizer=t5_rephrase_tokenizer)

# T5 model for paragraph analysis
t5_tokenizer = T5Tokenizer.from_pretrained(T5_PATH)
t5_model = T5ForConditionalGeneration.from_pretrained(T5_PATH)


# gpt2_pipeline = pipeline("text-generation", model="gpt2")

# 6) French Hate Speech Detection
french_tokenizer = AutoTokenizer.from_pretrained(FRENCH_MODEL_PATH)
french_model = AutoModelForSequenceClassification.from_pretrained(FRENCH_MODEL_PATH)
french_labels = ["non_hate", "hate"]  # 0=Non-Hate, 1=Hate


#                  Pydantic Models

class User(BaseModel):
    username: str
    email: str
    password: str

class Login(BaseModel):
    username: str
    password: str

class TextAnalysisRequest(BaseModel):
    text: str
    model: str

class BatchAnalysisRequest(BaseModel):
    texts: List[str]
    model: str

class TextAnalysisResponse(BaseModel):
    detected_label: str
    confidence: float
    rephrased_text: str

class FeedbackRequest(BaseModel):
    text: str
    rephrased_text: str
    feedback: str

class DisclaimerAgreement(BaseModel):
    agreed: bool

class Token(BaseModel):
    access_token: str
    token_type: str

class FrenchTextRequest(BaseModel):
    text: str

class ParagraphRequest(BaseModel):
    text: Optional[str] = None
    file_content: Optional[str] = None
    file_type: Optional[str] = None
    model: str = "T5"

class DatasetUploadRequest(BaseModel):
    filename: str
    file_content: str  # base64-encoded CSV


#                    USER & AUTH ENDPOINTS


@app.post("/register")
def register(user: User):
    conn = get_db_connection()
    cursor = conn.cursor()
    hashed_password = pwd_context.hash(user.password)
    try:
        cursor.execute(
            "INSERT INTO users (username, email, hashed_password) VALUES (%s, %s, %s)",
            (user.username, user.email, hashed_password),
        )
        conn.commit()
        return {"message": "User registered successfully"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=400, detail="User already exists.")
    finally:
        cursor.close()
        conn.close()

@app.post("/login")
def login(login: Login):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM users WHERE username = %s", (login.username,))
        user = cursor.fetchone()
        if not user or not pwd_context.verify(login.password, user["hashed_password"]):
            raise HTTPException(status_code=401, detail="Invalid username or password.")
        access_token = create_access_token(data={"sub": login.username})
        return {"access_token": access_token}
    finally:
        cursor.close()
        conn.close()

@app.post("/disclaimer")
def disclaimer_agreement(data: DisclaimerAgreement, token: str = Query(...)):
    verify_access_token(token)
    if data.agreed:
        return {"message": "Disclaimer accepted."}
    else:
        raise HTTPException(status_code=400, detail="You must accept the disclaimer to proceed.")


#               CHAT HISTORY RETRIEVAL ENDPOINT


@app.get("/history")
def get_history(token: str = Query(...), date: str = None):
    """
    Retrieve user's chat history, optionally filtered by date (YYYY-MM-DD).
    """
    payload = verify_access_token(token)
    username = payload["sub"]

    conn = get_db_connection()
    cursor = conn.cursor()
    if date:
        try:
            dt = datetime.strptime(date, "%Y-%m-%d").date()
            cursor.execute("""
                SELECT * FROM chat_history
                WHERE username = %s AND DATE(timestamp) = %s
                ORDER BY timestamp DESC
            """, (username, dt))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    else:
        cursor.execute("""
            SELECT * FROM chat_history
            WHERE username = %s
            ORDER BY timestamp DESC
        """, (username,))
    records = cursor.fetchall()
    cursor.close()
    conn.close()
    return records


#           ENGLISH TEXT ANALYSIS / REPHRASE ( /analyze )

@app.post("/analyze")
def analyze_text(data: TextAnalysisRequest, token: str = Query(...)):
    """
    Analyze English text with the chosen model (MBERT, BERT, RoBERTa, etc.).
    If labeled as hate/offensive, rephrase using the fine-tuned T5 model.
    """
    payload = verify_access_token(token)
    username = payload["sub"]

    # 1) Select model for classification
    if data.model == "MBERT":
        tokenizer, model = mbert_tokenizer, mbert_model
        label_set = ["hate_speech", "offensive", "neutral"]
    elif data.model == "Scratch Model":
        tokenizer, model = scratch_tokenizer, scratch_model
        label_set = ["hate_speech", "offensive", "neutral"]
    elif data.model == "BERT":
        tokenizer, model = bert_tokenizer, bert_model
        label_set = ["hate_speech", "offensive", "neutral"]
    elif data.model == "RoBERTa":
        tokenizer, model = roberta_tokenizer, roberta_model
        label_set = ["hate_speech", "offensive", "neutral"]
    else:
        raise HTTPException(status_code=400, detail="Invalid model selected")

    # 2) Perform Classification
    inputs = tokenizer(data.text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    # Move to device (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    confidence, pred_idx = torch.max(probs, dim=1)

    detected_label = label_set[pred_idx.item()]
    confidence_score = round(confidence.item(), 2)

    # 3) Rephrase using Fine-tuned T5 if the label is Hate Speech or Offensive
    if detected_label in ["hate_speech", "offensive"]:
        try:
            print(f"🔹 Original Text: {data.text}")

            # Using fine-tuned T5 pipeline for rephrasing
            rephrase_input = f"rephrase: {data.text[:200]}"  # Truncate to avoid overflow

            rephrase_output = t5_rephrase_pipeline(rephrase_input, max_length=128, num_return_sequences=1)
            rephrased_text = rephrase_output[0]['generated_text'] if rephrase_output else "Rephrase attempt failed."

            print(f"✅ Rephrased Text: {rephrased_text}")
        except Exception as e:
            print(f"❌ Error in T5 rephrasing: {e}")
            rephrased_text = "Rephrase attempt failed."
    else:
        rephrased_text = "No rephrasing needed for neutral text."

    # 4) Store in DB
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO chat_history (username, message, response, detected_label, confidence, timestamp)
            VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            """,
            (username, data.text, rephrased_text, detected_label, confidence_score),
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Database error: {e}")
    finally:
        cursor.close()
        conn.close()

    return {
        "detected_label": detected_label,
        "confidence": confidence_score,
        "rephrased_text": rephrased_text
    }


#              BATCH ANALYSIS  ( /batch_analyze )

@app.post("/batch_analyze")
def batch_analyze(data: BatchAnalysisRequest, token: str = Query(...)):
    """
    Analyze multiple texts with the chosen model.
    Calls /analyze logic for each text, collecting results.
    """
    verify_access_token(token)
    results = []
    for txt in data.texts:
        single_req = TextAnalysisRequest(text=txt, model=data.model)
        single_result = analyze_text(single_req, token)
        if isinstance(single_result, dict):
            results.append(single_result)
    return results

#              FRENCH TEXT ANALYSIS ( /analyze_french )


@app.post("/analyze_french")
def analyze_french(data: FrenchTextRequest, token: str = Query(...)):
    """
    Analyze French text for hate speech using 'Hate-speech-CNERG/dehatebert-mono-french'.
    If label == 'hate', attempt GPT-2 rephrasing or fallback.
    """
    payload = verify_access_token(token)
    username = payload["sub"]

    # 1) Classification
    in_fr = french_tokenizer(data.text, return_tensors="pt", truncation=True, padding=True)
    out_fr = french_model(**in_fr)
    logits = out_fr.logits

    pred_idx = torch.argmax(logits, dim=1).item()
    confidence_val = torch.nn.functional.softmax(logits, dim=-1)[0, pred_idx].item()
    detected_label = french_labels[pred_idx]
    confidence_score = round(confidence_val, 2)

    # 2) Rephrase if "hate"
    if detected_label == "hate":
        prompt = f"Rephrase: {data.text}"
        try:
            generation = t5_rephrase_pipeline(
                prompt,
                max_length=128,
                num_return_sequences=1,
                pad_token_id=t5_rephrase_tokenizer.eos_token_id
            )
            rephrased_text = generation[0]["generated_text"]
        except Exception as e:
            print(f"Error in GPT-2 rephrasing (French): {e}")
            rephrased_text = "Rephrase attempt failed."
    else:
        rephrased_text = "No rephrasing needed."

    # 3) Store in DB
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO chat_history
                (username, message, response, detected_label, confidence, timestamp)
            VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            """,
            (username, data.text, rephrased_text, detected_label, confidence_score),
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        print("DB Error storing french text:", e)
    finally:
        cursor.close()
        conn.close()

    return {
        "detected_label": detected_label,
        "confidence": confidence_score,
        "rephrased_text": rephrased_text
    }


#                FEEDBACK ENDPOINTS


@app.post("/feedback")
def submit_feedback(data: FeedbackRequest, token: str = Query(...)):
    """
    Store user feedback in the 'feedback' table.
    """
    payload = verify_access_token(token)
    username = payload["sub"]

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO feedback (username, text, rephrased_text, feedback, created_at)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (username, data.text, data.rephrased_text, data.feedback, datetime.utcnow())
        )
        conn.commit()
        return {"message": "Feedback submitted successfully!"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error storing feedback: {str(e)}")
    finally:
        cursor.close()
        conn.close()

@app.post("/save_feedback")
def save_feedback(data: FeedbackRequest, token: str = Query(...)):
    """
    Alternative feedback endpoint. Similar logic.
    """
    payload = verify_access_token(token)
    username = payload.get("sub")

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO feedback
                (username, text, rephrased_text, feedback, created_at)
            VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
            """,
            (username, data.text, data.rephrased_text, data.feedback),
        )
        conn.commit()
        return {"message": "Feedback submitted successfully!"}
    except psycopg2.Error as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error saving feedback: {e}")
    finally:
        cursor.close()
        conn.close()


#         PARAGRAPH ANALYSIS ( /analyze_paragraph )


@app.post("/analyze_paragraph")
def analyze_paragraph(data: ParagraphRequest, token: str = Query(...)):
    """
    Enhanced analysis:
    - Highlights hate/offensive words in text.
    - Generates statistics (hate/offensive word counts).
    """
    payload = verify_access_token(token)

    text_content = data.text or ""
    if not text_content.strip():
        raise HTTPException(status_code=400, detail="No valid text provided.")

    model_used = data.model
    highlighted_html = ""
    hate_count = 0
    offensive_count = 0
    words_list = text_content.split()

    # Define offensive and hate lexicons for basic fallback
    hate_lexicon = {"hate", "kill", "racist", "murder"}
    offensive_lexicon = {"fucker", "idiot", "stupid", "moron"}

    # Word-by-word analysis for T5
    if model_used == "T5":
        for word in words_list:
            input_text = f"classify: {word}"
            in_ids = t5_tokenizer(input_text, return_tensors="pt").input_ids
            out_gen = t5_model.generate(in_ids)
            label = t5_tokenizer.decode(out_gen[0], skip_special_tokens=True).lower()

            if "hate" in label:
                highlighted_html += f"<span style='color:red;'>{word}</span> "
                hate_count += 1
            elif "offensive" in label:
                highlighted_html += f"<span style='color:orange;'>{word}</span> "
                offensive_count += 1
            else:
                highlighted_html += f"{word} "
    # Full-text analysis for MBERT
    elif model_used == "MBERT":
        in_pt = mbert_tokenizer(text_content, return_tensors="pt", truncation=True, padding=True, max_length=512)
        out_pt = mbert_model(**in_pt)
        probs_pt = torch.nn.functional.softmax(out_pt.logits, dim=-1)
        conf_pt, idx_pt = torch.max(probs_pt, dim=1)

        english_labels = ["hate_speech", "offensive", "neutral"]
        label_val = english_labels[idx_pt.item()]

        if label_val in ["hate_speech", "offensive"]:
            for word in words_list:
                if word.lower() in hate_lexicon:
                    highlighted_html += f"<span style='color:red;'>{word}</span> "
                    hate_count += 1
                elif word.lower() in offensive_lexicon:
                    highlighted_html += f"<span style='color:orange;'>{word}</span> "
                    offensive_count += 1
                else:
                    highlighted_html += f"{word} "
        else:
            highlighted_html = text_content
    else:
        raise HTTPException(status_code=400, detail="Unsupported model used.")

    total_words = len(words_list)

    return {
        "original_text": text_content,
        "highlighted_html": highlighted_html.strip(),
        "hate_count": hate_count,
        "offensive_count": offensive_count,
        "total_words": total_words,
        "model_used": model_used
    }





#                 COMMUNITY SOLUTION FOR DATASETS


@app.post("/upload_dataset")
def upload_dataset(data: DatasetUploadRequest, token: str = Query(...)):
    """
    Allows user to upload a CSV dataset (base64-encoded).
    """
    payload = verify_access_token(token)
    username = payload["sub"]

    os.makedirs("uploads/datasets", exist_ok=True)

    # Decode file content (base64)
    try:
        file_bytes = base64.b64decode(data.file_content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 content: {str(e)}")

    dataset_uuid = str(uuid.uuid4())
    saved_path = f"uploads/datasets/{dataset_uuid}.csv"
    with open(saved_path, "wb") as f:
        f.write(file_bytes)

    # Insert DB metadata
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO datasets (filename, uploaded_by, file_path)
            VALUES (%s, %s, %s)
            RETURNING id
        """, (data.filename, username, saved_path))
        new_id = cursor.fetchone()["id"]
        conn.commit()
        return {"dataset_id": new_id, "message": "Dataset uploaded successfully"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error saving dataset info: {str(e)}")
    finally:
        cursor.close()
        conn.close()

@app.get("/list_datasets")
def list_datasets(token: str = Query(...)):
    """
    Returns a list of all datasets (id, filename, uploaded_by, created_at).
    """
    payload = verify_access_token(token)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT id, filename, uploaded_by, created_at
            FROM datasets
            ORDER BY created_at DESC
        """)
        records = cursor.fetchall()
        return records
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {str(e)}")
    finally:
        cursor.close()
        conn.close()

from fastapi.responses import FileResponse

@app.get("/download_dataset/{dataset_id}")
def download_dataset(dataset_id: int, token: str = Query(...)):
    """
    Download a previously uploaded dataset by ID.
    """
    payload = verify_access_token(token)
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT file_path, filename FROM datasets WHERE id = %s", (dataset_id,))
    record = cursor.fetchone()
    cursor.close()
    conn.close()

    if not record:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    file_path = record["file_path"]
    orig_name = record["filename"]

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File path missing on server.")

    return FileResponse(path=file_path, media_type="text/csv", filename=orig_name)


#                           QUIZ ENDPOINTS


@app.get("/quiz")
def get_quiz_questions(token: str = Query(...), limit: int = 10):
    """
    Fetch a specified number of random quiz questions from `quiz_questions`.
    """
    payload = verify_access_token(token)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM quiz_questions ORDER BY RANDOM() LIMIT %s", (limit,))
        questions = cursor.fetchall()
        return {"questions": questions}
    finally:
        cursor.close()
        conn.close()

@app.post("/quiz_results")
def save_quiz_results(username: str, score: int, total_questions: int):
    """
    Optional: Save quiz results to 'quiz_results' table if you have one.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO quiz_results (username, score, total_questions) 
            VALUES (%s, %s, %s)
            """,
            (username, score, total_questions),
        )
        conn.commit()
        return {"message": "Quiz results saved successfully."}
    finally:
        cursor.close()
        conn.close()

