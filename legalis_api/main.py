import os
import base64
import json
import jsonlines
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Initialize FastAPI app
app = FastAPI()

# Paths to your models
legalis_model_path = "Getes07/legalis_model"
faq_model_path = "Getes07/faq_model"

# Load tokenizers and models for both Legalis and FAQ
tokenizer_legalis = AutoTokenizer.from_pretrained(legalis_model_path)
model_legalis = AutoModel.from_pretrained(legalis_model_path)

tokenizer_faq = AutoTokenizer.from_pretrained(faq_model_path)
model_faq = AutoModel.from_pretrained(faq_model_path)

# Function to decode base64 and load the file
def decode_and_load_base64_file(env_var_name, local_file_path):
    encoded_data = os.getenv(env_var_name)
    if encoded_data:
        # If the environment variable is set, decode and load the file
        with open(local_file_path, 'wb') as f:
            f.write(base64.b64decode(encoded_data))
        print(f"File decoded from env var and saved as {local_file_path}")
    else:
        print(f"Loading file from local path: {local_file_path}")
    
    # Load the JSON data from the file
    with open(local_file_path, 'r') as f:
        return json.load(f)

# Load Legalis Data from JSON (via environment variable or local file)
cases_data = decode_and_load_base64_file('LEGALIS_TRAINING_JSON', 'legalis_training.json')

# Load FAQ Data from JSONL (via environment variable or local file)
faq_data = decode_and_load_base64_file('FAQ_TRAINING_JSON', 'faq_training.jsonl')

# Pydantic model for the request body
class TextRequest(BaseModel):
    text: str
    model_choice: str  # "legalis" or "faq"

# Function to encode text for both models
def encode_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Function to find relevant cases (Legalis)
def find_relevant_cases(user_input, cases_data, num_results=5):
    input_vector = encode_text(user_input, tokenizer_legalis, model_legalis)
    case_vectors = [encode_text(case["case_description"], tokenizer_legalis, model_legalis) for case in cases_data]
    
    # Cosine similarity expects 2D arrays (vector, matrix), hence reshape input_vector
    similarities = [cosine_similarity(input_vector, case_vec.reshape(1, -1))[0][0] for case_vec in case_vectors]
    
    # Get top N cases with highest similarity scores
    top_indices = np.argsort(similarities)[-num_results:][::-1]
    
    results = []
    for index in top_indices:
        case = cases_data[index]
        results.append({
            "case_id": case["case_id"],
            "case_title": case["case_title"],
            "case_link": case["case_link"],
            "similarity_score": float(similarities[index]),  # Convert to native Python float
            "sections": case["sections"],
            "strong_points": case["strong_points"],
            "weak_points": case["weak_points"]
        })

    return results

# Function to find relevant FAQs (FAQ Model)
def find_relevant_faq(query, faq_data, num_results=5):
    faq_prompts = [faq["prompt"] for faq in faq_data]
    faq_embeddings = np.array([encode_text(prompt, tokenizer_faq, model_faq) for prompt in faq_prompts])
    
    query_embedding = encode_text(query, tokenizer_faq, model_faq)
    
    # Cosine similarity expects 2D arrays (vector, matrix), hence reshape query_embedding
    similarities = [cosine_similarity(query_embedding.reshape(1, -1), faq_embedding.reshape(1, -1))[0][0] for faq_embedding in faq_embeddings]
    
    top_indices = np.argsort(similarities)[-num_results:][::-1]
    
    results = []
    for index in top_indices:
        faq = faq_data[index]
        results.append({
            "faq_prompt": faq["prompt"],
            "faq_completion": faq["completion"],
            "similarity_score": float(similarities[index])  # Convert to native Python float
        })
    
    return results

# Root endpoint for checking if the API is up
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Legalis AI API!"}

# Prediction endpoint (POST)
@app.post("/predict/")
async def predict(request: TextRequest):
    try:
        # Your existing prediction code
        if request.model_choice == "legalis":
            result = find_relevant_cases(request.text, cases_data)
            if result:
                return {"model": "Legalis", "results": result}
            else:
                raise HTTPException(status_code=404, detail="No relevant cases found.")

        elif request.model_choice == "faq":
            result = find_relevant_faq(request.text, faq_data)
            if result:
                return {"model": "FAQ", "results": result}
            else:
                raise HTTPException(status_code=404, detail="No relevant FAQs found.")
        
        else:
            raise HTTPException(status_code=400, detail="Invalid model_choice. Please choose either 'legalis' or 'faq'.")
    except Exception as e:
        print(f"Error: {e}")  # Log the error
        raise HTTPException(status_code=500, detail="Internal Server Error")
