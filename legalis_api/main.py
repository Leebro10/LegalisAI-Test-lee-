import json
import jsonlines
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import List
from pydantic import BaseModel, Field

# Initialize FastAPI app
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths to your models
legalis_model_path = "../legalis_model"
faq_model_path = "../faq_model"

# Load tokenizers and models for both Legalis and FAQ
tokenizer_legalis = AutoTokenizer.from_pretrained(legalis_model_path)
model_legalis = AutoModel.from_pretrained(legalis_model_path)

tokenizer_faq = AutoTokenizer.from_pretrained(faq_model_path)
model_faq = AutoModel.from_pretrained(faq_model_path)

# Load Legalis Data from JSON
try:
    with open('../Data/finalcases.json', 'r') as f:
        cases_data = json.load(f)  # This assumes the JSON is an array of case objects.
except FileNotFoundError:
    logger.error("Legalis data file not found.")
    cases_data = []

# Load FAQ Data from JSONL
faq_data = []
try:
    with jsonlines.open('../Data/QandA.jsonl') as reader:
        for obj in reader:
            faq_data.append(obj)
except FileNotFoundError:
    logger.error("FAQ data file not found.")

# Pydantic model for the request body
class TextRequest(BaseModel):
    text: str
    model_choice: str = Field(..., pattern="^(legalis|faq)$", example="legalis")

# Function to encode text for both models
def encode_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Function to find relevant cases (Legalis) with most similar sections
def find_relevant_cases(user_input, cases_data, num_results=5):
    input_vector = encode_text(user_input, tokenizer_legalis, model_legalis)
    case_vectors = [encode_text(case["case_description"], tokenizer_legalis, model_legalis) for case in cases_data]
    
    similarities = [cosine_similarity(input_vector, case_vec.reshape(1, -1))[0][0] for case_vec in case_vectors]
    
    # Get top N cases with highest similarity scores
    top_indices = np.argsort(similarities)[-num_results:][::-1]
    
    results = []
    for index in top_indices:
        case = cases_data[index]
        # Find most similar sections for the case
        case_sections = case["sections"]
        section_similarities = []

        for section in case_sections:
            section_vector = encode_text(section["section_description"], tokenizer_legalis, model_legalis)
            section_similarity = cosine_similarity(input_vector, section_vector.reshape(1, -1))[0][0]
            section_similarities.append((section, section_similarity))
        
        # Sort sections by similarity and pick top N similar sections
        sorted_sections = sorted(section_similarities, key=lambda x: x[1], reverse=True)[:3]

        results.append({
            "case_id": case["case_id"],
            "case_title": case["case_title"],
            "case_link": case["case_link"],
            "similarity_score": float(similarities[index]),
            "sections": [section[0] for section in sorted_sections],  # Include most similar sections
            "strong_points": case["strong_points"],
            "weak_points": case["weak_points"]
        })

    return results

# Function to find relevant FAQs (FAQ Model)
def find_relevant_faq(query, faq_data, num_results=5):
    faq_prompts = [faq["prompt"] for faq in faq_data]
    faq_embeddings = np.array([encode_text(prompt, tokenizer_faq, model_faq) for prompt in faq_prompts])
    
    query_embedding = encode_text(query, tokenizer_faq, model_faq)
    
    similarities = [cosine_similarity(query_embedding.reshape(1, -1), faq_embedding.reshape(1, -1))[0][0] for faq_embedding in faq_embeddings]
    
    top_indices = np.argsort(similarities)[-num_results:][::-1]
    
    results = []
    for index in top_indices:
        faq = faq_data[index]
        results.append({
            "faq_prompt": faq["prompt"],
            "faq_completion": faq["completion"],
            "similarity_score": float(similarities[index])
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
        # Input validation
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty.")

        if request.model_choice == "legalis":
            if not cases_data:
                raise HTTPException(status_code=404, detail="No legal cases available.")
            result = find_relevant_cases(request.text, cases_data)
            if result:
                return {"model": "Legalis", "results": result}
            else:
                raise HTTPException(status_code=404, detail="No relevant cases found.")

        elif request.model_choice == "faq":
            if not faq_data:
                raise HTTPException(status_code=404, detail="No FAQs available.")
            result = find_relevant_faq(request.text, faq_data)
            if result:
                return {"model": "FAQ", "results": result}
            else:
                raise HTTPException(status_code=404, detail="No relevant FAQs found.")
        
        else:
            raise HTTPException(status_code=400, detail="Invalid model_choice. Please choose either 'legalis' or 'faq'.")

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Testing Locally Command (for reference)
# curl -X POST "http://127.0.0.1:8000/predict/" -H "Content-Type: application/json" -d "{\"text\": \"What is the procedure for property registration?\", \"model_choice\": \"legalis\"}"
# curl -X POST "http://127.0.0.1:8000/predict/" -H "Content-Type: application/json" -d "{\"text\": \"How do I register a property in Maharashtra?\", \"model_choice\": \"faq\"}"

#Testing locally Command:

#curl -X POST "http://127.0.0.1:8000/predict/" -H "Content-Type: application/json" -d "{\"text\": \"What is the procedure for property registration?\", \"model_choice\": \"legalis\"}"
#curl -X POST "http://127.0.0.1:8000/predict/" -H "Content-Type: application/json" -d "{\"text\": \"How do I register a property in Maharashtra?\", \"model_choice\": \"faq\"}"