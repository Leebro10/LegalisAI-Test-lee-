import streamlit as st
import json
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from googletrans import Translator
from functools import lru_cache

translator = Translator()

@lru_cache(maxsize=100)
def translate_text(text, dest_language):
    if dest_language == "English":
        return text
    lang_code = {"Hindi": "hi", "Marathi": "mr"}.get(dest_language, "en")
    return translator.translate(text, dest=lang_code).text

# Load the trained models and tokenizers
legalis_model_path = "./legalis_model"
faq_model_path = "./faq_model"

tokenizer_legalis = AutoTokenizer.from_pretrained(legalis_model_path)
model_legalis = AutoModel.from_pretrained(legalis_model_path)

tokenizer_faq = AutoTokenizer.from_pretrained(faq_model_path)
model_faq = AutoModel.from_pretrained(faq_model_path)

# Load cases from JSON
with open("./Data/finalcases.json", "r", encoding="utf-8") as file:
    cases_data = json.load(file)

# Load FAQ data
def load_faq_data(file_path):
    faq_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            faq_data.append(json.loads(line))
    return faq_data

faq_data = load_faq_data("./Data/QandA.jsonl")

# Function to encode text using BERT
def encode_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Precompute and store case embeddings
case_vectors = {case["case_id"]: encode_text(case["case_description"], tokenizer_legalis, model_legalis) for case in cases_data}

# Precompute and store FAQ embeddings
faq_embeddings = {faq["prompt"]: encode_text(faq["prompt"], tokenizer_faq, model_faq) for faq in faq_data}


# Function to find relevant cases (Legalis)
def find_relevant_cases(user_input, cases, num_results=5, language="English"):
    if language in ["Hindi", "Marathi"]:
        user_input = translate_text(user_input, "English")

    input_vector = encode_text(user_input, tokenizer_legalis, model_legalis)
    #case_vectors = [encode_text(case["case_description"], tokenizer_legalis, model_legalis) for case in cases]
    similarities = cosine_similarity(input_vector, np.vstack(list(case_vectors.values()))).flatten()
    
    top_indices = np.argsort(similarities)[-num_results:][::-1]
    
    results = []
    for index in top_indices:
        case = cases[index]
        results.append({
            "case": case,
            "similarity_score": similarities[index]
        })
    
    return results

# Function to find relevant FAQ (FAQ Model)
def find_relevant_faq(query, faq_data, num_results=5, language="English"):
    if language in ["Hindi", "Marathi"]:
        query = translate_text(user_input, "English")

    faq_prompts = [faq["prompt"] for faq in faq_data]
    #faq_embeddings = np.array([encode_text(prompt, tokenizer_faq, model_faq) for prompt in faq_prompts])
    
    query_embedding = encode_text(query, tokenizer_faq, model_faq)
    
    similarities = [cosine_similarity(query_embedding, faq_embedding)[0][0] for faq_embedding in faq_embeddings]
    
    top_indices = np.argsort(similarities)[-num_results:][::-1]
    
    results = []
    for index in top_indices:
        faq = faq_data[index]
        results.append({
            "faq": faq,
            "similarity_score": similarities[index]
        })
    
    return results

# Streamlit UI with Sidebar
st.title("LegalisAI: Real Estate Legal Case Assistant ⚖️")

# Sidebar
with st.sidebar:
    # Language Selection
    language = st.selectbox("Choose Language", ["English", "Hindi", "Marathi"])
    # Model selection (Case or FAQ)
    model_choice = st.selectbox("Choose what you'd like to analyze:", ["Legal Cases", "FAQs"])

# Function to translate text
def translate_text(text, dest_language):
    if dest_language == "English":
        return text
    lang_code = {"Hindi": "hi", "Marathi": "mr"}.get(dest_language, "en")
    return translate_text(user_input, "English")

# Case analysis UI
if model_choice == "Legal Cases":
    user_input = st.text_area(translate_text("Enter your case description:", language), height=150)
    nombres = st.slider(translate_text("Select number of similar cases to retrieve:", language), min_value=1, max_value=10, value=5)

    if st.button(translate_text("Analyze Case", language)):
        if user_input.strip():
            st.session_state.results = find_relevant_cases(user_input, cases_data, nombres, language)
            st.session_state.case_index = 0  # Reset index when new search is made

    if st.session_state.get("results", []):
        result = st.session_state.results[st.session_state.case_index]
        best_case = result["case"]
        similarity_score = result["similarity_score"]

        st.subheader(translate_text("🔎 Case", language) + f" {st.session_state.case_index + 1} of {len(st.session_state.results)}")
        st.write(f"**{translate_text('Case ID:', language)}** {translate_text(best_case['case_title'], language)}")
        st.write(f"**{translate_text('Case Title:', language)}** {translate_text(best_case['case_title'], language)}")
        st.write(f"**{translate_text('Case PDF Link:', language)}** [{translate_text('Read More Here...', language)}]({best_case['case_link']})")
        st.write(f"**{translate_text('Relevancy Score:', language)}** {round(similarity_score, 2)}")

        st.write("---")

        st.subheader(translate_text("📜 Relevant Sections:", language))
        for section in best_case["sections"]:
            st.markdown(f"**🆔 {translate_text(section['section_id'], language)} - {translate_text(section['section_title'], language)}**")
            st.write(translate_text(section["section_description"], language))
            st.write("---")

        st.subheader(translate_text("✅ Top Strong Points:", language))
        for point in best_case["strong_points"][:5]:
            st.write(f"- {translate_text(point, language)}")

        st.subheader(translate_text("⚠️ Top Weak Points:", language))
        for point in best_case["weak_points"][:5]:
            st.write(f"- {translate_text(point, language)}")

        # Navigation Buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.case_index > 0:
                if st.button(translate_text("⬅️ Previous", language)):
                    st.session_state.case_index -= 1
                    st.rerun()

        with col2:
            if st.session_state.case_index < len(st.session_state.results) - 1:
                if st.button(translate_text("Next ➡️", language)):
                    st.session_state.case_index += 1
                    st.rerun()

# FAQ analysis UI
elif model_choice == "FAQs":
    faq_query = st.text_area(translate_text("Enter your question or query for FAQ:", language), height=150)
    faq_nombres = st.slider(translate_text("Select number of similar FAQs to retrieve:", language), min_value=1, max_value=10, value=5)

    if st.button(translate_text("Search FAQ", language)):
        if faq_query.strip():
            st.session_state.faq_results = find_relevant_faq(faq_query, faq_data, faq_nombres, language)
            st.session_state.faq_index = 0  # Reset index when new search is made

    if st.session_state.get("faq_results", []):
        result = st.session_state.faq_results[st.session_state.faq_index]
        best_faq = result["faq"]
        similarity_score = result["similarity_score"]

        st.subheader(translate_text(f"🔎 FAQ {st.session_state.faq_index + 1} of {len(st.session_state.faq_results)}", language))
        st.write(f"**{translate_text('FAQ Question:', language)}** {best_faq['prompt']}")
        st.write(f"**{translate_text('Relevancy Score:', language)}** {round(similarity_score, 2)}")
        st.write("---")

        st.subheader(translate_text("💡 Answer:", language))
        st.write(best_faq["completion"])  # Displaying the completion instead of answer

        # Navigation Buttons for FAQ
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.faq_index > 0:
                if st.button(translate_text("⬅️ Previous FAQ", language)):
                    st.session_state.faq_index -= 1
                    st.rerun()

        with col2:
            if st.session_state.faq_index < len(st.session_state.faq_results) - 1:
                if st.button(translate_text("Next FAQ ➡️", language)):
                    st.session_state.faq_index += 1
                    st.rerun()

footer = """
<style>
.footer {
    position: fixed;
    bottom: 10px;
    left: 0;
    width: 100%;
    text-align: center;
    padding: 10px;
    font-size: 14px;
    font-weight: bold;
    color: white;
}
</style>

<div class="footer">
    Developed by <span style="color:#4C9AFF;"><a href="https://github.com/AarDG10/LegalisAI-Test">Legalis Team 24-25</a></span>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)


# Legalis Team ~ Aarol D'Souza | Ananya Solanki | Leander Braganza | 24-25