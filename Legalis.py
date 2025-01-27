import streamlit as st
import json
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the trained model and tokenizer
model_path = "./legalis_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)

# Load cases from JSON
with open("finalcases.json", "r", encoding="utf-8") as file:
    cases_data = json.load(file)

# Function to encode text using BERT
def encode_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

nombres = st.slider("Select number of similar cases to retrieve:", min_value=1, max_value=10, value=5)
# Function to find multiple relevant cases
def find_relevant_cases(user_input, cases, num_results=nombres):
    input_vector = encode_text(user_input)
    case_vectors = [encode_text(case["case_description"]) for case in cases]
    similarities = [cosine_similarity(input_vector, case_vec)[0][0] for case_vec in case_vectors]
    
    # Get top N cases with highest similarity scores
    top_indices = np.argsort(similarities)[-num_results:][::-1]
    
    results = []
    for index in top_indices:
        case = cases[index]
        results.append({
            "case": case,
            "similarity_score": similarities[index]
        })
    
    return results

# Streamlit UI
st.title("LegalisAI: Real Estate Legal Case Assistant ⚖️ ")

if "case_index" not in st.session_state:
    st.session_state.case_index = 0

user_input = st.text_area("Enter your case description:", height=150)

if st.button("Analyse Case"):
    if user_input.strip():
        st.session_state.results = find_relevant_cases(user_input, cases_data)
        st.session_state.case_index = 0  # Reset index when new search is made

# Display Retrieved Cases
if "results" in st.session_state and st.session_state.results:
    result = st.session_state.results[st.session_state.case_index]
    best_case = result["case"]
    similarity_score = result["similarity_score"]

    st.subheader(f"🔎 Case {st.session_state.case_index + 1} of {len(st.session_state.results)}")
    st.write(f"**Case ID:** {best_case['case_id']}")
    st.write(f"**Case Title:** {best_case['case_title']}")
    st.write(f"**Case PDF Link:** [Read More Here...]( {best_case['case_link']} )")
    st.write(f"**Relevancy Score:** {round(similarity_score, 2)}")
    st.write("---")

    st.subheader("📜 Relevant Sections:")
    for section in best_case["sections"]:
        st.markdown(f"**🆔 {section['section_id']} - {section['section_title']}**")
        st.write(section["section_description"])
        st.write("---")

    st.subheader("✅ Top Strong Points:")
    for point in best_case["strong_points"][:5]:
        st.write(f"- {point}")

    st.subheader("⚠️ Top Weak Points:")
    for point in best_case["weak_points"][:5]:
        st.write(f"- {point}")

    # Navigation Buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.session_state.case_index > 0:
            if st.button("⬅️ Previous"):
                st.session_state.case_index -= 1
                st.rerun()

    with col2:
        if st.session_state.case_index < len(st.session_state.results) - 1:
            if st.button("Next ➡️"):
                st.session_state.case_index += 1
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
