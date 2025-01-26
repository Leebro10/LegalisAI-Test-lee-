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
with open("cases.json", "r", encoding="utf-8") as file:
    cases_data = json.load(file)

# Function to encode text using BERT
def encode_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Function to find the most relevant case
def find_relevant_case(user_input, cases):
    input_vector = encode_text(user_input)
    case_vectors = [encode_text(case["case_description"]) for case in cases]
    similarities = [cosine_similarity(input_vector, case_vec)[0][0] for case_vec in case_vectors]
    best_match_index = np.argmax(similarities)
    return cases[best_match_index], similarities[best_match_index]

# Streamlit UI
st.title("LegalisAI: Real Estate Legal Case Assistant")

user_input = st.text_area("Enter your case description:", height=150)

if st.button("Find Relevant Sections"):
    if user_input.strip():
        best_case, similarity_score = find_relevant_case(user_input, cases_data)

        st.subheader("🔎 Most Relevant Case Found:")
        st.write(f"**Case ID:** {best_case['case_id']}")
        st.write(f"**Case Title:** {best_case['case_title']}")
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

    else:
        st.warning("Please enter a case description to analyze.")

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
    color: white;  /* White text for the rest */
}
</style>

<div class="footer">
    Developed by <span style="color:#4C9AFF;"><a href="https://github.com/AarDG10/LegalisAI-Test">Legalis Team 24-25</a></span>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)


#Legalis Team ~ Aarol D'Souza | Ananya Solanki | Leander Braganza | 24-25