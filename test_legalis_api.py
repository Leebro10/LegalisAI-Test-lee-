import streamlit as st
import requests
import json
from googletrans import Translator

# Initialize the translator
translator = Translator()

# Define the URL of your FastAPI backend
FASTAPI_URL = "http://127.0.0.1:8000/predict/"

# Language Selection
language = st.selectbox("Choose Language", ["English", "Hindi", "Marathi"])

# Translate function to switch the language dynamically
def translate_text(text, dest_language):
    if dest_language == "English":
        return text
    lang_code = {"Hindi": "hi", "Marathi": "mr"}.get(dest_language, "en")
    return translator.translate(text, dest=lang_code).text

# Streamlit Sidebar for model choice
st.sidebar.title(translate_text("LegalisAI - Query Interface", language))
model_choice = st.sidebar.selectbox(translate_text("Select Model", language), ["legalis", "faq"])

# Main title of the app
st.title(translate_text("LegalisAI - Legal Query Assistant", language))

# Text input for user query
user_input = st.text_area(translate_text("Enter your query:", language), "")

# Function to send the request to the FastAPI server
def send_request(text, model_choice):
    payload = {"text": text, "model_choice": model_choice}
    
    try:
        response = requests.post(FASTAPI_URL, json=payload)
        response.raise_for_status()  # Raise an exception for non-2xx responses
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error: {e}")
        return None

# Button to trigger prediction
if st.button(translate_text("Get Results", language)):
    if user_input.strip() == "":
        st.warning(translate_text("Please enter a query to proceed.", language))
    else:
        # Send the request and get the response
        results = send_request(user_input, model_choice)
        
        if results:
            if "results" in results:
                st.subheader(f"{translate_text('Results from', language)} {results['model']} {translate_text('Model', language)}")
                for idx, result in enumerate(results["results"], 1):
                    if model_choice == "legalis":
                        st.write(f"**{translate_text('Case', language)} {idx}:** {result['case_title']}")
                        st.write(f"**{translate_text('Link', language)}:** {result['case_link']}")
                        st.write(f"**{translate_text('Similarity Score', language)}:** {result['similarity_score']}")
                        st.write(f"**{translate_text('Strong Points', language)}:** {result['strong_points']}")
                        st.write(f"**{translate_text('Weak Points', language)}:** {result['weak_points']}")
                        
                        # Displaying similar sections for the legalis model
                        st.subheader(f"{translate_text('Most Similar Sections', language)}:")
                        for section in result['sections']:
                            st.write(f"**{translate_text('Section Title', language)}:** {section['title']}")
                            st.write(f"**{translate_text('Section Content', language)}:** {section['content']}")
                        
                    elif model_choice == "faq":
                        st.write(f"**{translate_text('FAQ', language)} {idx}:** {result['faq_prompt']}")
                        st.write(f"**{translate_text('Answer', language)}:** {result['faq_completion']}")
                        st.write(f"**{translate_text('Similarity Score', language)}:** {result['similarity_score']}")
            else:
                st.warning(translate_text("No results found.", language))
