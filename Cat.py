import streamlit as st
import requests
import logging
import numpy as np
import cv2
from typing import Optional
from PIL import Image
from streamlit_lottie import st_lottie
import json
import keras  

# Constants
BASE_API_URL = "http://127.0.0.1:7860"
FLOW_ID = "03bd03f8-8200-4eec-9081-6f8ba2391f3b"
ENDPOINT = ""
TWEAKS = {
    "ChatInput-Osmkc": {},
    "ChatOutput-ZiTTv": {},
    "Prompt-PUvv4": {},
    "OpenAIModel-Z19mV": {},
    "Memory-YpgfI": {},
    "TextInput-y6BdN": {}
}

# Load Lottie animation
def load_lottie(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

# Function to run the flow
def run_flow(message: str, tweaks: Optional[dict] = None) -> dict:
    api_url = f"{BASE_API_URL}/api/v1/run/{ENDPOINT or FLOW_ID}"
    payload = {
        "input_value": message,
        "output_type": "chat",
        "input_type": "chat",
    }
    if tweaks:
        payload["tweaks"] = tweaks
    response = requests.post(api_url, json=payload)
    return response.json()

# Extract assistant's message
def extract_message(response: dict) -> str:
    try:
        return response['outputs'][0]['outputs'][0]['results']['message']['text']
    except (KeyError, IndexError):
        logging.error("No valid message found in response.")
        return "No valid message found in response."

@st.cache_resource
def load_model(filepath):
    model = keras.saving.load_model(filepath, custom_objects=None, compile=True, safe_mode=True)
    return model

# Load model and animation
model = load_model('model/model.keras')
lottie_cat = load_lottie("Animations/Animation - 1742280933412.json")

# App Styling
st.set_page_config(
    page_title="Cat Breed Prediction",
    page_icon="🐾",
    layout="wide"
)

# Main title and animation
with st.container():
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.markdown("<h1 style='color: #4F8BF9;'>🐾 Cat Breed Prediction</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #4F8BF9;'>Upload an image and predict the breed of your cat!</p>", unsafe_allow_html=True)
    with col2:
        st_lottie(lottie_cat, height=150, width=150, key="cat")

# Sidebar for uploading images
with st.sidebar:
    st.markdown("### 🖼️ Upload Cat Image")
    st.markdown("---")
    uploaded_image = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
        image = Image.open(uploaded_image)
        image = np.array(image)
        image = cv2.resize(image, (160, 160))
        image = np.expand_dims(image, axis=0)

        # Predict class
        predict = np.argmax(model.predict(image))
        class_names = [
            "Abyssinian", "Bombay", "Egyptian Mau", "Exotic Shorthair", "Himalayan", 
            "Maine Coon", "Regdoll", "Russian Blue", "Scottish Fold", "Siamese", "Sphynx"
        ]
        predicted_class = class_names[predict]

        st.success(f"**Predicted Class:** {predicted_class}")

        # Save predicted class in session state
        if not any(msg["content"] == f"What is {predicted_class}?" for msg in st.session_state.get("messages", [])):
            st.session_state.messages.append({
                "role": "user",
                "content": f"What is {predicted_class}?",
                "avatar": "🐾"
            })
            assistant_response = extract_message(run_flow(f"What is {predicted_class}?", tweaks=TWEAKS))
            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_response,
                "avatar": "🤖"
            })

# Chat Interface
st.markdown("#### 💬 Chat with Assistant")
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    bg_color = "#E8F0FE" if message["role"] == "assistant" else "#FCE8E6"
    text_color = "#202124"
    
    with st.container():
        st.markdown(
            f"""
            <div style="
                background-color: {bg_color};
                padding: 10px;
                border-radius: 10px;
                margin-bottom: 5px;
                color: {text_color};
            ">
                <b>{message['avatar']} {message['role'].capitalize()}</b><br>
                {message['content']}
            </div>
            """,
            unsafe_allow_html=True
        )

# Chat input box
query = st.chat_input("Ask me anything about cats...")
if query:
    with st.container():
        st.markdown(
            f"""
            <div style="
                background-color: #FCE8E6;
                padding: 10px;
                border-radius: 10px;
                color: #202124;
                margin-bottom: 5px;
            ">
                <b>🐾 You</b><br>
                {query}
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.session_state.messages.append({
        "role": "user",
        "content": query,
        "avatar": "🐾"
    })

    with st.spinner("Thinking..."):
        response = extract_message(run_flow(query, tweaks=TWEAKS))
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "avatar": "🤖"
        })
        st.markdown(
            f"""
            <div style="
                background-color: #E8F0FE;
                padding: 10px;
                border-radius: 10px;
                color: #202124;
                margin-bottom: 5px;
            ">
                <b>🤖 Assistant</b><br>
                {response}
            </div>
            """,
            unsafe_allow_html=True
        )