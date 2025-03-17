import streamlit as st
import requests
import logging
import numpy as np
import cv2
from typing import Optional
from PIL import Image

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

# Function to extract the assistant's message from the response
def extract_message(response: dict) -> str:
    try:
        return response['outputs'][0]['outputs'][0]['results']['message']['text']
    except (KeyError, IndexError):
        logging.error("No valid message found in response.")
        return "No valid message found in response."

@st.cache_resource
def load_model(filepath):
    import keras  
    model = keras.saving.load_model(filepath, custom_objects=None, compile=True, safe_mode=True)
    return model

# Load the model
model = load_model('model.keras')

# Main function
def main():
    st.title("Computer Vision + LLM 🤖")
    st.write("🢀 Please capture an image first before you start asking questions.")

    # Sidebar for file uploader
    with st.sidebar:
        st.header("Upload Image")
        image = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])

        if image is not None:
            st.image(image, caption="Uploaded Image", use_container_width=True)
            image = Image.open(image)
            image = np.array(image)
            image = cv2.resize(image, (160, 160))
            image = np.expand_dims(image, axis=0)

            # Predict image class
            predict = np.argmax(model.predict(image))
            class_names = [
                "Abyssinian", "Bombay", "Egyptian Mau", "Exotic Shorthair", "Himalayan", 
                "Maine Coon", "Regdoll", "Russian Blue", "Scottish Fold", "Siamese", "Sphynx"
            ]
            predicted_class = class_names[predict]
            st.sidebar.success(f"**Predicted Class:** {predicted_class}")

            # Update tweaks for flow input
            TWEAKS["TextInput-y6BdN"]["input_value"] = predicted_class

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all previous messages with avatars
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.write(message["content"])

    # Chat input for user query
    if query := st.chat_input("Ask me anything"):
        # Display user message in chat
        with st.chat_message("user", avatar="☠️"):
            st.write(query)

        # Save user message to session state
        st.session_state.messages.append({
            "role": "user",
            "content": query,
            "avatar": "☠️"
        })

        # Get assistant's response
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                assistant_response = extract_message(run_flow(query, tweaks=TWEAKS))
                message_placeholder.write(assistant_response)

        # Save assistant's response to session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_response,
            "avatar": "🤖"
        })

if __name__ == "__main__":
    main()