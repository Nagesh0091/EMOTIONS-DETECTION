import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import openai
import requests

st.set_page_config(page_title="Emotion Chat", page_icon="🧠", layout="centered")
st.markdown("<h1 style='text-align:center;'>😄 Emotion Detection & Chat App</h1>", unsafe_allow_html=True)


# ----------------------------
# ENVIRONMENT SETUP
# ----------------------------
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ----------------------------
# OPENAI API KEY SETUP
# ----------------------------
openai.api_key = st.secrets["openai_api_key"]  # Add this key in .streamlit/secrets.toml

# ----------------------------
# LOAD CNN MODEL (CACHED)
# ----------------------------
import os
import urllib.request
import tensorflow as tf
import streamlit as st

# Define model path
model_path = "emotion_model.h5"

# Download the model if not present
if not os.path.exists(model_path):
    print("Downloading model...")
    urllib.request.urlretrieve(
        "https://huggingface.co/Slytherinsoul/emotion-model/resolve/main/emotion_model.h5",
        model_path
    )
    print("Download complete.")

# Cache the model loading
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)

# Load the model
model = load_model()


# Emotion labels
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ----------------------------
# EMOTION PREDICTION FUNCTION
# ----------------------------
def predict_emotion(image):
    img = image.resize((48, 48)).convert('L')
    img_array = np.array(img).reshape(1, 48, 48, 1) / 255.0
    predictions = model.predict(img_array)[0]
    top_idx = np.argmax(predictions)
    predicted_class = class_names[top_idx]
    return predicted_class, predictions

# ----------------------------
# HUGGING FACE CHAT FUNCTION
# ----------------------------
def hf_chat(messages, hf_token):
    url = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
    headers = {"Authorization": f"Bearer {hf_token}"}
    # Convert messages to a single prompt (simple format)
    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant:"
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 128, "return_full_text": False}}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data[0]["generated_text"].split("assistant:")[-1].strip()
    else:
        return f"[HF API Error {response.status_code}] {response.text}"

# ----------------------------
# STREAMLIT PAGE CONFIGURATION
# ----------------------------

# Input option
option = st.radio("Choose input method:", ("Upload Image", "Use Camera"))

image = None
predicted_class = None
probabilities = None

# Upload option
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload a facial image...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        predicted_class, probabilities = predict_emotion(image)

# Camera option
elif option == "Use Camera":
    picture = st.camera_input("Take a picture")
    if picture:
        image = Image.open(picture)
        st.image(image, caption='Captured Image', use_column_width=True)
        predicted_class, probabilities = predict_emotion(image)

# ----------------------------
# DISPLAY PREDICTION & START CHAT
# ----------------------------
if predicted_class:
    st.subheader(f"🧠 Predicted Emotion: **{predicted_class}**")

    # Show emotion probabilities
    st.markdown("### Emotion Probabilities")
    st.bar_chart({class_names[i]: probabilities[i] for i in range(len(class_names))})

    st.markdown("📜 *Tell me more about what you're feeling...*")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "You are a friendly emotional support assistant. Have casual conversations and offer support to users based on their mood."},
            {"role": "user", "content": f"I'm feeling {predicted_class.lower()}."},
        ]

    user_reason = st.text_area("Why do you feel this way?", key="user_reason")

    if st.button("💬 Start Chat"):
        if user_reason.strip():
            st.session_state.messages.append(
                {"role": "user", "content": f"I feel {predicted_class.lower()} because {user_reason.strip()}."}
            )
        else:
            st.warning("Please write something about your emotion.")

# ----------------------------
# LIVE CHAT WITH OPENAI
# ----------------------------
if "messages" in st.session_state and len(st.session_state.messages) > 1:
    user_input = st.chat_input("Type your message...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("Assistant is typing..."):
            try:
                hf_token = st.secrets["hf_token"]
                reply = hf_chat(st.session_state.messages, hf_token)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"Chat Error: {e}")

    for msg in st.session_state.messages[1:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
