import streamlit as st
import replicate
import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables (for Replicate API key)
load_dotenv(find_dotenv())

# List of available Replicate models
replicate_model_names = [
    'meta/meta-llama-3-8b',
    'google-deepmind/gemma2-9b-it',
    'microsoft/phi-3-medium-4k-instruct',
    'mistralai/mixtral-8x7b-instruct-v0.1'
]

# Streamlit app
st.title("Replicate Model Stream Interface")

# Sidebar for selecting model and input parameters
with st.sidebar:
    st.header("Model Selection & Settings")

    # Dropdown for selecting Replicate model
    selected_model = st.selectbox("Select Replicate Model:", replicate_model_names)

    # Text area for custom prompt input
    prompt_input = st.text_area("Enter the prompt:", value="Story title: 3 llamas go for a walk\nSummary: The 3 llamas crossed a bridge and something unexpected happened\n\nOnce upon a time")

    # Optional settings for the model
    temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.6)
    top_p = st.slider("Top P:", min_value=0.0, max_value=1.0, value=0.9)
    presence_penalty = st.slider("Presence Penalty:", min_value=0.0, max_value=2.0, value=1.15)

# Prepare the input for the Replicate model
input = {
    "prompt": prompt_input,
    "min_tokens": 0,
    "temperature": temperature,
    "top_p": top_p,
    "presence_penalty": presence_penalty
}

# Button to trigger the Replicate model
if st.button("Generate Response"):
    st.write(f"Using model: {selected_model}")
    st.write("Response:")

    # Streaming response from replicate using the selected model
    response = ""
    for event in replicate.stream(selected_model, input=input):
        response += event['text']
        st.write(event['text'], end="")  # Show real-time output
