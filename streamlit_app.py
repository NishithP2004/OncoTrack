import streamlit as st
from openai import OpenAI
import docx2txt
import os
import pandas as pd
import matplotlib.pyplot as plt
import re

with st.sidebar:
    kaggle_server_url = st.text_input("Kaggle Server URL", "https://dominant-usually-oyster.ngrok-free.app", key="kaggle-server-url", type="default")
    model_name = st.text_input("Model Name", "NishithP2004/tsa_oral_cancer_data_extraction_Meta-Llama-3.1-8B-bnb-4bit_10e", key="model-name")
    # Navigation Pane (Creative Sidebar)
    page = st.selectbox("Navigation", ["📤 Upload & Extract", "📊 Time Series Analysis", "📝 Summary", "💬 Chat with AI"])

st.title("🦠 OncoTrack: Time Series Analysis of Oral Cancer Progression")
st.caption("🔍 Analyze patient records and track disease progression over time")

# Session State for storing extracted text
if "extracted_text" not in st.session_state:
    st.session_state["extracted_text"] = ""

# Page 1: Upload & Extract
if page == "📤 Upload & Extract":
    st.header("Upload Patient Records")
    uploaded_files = st.file_uploader("Upload multiple Word Documents", accept_multiple_files=True, type=["docx"])
    
    if uploaded_files:
        extracted_texts = {}
        for file in uploaded_files:
            text = docx2txt.process(file)
            # Clean the extracted text
            cleaned_text = re.sub(r'(\r\n)+', "\n", text)
            cleaned_text = re.sub(r'\n{3,}', "\n", cleaned_text)
            cleaned_text = re.sub(r'\t+', " ", cleaned_text)
            extracted_texts[file.name] = cleaned_text
            st.success(f"Text extracted from {file.name} successfully!")
            st.text_area(f"Extracted Text Preview - {file.name}", cleaned_text, height=300)
        
        st.session_state["extracted_text"] = extracted_texts

# Page 2: Time Series Analysis
elif page == "📊 Time Series Analysis":
    st.header("Time Series Analysis")
    
    if not st.session_state["extracted_text"]:
        st.warning("Please upload and extract data first.")
    else:
        # Mock Data - Replace with actual processing logic
        data = {
            "Date": pd.date_range(start="1/1/2022", periods=10, freq='M'),
            "Tumor Size": [2.1, 2.5, 2.9, 3.2, 3.8, 4.1, 4.4, 4.9, 5.2, 5.6]
        }
        df = pd.DataFrame(data)
        
        st.line_chart(df.set_index("Date"))

# Page 3: Summary
elif page == "📝 Summary":
    st.header("Generated Summary")
    
    if not st.session_state["extracted_text"]:
        st.warning("Please upload and extract data first.")
    else:
        st.write("📄 **AI-Generated Report**")
        st.write("This section will generate a structured summary based on extracted patient records.")
        st.info("🚧 Feature under development!")

# Page 4: Chat with AI
elif page == "💬 Chat with AI":
    st.header("💬 Chatbot")
    st.caption("🚀 AI-powered chatbot for medical insights")
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    if prompt := st.chat_input():
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        
        client = OpenAI(api_key=openai_api_key)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
        msg = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)
