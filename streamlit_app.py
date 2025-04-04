import streamlit as st
from openai import OpenAI
import pandas as pd
import matplotlib.pyplot as plt
import re
import mammoth
import requests  # New import for sending POST requests

with st.sidebar:
    kaggle_server_url = st.text_input("Kaggle Server URL", "https://dominant-usually-oyster.ngrok-free.app", key="kaggle-server-url", type="default")
    model_name = st.text_input("Model Name", "NishithP2004/tsa_oral_cancer_data_extraction_Meta-Llama-3.1-8B-bnb-4bit_10e", key="model-name")
    # Navigation Pane (Creative Sidebar)
    page = st.selectbox("Navigation", ["📤 Upload & Extract", "📊 Time Series Analysis", "📝 Summary", "💬 Chat with AI"])
    
    # Add a reset counter to dynamically reset widgets
    if "reset_counter" not in st.session_state:
        st.session_state["reset_counter"] = 0

    if st.button("Reset Session"):
        st.session_state["extracted_text"] = ""
        st.session_state["messages"] = []
        st.session_state["edited_responses"] = None  # Clear edited responses
        st.session_state["all_responses"] = None  # Clear all responses
        st.session_state["reset_counter"] += 1  # Increment the reset counter
        st.success("Session has been reset.")
        st.rerun()

st.title("🦠 OncoTrack: Time Series Analysis of Oral Cancer Progression")
st.caption("🔍 Analyze patient records and track disease progression over time")

# Session State for storing extracted text (persisted across page switches)
if "extracted_text" not in st.session_state:
    st.session_state["extracted_text"] = ""

# Page 1: Upload & Extract
if page == "📤 Upload & Extract":
    st.header("Upload Patient Records")
    
    # Use the reset counter to dynamically change the key for the file uploader
    uploaded_files = st.file_uploader(
        "Upload multiple Word Documents", 
        accept_multiple_files=True, 
        type=["docx"], 
        key=f"uploaded_files_{st.session_state['reset_counter']}"
    )
    
    if uploaded_files:
        extracted_texts = {}
        all_progress_notes = []  # List to accumulate (file_name, note) tuples

        # First pass: Process each file and extract progress notes
        for file in uploaded_files:
            try:
                # Extract raw text using Mammoth without manually reading bytes
                result = mammoth.extract_raw_text(file)
                text = result.value

                # Clean the extracted text
                cleaned_text = re.sub(r'(\n){3,}', "\n", text)
                cleaned_text = re.sub(r'(\t)+', " ", cleaned_text)
                cleaned_text = re.sub(r'(\r\n)+', "\n", cleaned_text)

                # Extract progress notes using the regex pattern
                progress_notes = re.findall(
                    r"Date\s*:\s*\d{2}\/\d{2}\/\d{4}\s*ProgressNotes\s*:(?:.*?(?=Date\s*:\s*\d{2}\/\d{2}\/\d{4}\s*ProgressNotes\s*:|Signed By|$))", 
                    cleaned_text,
                    flags=re.DOTALL
                )

                st.success(f"Text extracted from {file.name} successfully!")
                st.info(f"Found {len(progress_notes)} progress note(s) in {file.name}")
                st.text_area(f"Extracted Text Preview - {file.name}", cleaned_text, height=300)

                # Save the extracted text and progress notes
                extracted_texts[file.name] = {
                    "cleaned_text": cleaned_text,
                    "progress_notes": progress_notes,
                    "responses": []  # Will be populated later
                }

                # Accumulate notes for total count
                for note in progress_notes:
                    all_progress_notes.append((file.name, note))

            except Exception as err:
                st.error(f"Error processing {file.name}: {err}")

        total_notes = len(all_progress_notes)
        progress_bar = st.progress(0)
        progress_text = st.empty()
        note_counter = 0

        # Second pass: Process each progress note and update the progress bar
        for file in uploaded_files:
            responses = []
            for note in extracted_texts[file.name]["progress_notes"]:
                note_counter += 1  # Increment counter for each note
                payload = {"text": note}
                r = requests.post(f"{kaggle_server_url}/extract", json=payload)
                if r.status_code == 200:
                    response_data = r.json()["response"]
                    responses.append({
                        "order": note_counter,
                        "note": note,
                        "response": response_data
                    })
                else:
                    st.error(f"Error: {r.text}")
                    responses.append({
                        "order": note_counter,
                        "note": note,
                        "response": f"Error: Status code {r.status_code}"
                    })

                # Update the progress bar and display the remaining counter
                progress = note_counter / total_notes if total_notes > 0 else 1
                progress_bar.progress(progress)
                remaining = total_notes - note_counter
                progress_text.text(f"Remaining progress notes: {remaining}")

            extracted_texts[file.name]["responses"] = responses

        # Cache the extracted responses in session state to avoid re-sending POST requests
        st.session_state["extracted_text"] = extracted_texts
        st.session_state["all_responses"] = [resp for file in extracted_texts.values() for resp in file["responses"]]

    # If there are cached POST responses, display them in order
    if st.session_state.get("all_responses"):
        st.markdown("## Extracted Features")
        # Sort responses by the "order" key to preserve order
        ordered_responses = sorted(st.session_state["all_responses"], key=lambda x: x["order"])
        # Extract only the response dict from each entry
        response_dicts = [entry["response"] for entry in ordered_responses]
        # Create a DataFrame from the list of dicts; each row will contain the 24 key/value pairs
        df_responses = pd.DataFrame(response_dicts)
        st.session_state["edited_responses"] = st.data_editor(
            df_responses,
            num_rows="dynamic",
            use_container_width=True,
            disabled=True,
            key="data_editor_unique"
        )

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
        if not kaggle_server_url:  # Assuming API key variable was meant to be kaggle_server_url
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        
        client = OpenAI(api_key=kaggle_server_url)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
        msg = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)