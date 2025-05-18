import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import mammoth
import requests  # For making HTTP requests
import base64
import uuid
from st_audiorec import st_audiorec # Corrected import based on user preference

with st.sidebar:
    kaggle_server_url = st.text_input("Kaggle Server URL", "https://dominant-usually-oyster.ngrok-free.app", key="kaggle-server-url", type="default")
    model_name = st.text_input("Model Name", "NishithP2004/tsa_oral_cancer_data_extraction_Meta-Llama-3.1-8B-bnb-4bit_10e", key="model-name")
    page = st.selectbox("Navigation", ["📤 Upload & Extract", "📝 Summary", "💬 Chat with AI"])
    
    if "reset_counter" not in st.session_state:
        st.session_state["reset_counter"] = 0
    if "session_id" not in st.session_state: # This implies a new session start
        st.session_state["session_id"] = str(uuid.uuid4())
        st.session_state.summary_text = None
        st.session_state.summary_error_text = None
        st.session_state.summary_loading = False
        st.session_state.generated_summaries_list = []
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
        st.session_state.last_submitted_audio_bytes = None # Initialize for audio loop fix
    
    if "summary_text" not in st.session_state: st.session_state.summary_text = None
    if "summary_error_text" not in st.session_state: st.session_state.summary_error_text = None
    if "summary_loading" not in st.session_state: st.session_state.summary_loading = False
    if "generated_summaries_list" not in st.session_state: st.session_state.generated_summaries_list = []
    if "messages" not in st.session_state: st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
    if "last_submitted_audio_bytes" not in st.session_state: st.session_state.last_submitted_audio_bytes = None # Ensure initialized


    if st.button("Reset Session"):
        st.session_state["extracted_text"] = ""
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
        st.session_state["edited_responses"] = None
        st.session_state["all_responses"] = None
        st.session_state["reset_counter"] += 1
        st.session_state["session_id"] = str(uuid.uuid4())
        st.session_state.summary_text = None
        st.session_state.summary_error_text = None
        st.session_state.summary_loading = False
        st.session_state.generated_summaries_list = []
        st.session_state.last_submitted_audio_bytes = None # Reset for audio loop fix
        if "pending_audio_payload" in st.session_state: st.session_state.pop("pending_audio_payload")
        if "pending_audio_payload_source_bytes" in st.session_state: st.session_state.pop("pending_audio_payload_source_bytes")
        if "pending_text_payload" in st.session_state: st.session_state.pop("pending_text_payload")
        st.success("Session has been reset.")
        st.rerun()

st.title("🦠 OncoTrack: Time Series Analysis of Oral Cancer Progression") # Title might need update if TS is gone
st.caption("🔍 Analyze patient records and track disease progression over time")

if "extracted_text" not in st.session_state:
    st.session_state["extracted_text"] = ""

if page == "📤 Upload & Extract":
    st.header("Upload Patient Records")
    uploaded_files = st.file_uploader(
        "Upload multiple Word Documents", 
        accept_multiple_files=True, 
        type=["docx"], 
        key=f"uploaded_files_{st.session_state['reset_counter']}"
    )
    
    if uploaded_files:
        extracted_texts = {}
        all_progress_notes = []
        for file in uploaded_files:
            try:
                result = mammoth.extract_raw_text(file)
                text = result.value
                cleaned_text = re.sub(r'(\n){3,}', "\n", text)
                cleaned_text = re.sub(r'(\t)+', " ", cleaned_text)
                cleaned_text = re.sub(r'(\r\n)+', "\n", cleaned_text)
                progress_notes = re.findall(
                    r"Date\s*:\s*\d{2}\/\d{2}\/\d{4}\s*ProgressNotes\s*:(?:.*?(?=Date\s*:\s*\d{2}\/\d{2}\/\d{4}\s*ProgressNotes\s*:|Signed By|$))", 
                    cleaned_text,
                    flags=re.DOTALL
                )
                st.success(f"Text extracted from {file.name} successfully!")
                st.info(f"Found {len(progress_notes)} progress note(s) in {file.name}")
                st.text_area(f"Extracted Text Preview - {file.name}", cleaned_text, height=300)
                extracted_texts[file.name] = {
                    "cleaned_text": cleaned_text,
                    "progress_notes": progress_notes,
                    "responses": []
                }
                for note in progress_notes:
                    all_progress_notes.append((file.name, note))
            except Exception as err:
                st.error(f"Error processing {file.name}: {err}")

        total_notes = len(all_progress_notes)
        progress_bar = st.progress(0)
        progress_text = st.empty()
        note_counter = 0

        for file_obj in uploaded_files: # Renamed 'file' to 'file_obj' to avoid conflict
            responses = []
            if file_obj.name in extracted_texts: # Check if file was successfully processed earlier
                for note in extracted_texts[file_obj.name]["progress_notes"]:
                    note_counter += 1
                    payload = {
                        "text": note,
                        "session_id": st.session_state.session_id # Add session_id for vector store
                    }
                    # Use a try-except block for the API call
                    try:
                        if not kaggle_server_url:
                            st.error("Kaggle Server URL not set. Cannot extract features.")
                            responses.append({"order": note_counter, "note": note, "response": "Error: Kaggle Server URL not set."})
                            continue

                        api_url = f"{kaggle_server_url.rstrip('/')}/extract"
                        r = requests.post(api_url, json=payload, timeout=60) # Added timeout
                        r.raise_for_status() # Raise HTTPError for bad responses (4XX or 5XX)
                        response_data = r.json().get("response", f"Error: 'response' key missing in API result from {api_url}")
                        responses.append({"order": note_counter, "note": note, "response": response_data})
                    except requests.exceptions.RequestException as e:
                        st.error(f"API Error for note {note_counter}: {e}")
                        responses.append({"order": note_counter, "note": note, "response": f"Error: API request failed - {e}"})
                    except Exception as e: # Catch other potential errors like JSON parsing
                        st.error(f"Processing Error for note {note_counter}: {e}")
                        responses.append({"order": note_counter, "note": note, "response": f"Error: Processing failed - {e}"})
                    
                    progress = note_counter / total_notes if total_notes > 0 else 1
                    progress_bar.progress(progress)
                    remaining = total_notes - note_counter
                    progress_text.text(f"Remaining progress notes: {remaining}")
                extracted_texts[file_obj.name]["responses"] = responses
        
        st.session_state["extracted_text"] = extracted_texts
        st.session_state["all_responses"] = [resp for file_data in extracted_texts.values() for resp in file_data["responses"]]

    if st.session_state.get("all_responses"):
        st.markdown("## Extracted Features")
        ordered_responses = sorted(st.session_state["all_responses"], key=lambda x: x["order"])
        
        # Filter out responses that are error strings before creating DataFrame
        valid_response_dicts = []
        for entry in ordered_responses:
            if isinstance(entry["response"], dict):
                 valid_response_dicts.append(entry["response"])
            else:
                # Optionally log or display that some notes had feature extraction errors
                st.warning(f"Note (Order: {entry['order']}) had a feature extraction error: {entry['response']}")

        if valid_response_dicts:
            df_responses = pd.DataFrame(valid_response_dicts)
            st.session_state["edited_responses"] = st.data_editor(
                df_responses,
                num_rows="dynamic",
                use_container_width=True,
                disabled=True, 
                key=f"data_editor_unique_{st.session_state['reset_counter']}" # Dynamic key
            )
        else:
            st.info("No valid features extracted to display.")


elif page == "📝 Summary":
    st.header("Generate Summary")

    source_df = st.session_state.get("edited_responses")

    if source_df is None or not isinstance(source_df, pd.DataFrame) or source_df.empty:
        st.warning("Please upload and extract features from documents first. The extracted features will be used for summarization.")
    else:
        st.info(f"Found {len(source_df)} extracted feature sets. Unique sets will be summarized.")
        st.dataframe(source_df.head()) # Show a preview of what will be summarized

        if st.button("Generate", key=f"generate_summaries_btn_{st.session_state['reset_counter']}", disabled=st.session_state.summary_loading):
            if not kaggle_server_url:
                st.error("Kaggle Server URL not set. Cannot generate summaries.")
            else:
                st.session_state.summary_loading = True
                st.session_state.generated_summaries_list = [] # Clear previous summaries
                st.session_state.summary_error_text = None # Clear overall errors
                st.rerun() # Show loading state

        if st.session_state.summary_loading:
            with st.spinner("Generating summaries for unique feature sets... Please wait."):
                unique_feature_df = source_df.drop_duplicates()
                total_unique_sets = len(unique_feature_df)
                
                if total_unique_sets == 0:
                    st.session_state.summary_error_text = "No unique feature sets found to summarize."
                    st.session_state.summary_loading = False
                    st.rerun()
                else:
                    progress_bar_summaries = st.progress(0)
                    temp_summaries_list = []

                    for i, row_tuple in enumerate(unique_feature_df.iterrows()):
                        index, row_series = row_tuple
                        feature_payload = row_series.to_dict()
                        
                        # The backend filters by `cols`. Client sends all, backend filters.
                        # Ensure no NaN values that might cause JSON issues if not handled by backend.
                        # df.to_dict usually handles this, but good to be aware.
                        # Pandas NaNs become `null` in JSON, which is fine.

                        api_url = f"{kaggle_server_url.rstrip('/')}/summarise_features?session_id={st.session_state.session_id}"
                        summary_item = {"original_features": feature_payload} # Store original for context

                        try:
                            r = requests.post(api_url, json=feature_payload, timeout=120) 
                            r.raise_for_status()
                            response_data = r.json()
                            if "summary" in response_data:
                                summary_item["summary"] = response_data["summary"]
                            elif "error" in response_data:
                                summary_item["error"] = response_data["error"]
                            else:
                                summary_item["error"] = "Unexpected response format from summary API."
                        except requests.exceptions.HTTPError as e:
                            error_detail = f"API Error: {e.response.status_code}"
                            try:
                                error_json = e.response.json()
                                error_detail += f" - {error_json.get('error', e.response.text)}"
                            except ValueError:
                                error_detail += f" - {e.response.text}"
                            summary_item["error"] = error_detail
                        except requests.exceptions.RequestException as e:
                            summary_item["error"] = f"Connection/Request Error: {e}"
                        except Exception as e:
                            summary_item["error"] = f"An unexpected error occurred: {e}"
                        
                        temp_summaries_list.append(summary_item)
                        progress_bar_summaries.progress((i + 1) / total_unique_sets)
                    
                    st.session_state.generated_summaries_list = temp_summaries_list
                    st.session_state.summary_loading = False
                    st.rerun() # Update UI with results

        if st.session_state.generated_summaries_list:
            st.markdown("### Generated Summaries")
            for idx, item in enumerate(st.session_state.generated_summaries_list):
                st.markdown(f"--- ")
                st.markdown(f"**Summary for Feature Set {idx + 1}:**")
                # Optionally display the features that generated this summary
                # with st.expander("View Features for this Summary"):
                #    st.json(item.get("original_features", {}))
                
                if "summary" in item:
                    st.success(item["summary"])
                elif "error" in item:
                    st.error(f"Failed to generate summary for this feature set: {item.get('error', 'Unknown error')}")
        
        if st.session_state.summary_error_text: # For overall process errors
            st.error(st.session_state.summary_error_text)


elif page == "💬 Chat with AI":
    st.header("💬 Chatbot")
    st.caption("🚀 AI-powered chatbot for medical insights")

    if "messages" not in st.session_state: st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    if "last_submitted_audio_bytes" not in st.session_state: st.session_state.last_submitted_audio_bytes = None

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if isinstance(msg.get("audio_details"), dict):
                details = msg["audio_details"]
                with st.expander("See transcription details", expanded=False):
                    if "transcribed_text" in details and details["transcribed_text"]:
                        st.caption(f"🗣️ You said (transcribed): {details['transcribed_text']}")
                    if "translated_text" in details and details["translated_text"] and details.get("transcribed_text") != details["translated_text"]:
                        st.caption(f"🌐 Translated to English: {details['translated_text']}")
                    if "detected_language" in details:
                        st.caption(f"🌍 Detected language: {details['detected_language']}")
    
    st.markdown("---")
    
    # Define chat submission handler function
    def handle_chat_submission(payload):
        if not kaggle_server_url:
            if st.session_state.messages and st.session_state.messages[-1]["content"] == "⌛ Assistant is thinking...":
                st.session_state.messages.pop()
            st.session_state.messages.append({"role": "assistant", "content": "⚠️ Kaggle Server URL not set. Cannot send message."})
            st.rerun()
            return

        api_url = f"{kaggle_server_url.rstrip('/')}/chat_interaction?session_id={st.session_state.session_id}"
        try:
            r = requests.post(api_url, json=payload, timeout=60)
            r.raise_for_status()
            response_data = r.json()
            if st.session_state.messages and st.session_state.messages[-1]["content"] == "⌛ Assistant is thinking...":
                st.session_state.messages.pop()
            assistant_response_content = response_data.get("response", "No response content from assistant.")
            audio_details_for_display = None
            if response_data.get("input_type") == "audio":
                audio_details_for_display = {
                    "transcribed_text": response_data.get("transcribed_text"),
                    "translated_text": response_data.get("translated_text"),
                    "detected_language": response_data.get("detected_language")
                }
            st.session_state.messages.append({
                "role": "assistant", 
                "content": assistant_response_content,
                "audio_details": audio_details_for_display
            })
        except requests.exceptions.HTTPError as e:
            if st.session_state.messages and st.session_state.messages[-1]["content"] == "⌛ Assistant is thinking...":
                st.session_state.messages.pop()
            error_message = f"API Error: {e.response.status_code}"
            try:
                error_json = e.response.json()
                error_message += f" - {error_json.get('error', e.response.text)}"
            except ValueError:
                error_message += f" - {e.response.text}"
            st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {error_message}"})
        except requests.exceptions.RequestException as e:
            if st.session_state.messages and st.session_state.messages[-1]["content"] == "⌛ Assistant is thinking...":
                st.session_state.messages.pop()
            st.session_state.messages.append({"role": "assistant", "content": f"⚠️ Connection Error: {e}"})
        except Exception as e:
            if st.session_state.messages and st.session_state.messages[-1]["content"] == "⌛ Assistant is thinking...":
                st.session_state.messages.pop()
            st.session_state.messages.append({"role": "assistant", "content": f"⚠️ An unexpected error occurred: {e}"})
        finally:
            st.rerun()

    # Audio Recorder Input - no key, state-based loop prevention
    st.write("🎤 **Record your message** or type below:")
    wav_audio_data = st_audiorec() # Call without key
    chat_input_disabled = not kaggle_server_url

    # Process pending audio payload (set in a previous run)
    if "pending_audio_payload" in st.session_state and st.session_state.pending_audio_payload:
        payload_to_send = st.session_state.pop("pending_audio_payload")
        # Crucially, now mark the source bytes (that created this payload) as processed
        if "pending_audio_payload_source_bytes" in st.session_state:
             st.session_state.last_submitted_audio_bytes = st.session_state.pop("pending_audio_payload_source_bytes")
        handle_chat_submission(payload_to_send)

    # Detect new audio from the component in the current run
    elif wav_audio_data is not None and wav_audio_data != st.session_state.last_submitted_audio_bytes:
        # This block is entered ONLY if new audio is recorded and it's different from the last *submitted* one.
        if chat_input_disabled:
            st.warning("Cannot send audio: Kaggle Server URL missing.")
        else:
            st.audio(wav_audio_data, format='audio/wav')
            audio_b64 = base64.b64encode(wav_audio_data).decode('utf-8')
            
            st.session_state.messages.append({"role": "user", "content": "🎤 User sent an audio message."})
            st.session_state.messages.append({"role": "assistant", "content": "⌛ Assistant is thinking..."})
            
            # Store the actual audio data bytes that will form this payload
            st.session_state.pending_audio_payload_source_bytes = wav_audio_data 
            st.session_state.pending_audio_payload = {"audio": audio_b64} # Set payload for next run
            
            st.rerun() # Rerun to show user message & thinking, then process on next run

    # Text Input
    if "pending_text_payload" in st.session_state and st.session_state.pending_text_payload:
        payload = st.session_state.pop("pending_text_payload")
        handle_chat_submission(payload)
    elif prompt := st.chat_input("Type your message...", disabled=chat_input_disabled, key=f"chat_input_{st.session_state.get('reset_counter', 0)}"):
        if chat_input_disabled:
            st.warning("Cannot send message: Kaggle Server URL missing.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": "⌛ Assistant is thinking..."})
            st.session_state.pending_text_payload = {"text": prompt}
            st.rerun()

else:
    st.error("Page not found.")