import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO
from fpdf import FPDF
import base64
import PyPDF2
from docx import Document


API_URL = "http://127.0.0.1:8000"

# Set page configuration for Streamlit
st.set_page_config(
    page_title="Hate Speech Detection Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


#                  INITIALIZE SESSION STATE


if "access_token" not in st.session_state:
    st.session_state["access_token"] = None
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = "MBERT"
if "dashboard_option" not in st.session_state:
    st.session_state["dashboard_option"] = "Chatbot"
if "conversation" not in st.session_state:
    st.session_state["conversation"] = []
if "disclaimer_accepted" not in st.session_state:
    st.session_state["disclaimer_accepted"] = False
if "background_image" not in st.session_state:
    st.session_state["background_image"] = None
if "para_results" not in st.session_state:
    st.session_state["para_results"] = None
if "background_theme" not in st.session_state:
    st.session_state["background_theme"] = "Default"
if "community_datasets" not in st.session_state:
    st.session_state["community_datasets"] = []
# Quiz-related states:
if "quiz_started" not in st.session_state:
    st.session_state["quiz_started"] = False
if "quiz_score" not in st.session_state:
    st.session_state["quiz_score"] = 0
if "current_question_index" not in st.session_state:
    st.session_state["current_question_index"] = 0
if "quiz_questions" not in st.session_state:
    st.session_state["quiz_questions"] = []


#                  HELPER FUNCTIONS


def label_to_emoji(label: str) -> str:
    """Map detected labels to emojis for better UI experience."""
    if label == "hate_speech":
        return "😡"
    elif label == "offensive":
        return "😠"
    elif label == "neutral":
        return "🙂"
    elif label == "hate":       # for French
        return "🇫🇷😡"
    elif label == "non_hate":   # for French
        return "🇫🇷🙂"
    return ""


#                  DISCLAIMER & AUTHENTICATION


def disclaimer():
    st.subheader("User Disclaimer")
    st.warning("By continuing, you agree that this tool is for research purposes only and should not be misused.")
    if st.button("I Agree", key="agree_button"):
        st.session_state["disclaimer_accepted"] = True
        st.rerun()

def register_user():
    st.subheader("Create a New Account")
    username = st.text_input("Username")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Register"):
        payload = {"username": username, "email": email, "password": password}
        resp = requests.post(f"{API_URL}/register", json=payload)
        if resp.status_code == 200:
            st.success("Registration successful!")
        else:
            st.error("Registration failed.")

def login_user():
    if st.session_state["logged_in"]:
        st.success("You are already logged in!")
    else:
        st.subheader("Log In to Your Account")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            payload = {"username": username, "password": password}
            resp = requests.post(f"{API_URL}/login", json=payload)
            if resp.status_code == 200:
                st.session_state["logged_in"] = True
                st.session_state["access_token"] = resp.json()["access_token"]
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password.")


#                      QUIZ FEATURE


def quiz():
    """
    Enhanced Quiz:
      - Immediate feedback on correct/incorrect answers.
      - 'Quit Quiz' flows into partial score display, then user can return to main dashboard.
      - Custom background image only for the quiz interface.
    """

    # 1) Apply a custom background for the quiz
    st.markdown(
        """
        <style>
        .stApp {
            background: url("https://images.unsplash.com/photo-1573497019571-bc9e2531ea3d?fit=crop&w=1600&q=80");
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # 2) Initialize quiz-related session state if not present
    if "quiz_questions" not in st.session_state:
        st.session_state["quiz_questions"] = None
    if "quiz_index" not in st.session_state:
        st.session_state["quiz_index"] = 0
    if "quiz_score" not in st.session_state:
        st.session_state["quiz_score"] = 0
    if "show_answer_feedback" not in st.session_state:
        st.session_state["show_answer_feedback"] = False
    if "answer_feedback_msg" not in st.session_state:
        st.session_state["answer_feedback_msg"] = ""
    if "quiz_completed" not in st.session_state:
        st.session_state["quiz_completed"] = False
    # Tracks whether user chose to quit, so we can show partial score
    if "quiz_quitting" not in st.session_state:
        st.session_state["quiz_quitting"] = False

    # 3) Two-step logic for quitting
    def quit_quiz():
        # Step 1: Indicate the user is quitting
        st.session_state["quiz_quitting"] = True
        st.rerun()

    def reset_quiz_and_return():
        # Step 2: Reset states and return to dashboard
        st.session_state["quiz_questions"] = None
        st.session_state["quiz_index"] = 0
        st.session_state["quiz_score"] = 0
        st.session_state["quiz_completed"] = False
        st.session_state["show_answer_feedback"] = False
        st.session_state["answer_feedback_msg"] = ""
        st.session_state["quiz_quitting"] = False
        # Navigate back to your default dashboard (e.g. Chatbot)
        st.session_state["dashboard_option"] = "Chatbot"
        st.rerun()

    st.title("Hate Speech Awareness Quiz")

    # If user is in the "quit" state, display partial score
    if st.session_state["quiz_quitting"]:
        # Show partial or final score
        total_questions = len(st.session_state["quiz_questions"] or [])
        current_index = st.session_state["quiz_index"]
        st.warning("You have chosen to quit the quiz.")
        st.info(f"Your partial score is {st.session_state['quiz_score']} out of {current_index} attempted question(s).")

        # Button to confirm returning to dashboard
        if st.button("Return to Dashboard"):
            reset_quiz_and_return()
        return

    # Check if quiz questions are loaded
    if st.session_state["quiz_questions"] is None:
        # Fetch questions only once
        token = st.session_state.get("access_token")
        if not token:
            st.error("Please log in to access the quiz.")
            return
        try:
            resp = requests.get(f"{API_URL}/quiz", params={"limit": 10, "token": token})
            if resp.status_code == 200:
                st.session_state["quiz_questions"] = resp.json()["questions"]
                st.session_state["quiz_index"] = 0
                st.session_state["quiz_score"] = 0
                st.session_state["quiz_completed"] = False
            else:
                st.error("Failed to load quiz questions.")
                return
        except requests.exceptions.RequestException as e:
            st.error(f"Error loading quiz questions: {str(e)}")
            return

    # If quiz is completed
    if st.session_state["quiz_completed"]:
        st.success(f"Quiz completed! Your score: {st.session_state['quiz_score']} / {len(st.session_state['quiz_questions'])}")
        if st.button("Return to Dashboard"):
            reset_quiz_and_return()
        return

    # 4) Fetch current question
    questions = st.session_state["quiz_questions"]
    q_index = st.session_state["quiz_index"]

    # If user answered all questions
    if q_index >= len(questions):
        st.session_state["quiz_completed"] = True
        st.rerun()

    current_question = questions[q_index]
    st.markdown(f"**Question {q_index + 1}: {current_question['question_text']}**")
    options = current_question["options"]
    correct_opt_char = current_question["correct_option"]
    correct_opt_text = options[correct_opt_char]

    # 5) Radio input for user answer
    key_for_radio = f"quiz_answer_{q_index}"
    if key_for_radio not in st.session_state:
        st.session_state[key_for_radio] = None

    user_answer = st.radio(
        "Choose your answer:",
        [options["A"], options["B"], options["C"], options["D"]],
        key=key_for_radio
    )

    # 6) Display feedback if available
    if st.session_state["show_answer_feedback"]:
        st.info(st.session_state["answer_feedback_msg"])

    # Submit or Next logic
    submit_label = "Submit Answer" if not st.session_state["show_answer_feedback"] else "Next Question"

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button(submit_label):
            if not st.session_state["show_answer_feedback"]:
                # First click => check user answer
                if user_answer is None or user_answer == "":
                    st.warning("Please select an answer before submitting!")
                else:
                    # Compare user answer to correct one
                    if user_answer == correct_opt_text:
                        st.session_state["quiz_score"] += 1
                        st.session_state["answer_feedback_msg"] = f"✅ Correct! The answer is: {correct_opt_text}"
                    else:
                        st.session_state["answer_feedback_msg"] = f"❌ Incorrect. The correct answer is: {correct_opt_text}"
                    st.session_state["show_answer_feedback"] = True
                    st.rerun()
            else:
                # Second click => move to next question
                st.session_state["quiz_index"] += 1
                st.session_state["show_answer_feedback"] = False
                st.session_state["answer_feedback_msg"] = ""
                st.rerun()

    with col2:
        if st.button("Quit Quiz"):
            quit_quiz()




def fetch_quiz_questions():
    """Fetch quiz questions from the backend."""
    token = st.session_state.get("access_token")
    if not token:
        st.error("You need to log in to access the quiz.")
        return
    try:
        # Example: fetch 10 random questions
        resp = requests.get(f"{API_URL}/quiz", params={"token": token, "limit": 10})
        if resp.status_code == 200:
            st.session_state["quiz_questions"] = resp.json()["questions"]
        else:
            st.error(f"Failed to fetch quiz questions: {resp.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch quiz questions: {str(e)}")


#                   SIDEBAR MENU & THEMES


def customize_background():
    uploaded_file = st.file_uploader("Upload a background image:", type=["png", "jpg", "jpeg"], key="background_upload")

    if uploaded_file:
        # Convert image to Base64
        image_bytes = uploaded_file.read()
        encoded = base64.b64encode(image_bytes).decode()
        st.session_state["background_image"] = f"data:image/png;base64,{encoded}"

        # Display confirmation
        st.success("Background image updated successfully!")
        st.rerun()  # Force UI update

    # Apply background image if available
    if "background_image" in st.session_state and st.session_state["background_image"]:
        background_css = f"""
        <style>
            .stApp {{
                background-image: url({st.session_state["background_image"]});
                background-size: cover;
            }}
        </style>
        """
        st.markdown(background_css, unsafe_allow_html=True)


def sidebar_menu():
    with st.sidebar:
        st.title("Dashboard Menu")

        # Feature selection radio buttons
        st.session_state["dashboard_option"] = st.radio(
            "Select Option",
            [
                "Chatbot",
                "Paragraph Analysis",
                "Choose Model",
                "Batch Analysis",
                "Statistics",
                "French Hate Speech",
                "Community Datasets",
                "Quiz"
            ],
            key="dashboard_menu_radio",
        )

        st.markdown("---")

        # Contact Support Section
        st.subheader("Contact Support")
        st.markdown(
            """
            For assistance, reach out to us:
            - 📧 **Email**: support@hatespeechapp.com
            - 📞 **Phone**: +123-456-7890
            """
        )

        # Initialize session state for email and message
        if "user_email_input" not in st.session_state:
            st.session_state["user_email_input"] = ""
        if "user_message_input" not in st.session_state:
            st.session_state["user_message_input"] = ""

        # Input fields
        user_email = st.text_input(
            "Your email",
            value=st.session_state["user_email_input"],
            placeholder="Enter your email..."
        )
        support_message = st.text_area(
            "Message",
            value=st.session_state["user_message_input"],
            placeholder="Type your message here..."
        )

        def clear_fields():
            """Clear email and message fields."""
            st.session_state["user_email_input"] = ""
            st.session_state["user_message_input"] = ""

        if st.button("Send Message"):
            if support_message.strip():
                # Display success message and clear fields using callback
                st.success("Thank you! Our team will get back to you shortly.")
                clear_fields()
            else:
                st.warning("Please enter a message before sending.")

        st.markdown("---")

        # Background Customization Section
        st.subheader("Customize Background")
        uploaded_file = st.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            # Convert image to base64
            encoded_image = base64.b64encode(uploaded_file.read()).decode()
            st.session_state["background_image"] = f"data:image/png;base64,{encoded_image}"
            st.success("Background updated! Refresh if not visible.")




#                   CHATBOT FUNCTIONALITY

def chatbot():
    st.subheader("Chatbot for Hate Speech Detection")
    st.write(f"**Currently Selected Model:** {st.session_state['selected_model']}")

    st.markdown("### Conversation")
    for convo in st.session_state["conversation"]:
        confidence = convo.get("confidence", "N/A")
        conf_str = f"{confidence:.2f}" if isinstance(confidence, (float, int)) else "N/A"
        st.markdown(f"**You:** {convo['user_input']}")
        st.markdown(f"**Bot:** {label_to_emoji(convo['label'])} {convo['response']} "
                    f"(Label: {convo['label']}, Confidence: {conf_str})")

    user_input = st.text_input("Your message:", key="chat_input", placeholder="Type your message here...")
    if st.button("Send", key="send_button"):
        if user_input.strip():
            token = st.session_state.get("access_token")
            if not token:
                st.error("Please log in first.")
                return

            payload = {"text": user_input, "model": st.session_state["selected_model"]}
            params = {"token": token}

            try:
                resp = requests.post(f"{API_URL}/analyze", params=params, json=payload)
                if resp.status_code == 200:
                    result = resp.json()
                    st.session_state["conversation"].append({
                        "user_input": user_input,
                        "response": result.get("rephrased_text", "No rephrasing available."),
                        "label": result.get("detected_label", "N/A"),
                        "confidence": result.get("confidence"),
                        "feedback_given": False
                    })
                    st.rerun()
                else:
                    st.error(f"Failed to analyze text. Code {resp.status_code}: {resp.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {str(e)}")
        else:
            st.warning("Please enter a message before sending.")

    # Prompt for feedback if the last response was hate/offensive
    if st.session_state["conversation"]:
        last_msg = st.session_state["conversation"][-1]
        if last_msg["label"] in ["hate_speech", "offensive"] and not last_msg["feedback_given"]:
            st.markdown(f"**Rephrased Text:** {last_msg['response']}")
            st.markdown("**Was this rephrasing helpful?**")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("Yes", key="feedback_yes"):
                    send_feedback(last_msg, "yes")
            with col_no:
                if st.button("No", key="feedback_no"):
                    send_feedback(last_msg, "no")

def send_feedback(chat_obj, fb_value):
    """Submit feedback for the last chat response."""
    token = st.session_state.get("access_token")
    payload = {
        "text": chat_obj["user_input"],
        "rephrased_text": chat_obj["response"],
        "feedback": fb_value
    }
    try:
        fb_resp = requests.post(f"{API_URL}/feedback", params={"token": token}, json=payload)
        if fb_resp.status_code == 200:
            st.success("Thanks for your feedback!")
            chat_obj["feedback_given"] = True
            st.rerun()
        else:
            st.error(f"Error submitting feedback: {fb_resp.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to submit feedback: {str(e)}")


#                 PARAGRAPH ANALYSIS FEATURE


def paragraph_analysis():
    """
    Enhanced Paragraph Analysis:
    - Detects and highlights hate/offensive words in a paragraph or uploaded file.
    - Displays counts and statistics.
    """
    st.subheader("Paragraph Analysis")

    # 1) Model Selection
    model_choice = st.radio(
        "Choose a Model for Paragraph Analysis:",
        ["T5", "MBERT"],
        key="paragraph_model_radio",
    )

    col1, col2 = st.columns(2)
    file_data = None

    with col1:
        # File upload or text input
        st.markdown("### Upload a Document")
        up_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"], key="upload_paragraph")
        if up_file:
            try:
                file_ext = up_file.name.split(".")[-1].lower()
                if file_ext == "pdf":
                    reader = PyPDF2.PdfReader(up_file)
                    file_data = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
                elif file_ext == "docx":
                    doc = Document(up_file)
                    file_data = "\n".join(para.text for para in doc.paragraphs if para.text.strip())
                elif file_ext == "txt":
                    file_data = up_file.read().decode("utf-8")
                else:
                    st.error("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
            except Exception as e:
                st.error(f"Failed to read the uploaded file: {str(e)}")

        st.markdown("### OR")
        user_paragraph = st.text_area("Enter text directly:", height=200)

    # Combine file and user input (priority given to uploaded file)
    paragraph_text = file_data if file_data else user_paragraph

    if not paragraph_text.strip():
        st.warning("Please provide text or upload a document to analyze.")
        return

    with col2:
        if st.button("Analyze Paragraph"):
            token = st.session_state.get("access_token")
            if not token:
                st.error("Please log in first.")
                return

            payload = {
                "text": paragraph_text,
                "model": model_choice,
            }
            try:
                resp = requests.post(f"{API_URL}/analyze_paragraph", json=payload, params={"token": token})
                if resp.status_code == 200:
                    st.session_state["para_results"] = resp.json()
                else:
                    st.error(f"Analysis failed: {resp.status_code} - {resp.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Error during analysis: {str(e)}")

    if st.session_state.get("para_results"):
        results = st.session_state["para_results"]
        st.markdown("---")
        st.markdown(f"**Model Used:** {results.get('model_used', model_choice)}")

        hate_count = results.get("hate_count", 0)
        offensive_count = results.get("offensive_count", 0)
        total_words = results.get("total_words", 0)
        highlighted_html = results.get("highlighted_html", "")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("🚨 Hate Speech Count", hate_count)
            st.metric("📝 Total Words", total_words)
        with col2:
            st.metric("⚠️ Offensive Words Count", offensive_count)

        st.markdown("### Highlighted Text")
        st.markdown(
            f"""<div style="
                border: 1px solid #e0e0e0;
                padding: 15px;
                border-radius: 10px;
                line-height: 1.8;
                background-color: #f9f9f9;"
            >{highlighted_html}</div>""",
            unsafe_allow_html=True,
        )

        if st.button("Clear Results"):
            st.session_state["para_results"] = None





#                   MODEL SELECTION FEATURE


def choose_model():
    st.subheader("Select a Model for Hate Speech Detection")
    choice = st.radio(
        "Choose a Model:",
        ["MBERT", "Scratch Model", "BERT", "RoBERTa"],
        key="model_selection_radio",
    )
    if st.button("Save Model", key="save_model_button"):
        st.session_state["selected_model"] = choice
        st.success(f"Model switched to: {choice}")


#                   BATCH ANALYSIS FEATURE


def batch_analysis():
    st.subheader("Batch Text Analysis")
    up_file = st.file_uploader("Upload CSV with a 'text' column:", type="csv", key="batch_file")

    if up_file:
        try:
            df = pd.read_csv(up_file)
        except Exception as e:
            st.error(f"Unable to read CSV: {str(e)}")
            return
        if "text" not in df.columns:
            st.error("Uploaded file must contain a 'text' column.")
            return
        st.write("### File Preview:")
        st.dataframe(df.head())

        if st.button("Analyze Batch"):
            token = st.session_state.get("access_token")
            if not token:
                st.error("Please log in first.")
                return

            payload = {
                "texts": df["text"].tolist(),
                "model": st.session_state["selected_model"]
            }
            resp = requests.post(f"{API_URL}/batch_analyze",
                                 params={"token": token}, json=payload)
            if resp.status_code == 200:
                results = resp.json()
                out_df = pd.DataFrame(results)
                st.write("### Batch Analysis Results")
                st.dataframe(out_df)
                st.download_button("Download Results",
                                   out_df.to_csv(index=False),
                                   "batch_results.csv")
            else:
                st.error(f"Batch analysis failed: {resp.status_code} - {resp.text}")


#                 STATISTICS FEATURE

def display_statistics():
    st.subheader("Your Analysis Statistics")
    token = st.session_state.get("access_token")
    if not token:
        st.error("Please log in first.")
        return

    try:
        resp = requests.get(f"{API_URL}/history", params={"token": token})
        resp.raise_for_status()
        data = resp.json()

        if not data:
            st.info("No analysis history available.")
            return

        df = pd.DataFrame(data)
        st.write("### Analysis History")
        st.dataframe(df)

        # Summaries
        st.write("### Summary Statistics")
        st.write(f"**Total Analyses:** {len(df)}")

        # English-based labels
        hs_ct = df[df['detected_label'] == 'hate_speech'].shape[0]
        off_ct = df[df['detected_label'] == 'offensive'].shape[0]
        neu_ct = df[df['detected_label'] == 'neutral'].shape[0]

        # French-based labels
        fr_hate = df[df['detected_label'] == 'hate'].shape[0]
        fr_non = df[df['detected_label'] == 'non_hate'].shape[0]

        st.write(f"**Hate Speech (English):** {hs_ct}")
        st.write(f"**Offensive (English):** {off_ct}")
        st.write(f"**Neutral (English):** {neu_ct}")
        st.write(f"**Hate (French):** {fr_hate}")
        st.write(f"**Non-Hate (French):** {fr_non}")

        # Distribution pie chart
        st.write("### Detected Labels Distribution")
        if len(df) > 0 and 'detected_label' in df.columns:
            label_counts = df['detected_label'].value_counts()
            fig, ax = plt.subplots()
            ax.pie(label_counts,
                   labels=label_counts.index,
                   autopct="%1.1f%%",
                   startangle=90,
                   colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'])
            ax.axis('equal')
            st.pyplot(fig)
        else:
            st.write("No data available for visualization.")

    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to server: {str(e)}")
    except Exception as exc:
        st.error(f"An unexpected error occurred: {str(exc)}")


#                CHAT HISTORY EXPORT

def export_chat_history():
    st.subheader("Export Chat History")
    chat_data = pd.DataFrame(st.session_state.get("conversation", []))
    if chat_data.empty:
        st.write("No chat history to export.")
        return

    # CSV
    csv_buf = chat_data.to_csv(index=False)
    st.download_button("Download as CSV", csv_buf, "chat_history.csv", mime="text/csv")

    # PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Chat History", ln=True, align="C")

    for idx, row in chat_data.iterrows():
        pdf.cell(200, 10, txt=f"You: {row.get('user_input','N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Bot: {row.get('response','N/A')} (Label: {row.get('label','N/A')})", ln=True)

    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    st.download_button("Download as PDF", pdf_bytes, "chat_history.pdf", mime="application/pdf")


#        RIGHT-SIDE PANEL (Chat History, FAQ, Feedback)

def right_side_panel():
    st.markdown("### Recent Chat History")

    if st.session_state["logged_in"]:
        selected_date = st.date_input("Select Date:", value=datetime.today())
        token = st.session_state["access_token"]
        params = {"token": token, "date": selected_date.strftime("%Y-%m-%d")}

        try:
            r = requests.get(f"{API_URL}/history", params=params)
            if r.status_code == 200:
                hist = r.json()
                if hist:
                    for c in hist:
                        conf_s = c.get("confidence","N/A")
                        conf_disp = f"{conf_s:.2f}" if isinstance(conf_s,(int,float)) else "N/A"
                        st.write(
                            f"🔹 **You:** {c['message']} → "
                            f"{label_to_emoji(c['detected_label'])} **{c['detected_label']}** "
                            f"(Confidence: {conf_disp})"
                        )
                else:
                    st.write("No chat history for this date.")
            else:
                st.error(f"Failed to fetch history. Code {r.status_code}: {r.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to retrieve chat history: {str(e)}")

    st.markdown("---")
    st.subheader("Export Current Conversation")
    export_chat_history()

    st.markdown("---")
    st.markdown("### FAQ")
    faqs = {
        "What is this tool for?": "It detects hate/offensive speech in English & French texts.",
        "Which models are used?": "MBERT, BERT, RoBERTa, Scratch, plus T5 for rephrasing.",
        "How does rephrasing work?": "It uses T5 (English) or GPT-2 for advanced usage/fallback.",
        "How do I report incorrect results?": "Use the feedback form or chatbot's feedback prompt."
    }
    question = st.selectbox("Select a question:", list(faqs.keys()), key="faq_selectbox")
    st.write(f"💡 {faqs[question]}")

    st.markdown("---")
    st.subheader("Feedback Form")
    fb_text = st.text_area("Your feedback:", placeholder="Type your feedback here...")
    if st.button("Submit Feedback", key="feedback_submit"):
        if fb_text.strip():
            st.success("Thank you for your feedback!")
        else:
            st.warning("Please enter feedback before submitting.")


#        FRENCH HATE SPEECH FRONTEND


def french_hate_speech():
    st.subheader("French Hate Speech Detection 🇫🇷")
    fr_text = st.text_area("Entrez votre texte en français:", placeholder="Tapez quelque chose en français...")

    if st.button("Analyze French Text"):
        if fr_text.strip():
            token = st.session_state.get("access_token")
            if not token:
                st.error("You need to log in first.")
                return
            payload = {"text": fr_text}
            params = {"token": token}

            try:
                r = requests.post(f"{API_URL}/analyze_french", params=params, json=payload)
                if r.status_code == 200:
                    result = r.json()
                    st.markdown(
                        f"**Label:** {label_to_emoji(result['detected_label'])} {result['detected_label']}"
                    )
                    st.markdown(f"**Confidence:** {result['confidence']:.2f}")
                    st.markdown(f"**Rephrased Text:** {result['rephrased_text']}")
                else:
                    st.error(f"French analysis failed: {r.status_code} - {r.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Request error: {str(e)}")
        else:
            st.warning("Veuillez saisir du texte en français.")


#             COMMUNITY DATASETS FEATURE


def community_datasets():
    st.subheader("Community Datasets")

    # 1) Upload section
    st.markdown("### Upload a New Dataset")
    upfile = st.file_uploader("Choose CSV file:")
    ds_name = st.text_input("Dataset Name (e.g. my_data.csv):", value="my_data.csv")

    if st.button("Upload Dataset"):
        if upfile is not None and ds_name.strip():
            token = st.session_state.get("access_token")
            if not token:
                st.error("Please log in to upload a dataset.")
                return

            # read file content
            file_bytes = upfile.read()
            encoded = base64.b64encode(file_bytes).decode()

            payload = {"filename": ds_name, "file_content": encoded}
            params = {"token": token}

            try:
                resp = requests.post(f"{API_URL}/upload_dataset", params=params, json=payload)
                if resp.status_code == 200:
                    st.success("Dataset uploaded successfully!")
                else:
                    st.error(f"Failed to upload: {resp.status_code} - {resp.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {str(e)}")
        else:
            st.warning("Please select a CSV file and provide a dataset name.")

    st.markdown("---")

    # 2) List & Download
    st.markdown("### Available Datasets")
    if st.button("Refresh List"):
        fetch_community_datasets()

    for ds in st.session_state["community_datasets"]:
        ds_id = ds["id"]
        ds_name = ds["filename"]
        uploader = ds["uploaded_by"]
        created_at = ds.get("created_at", "N/A")

        cA, cB = st.columns([3,1])
        with cA:
            st.write(f"**ID {ds_id}**: {ds_name} (by {uploader}, at {created_at})")
        with cB:
            if st.button(f"Download {ds_name}", key=f"ds_dl_{ds_id}"):
                download_community_dataset(ds_id, ds_name)

def fetch_community_datasets():
    token = st.session_state.get("access_token")
    if not token:
        st.error("Please log in to see datasets.")
        return
    try:
        r = requests.get(f"{API_URL}/list_datasets", params={"token": token})
        if r.status_code == 200:
            st.session_state["community_datasets"] = r.json()
        else:
            st.error(f"Failed to list datasets: {r.status_code} - {r.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching datasets: {str(e)}")

def download_community_dataset(ds_id, ds_name):
    token = st.session_state.get("access_token")
    if not token:
        st.error("Please log in first.")
        return

    url = f"{API_URL}/download_dataset/{ds_id}?token={token}"
    try:
        with requests.get(url, stream=True) as resp:
            if resp.status_code == 200:
                st.download_button(
                    label=f"Download {ds_name}",
                    data=resp.content,
                    file_name=ds_name,
                    mime="text/csv"
                )
            else:
                st.error(f"Download error: {resp.status_code} - {resp.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {str(e)}")


#                 MAIN FUNCTION


def main():
    menu = ["Login", "Register", "Logout"]
    pick = st.sidebar.selectbox("Main Menu", menu, key="menu")

    if not st.session_state["disclaimer_accepted"]:
        disclaimer()
    else:
        if pick == "Register":
            register_user()
        elif pick == "Login":
            login_user()

        if st.session_state["logged_in"]:
            col1, col2 = st.columns([3,1])
            with col1:
                sidebar_menu()

                # Routing for the main dashboard options
                if st.session_state["dashboard_option"] == "Chatbot":
                    chatbot()
                elif st.session_state["dashboard_option"] == "Paragraph Analysis":
                    paragraph_analysis()
                elif st.session_state["dashboard_option"] == "Choose Model":
                    choose_model()
                elif st.session_state["dashboard_option"] == "Batch Analysis":
                    batch_analysis()
                elif st.session_state["dashboard_option"] == "Statistics":
                    display_statistics()
                elif st.session_state["dashboard_option"] == "French Hate Speech":
                    french_hate_speech()
                elif st.session_state["dashboard_option"] == "Community Datasets":
                    community_datasets()
                elif st.session_state["dashboard_option"] == "Quiz":
                    quiz()

            # The right side panel (chat history, FAQ, feedback, etc.)
            with col2:
                right_side_panel()

if __name__ == "__main__":
    # Default background
    background_style = """
        <style>
            .stApp {
                background: url("https://source.unsplash.com/random/1600x900/?technology,abstract");
                background-size: cover;
            }
        </style>
    """

    # If user uploaded a custom image, apply it
    if "background_image" in st.session_state and st.session_state["background_image"]:
        background_style = f"""
            <style>
                .stApp {{
                    background: url("{st.session_state['background_image']}");
                    background-size: cover;
                    background-position: center;
                    background-repeat: no-repeat;
                }}
            </style>
        """

    # Apply the background style
    st.markdown(background_style, unsafe_allow_html=True)

    # Run the main application
    main()

