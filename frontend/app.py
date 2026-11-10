import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000/match-pdf")

st.set_page_config(page_title="Resume Matcher", layout="wide")
st.title("Intelligent Resume Matcher")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Resume (PDF)")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

with col2:
    st.subheader("Paste Job Description")
    job_description = st.text_area("Target Job Description", height=300)


def validate_inputs(file, jd):
    return file is not None and bool(jd.strip())


def send_request(file, jd):
    files = {"file": (file.name, file, "application/pdf")}
    data = {"job_description": jd}
    # generous: Ollama on CPU (the Docker default) takes 2-3 minutes per match
    return requests.post(API_URL, files=files, data=data, timeout=600)


if st.button("Analyze Match"):
    if not validate_inputs(uploaded_file, job_description):
        st.warning("Resume and job description are required.")
        st.stop()

    with st.spinner("Processing..."):
        try:
            response = send_request(uploaded_file, job_description)

            if response.status_code != 200:
                try:  # the API explains 400/413/500 in "detail", e.g. "Failed to extract text from PDF."
                    detail = response.json().get("detail", "")
                except ValueError:
                    detail = ""
                st.error(
                    f"API Error {response.status_code}: {detail}" if detail else f"API Error {response.status_code}"
                )
                st.stop()

            result = response.json()
            analysis = result["analysis"]
            score = analysis["match_percentage"]  # the API's schema guarantees an int from 0 to 100

            st.success("Analysis Complete")
            st.metric("Match Score", f"{score}%")
            st.progress(score / 100)

            recommendation = analysis["recommendation"]

            if recommendation == "Hire":
                st.success(f"Recommendation: {recommendation}")
            elif recommendation == "Interview":
                st.warning(f"Recommendation: {recommendation}")
            else:
                st.error(f"Recommendation: {recommendation}")

            st.subheader("Missing Skills")
            missing = analysis["missing_keywords"]

            if missing:
                st.write(", ".join(missing))
            else:
                st.write("No major skill gaps detected.")

            if analysis["analysis_summary"]:
                st.info(analysis["analysis_summary"])

            with st.expander("Resume text sent to the model (only the most relevant parts in RAG mode)"):
                st.text(result["parsed_text_preview"])

        except requests.exceptions.Timeout:
            st.error("The backend took longer than 10 minutes to respond. Is Ollama running?")
        except requests.exceptions.ConnectionError:
            st.error("Backend service is not running.")
        except Exception as e:
            st.error(f"Unexpected error: {e!s}")
