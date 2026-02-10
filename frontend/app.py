# app.py
import streamlit as st
import requests

API_URL = "http://localhost:8000/match-pdf"

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
    return requests.post(API_URL, files=files, data=data)

if st.button("Analyze Match"):

    if not validate_inputs(uploaded_file, job_description):
        st.warning("Resume and job description are required.")
        st.stop()

    with st.spinner("Processing..."):
        try:
            response = send_request(uploaded_file, job_description)

            if response.status_code != 200:
                st.error(f"API Error {response.status_code}")
                st.stop()

            result = response.json()
            analysis = result.get("analysis", {})

            score = analysis.get("match_percentage", 0)
            if not isinstance(score, (int, float)):
                score = 0
            score = max(0, min(100, score))

            st.success("Analysis Complete")
            st.metric("Match Score", f"{score}%")
            st.progress(score / 100)

            recommendation = analysis.get("recommendation", "Unknown")

            if recommendation == "Hire":
                st.success(f"Recommendation: {recommendation}")
            elif recommendation == "Interview":
                st.warning(f"Recommendation: {recommendation}")
            else:
                st.error(f"Recommendation: {recommendation}")

            st.subheader("Missing Skills")
            missing = analysis.get("missing_keywords", [])

            if missing:
                st.write(", ".join(missing))
            else:
                st.write("No major skill gaps detected.")

            summary = analysis.get("analysis_summary")
            if summary:
                st.info(summary)

            with st.expander("Extracted Resume Text Preview"):
                st.text(result.get("parsed_text_preview", "No preview available."))

        except requests.exceptions.ConnectionError:
            st.error("Backend service is not running.")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
