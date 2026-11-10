"""Local labelling tool for eval cases: one resume–job pair at a time, saved straight back to the CSV.

Run from the repo root:
    streamlit run eval/label_app.py -- --cases eval/private/cases.csv --resumes resumes --jds eval/private/jds
Everything stays on this machine: resumes are read with pypdf, never a cloud parser. Names, contact details
and personal details are redacted, so labels judge skills and experience only.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import streamlit as st
from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from ai_logic import BANDS, recommendation_for  # noqa: E402  (the app's own policy, so labels follow it)
from redact import filename_names, redact  # noqa: E402

FIELDS = ["resume", "jd", "expected_score", "expected_recommendation", "notes"]

parser = argparse.ArgumentParser()
parser.add_argument("--cases", default="eval/private/cases.csv")
parser.add_argument("--resumes", default="resumes")
parser.add_argument("--jds", default="eval/private/jds")
parser.add_argument("--names-map", default="eval/private/resume_names_map.csv", help="generic -> original filename")
args = parser.parse_args()
cases_path, resumes_dir, jds_dir = Path(args.cases), Path(args.resumes), Path(args.jds)


@st.cache_data
def original_names() -> dict[str, str]:
    """Original filenames often contain the candidate's name, so they're redacted too (if the map exists)."""
    path = Path(args.names_map)
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["generic_name"]: r["original_name"] for r in csv.DictReader(f)}


@st.cache_data
def resume_text(path: str) -> str:
    known = filename_names(original_names().get(Path(path).stem, ""))
    return redact("\n".join(page.extract_text() or "" for page in PdfReader(path).pages[:10]), known)


def load_rows() -> list[dict]:
    with open(cases_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_rows(rows: list[dict]) -> None:
    tmp = cases_path.with_suffix(".tmp")  # write then swap, so a crash can't leave a half-written sheet
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, cases_path)


def labelled(row: dict) -> bool:
    return bool(row["expected_score"].strip())


st.set_page_config(page_title="Label pairs", layout="wide")
rows = load_rows()
if "i" not in st.session_state:
    st.session_state.i = next((n for n, r in enumerate(rows) if not labelled(r)), 0)
i = st.session_state.i
row = rows[i]

done = sum(labelled(r) for r in rows)
st.progress(done / len(rows), text=f"{done} of {len(rows)} labelled · pair {i + 1}")
if done == len(rows):  # saving the last pair has nowhere to jump, so say so instead of looking stuck
    st.success("All pairs labelled and saved. Use Previous/Next to review or change any label.")

nav = st.columns(4)
if nav[0].button("← Previous", disabled=i == 0):
    st.session_state.i -= 1
    st.rerun()
if nav[1].button("Next →", disabled=i == len(rows) - 1):
    st.session_state.i += 1
    st.rerun()
if nav[2].button("Next unlabelled", disabled=done == len(rows)):
    st.session_state.i = next((n for n, r in enumerate(rows) if not labelled(r)), i)
    st.rerun()

pdf = resumes_dir / f"{row['resume']}.pdf"
left, right = st.columns(2)
with left:
    st.subheader(f"Job: {row['jd']}")
    st.text_area(
        "job",
        (jds_dir / f"{row['jd']}.txt").read_text(encoding="utf-8"),
        height=520,
        disabled=True,
        label_visibility="collapsed",
    )
with right:
    st.subheader(f"Resume: {row['resume']}")
    # no "open the PDF" button: the original shows the name, contacts and photo the redaction removes
    st.text_area("resume", resume_text(str(pdf)), height=520, disabled=True, label_visibility="collapsed")

st.divider()
st.caption(" · ".join(f"{low}+ {label}" for low, label, _ in BANDS))
# starts empty for unlabelled pairs, so saving without typing a score is impossible (no accidental default labels)
current = int(row["expected_score"]) if labelled(row) else None
score = st.number_input("Expected score (0–100)", min_value=0, max_value=100, value=current, step=1, key=f"score-{i}")
notes = st.text_input("Reason (one line)", row["notes"], key=f"notes-{i}")
if score is not None:
    st.write(f"Recommendation for this score: **{recommendation_for(int(score))}**")
if st.button("Save and go to next unlabelled", type="primary", disabled=score is None):
    row.update(expected_score=str(int(score)), expected_recommendation=recommendation_for(int(score)), notes=notes)
    save_rows(rows)
    st.session_state.i = next((n for n, r in enumerate(rows) if not labelled(r)), i)
    st.rerun()
