"""Strip personal details from resume text so labelling judges skills and experience only.

Heuristic, not a guarantee: it removes contact details, personal-details lines and the candidate's name
(taken from the first line), which covers the usual resume layouts.
"""

import re
from collections.abc import Sequence

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")
URL = re.compile(r"(https?://\S+|www\.\S+|\b[\w-]+\.(com|in|io|dev|me|org|net)/\S*)", re.I)
PHONE_CANDIDATE = re.compile(r"\+?\d[\d\s().-]{8,}\d")
PERSONAL_LINE = re.compile(
    r"^\s*(date of birth|d\.?o\.?b|birth ?date|age|gender|sex|marital status|nationality|religion|"
    r"father'?s name|mother'?s name|spouse|address|permanent address|current address|passport)\b",
    re.I,
)
# words that start a job title rather than a name (Kaggle resumes open with the role, e.g. "NETWORK ADMINISTRATOR")
ROLE_WORDS = re.compile(
    r"\b(engineer|developer|manager|specialist|analyst|accountant|teacher|chef|consultant|director|"
    r"administrator|technician|representative|assistant|officer|sales|designer|information|technology|"
    r"intern|lead|executive|associate|coordinator|supervisor|resume|curriculum|vitae|cv|profile|summary|"
    r"network|systems?|it|support|help|desk|data|database|web|software|senior|junior|technical|operations|"
    r"project|program|business|quality|security|applications?|services?|computer|desktop|field|cloud|"
    r"infrastructure|solutions|architect|programmer|tester|scientist|head|chief|vice|president|staff|"
    r"principal|product|customer|account|financial|finance|marketing|human|resources|office|clerk|cashier|"
    r"cook|sous|pastry|instructor|educator|professor|tutor|banker|teller|loan|credit|auditor|bookkeeper|"
    r"payroll|recruiter|generalist|partner|full|stack|frontend|backend|front|end|back|mobile|devops|qa|ml|ai|"
    r"substitute|trainee|volunteer|freelance|graduate|certified|registered|licensed|teaching|accounting|"
    r"banking|hr|professional|experience|skills|objective|career|highlights|qualifications|education|"
    r"accomplishments|overview|personal|contact|about|work|history|employment)\b",
    re.I,
)
NAME_WORD = re.compile(r"^[A-Z][A-Za-z.'-]*$")  # Capitalised or ALL-CAPS word


def _phone(match: re.Match) -> str:
    # only long digit runs are phone numbers; "2021 - 2023" keeps its dates
    return "[phone]" if sum(c.isdigit() for c in match.group()) >= 10 else match.group()


def name_tokens(first_line: str) -> list[str]:
    """The leading Capitalised words of the first line, up to a separator, digit or job-title word."""
    head = re.split(r"[|•·,:;@()\d–—]| - ", first_line, maxsplit=1)[0]
    tokens = []
    for word in head.split():
        if ROLE_WORDS.fullmatch(word.strip(".")) or not NAME_WORD.match(word) or len(tokens) == 4:
            break
        tokens.append(word.strip("."))
    return [t for t in tokens if len(t) >= 3]  # initials ("U.", "K") are too short to redact safely


def filename_names(original_name: str) -> list[str]:
    """Name words from a resume's original filename, e.g. "Jane_Doe_Frontend_CV" -> ["Jane", "Doe"]."""
    words = re.split(r"[^A-Za-z]+", original_name)
    return [w for w in words if len(w) >= 3 and not ROLE_WORDS.fullmatch(w)]


def redact(text: str, known_names: Sequence[str] = ()) -> str:
    lines = [line for line in text.splitlines() if not PERSONAL_LINE.match(line)]
    top = [line.strip() for line in lines if line.strip()][:2]
    tokens = name_tokens(top[0]) if top else []
    if len(tokens) == 1 and len(top) == 2:  # name split over two lines: "FIRSTNAME" then "SURNAME | email ..."
        tokens += name_tokens(top[1])
    tokens += [n for n in known_names if len(n) >= 3]

    out = "\n".join(lines)
    out = EMAIL.sub("[email]", out)
    out = URL.sub("[link]", out)
    out = PHONE_CANDIDATE.sub(_phone, out)
    if tokens:  # the name, wherever it appears again (footer, declaration, signature)
        out = re.sub(r"\b(" + "|".join(map(re.escape, tokens)) + r")\b", "[name]", out, flags=re.I)
    return out
