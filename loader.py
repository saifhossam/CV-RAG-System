import json
import hashlib
import re
import os
from pypdf import PdfReader
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def extract_text_from_pdf(file) -> str:
    """Extract raw text from all PDF pages."""
    file.seek(0)
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def llm_structural_chunking(text: str):
    """
    Fully LLM-driven chunking — the model reads the CV and decides what
    the sections are, regardless of their names or formatting.
    Returns (chunks, candidate_name).
    """
    prompt = f"""You are an expert CV parser.

Read the CV below and do two things:
1. Extract the candidate's full name.
2. Split the CV into its logical sections. Each section should have:
   - A short descriptive title (e.g. "Education", "Work Experience", "Skills", "Projects", "Summary", or whatever the CV actually contains).
   - The full text content of that section, exactly as it appears.

Important rules:
- Do NOT skip any section, even if it has an unusual name.
- Do NOT invent or summarize content — copy the text as-is.
- Every part of the CV must belong to exactly one section.

Respond ONLY with valid JSON — no markdown, no code fences, no extra text:
{{
  "candidate_name": "Full Name Here",
  "sections": [
    {{"section_title": "Summary", "content": "...full text of this section..."}},
    {{"section_title": "Education", "content": "...full text of this section..."}},
    {{"section_title": "Work Experience", "content": "...full text of this section..."}}
  ]
}}

CV Text:
{text[:6000]}
"""
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if the model adds them despite instructions
        raw = re.sub(r"```(?:json)?", "", raw).strip("` \n")
        json_match = re.search(r"\{[\s\S]*\}", raw)

        if not json_match:
            print("[loader] LLM returned no valid JSON — using fallback.")
            return fallback_chunking(text)

        data = json.loads(json_match.group(0))
        candidate_name = data.get("candidate_name", "Unknown").strip()
        sections = data.get("sections", [])

        chunks = [
            {
                "content": s["content"],
                "metadata": {
                    "section": s.get("section_title", "General"),
                    "candidate_name": candidate_name,
                },
            }
            for s in sections
            if len(s.get("content", "").strip()) > 40
        ]

        if chunks:
            return chunks, candidate_name

        print("[loader] LLM returned empty sections — using fallback.")
        return fallback_chunking(text, candidate_name)

    except Exception as e:
        print(f"[loader] LLM chunking failed: {e}")
        return fallback_chunking(text)


def fallback_chunking(text: str, candidate_name: str = "Unknown"):
    """Last resort: store the whole CV as one chunk."""
    chunks = [
        {
            "content": text,
            "metadata": {
                "section": "Full Text",
                "candidate_name": candidate_name,
            },
        }
    ]
    return chunks, candidate_name


def calculate_file_hash(file) -> str:
    file.seek(0)
    file_bytes = file.read()
    file.seek(0)
    return hashlib.sha256(file_bytes).hexdigest()