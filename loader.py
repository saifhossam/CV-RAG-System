import json
import hashlib
import re
import os
from pypdf import PdfReader
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT")

azure_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)


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

    prompt = f"""You are an expert CV parser.

Read the CV below and do two things:
1. Extract the candidate's full name always in the first line.
2. Split the CV into its logical sections. Each section should have:
   - Its title (e.g. "Education", "Work Experience", "Skills", "Projects", "Summary", or whatever the CV actually contains).
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
{text}
"""
    try:
        response = azure_client.chat.completions.create(
            model=AZURE_CHAT_DEPLOYMENT,
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