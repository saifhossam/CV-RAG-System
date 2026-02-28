import streamlit as st
from dotenv import load_dotenv

from loader import extract_text_from_pdf, llm_structural_chunking, calculate_file_hash
from rag import (
    document_exists,
    index_documents,
    retrieve,
    generate_answer,
    generate_cv_strength_report,   # NEW
    qdrant,
    COLLECTION_NAME,
)
from qdrant_client.models import Filter, FieldCondition, MatchAny

load_dotenv()

st.set_page_config(page_title="CV RAG System", layout="wide")
st.title("📄 CV Intelligence System")

# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs([
    "💬 Chat with CVs",
    "📊 CV Strength & Weakness Report"
])

# ──────────────────────────────────────────────────────────────────────────────
# Session state (shared across tabs)
# ──────────────────────────────────────────────────────────────────────────────
if "session_file_hashes" not in st.session_state:
    st.session_state.session_file_hashes = []

if "file_info" not in st.session_state:
    st.session_state.file_info = {}

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — Chat with CVs (Original RAG System)
# ──────────────────────────────────────────────────────────────────────────────
with tab1:

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Session CVs")
        if st.session_state.session_file_hashes:
            st.write(f"**{len(st.session_state.session_file_hashes)}** CV(s) in scope")
            for h in st.session_state.session_file_hashes:
                info = st.session_state.file_info.get(h, {})
                st.write(f"• {info.get('name', '—')} — *{info.get('candidate', 'Unknown')}*")

            if st.button("🗑️ Clear Session", type="primary"):
                st.session_state.session_file_hashes = []
                st.session_state.file_info = {}
                st.rerun()
        else:
            st.info("No CVs uploaded yet.")

    # ── Upload section ───────────────────────────────────────────────────────
    st.header("1. Upload CV PDFs")
    uploaded_files = st.file_uploader(
        "Select one or more CV PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_hash = calculate_file_hash(uploaded_file)

            if file_hash in st.session_state.session_file_hashes:
                continue

            if document_exists(file_hash):
                points, _ = qdrant.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=Filter(
                        must=[FieldCondition(key="file_hash", match=MatchAny(any=[file_hash]))]
                    ),
                    limit=1,
                    with_payload=True,
                )
                candidate_name = (
                    points[0].payload.get("candidate_name", "Unknown") if points else "Unknown"
                )
                st.session_state.session_file_hashes.append(file_hash)
                st.session_state.file_info[file_hash] = {
                    "name": uploaded_file.name,
                    "candidate": candidate_name,
                }
                st.info(
                    f"✅ **{uploaded_file.name}** already indexed — added to session as *{candidate_name}*."
                )
                continue

            with st.spinner(f"Processing **{uploaded_file.name}**…"):
                text = extract_text_from_pdf(uploaded_file)
                if not text.strip():
                    st.warning(
                        f"⚠️ Could not extract text from **{uploaded_file.name}**. Skipping."
                    )
                    continue

                chunks, candidate_name = llm_structural_chunking(text)
                if not chunks:
                    st.warning(
                        f"⚠️ No usable content extracted from **{uploaded_file.name}**. Skipping."
                    )
                    continue

                index_documents(chunks, file_hash, candidate_name=candidate_name)
                st.session_state.session_file_hashes.append(file_hash)
                st.session_state.file_info[file_hash] = {
                    "name": uploaded_file.name,
                    "candidate": candidate_name,
                }

                section_names = list({c["metadata"]["section"] for c in chunks})
                st.success(
                    f"✅ **{uploaded_file.name}** indexed as *{candidate_name}* "
                    f"— {len(chunks)} section chunk(s): {', '.join(section_names)}"
                )

    # ── Query section ─────────────────────────────────────────────────────────
    st.header("2. Ask questions")
    query = st.text_input(
        "Your question",
        placeholder="e.g. Who has Python experience?  /  What is Saif's education?",
    )

    if st.button("🔍 Search", type="primary") and query.strip():
        if not st.session_state.session_file_hashes:
            st.error("Please upload at least one CV first.")
        else:
            q_lower = query.lower()
            mentioned_candidates: list[str] = []

            for info in st.session_state.file_info.values():
                name = info.get("candidate", "").strip()
                if not name or name.lower() in {"unknown", "existing"}:
                    continue
                name_parts = name.lower().split()
                if name.lower() in q_lower or any(part in q_lower for part in name_parts):
                    if name not in mentioned_candidates:
                        mentioned_candidates.append(name)

            with st.spinner("Searching…"):
                contexts = retrieve(
                    query,
                    file_hashes=st.session_state.session_file_hashes,
                    candidate_names=mentioned_candidates if mentioned_candidates else None,
                )

            if mentioned_candidates:
                st.caption(
                    f"🔎 Filtered by candidate(s): **{', '.join(mentioned_candidates)}**"
                )

            if contexts:
                with st.spinner("Generating answer…"):
                    candidates_in_scope = [
                        info.get("candidate", "Unknown")
                        for info in st.session_state.file_info.values()
                    ]
                    answer = generate_answer(query, contexts, candidates_in_scope)

                st.subheader("Answer")
                st.markdown(answer)

                with st.expander("📚 View Retrieved Sources"):
                    for c in contexts:
                        st.markdown(f"**{c['candidate_name']}** — *{c['section']}*")
                        st.text(
                            c["content"][:300]
                            + ("…" if len(c["content"]) > 300 else "")
                        )
                        st.divider()
            else:
                st.warning(
                    "No relevant information found for this query in the uploaded CVs."
                )

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — CV Strength & Weakness Report
# ──────────────────────────────────────────────────────────────────────────────
with tab2:

    st.header("📊 CV Strength & Weakness Report")

    uploaded_cv = st.file_uploader(
        "Upload a single CV (PDF)",
        type=["pdf"],
        key="strength_cv"
    )

    job_description = st.text_area(
        "Paste Job Description",
        height=250,
        placeholder="Paste the full job description here..."
    )

    if st.button("Analyze CV", type="primary"):

        if not uploaded_cv:
            st.error("Please upload a CV.")
        elif not job_description.strip():
            st.error("Please paste the job description.")
        else:
            with st.spinner("Analyzing CV against Job Description..."):

                cv_text = extract_text_from_pdf(uploaded_cv)

                if not cv_text.strip():
                    st.error("Could not extract text from CV.")
                else:
                    report = generate_cv_strength_report(
                        cv_text=cv_text,
                        job_description=job_description
                    )

                    st.subheader("📋 Evaluation Report")
                    st.markdown(report)