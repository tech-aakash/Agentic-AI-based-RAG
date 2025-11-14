from flask import Flask, render_template, request, jsonify
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os
import json
import re
import time
from datetime import datetime, timezone

# ---------- NEW: imports for PDF citation / search ----------
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery

# ============================================================
#  ENV + FLASK SETUP
# ============================================================
load_dotenv()

app = Flask(__name__)
app.secret_key = "supersecretkey"  # TODO: rotate in production

# ------------------ Azure Chat Config ------------------
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# ------------------ Azure Embedding + Search (for citations) ------------------
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMB_DEPLOYMENT_NAME")
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX_NAME")
SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")

# ============================================================
#  HELPER FACTORIES FOR LLMs
# ============================================================
def llm_zero_temp():
    """Low-temperature model for control / grading."""
    return AzureChatOpenAI(
        azure_deployment=AZURE_DEPLOYMENT,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        temperature=0,
    )


def llm_brief():
    """Short answers model."""
    return AzureChatOpenAI(
        azure_deployment=AZURE_DEPLOYMENT,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        temperature=0.1,
    )


def llm_deep():
    """Deep explanation model (used as fallback when no refs)."""
    return AzureChatOpenAI(
        azure_deployment=AZURE_DEPLOYMENT,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        temperature=0.1,
    )

# ============================================================
#  RAW AZURE CLIENTS FOR EMBEDDINGS + SEARCH
# ============================================================
openai_client = None
if AZURE_API_KEY and AZURE_ENDPOINT and AZURE_API_VERSION:
    openai_client = AzureOpenAI(
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
    )

search_client = None
if SEARCH_ENDPOINT and SEARCH_INDEX and SEARCH_API_KEY:
    search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=SEARCH_INDEX,
        credential=AzureKeyCredential(SEARCH_API_KEY),
    )

# ============================================================
#  GENERIC HELPERS
# ============================================================
def timestamp():
    """UTC timestamp with timezone (avoids deprecation warning)."""
    return datetime.now(timezone.utc).isoformat()


def extract_json(text: str):
    """Extract first JSON object from LLM response."""
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"[extract_json] Failed to parse JSON: {e}")
    return None


def is_greeting(text: str) -> bool:
    """Very small heuristic so greetings never go to deep_answer."""
    q = re.sub(r"[^\w\s]", "", text.lower()).strip()
    if not q:
        return False
    GREETINGS = {
        "hi",
        "hello",
        "hey",
        "yo",
        "sup",
        "good morning",
        "good afternoon",
        "good evening",
        "hi there",
        "hey there",
    }
    return any(q == g or q.startswith(g + " ") for g in GREETINGS)


# ============================================================
#  GLOBAL STATE (same as your original)
# ============================================================
GLOBAL_STATE = {
    "history": [],
    "events": [],
    "quiz_state": None,
}


def persist_log():
    """Save GLOBAL_STATE to JSON file."""
    payload = {
        "updated_at": timestamp(),
        "history": GLOBAL_STATE["history"],
        "events": GLOBAL_STATE["events"],
        "quiz": GLOBAL_STATE["quiz_state"],
    }
    with open("chat_log.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def log_event(kind: str, detail: dict):
    """Log event into global events and persist."""
    entry = {"ts": timestamp(), "type": kind, "detail": detail}
    GLOBAL_STATE["events"].append(entry)
    print(f"[EVENT] {kind}: {detail}")
    persist_log()


# Load previous memory if exists
if os.path.exists("chat_log.json"):
    with open("chat_log.json", "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            GLOBAL_STATE["history"] = data.get("history", [])
            GLOBAL_STATE["events"] = data.get("events", [])
            GLOBAL_STATE["quiz_state"] = data.get("quiz", None)
            print("[INIT] Loaded previous chat_log.json")
        except Exception as e:
            print(f"[INIT] Failed to load chat_log.json: {e}")

# ============================================================
#  PDF / REFERENCE HELPERS — USED ONLY BY deep_answer
# ============================================================
def generate_embedding(text: str):
    """Returns embedding vector from Azure OpenAI."""
    if not openai_client or not EMBEDDING_DEPLOYMENT:
        raise RuntimeError("Embedding client or deployment is not configured.")
    resp = openai_client.embeddings.create(
        input=text,
        model=EMBEDDING_DEPLOYMENT,
        # dimensions optional; uncomment if your deployment expects it
        # dimensions=3072
    )
    return resp.data[0].embedding


def search_matching_documents(embedding, threshold: float = 0.5):
    """Queries Azure Cognitive Search vector index and returns top matches."""
    if not search_client:
        print("[SEARCH] search_client is not configured. Skipping vector search.")
        return []

    vector_query = VectorizedQuery(
        vector=embedding,
        k_nearest_neighbors=5,
        fields="embedding",
    )

    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=["document_name", "page_number", "sas_url"],
        top=5,
    )

    matches = []
    for r in results:
        score = r.get("@search.score", 0)
        if score >= threshold:
            matches.append(
                {
                    "document_name": r["document_name"],
                    "page_number": r["page_number"],
                    "sas_url": r.get("sas_url"),
                    "score": float(score),
                }
            )
    print(f"[SEARCH] Found {len(matches)} matches (>= {threshold}).")
    return matches


def generate_refined_response_with_refs(user_prompt: str, matching_documents: list) -> str:
    """
    Generate a citation-based answer using the reference docs.
    Ensures that all links follow strict markdown format:
    [Document p.X](sas_url)
    """

    # Build reference list markdown (STRICT format)
    ref_lines = []
    for doc in matching_documents:
        doc_name = doc['document_name']
        page = doc['page_number']
        url = doc.get("sas_url")

        if url:
            link = f"[{doc_name} p.{page}]({url})"
        else:
            link = f"{doc_name} p.{page}"

        ref_lines.append(f"- {link}")

    ref_block = "\n".join(ref_lines)

    # --- VERY IMPORTANT FIX ---
    # Tell the LLM EXACTLY how to format links
    system_prompt = """
You are an AI tutor.

STRICT RULES (must follow):
1. Cite documents inside the explanation using ONLY this inline format:
   (Document Name, p.X)

2. At the end, output a section titled "References" containing ONLY markdown links:
   - [Document Name p.X](URL)

3. NEVER output a raw URL directly after text like:
   Document.pdf(URL)
   This breaks markdown. Always use [text](url) format.

4. NEVER wrap URLs manually. Do NOT add line breaks inside URLs.

5. Keep explanations clear and student-friendly.

6. If the user is greeting or chatting, ignore references.
"""

    user_message = f"""
User question:
{user_prompt}

Relevant reference documents (use STRICT markdown format):
{ref_block}
"""

    if not openai_client:
        raise RuntimeError("openai_client is not configured for chat.completions.")

    response = openai_client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_message.strip()},
        ],
        temperature=0.1,
        max_tokens=2000,
    )

    return response.choices[0].message.content

# ============================================================
#  TOOLS
# ============================================================
@tool
def deep_answer(question: str) -> str:
    """
    Detailed, comprehensive answer.

    Uses PDF citation logic via Azure Cognitive Search when references exist.
    Falls back to normal deep LLM explanation if search/embedding fails
    or returns no relevant documents.
    """

    print(f"[deep_answer] Called with question: {question!r}")

    # Extra safety: greetings should not go deep.
    if is_greeting(question):
        print("[deep_answer] Greeting detected inside deep_answer – returning short reply.")
        return "Hi! How can I help you today? 😊"

    # Fallback: original deep behavior
    def _fallback_deep():
        print("[deep_answer] Using fallback deep LLM (no references).")
        prompt = f"Give a deep, detailed explanation for: {question}"
        resp = llm_deep().invoke(prompt)
        return resp.content

    # If embedding/search not configured, just fallback
    if not (openai_client and EMBEDDING_DEPLOYMENT and search_client):
        print("[deep_answer] Embedding/Search env not fully configured → fallback.")
        return _fallback_deep()

    try:
        # Step 1 – embedding
        embedding = generate_embedding(question)
    except Exception as e:
        print(f"[deep_answer] Embedding error: {e} → fallback.")
        log_event("deep_answer_embedding_error", {"error": str(e)})
        return _fallback_deep()

    try:
        # Step 2 – vector search
        matches = search_matching_documents(embedding, threshold=0.5)
        print("[DEBUG] Matching docs:", matches)
    except Exception as e:
        print(f"[deep_answer] Search error: {e} → fallback.")
        log_event("deep_answer_search_error", {"error": str(e)})
        return _fallback_deep()

    # Step 3 – no matches → fallback
    if not matches:
        print("[deep_answer] No matching documents found → fallback deep LLM.")
        log_event("deep_answer_no_refs", {"info": "No matching documents, using fallback."})
        return _fallback_deep()

    # Step 4 – generate refined answer with references
    try:
        answer = generate_refined_response_with_refs(question, matches)
        log_event("deep_answer_with_refs", {"matches": matches})
        print("[deep_answer] Returning answer WITH references.")
        return answer
    except Exception as e:
        print(f"[deep_answer] Refinement error: {e} → fallback.")
        log_event("deep_answer_refine_error", {"error": str(e)})
        return _fallback_deep()


@tool
def brief_answer(question: str) -> str:
    """Short, concise answer (2–3 sentences)."""
    print(f"[brief_answer] Called with question: {question!r}")
    # Friendly greeting override here as well
    if is_greeting(question):
        return "Hi! I'm your AI tutor. Ask me anything about the course. 👋"
    prompt = f"Answer briefly (2-3 sentences): {question}"
    resp = llm_brief().invoke(prompt)
    return resp.content


# ============================================================
#  CONTROLLERS (GRADER / QUIZ / VERIFY)
# ============================================================
def grade_question_with_llm(question: str, history: list):
    """
    Grade into Strong / Normal / CounterCue.

    - GREETINGS are ALWAYS forced to "Normal".
    - If JSON parsing fails, default to Normal with explicit reason.
    """

    if is_greeting(question):
        reason = "Greeting / chit-chat detected."
        print(f"[GRADER] Forced grade=Normal for greeting. Reason: {reason}")
        return {"grade": "Normal", "reason": reason}

    # Find last Q/A pair in history (if any)
    last_q, last_a = "", ""
    for item in reversed(history):
        if "question" in item and "answer" in item:
            last_q = item.get("question", "")
            last_a = item.get("answer", "")
            break

    sys = """
You are a grader. Classify the user's question into exactly one of:
- "Strong": conceptual/advanced question showing deep understanding, as if user is an expert from the domain
- "Normal": foundational or basic question or casual chit-chat
- "CounterCue": follow-up question based on prior context

Return STRICT JSON:
{
  "grade": "Strong" | "Normal" | "CounterCue",
  "reason": "short reason"
}
"""
    user = f"Last Q: {last_q}\nLast A: {last_a}\nUser Question: {question}"
    resp_text = llm_zero_temp().invoke(sys + "\n" + user).content.strip()
    parsed = extract_json(resp_text)

    if not parsed or "grade" not in parsed:
        print(f"[GRADER] Failed to parse grade JSON. Raw: {resp_text!r}")
        return {
            "grade": "Normal",
            "reason": "Fallback to Normal because grading JSON could not be parsed.",
        }

    grade = parsed.get("grade", "Normal")
    reason = parsed.get("reason", "")

    # Additional safety: if for some reason the model marked a greeting as Strong
    if is_greeting(question) and grade != "Normal":
        print(f"[GRADER] Overriding grade={grade} to Normal for greeting.")
        grade = "Normal"
        reason = "Overridden to Normal: greeting."

    print(f"[GRADER] Question graded as {grade!r} with reason: {reason}")
    return {"grade": grade, "reason": reason}


def should_launch_quiz(history):
    """Launch quiz if ≥3 Strong questions and no quiz active."""
    if GLOBAL_STATE.get("quiz_state"):
        return False
    strong = sum(1 for h in history if h.get("grade") == "Strong")
    return strong >= 3


def generate_quiz_from_strong(history):
    """Make quiz question based on last 3 Strong questions."""
    strong_items = [h for h in history if h.get("grade") == "Strong"][-3:]
    if not strong_items:
        return None

    context = "\n".join(f"- {it['question']}" for it in strong_items)
    sys = f"""
You are a tutor. Create ONE short quiz question from these strong questions:
{context}
Return STRICT JSON:
{{
  "question": "quiz question",
  "expected_answer": "concise correct answer",
  "explanation": "2-3 sentence explanation"
}}
"""
    resp_text = llm_zero_temp().invoke(sys).content.strip()
    data = extract_json(resp_text) or {}

    quiz = {
        "question": data.get("question", "Quiz generation failed."),
        "answer": data.get("expected_answer", ""),
        "explanation": data.get("explanation", ""),
    }
    print(f"[QUIZ] Generated quiz: {quiz}")
    return quiz


def evaluate_quiz_answer(user_answer, expected_answer):
    sys = f"""
You evaluate a student's short answer.
Expected: {expected_answer}
Return STRICT JSON:
{{
  "verdict": "Satisfactory" | "Unsatisfactory",
  "reason": "short reason"
}}
"""
    resp_text = llm_zero_temp().invoke(sys + "\nStudent Answer: " + user_answer).content.strip()
    data = extract_json(resp_text) or {
        "verdict": "Unsatisfactory",
        "reason": "Parse fallback.",
    }
    print(f"[QUIZ_EVAL] verdict={data.get('verdict')} reason={data.get('reason')}")
    return data


def verify_conceptual_relevance(question, answer, quiz=None):
    quiz_text = ""
    if quiz:
        quiz_text = f"QuizQ: {quiz.get('question','')}\nExpected: {quiz.get('answer','')}"
    sys = """
You are a verifier. Determine if the Answer conceptually aligns with the Question.
If quiz info is present, ensure they are about the same concept.
Return STRICT JSON:
{
  "relevant": true|false,
  "notes": "short reason"
}
"""
    user = f"Question: {question}\nAnswer: {answer}\n{quiz_text}"
    resp_text = llm_zero_temp().invoke(sys + "\n" + user).content.strip()
    data = extract_json(resp_text) or {"relevant": True, "notes": "Default true."}
    print(f"[VERIFY] relevant={data.get('relevant')} notes={data.get('notes')}")
    return data


# ============================================================
#  FLASK ROUTES
# ============================================================
@app.route("/")
def home():
    return render_template("chat.html")


@app.route("/reset", methods=["POST"])
def reset():
    GLOBAL_STATE["history"].clear()
    GLOBAL_STATE["events"].clear()
    GLOBAL_STATE["quiz_state"] = None
    persist_log()
    print("[RESET] Memory cleared.")
    return jsonify({"message": "Memory cleared ✅"})

# Delete previous json completely
if os.path.exists("chat_log.json"):
    os.remove("chat_log.json")


@app.route("/ask", methods=["POST"])
def ask():
    user_text = request.form.get("question", "").strip()
    print(f"\n[ASK] User question: {user_text!r}")

    if not user_text:
        return jsonify({"error": "Please enter a question"}), 400

    history = GLOBAL_STATE["history"]
    quiz_state = GLOBAL_STATE.get("quiz_state")

    # ========================================================
    # 1) QUIZ FLOW (if quiz active)
    # ========================================================
    if quiz_state:
        print("[FLOW] Quiz mode active.")
        quiz_state["attempts"] += 1
        verdict = evaluate_quiz_answer(user_text, quiz_state["answer"])
        log_event("quiz_evaluate", {"attempt": quiz_state["attempts"], **verdict})

        if verdict["verdict"] == "Satisfactory":
            msg = f"✅ Correct! {verdict['reason']}"
            history.append(
                {
                    "type": "quiz_attempt",
                    "question": quiz_state["question"],
                    "user_answer": user_text,
                    "verdict": "Satisfactory",
                    "ts": timestamp(),
                }
            )
            GLOBAL_STATE["quiz_state"] = None
            persist_log()
            return jsonify({"response": msg})

        if quiz_state["attempts"] < quiz_state["max_attempts"]:
            GLOBAL_STATE["quiz_state"] = quiz_state
            persist_log()
            return jsonify(
                {"response": f"❌ Not quite: {verdict['reason']} Try again."}
            )

        msg = (
            f"❌ Incorrect again.\n\n✅ **Correct:** {quiz_state['answer']}\n"
            f"💡 **Explanation:** {quiz_state['explanation']}"
        )
        history.append(
            {
                "type": "quiz_reveal",
                "question": quiz_state["question"],
                "user_answer": user_text,
                "correct_answer": quiz_state["answer"],
                "explanation": quiz_state["explanation"],
                "ts": timestamp(),
            }
        )
        GLOBAL_STATE["quiz_state"] = None
        persist_log()
        return jsonify({"response": msg})

    # ========================================================
    # 2) NORMAL QA FLOW
    # ========================================================
    grade_info = grade_question_with_llm(user_text, history)
    grade = grade_info["grade"]
    grade_reason = grade_info["reason"]
    log_event("question_graded", {"grade": grade, "reason": grade_reason})

    # Choose tool based on grade (router)
    tool_obj = deep_answer if grade in ("Strong", "CounterCue") else brief_answer
    print(
        f"[ROUTER] grade={grade} → using tool={tool_obj.name} "
        f"(reason: {grade_reason})"
    )

    # Call tool
    try:
        start = time.time()
        answer = tool_obj.func(user_text)
        duration = round(time.time() - start, 2)
        log_event("tool_used", {"name": tool_obj.name, "duration_s": duration})
        print(f"[TOOL] {tool_obj.name} completed in {duration}s")
    except Exception as e:
        log_event("tool_error", {"name": tool_obj.name, "error": str(e)})
        print(f"[ERROR] Tool {tool_obj.name} raised: {e}")
        return jsonify({"error": str(e)}), 500

    verify = verify_conceptual_relevance(user_text, answer, None)
    log_event("verify", verify)

    history.append(
        {
            "type": "qa",
            "question": user_text,
            "answer": answer,
            "grade": grade,
            "grade_reason": grade_reason,
            "verification": verify,
            "tools_used": [tool_obj.name],
            "ts": timestamp(),
        }
    )

    # ========================================================
    # 3) QUIZ TRIGGER
    # ========================================================
    if should_launch_quiz(history):
        quiz = generate_quiz_from_strong(history)
        if quiz:
            GLOBAL_STATE["quiz_state"] = {
                "question": quiz["question"],
                "answer": quiz["answer"],
                "explanation": quiz["explanation"],
                "attempts": 0,
                "max_attempts": 2,
            }
            log_event("quiz_launched", {"question": quiz["question"]})
            persist_log()
            combined = (
                f"{answer}\n\n---\n🧪 **Quick Check:** {quiz['question']}\n"
                "Reply with your answer."
            )
            return jsonify({"response": combined})

    persist_log()
    return jsonify({"response": answer})


# ============================================================
#  ENTRY POINT
# ============================================================
if __name__ == "__main__":
    print("[START] AI Tutor server starting on http://127.0.0.1:5000")
    app.run(debug=True)