from flask import Flask, render_template, request, jsonify, session
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os, json, re

# ------------------ Load Environment ------------------
load_dotenv()

app = Flask(__name__)
app.secret_key = "supersecretkey"

# ------------------ Azure Config ------------------
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# ------------------ Memory Reset ------------------
@app.before_request
def ensure_fresh_session():
    if "initialized" not in session:
        session.clear()
        session["initialized"] = True
        session["history"] = []
        print("✨ New session initialized, memory flushed.")

# ------------------ Tools ------------------
@tool
def deep_answer(question: str) -> str:
    llm = AzureChatOpenAI(
        azure_deployment=AZURE_DEPLOYMENT,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        temperature=0.7,
    )
    prompt = f"Give a deep, detailed explanation for: {question}"
    return llm.invoke(prompt).content


@tool
def brief_answer(question: str) -> str:
    llm = AzureChatOpenAI(
        azure_deployment=AZURE_DEPLOYMENT,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        temperature=0.3,
    )
    prompt = f"Answer briefly in 2-3 sentences: {question}"
    return llm.invoke(prompt).content


# ------------------ Memory-based Reasoning ------------------
def check_relevance_with_llm(question: str, history: list) -> bool:
    if not history:
        return False

    controller_llm = AzureChatOpenAI(
        azure_deployment=AZURE_DEPLOYMENT,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        temperature=0,
    )

    history_text = "\n".join(
        [f"Q: {item['question']}\nA: {item['answer']}" for item in history[-5:]]
    )

    schema = """
    You are a decision engine that checks if a new user question is semantically related
    to any previously asked questions. Respond in *strict JSON* using this schema:
    {
      "related": true or false,
      "reason": "Short one-line reason for your decision"
    }
    """

    prompt = f"""
{schema}

Previous conversation:
{history_text}

New question:
{question}
"""

    try:
        response = controller_llm.invoke(prompt).content.strip()
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            print(f"🧠 Memory relevance check: {data}")
            return data.get("related", False)
        else:
            print("⚠️ No JSON found in LLM output.")
            return False
    except Exception as e:
        print(f"⚠️ LLM relevance check failed: {e}")
        return False


# ------------------ Flask Routes ------------------
@app.route("/")
def home():
    return render_template("chat.html")


@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question")
    if not question:
        return jsonify({"error": "Please enter a question"}), 400

    memory = session.get("history", [])

    is_related = check_relevance_with_llm(question, memory)
    chosen_tool = deep_answer if is_related else brief_answer
    print(f"🤖 Tool chosen: {'deep_answer' if is_related else 'brief_answer'}")

    try:
        response = chosen_tool.func(question)
        memory.append({"question": question, "answer": response})
        session["history"] = memory
        session.modified = True
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reset", methods=["POST"])
def reset_session():
    session.clear()
    return jsonify({"message": "🧹 Memory cleared successfully."})


# ------------------ Main ------------------
if __name__ == "__main__":
    app.run(debug=True)