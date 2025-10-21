from flask import Flask, render_template, request, jsonify
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Azure OpenAI config
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# Tools
@tool
def deep_answer(question: str) -> str:
    """Detailed, comprehensive answer"""
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
    """Short, concise answer"""
    llm = AzureChatOpenAI(
        azure_deployment=AZURE_DEPLOYMENT,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        temperature=0.3,
    )
    prompt = f"Answer briefly in 2-3 sentences: {question}"
    return llm.invoke(prompt).content

# Controller LLM (decides which tool to use)
controller_llm = AzureChatOpenAI(
    azure_deployment=AZURE_DEPLOYMENT,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    temperature=0,
)

# Initialize the agent
agent = initialize_agent(
    tools=[deep_answer, brief_answer],
    llm=controller_llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=False,
)

# Flask routes
@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question")
    if not question:
        return jsonify({"error": "Please enter a question"}), 400
    try:
        response = agent.run(question)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
