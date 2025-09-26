# app.py
from flask import Flask, render_template, request, jsonify
from langchain.chains import RetrievalQA
from langchain.vectorstores import DocArrayInMemorySearch
from langchain_community.llms import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings

app = Flask(__name__)

# --- LangChain Setup ---
# A sample of college data to be used by the chatbot
college_data = [
    "The College of Engineering offers undergraduate degrees in Computer Science, Mechanical Engineering, and Electrical Engineering.",
    "Admissions requirements for undergraduate programs include a high school diploma, a minimum GPA of 3.0, and SAT/ACT scores.",
    "The campus has a library, a gymnasium, and several dining halls.",
    "The academic calendar consists of two semesters: Fall and Spring.",
    "Dr. Jane Doe is a professor in the Computer Science department.",
]

# Create an in-memory vector store with the college data
embeddings = OpenAIEmbeddings()
db = DocArrayInMemorySearch.from_texts(college_data, embeddings)

# Create a retrieval-based Q&A chain
retriever = db.as_retriever()
llm = OpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

@app.route("/")
def index():
    """Renders the main page of the website."""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Handles the chatbot's questions and returns a response."""
    user_question = request.json.get("question")
    if not user_question:
        return jsonify({"answer": "Please enter a question."})

    # Get the answer from the LangChain model
    try:
        response = qa_chain.run(user_question)
        return jsonify({"answer": response})
    except Exception as e:
        return jsonify({"answer": f"An error occurred: {e}"})

if __name__ == "__main__":
    app.run(debug=True)

# Lang-chain
