from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import datetime
import os

app = Flask(__name__)

# RAG Setup
print("Loading RAG system...")

loader = PyMuPDFLoader("/Users/yashshakya787/genAI/data/customer_support_kb.pdf")
docs   = loader.load()
print(f"✅ PDF loaded — {len(docs)} pages")

splitter = RecursiveCharacterTextSplitter(
    chunk_size    = 300,
    chunk_overlap = 50,
    separators    = ["\n\n", "\n", "Q:", "A:", " "]
)
chunks = splitter.split_documents(docs)
print(f"✅ Chunks: {len(chunks)}")

embedding   = OllamaEmbeddings(model="nomic-embed-text")
chroma_path = "/Users/yashshakya787/genAI/data/chroma_whatsapp"

# ── ChromaDB — duplicate vectors fix ──────────────────────────
if os.path.exists(chroma_path) and os.listdir(chroma_path):
    vectorstore = Chroma(
        collection_name    = "whatsapp_bot",
        embedding_function = embedding,
        persist_directory  = chroma_path
    )
    print(f"✅ ChromaDB loaded — {vectorstore._collection.count()} vectors")
else:
    vectorstore = Chroma.from_documents(
        documents         = chunks,
        embedding         = embedding,
        collection_name   = "whatsapp_bot",
        persist_directory = chroma_path
    )
    print(f"✅ ChromaDB created — {vectorstore._collection.count()} vectors")

retriever = vectorstore.as_retriever(
    search_type   = "similarity",
    search_kwargs = {"k": 3}
)

# ── LLM ───────────────────────────────────────────────────────
llm = OllamaLLM(model="llama3.1:8b", temperature=0.2)

# ── Prompt ────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_template("""
You are a helpful WhatsApp customer support agent for TechCorp.
Use ONLY the context below to answer.
Keep answer SHORT — max 3 lines only.
If not found say: I don't have info on that. A human agent will contact you soon.

Context: {context}
Question: {question}
Response:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context" : retriever | format_docs,
        "question": lambda x: x
    }
    | prompt
    | llm
    | StrOutputParser()
)
print("✅ RAG Chain ready!")

# ── Ticket System ─────────────────────────────────────────────
ticket_db    = {}
ticket_count = 1000

def create_ticket(question, sender):
    global ticket_count
    ticket_count += 1
    tid = "TKT-" + str(ticket_count)
    ticket_db[tid] = {
        "id"      : tid,
        "question": question,
        "sender"  : sender,
        "status"  : "Open",
        "time"    : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return tid

# ── Flask ─────────────────────────────────────────────────────
@app.after_request
def add_header(response):
    response.headers['ngrok-skip-browser-warning'] = 'true'
    return response

@app.route("/", methods=["GET"])
def home():
    return f"✅ Bot running! Total tickets: {len(ticket_db)}", 200

@app.route("/whatsapp", methods=["POST"])
def whatsapp():
    print("\n" + "="*50)
    incoming_msg = request.form.get("Body", "").strip()
    sender       = request.form.get("From", "")
    print(f"From   : {sender}")
    print(f"Message: {incoming_msg}")

    resp = MessagingResponse()
    msg  = resp.message()

    # ── Greeting ──────────────────────────────────────────────
    if any(w in incoming_msg.lower() for w in ['hi','hello','hey','hii','helo']):
        msg.body(
            "Hey! Welcome to TechCorp Support!\n\n"
            "I can help you with:\n"
            "📦 Order & Delivery\n"
            "💰 Refunds & Returns\n"
            "🔐 Account & Login\n"
            "💳 Payment issues\n\n"
            "Type your problem!"
        )

    # ── Tickets check ─────────────────────────────────────────
    elif incoming_msg.lower() == 'tickets':
        if not ticket_db:
            msg.body("No tickets yet!")
        else:
            lines = [
                f"{t['id']}: {t['status']}"
                for t in list(ticket_db.values())[-5:]
            ]
            msg.body("Your recent tickets:\n" + "\n".join(lines))

    # ── Close ticket ──────────────────────────────────────────
    elif incoming_msg.lower().startswith('close '):
        tid = incoming_msg.split(' ', 1)[1].strip().upper()
        if tid in ticket_db:
            ticket_db[tid]['status'] = 'Resolved'
            msg.body(f"Ticket {tid} marked as Resolved!")
        else:
            msg.body(f"Ticket {tid} not found.")

    # ── Thanks / Bye ──────────────────────────────────────────
    elif any(w in incoming_msg.lower() for w in ['thanks','thank you','bye','done','resolved']):
        msg.body("Glad I could help! Have a great day! Type 'hi' anytime. 😊")

    # ── RAG Answer ────────────────────────────────────────────
    else:
        try:
            print("RAG thinking...")
            answer = rag_chain.invoke(incoming_msg)
            tid    = create_ticket(incoming_msg, sender)
            print(f"Answer : {answer[:100]}...")
            print(f"Ticket : {tid}")
            msg.body(f"{answer}\n\n🎫 Ticket ID: {tid}")
        except Exception as e:
            print(f"Error: {e}")
            msg.body("Sorry, having trouble right now. Please try again!")

    print("="*50)
    return str(resp), 200

if __name__ == "__main__":
    app.run(port=5001, debug=False)