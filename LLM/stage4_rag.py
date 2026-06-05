import argparse
import os
import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True, help='Run version tag, e.g. v1')
args = parser.parse_args()
version = args.version

# SA results
N = 300  # hardcoded
results_fid = Path(__file__).parent.parent / 'SA' / 'results'
sa_results = (results_fid / 'sa_results_formatted.txt').read_text()

# Load stage 3 conversation
stage3_path = Path(__file__).parent / f'stage3_rag_{version}.json'
if not stage3_path.exists():
    raise FileNotFoundError(f"stage3_rag_{version}.json not found. Run stage3_rag.py first.")
stage3 = json.loads(stage3_path.read_text())
print("Stage 3 conversation loaded.")

# Configuration
EMBEDDING_BASE_URL = "http://localhost:1234/v1"
LLM_BASE_URL       = "http://localhost:1234/v1"
EMBEDDING_MODEL    = "text-embedding-nomic-embed-text-v1.5"
LLM_MODEL          = "qwen/qwen3-vl-8b"
API_KEY            = "lm-studio"

PDF_PATHS = [
    Path(__file__).parent / "Nossent, Elsen & Bauwens (2011).pdf",
    Path(__file__).parent / "Wagener, Reinecke & Pianosi (2022).pdf",
]
DB_PATH       = str(Path(__file__).parent / "chroma_db_stage3")  # shared with stage 3
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 50
RETRIEVER_K   = 6

# Vector store
def build_or_load_vectorstore():
    """Build vector DB from PDFs or load existing one from disk."""
    embeddings = OpenAIEmbeddings(
        base_url=EMBEDDING_BASE_URL,
        api_key=API_KEY,
        model=EMBEDDING_MODEL,
        check_embedding_ctx_length=False,
    )

    if os.path.exists(DB_PATH) and len(os.listdir(DB_PATH)) > 0:
        print("Loading existing vector database...")
        return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    print("Building new vector database from PDFs...")
    all_docs = []
    for pdf_path in PDF_PATHS:
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        print(f"  Loaded {len(docs)} pages from {pdf_path.name}.")
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(all_docs)
    print(f"  Created {len(chunks)} chunks.")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH,
    )
    print(f"  Vector database saved to {DB_PATH}")
    return vectorstore


# Retrieval
def retrieve_and_log(vectorstore, query: str) -> tuple[list, str]:
    """
    Retrieve top-k chunks using cosine similarity and log them to console.
    Returns the list of Document objects and the formatted context string.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )
    retrieved_docs = retriever.invoke(query)

    print(f"\nRETRIEVED CHUNKS (k={RETRIEVER_K}):")
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "?")
        print(f"\n[Chunk {i} | {Path(source).name}, p.{page}]")
        print(doc.page_content)

    context = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)
    return retrieved_docs, context

# LLM
def build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=API_KEY,
        model=LLM_MODEL,
        temperature=0.0,
        max_tokens=2048,
    )

# Prompts
SYSTEM_PROMPT = """You are analysing Sobol sensitivity analysis results for a \
water systems model. Use the following excerpts from reference documents to \
inform your interpretation where relevant.

--- REFERENCE DOCUMENT EXCERPTS ---
{context}
--- END OF EXCERPTS ---

When interpreting results, draw on the above where applicable. \
Do not fabricate citations."""

USER_PROMPT = """Action Recommendations: Based on the Sobol sensitivity analysis \
results interpreted above, identify 3 to 5 concrete actions that should be taken \
to improve confidence in the model results or address identified limitations.

For each action, state:
- What the action is
- Which specific numerical result justifies it (cite the exact index value and \
confidence interval)
- What methodological or literature-based reasoning supports it, if available
- What outcome is expected

Constraints:
- Every action must be traceable to a specific numerical result from the SA output. \
Do not recommend actions that are not directly supported by the values provided.
- Where retrieved literature provides relevant methodological guidance (e.g. on \
convergence thresholds, sampling requirements, or parameter prioritisation), cite \
it explicitly by author and finding. Do not cite literature that is not available \
to you.
- Do not use thresholds or benchmarks from case-specific studies unless you can \
identify the exact source. If no literature source is available, state that the \
justification is based on general SA practice.
- Do not recommend actions related to parameter leverage or model complexity unless \
directly supported by the ST values provided."""

# Main
def main():
    # 1. Build or load vector store
    vectorstore = build_or_load_vectorstore()

    # 2. Format the user question
    question = USER_PROMPT

    # 3. Retrieve chunks explicitly and log them
    retrieved_docs, context = retrieve_and_log(vectorstore, question)

    # 4. Build conversation history from stages 1, 2, and 3
    chat_history = [
        ("human", stage3["stage1"]["question"]),
        ("ai",    stage3["stage1"]["response"]),
        ("human", stage3["stage2"]["question"]),
        ("ai",    stage3["stage2"]["response"]),
        ("human", stage3["stage3"]["question"]),
        ("ai",    stage3["stage3"]["response"]),
    ]

    # 5. Build prompt with history and call LLM
    llm = build_llm()
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        *chat_history,
        ("human", "{question}"),
    ])
    chain = prompt_template | llm | StrOutputParser()

    print("\n--- STAGE 4 RESPONSE (RAG) ---\n")
    response = chain.invoke({"context": context, "question": question})
    print(response)

    # 6. Save conversation
    conversation = {
        "version": version,
        "stage1": stage3["stage1"],
        "stage2": stage3["stage2"],
        "stage3": stage3["stage3"],
        "stage4": {
            "retrieved_chunks": [
                {
                    "chunk_index": i + 1,
                    "source": doc.metadata.get("source", "unknown"),
                    "page":   doc.metadata.get("page", None),
                    "content": doc.page_content,
                }
                for i, doc in enumerate(retrieved_docs)
            ],
            "question": question,
            "response": response,
        },
    }

    conversation_path = Path(__file__).parent / f"stage4_rag_{version}.json"
    conversation_path.write_text(json.dumps(conversation, indent=2))
    print(f"\nStage 4 conversation saved to {conversation_path}")

if __name__ == "__main__":
    main()