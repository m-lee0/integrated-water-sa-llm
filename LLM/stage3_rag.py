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

# Load stage 2 conversation
stage2_path = Path(__file__).parent / f'stage2_rag_{version}.json'
if not stage2_path.exists():
    raise FileNotFoundError(f"stage2_rag_{version}.json not found. Run stage2_rag.py first.")
stage2 = json.loads(stage2_path.read_text())
print("Stage 2 conversation loaded.")

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
DB_PATH       = str(Path(__file__).parent / "chroma_db_stage3")  # separate DB: Nossent + Wagener
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

USER_PROMPT = """Literature Contextualisation: Assess whether the parameter \
ranking and interaction pattern identified in Stage 1 are consistent with \
findings reported in the GSA literature for comparable water system models. \
Then evaluate whether the result supports the consistency criterion: that the \
parameters governing surface runoff and subsurface percolation exert physically \
expected levels of control on mean flow.

Constraints:
- Do not fabricate citations. Only cite sources you can identify with confidence, \
including author and year.
- If citing a retrieved document, refer to it explicitly by author and year.
- If no source is available to support a claim, state that the assessment is \
based on general GSA practice.
- Each section must be 2-3 sentences. Do not write more or less."""


# Main
def main():
    # 1. Build or load vector store
    vectorstore = build_or_load_vectorstore()

    # 2. Format the user question
    question = USER_PROMPT

    # 3. Retrieve chunks explicitly and log them
    retrieved_docs, context = retrieve_and_log(vectorstore, question)

    # 4. Build conversation history from stages 1 and 2
    chat_history = [
        ("human", stage2["stage1"]["question"]),
        ("ai",    stage2["stage1"]["response"]),
        ("human", stage2["stage2"]["question"]),
        ("ai",    stage2["stage2"]["response"]),
    ]

    # 5. Build prompt with history and call LLM
    llm = build_llm()
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        *chat_history,
        ("human", "{question}"),
    ])
    chain = prompt_template | llm | StrOutputParser()

    print("\n--- STAGE 3 RESPONSE (RAG) ---\n")
    response = chain.invoke({"context": context, "question": question})
    print(response)

    # 6. Save conversation
    conversation = {
        "version": version,
        "stage1": stage2["stage1"],
        "stage2": stage2["stage2"],
        "stage3": {
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

    conversation_path = Path(__file__).parent / f"stage3_rag_{version}.json"
    conversation_path.write_text(json.dumps(conversation, indent=2))
    print(f"\nStage 3 conversation saved to {conversation_path}")


if __name__ == "__main__":
    main()