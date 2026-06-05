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

# Args parsing for version tag
parser = argparse.ArgumentParser() 
parser.add_argument('--version', type=str, required=True, help='Run version tag, e.g. v1')
args = parser.parse_args()
version = args.version

# SA results
N=300 # hardcoded
results_fid = Path(__file__).parent.parent / 'SA' / 'results'
sa_results = (results_fid / 'sa_results_formatted.txt').read_text()

# Configuration
EMBEDDING_BASE_URL = "http://localhost:1234/v1"
LLM_BASE_URL       = "http://localhost:1234/v1"
EMBEDDING_MODEL    = "text-embedding-nomic-embed-text-v1.5"   # exact name shown in LM Studio
LLM_MODEL          = "qwen/qwen3-vl-8b"                       # exact name shown in LM Studio  
API_KEY            = "lm-studio"

PDF_PATH   = Path(__file__).parent / "Nossent, Elsen & Bauwens (2011).pdf"
DB_PATH    = str(Path(__file__).parent / "chroma_db")
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 50 
RETRIEVER_K   = 6

# Vector store
def build_or_load_vectorstore():
    """Build vector DB from PDF or load existing one from disk."""
    embeddings = OpenAIEmbeddings(
        base_url=EMBEDDING_BASE_URL,
        api_key=API_KEY,
        model=EMBEDDING_MODEL,
        check_embedding_ctx_length=False,
    )

    if os.path.exists(DB_PATH) and len(os.listdir(DB_PATH)) > 0:
        print("Loading existing vector database...")
        return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    loader = PyPDFLoader(str(PDF_PATH))
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from PDF.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print(f"Vector database saved to {DB_PATH}")
    return vectorstore
    
# Retrieval 
def retrieve_and_log(vectorstore, query: str) -> tuple[list, str]:
    """
    Retrieve top-k chunks using cosine similarity and log them to console.
    Returns the list of Document objects and the formatted context string.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",           # cosine similarity (Chroma default)
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

# Prompt
SYSTEM_PROMPT = """You are analysing Sobol sensitivity analysis results for a 
    water systems model. Use the following excerpts from a reference document to 
    inform your interpretation where relevant.

    --- REFERENCE DOCUMENT EXCERPTS ---
    {context}
    --- END OF EXCERPTS ---

    When interpreting results, draw on the above where applicable. 
    Do not fabricate citations."""

USER_PROMPT = """You are analysing Sobol sensitivity analysis results for a hydrology model \
with {sa_results}

Respond with exactly these four sections, in this order:
Parameter Ranking: Rank the parameters by ST only. Cite the values. State that \
ranking is uncertain if ST confidence intervals overlap.
Interaction Contribution: For each parameter, compute ST − S1. Cite both input \
values and the result. Interpret which parameter's output variance is more \
interaction-driven.
Interaction Assessment: Assess the pairwise interaction between parameters using \
S2. Cite the value.
Reliability: Assess reliability for ST, S1, and S2 individually using the \
criterion CI / index > 0.5. CI / index > 0.5 is unreliable. Do not assess reliability \
for ST − S1. Based on this assessment, comment whether N={N} appears sufficient for \
convergence, noting that this criterion serves as a proxy for convergence rather than \
a formal convergence test.

Constraints:
- Do not write anything outside these four sections.
- Each section must be 2-3 sentences. Do not write more or less.
- Report all numerical values with the format: index = value ± confidence interval. \
Do not deviate from this format."""

# main
def main():
    # 1. Build or load vector store
    vectorstore = build_or_load_vectorstore()

    # 2. Format the user question (needed before retrieval so the query is meaningful)
    question = USER_PROMPT.format(sa_results=sa_results, N=N)

    # 3. Retrieve chunks explicitly and log them
    _, context = retrieve_and_log(vectorstore, question)

    # 4. Build prompt and call LLM
    llm = build_llm()
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
    chain = prompt_template | llm | StrOutputParser()

    print("--- STAGE 1 RESPONSE (RAG) ---\n")
    response = chain.invoke({"context": context, "question": question})
    print(response)

    # 5. Save conversation
    conversation = {
        "version": version,
        "retrieved_chunks": [
            {
                "chunk_index": i + 1,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", None),
                "content": doc.page_content,
            }
            for i, doc in enumerate(
                vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": RETRIEVER_K},
                ).invoke(question)
            )
        ],
        "question": question,
        "response": response,
    }

    conversation_path = Path(__file__).parent / f"stage1_rag_{version}.json"
    conversation_path.write_text(json.dumps(conversation, indent=2))
    print(f"\nStage 1 conversation saved to {conversation_path}")


if __name__ == "__main__":
    main()