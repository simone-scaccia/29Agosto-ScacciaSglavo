"""RAG tools built on LangChain, FAISS, and Azure OpenAI.

This module provides utilities to load and split local documents, build or
load a FAISS vector store, configure a retriever, and compose a simple RAG
chain that answers questions grounded in retrieved context. It also exposes a
`rag_tool` entry point suitable for use with CrewAI.
"""

import os
from dotenv import load_dotenv
from crewai.tools import tool

from pathlib import Path
from typing import List
from dataclasses import dataclass

import faiss
from langchain.chat_models import init_chat_model
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_API_BASE")
os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = os.getenv("LLM_DEPLOYMENT_NAME")

@dataclass
class Settings:
    """Configuration for RAG components.

    Attributes:
        persist_dir: Directory where the FAISS index is persisted.
        chunk_size: Target character length for text chunks.
        chunk_overlap: Overlap in characters between consecutive chunks.
        search_type: Retrieval mode, either "mmr" or "similarity".
        k: Number of final results returned by the retriever.
        fetch_k: Number of initial candidates considered (used by MMR).
        mmr_lambda: Diversification parameter for MMR (0=max diversity, 1=max relevance).
        llm_model_name: Deployment/model name for the chat model.
    """
    # Persistenza FAISS
    persist_dir: str = "faiss_index_example"
    # Text splitting
    chunk_size: int = 700
    chunk_overlap: int = 100
    # Retriever (MMR)
    search_type: str = "mmr"        # "mmr" o "similarity"
    k: int = 4                      # risultati finali
    fetch_k: int = 20               # candidati iniziali (per MMR)
    mmr_lambda: float = 0.3         # 0 = diversificazione massima, 1 = pertinenza massima
    # LM Studio (OpenAI-compatible)
    llm_model_name: str = "gpt-4o"  # nome del modello in LM Studio, via env var

SETTINGS = Settings()

# =========================
# Componenti di base
# =========================

def get_embeddings(settings: Settings) -> AzureOpenAIEmbeddings:
    """Create an Azure OpenAI embeddings client.

    Args:
        settings: RAG configuration.

    Returns:
        AzureOpenAIEmbeddings: Configured embeddings client.
    """
    return AzureOpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=settings.chunk_size)

def get_llm(settings: Settings):
    """Initialize the chat model client.

    Args:
        settings: RAG configuration.

    Returns:
        Any: A chat model instance initialized via LangChain.
    """
    return init_chat_model(settings.llm_model_name, model_provider="azure_openai", api_version="2024-12-01-preview")

def load_real_documents_from_folder(folder_path: str) -> List[Document]:
    """Load `.txt` and `.md` files from a folder as LangChain documents.

    Each document is tagged with a `source` metadata field for citation.

    Args:
        folder_path: Absolute or relative path to the folder containing text files.

    Returns:
        list[Document]: Loaded documents with `source` metadata populated.

    Raises:
        ValueError: If the folder does not exist or is not a directory.
    """
    folder = Path(folder_path)
    documents: List[Document] = []

    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"La cartella '{folder_path}' non esiste o non è una directory.")

    for file_path in folder.glob("**/*"):
        if file_path.suffix.lower() not in [".txt", ".md"]:
            continue  # ignora file non supportati

        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()

        # Aggiunge il metadato 'source' per citazioni (es. nome del file)
        for doc in docs:
            doc.metadata["source"] = file_path.name
            print(doc.metadata["source"])

        documents.extend(docs)

    return documents

def split_documents(docs: List[Document], settings: Settings) -> List[Document]:
    """Split documents into semantically coherent chunks.

    Uses a recursive character splitter with multiple separators and the
    configured chunk size and overlap to optimize retrieval quality.

    Args:
        docs: Documents to split.
        settings: RAG configuration.

    Returns:
        list[Document]: Chunked documents.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=[
            "\n\n", "\n", ". ", "? ", "! ", "; ", ": ",
            ", ", " ", ""  # fallback aggressivo
        ],
    )
    return splitter.split_documents(docs)

def build_faiss_vectorstore(chunks: List[Document], embeddings: AzureOpenAIEmbeddings, persist_dir: str) -> FAISS:
    """Build a FAISS vector store from chunked documents and persist it.

    Args:
        chunks: Chunked documents to index.
        embeddings: Embeddings model to encode chunks.
        persist_dir: Directory where the FAISS index will be saved.

    Returns:
        FAISS: The created FAISS vector store instance.
    """
    # Determina la dimensione dell'embedding
    vs = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    vs.save_local(persist_dir)
    return vs

def load_or_build_vectorstore(settings: Settings, embeddings: AzureOpenAIEmbeddings, docs: List[Document]) -> FAISS:
    """Load a persisted FAISS vector store or build it if missing.

    This will attempt to load the index from ``settings.persist_dir``. If the
    files are not present, the function splits documents and builds a new
    index, saving it to disk.

    Args:
        settings: RAG configuration.
        embeddings: Embeddings model used for encoding.
        docs: Source documents to (re)build the index from if needed.

    Returns:
        FAISS: Loaded or newly created FAISS vector store.
    """
    persist_path = Path(settings.persist_dir)
    index_file = persist_path / "index.faiss"
    meta_file = persist_path / "index.pkl"

    if index_file.exists() and meta_file.exists():
        # Dal 2024/2025 molte build richiedono il flag 'allow_dangerous_deserialization' per caricare pkl locali
        return FAISS.load_local(
            settings.persist_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )

    chunks = split_documents(docs, settings)
    return build_faiss_vectorstore(chunks, embeddings, settings.persist_dir)

def make_retriever(vector_store: FAISS, settings: Settings):
    """Configure the retriever for the given vector store.

    Uses MMR (Maximal Marginal Relevance) when ``settings.search_type == 'mmr'``
    to reduce redundancy and improve coverage; otherwise uses plain similarity.

    Args:
        vector_store: The FAISS vector store to wrap as a retriever.
        settings: RAG configuration controlling retrieval behavior.

    Returns:
        BaseRetriever: A configured retriever instance.
    """
    if settings.search_type == "mmr":
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": settings.k, "fetch_k": settings.fetch_k, "lambda_mult": settings.mmr_lambda},
        )
    else:
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.k},
        )

def format_docs_for_prompt(docs: List[Document]) -> str:
    """Format retrieved documents into a prompt-ready context string.

    Each document is prefixed with a bracketed source tag for downstream
    citation, e.g. ``[source:FILE]``.

    Args:
        docs: Documents returned by the retriever.

    Returns:
        str: A newline-separated string suitable for inclusion in the prompt.
    """
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", f"doc{i}")
        lines.append(f"[source:{src}] {d.page_content}")
    return "\n\n".join(lines)

def build_rag_chain(llm, retriever):
    """Compose the RAG chain: retrieval -> prompt -> LLM -> parsing.

    The chain injects the retrieved context and the user question into a
    system+human prompt and parses the model output to a string.

    Args:
        llm: Chat model instance to generate the final answer.
        retriever: Retriever used to fetch relevant documents.

    Returns:
        Runnable: A LangChain runnable that accepts a question string and
        returns a grounded answer string.
    """
    system_prompt = (
        "You are a personal assistant. Please respond in English."
        "Use only the CONTENT and INFORMATION in context."
        "Include citations in square brackets in the format [source:...]."
        "If the information is missing, state that it is not available."
    )

    prompt_messages = [("system", system_prompt)]

    # Aggiunge ultima domanda con placeholder {question} e {context}
    prompt_messages.append(
        ("human",
        "Question:\n{question}\n\n"
        "Context:\n{context}\n\n"
        "Instructions:\n"
        "1) Answer only with information contained in the context.\n"
        "2) Always cite relevant sources in the format [source:FILE]."
    ))

    prompt_template = ChatPromptTemplate.from_messages(prompt_messages)

    # LCEL: dict -> prompt -> llm -> parser
    chain = (
        {
            "context": retriever | format_docs_for_prompt,
            "question": RunnablePassthrough(),
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return chain

def rag_answer(question: str, chain) -> str:
    """Execute the RAG chain for a single question.

    Args:
        question: The user query to answer.
        chain: The runnable chain produced by ``build_rag_chain``.

    Returns:
        str: The generated answer text.
    """
    return chain.invoke(question)

@tool("RAG tool")
def rag_tool(query: str):
    """Execute RAG (Retrieval-Augmented Generation) for a given query.

    Loads or builds the FAISS index from local documents, configures the
    retriever and chat model, and returns an answer grounded in retrieved
    context.

    Args:
        query: The user question to answer.

    Returns:
        str: The grounded answer including inline source citations.
    """
    settings = SETTINGS
    print(f"TEST ######################################################## k: {settings.k}")

    # 1) Componenti
    embeddings = get_embeddings(settings)
    llm = get_llm(settings)
    
    # 2) Dati simulati e indicizzazione (load or build)
    docs = load_real_documents_from_folder("C:\\Users\\KB316GR\\OneDrive - EY\\Desktop\\Academy\\26_agosto\\es3_research_rag\\rag_context")
    #print(f"Numero documenti caricati: {len(docs)}")
    #print("Contenuto primo documento:", docs[0].page_content[:500])
    vector_store = load_or_build_vectorstore(settings, embeddings, docs)

    # 3) Retriever ottimizzato
    retriever = make_retriever(vector_store, settings)

    # 4) Catena RAG
    chain = build_rag_chain(llm, retriever)

    ans = rag_answer(query, chain)

    return ans