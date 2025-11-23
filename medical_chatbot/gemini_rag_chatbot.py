#!/usr/bin/env -S uv run --with-requirements requirements.txt
# -*- coding: utf-8 -*-
"""Simple RAG Chain with Gemini and ChromaDB.

A basic Retrieval-Augmented Generation (RAG) system that:
- Loads a medical Q&A dataset
- Stores embeddings in ChromaDB
- Answers questions using retrieved context

Dependencies: langchain, langchain-google-genai, chromadb
"""

import os
from pathlib import Path

import gdown
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

BASE_DIR = Path(__file__).resolve().parent
DATASET_ID = "1FpC7_DaxWQJf4JoVQLlbSYcSTZSz86C6"
DATASET_PATH = BASE_DIR / "inputs" / "ai-medical-chatbot.csv"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "medical_docs"
SYSTEM_PROMPT = (
    "You are a medical assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise. If the question is not clear, ask follow-up questions."
    "\n\n"
    "{context}"
)


def init_api_keys() -> str:
    """Load API keys from .env and return GOOGLE_API_KEY."""
    load_dotenv(BASE_DIR.parent / ".env")

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to .env before running the script.")

    return google_api_key


def load_dataset() -> pd.DataFrame:
    """Download dataset if needed and return DataFrame with combined 'document' column.

    The dataset contains medical Q&A data with columns: Description, Patient, Doctor.
    These are combined into a single 'document' column for embedding.
    """
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATASET_PATH.exists():
        print("Downloading dataset...")
        gdown.download(id=DATASET_ID, output=str(DATASET_PATH), quiet=False)

    df = pd.read_csv(DATASET_PATH)

    # Combine the fields into one document for embedding
    df['document'] = (
        df['Description'].fillna('') + '\n\n' +
        'Patient: ' + df['Patient'].fillna('') + '\n\n' +
        'Doctor: ' + df['Doctor'].fillna('')
    )
    return df


def init_vector_store(df: pd.DataFrame, num_docs: int | None = None, batch_size: int = 1000):
    """Initialize ChromaDB vector store with Google embeddings.

    Args:
        df: DataFrame with 'document' column
        num_docs: Number of documents to embed (None = all documents)
        batch_size: Number of documents per batch for embedding (default 1000)

    Returns:
        Tuple of (vectorstore, retriever)
    """
    # task_type optimizes embeddings for specific use cases:
    # - "retrieval_document": for documents to be searched
    # - "retrieval_query": for search queries
    # - "semantic_similarity": for comparing text similarity
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        task_type="retrieval_document"
    )

    persist_dir = str(CHROMA_DIR)

    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        # Load existing DB
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
    else:
        # Create new DB from documents in batches
        rows = df.head(num_docs) if num_docs else df
        total_docs = len(rows)
        print(f"Indexing {total_docs} documents in batches of {batch_size}...")

        # Create empty vectorstore first
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )

        # Process in batches
        for start_idx in tqdm(range(0, total_docs, batch_size), desc="Embedding batches"):
            end_idx = min(start_idx + batch_size, total_docs)
            batch_rows = rows.iloc[start_idx:end_idx]
            batch_docs = [
                Document(page_content=row['document'])
                for _, row in batch_rows.iterrows()
            ]
            vectorstore.add_documents(batch_docs)

        vectorstore.persist()
        print(f"Created vector store with {vectorstore._collection.count()} documents")

    # Wrap vectorstore as a Retriever for use in LangChain RAG chains
    retriever = vectorstore.as_retriever()
    return vectorstore, retriever


def build_rag_chain(retriever, model: str = "gemini-2.0-flash", temperature: float = 0.2):
    """Build a Retrieval-Augmented Generation (RAG) chain.

    The RAG chain retrieves relevant documents based on user queries and uses
    an LLM to generate answers based on the retrieved context.

    Components:
    - system_prompt: Tells the LLM how to use the retrieved context
    - create_stuff_documents_chain: Combines retrieved documents into LLM input
    - create_retrieval_chain: Connects retriever and processing chain

    Args:
        retriever: The document retriever from vector store
        model: Gemini model name
        temperature: LLM temperature (0.0 = deterministic, 1.0 = creative)

    Returns:
        Tuple of (rag_chain, llm)
    """
    # Create a chat prompt template with two message types:
    # - "system": Sets the LLM's behavior/role and includes {context} placeholder
    #             where retrieved documents will be injected
    # - "human": The user's question, with {input} placeholder
    # These are combined into a single prompt that the LLM processes together
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}")
    ])

    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)

    # Create the RAG chain
    # - create_stuff_documents_chain: combines retrieved docs into LLM input
    # - create_retrieval_chain: connects retriever with the processing chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain, llm


def test_rag_chain(rag_chain, queries: list[str] | None = None, show_context: bool = True):
    """Test the RAG system with sample queries.

    Invoke triggers the retrieval and generation process:
    1. Retriever finds relevant documents from vector store
    2. Documents are injected into the prompt as context
    3. LLM generates an answer based on the context

    Args:
        rag_chain: The configured RAG chain
        queries: List of test queries (uses defaults if None)
        show_context: Whether to print retrieved context
    """
    if queries is None:
        queries = [
            "what causes acne on face?",
            "Can you tell me about the top tourist places in Japan?",  # Out of domain test
        ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        response = rag_chain.invoke({"input": query})
        print(f"Answer: {response['answer']}")
        if show_context:
            print(f"\nContext: {response['context'][:200]}...")


def start_interactive_chat(rag_chain):
    """Run an interactive chat session with the RAG chain.

    Continuously prompts for user input and displays RAG responses.
    Type 'quit' or 'exit' to end the session, or press Enter with empty input.
    """
    print("\n" + "=" * 50)
    print("Interactive RAG Chat")
    print("Type 'quit' or 'exit' to end, or press Enter to exit")
    print("=" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input or user_input.lower() in ('quit', 'exit'):
            print("Goodbye!")
            break

        response = rag_chain.invoke({"input": user_input})
        print(f"\nAssistant: {response['answer']}")


if __name__ == "__main__":
    # Initialize API keys
    init_api_keys()

    # Load dataset and initialize vector store
    df = load_dataset()
    vectorstore, retriever = init_vector_store(df, num_docs=10000)

    # Sanity check - embedding dimensions
    print(f"\nVector store contains {vectorstore._collection.count()} documents")
    temp_embeddings_q = np.array((vectorstore.embeddings.embed_query("What is nerve damage?")))
    print(f"Embedding dimensions: {temp_embeddings_q.shape}")
    print(f"Sample embedding: {temp_embeddings_q[:5]}...")

    # Build and test RAG chain
    rag_chain, llm = build_rag_chain(retriever)
    test_rag_chain(rag_chain)

    # Start interactive chat
    start_interactive_chat(rag_chain)
