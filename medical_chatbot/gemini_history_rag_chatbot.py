#!/usr/bin/env -S uv run --with-requirements requirements.txt
# -*- coding: utf-8 -*-
"""History-Aware RAG Chain with Gemini and ChromaDB.

A Retrieval-Augmented Generation (RAG) system that maintains conversation context:
- Loads a medical Q&A dataset
- Stores embeddings in ChromaDB
- Rephrases follow-up questions using chat history before retrieval
- Handles context-dependent questions like "What is the treatment for it?"

Dependencies: langchain, langchain-google-genai, chromadb
"""

import os
from pathlib import Path

import gdown
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
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


def build_history_aware_rag_chain(retriever, model: str = "gemini-2.0-flash", temperature: float = 0.2):
    """Build a history-aware RAG chain that considers conversation context.

    Enhances the RAG chain to consider conversation history, allowing the assistant
    to understand follow-up questions that refer to previous turns.

    How it works:
    - Contextualization Prompt: A system prompt specifically for rephrasing a user's
      follow-up question into a standalone question, using chat history for context.
      MessagesPlaceholder("chat_history") holds the conversation history.

    - History-Aware Retriever: Uses create_history_aware_retriever to create a retriever
      that first uses the LLM and contextualization prompt to potentially rephrase the
      user's input based on history, *before* querying the vector store.

    Behavior:
    - If there is no chat_history, then the input is passed directly to the retriever.
    - If there is chat_history, then the prompt and LLM will be used to generate a
      search query. That search query is then passed to the retriever.

    This chain prepends a rephrasing of the input query to our retriever,
    so that the retrieval incorporates the context of the conversation.

    Args:
        retriever: The base retriever from vector store
        model: Gemini model name
        temperature: LLM temperature (0.0 = deterministic, 1.0 = creative)

    Returns:
        A history-aware RAG chain that can handle follow-up questions
    """
    llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)

    # Step 1: Create the contextualization prompt for rephrasing follow-up questions
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # Step 2: Create history-aware retriever that rephrases queries before retrieval
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Step 3: Build the QA chain with history-aware prompt
    history_aware_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    question_answer_chain = create_stuff_documents_chain(llm, history_aware_prompt)

    # Step 4: Combine into full RAG chain
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


def test_history_aware_rag_chain(rag_chain):
    """Test the history-aware RAG chain with a follow-up question.

    Demonstrates how the chain handles context-dependent questions:
    1. Ask an initial question about optical nerve damage
    2. Store the Q&A in chat_history using HumanMessage/AIMessage format
    3. Ask a follow-up question ("What is the treatment for it?")
    4. The chain rephrases "it" to "optical nerve damage" before retrieval
    """
    print("\n--- Testing History-Aware RAG Chain ---")
    chat_history = []

    # First question
    question1 = "what is optical nerve damage"
    response1 = rag_chain.invoke({"input": question1, "chat_history": chat_history})
    print(f"Q1: {question1}")
    print(f"A1: {response1['answer']}\n")

    # Update chat history with HumanMessage/AIMessage (required format for MessagesPlaceholder)
    chat_history.extend([
        HumanMessage(content=question1),
        AIMessage(content=response1["answer"])
    ])

    # Follow-up question - "it" refers to optical nerve damage from context
    question2 = "What is the treatment for it?"
    response2 = rag_chain.invoke({"input": question2, "chat_history": chat_history})
    print(f"Q2: {question2}")
    print(f"A2: {response2['answer']}\n")


def start_interactive_chat(rag_chain):
    """Run an interactive chat session with the history-aware RAG chain.

    Continuously prompts for user input and displays RAG responses.
    Maintains conversation history for context-aware follow-up questions.
    Type 'quit' or 'exit' to end the session, or press Enter with empty input.
    """
    print("\n" + "=" * 50)
    print("Interactive History-Aware RAG Chat")
    print("Type 'quit' or 'exit' to end, or press Enter to exit")
    print("=" * 50)

    chat_history = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input or user_input.lower() in ('quit', 'exit'):
            print("Goodbye!")
            break

        response = rag_chain.invoke({"input": user_input, "chat_history": chat_history})
        print(f"\nAssistant: {response['answer']}")

        # Update chat history for context-aware follow-up questions
        chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=response["answer"])
        ])


if __name__ == "__main__":
    # Initialize API keys
    init_api_keys()

    # Load dataset and initialize vector store
    df = load_dataset()
    vectorstore, retriever = init_vector_store(df)

    # Sanity check
    print(f"\nVector store contains {vectorstore._collection.count()} documents")
    temp_embeddings_q = np.array((vectorstore.embeddings.embed_query("What is nerve damage?")))
    print(f"Embedding dimensions: {temp_embeddings_q.shape}")

    # Build and test history-aware RAG chain
    history_aware_rag_chain = build_history_aware_rag_chain(retriever)
    test_history_aware_rag_chain(history_aware_rag_chain)

    # Start interactive chat
    start_interactive_chat(history_aware_rag_chain)
