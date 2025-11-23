#!/usr/bin/env -S uv run --with-requirements requirements.txt
# -*- coding: utf-8 -*-
"""Medical RAG-Based AI Agent with Web Search.

RAG-based Medical Assistant that:
- Answers questions using a medical dataset stored in ChromaDB
- Optionally fetches real-time info via Tavily web search
- Maintains conversation context with LangChain

Dependencies: langchain, langchain-google-genai, chromadb, tavily
"""

import os
from pathlib import Path

import gdown
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain.chains import ConversationalRetrievalChain, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.tools import Tool
from langchain.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
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
AGENT_TEMPLATE = """
You are a helpful medical assistant. Get an answer based only on the internal (RAG) and web (Tavily) context.
Do not answer it on your own.

If the question is non-medical, respond:
Final Answer: I'm not allowed to discuss it.

Tools:
{tools}

Use the following format:

Question: {input}

Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action


Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""


def init_api_keys() -> tuple[str, str]:
    """Load API keys from .env and return (GOOGLE_API_KEY, TAVILY_API_KEY)."""
    load_dotenv(BASE_DIR.parent / ".env")

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to .env before running the script.")

    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise RuntimeError("TAVILY_API_KEY is not set. Add it to .env before running the script.")

    return google_api_key, tavily_api_key


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

    Args:
        retriever: The document retriever from vector store
        model: Gemini model name
        temperature: LLM temperature (0.0 = deterministic, 1.0 = creative)

    Returns:
        Tuple of (rag_chain, llm)
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}")
    ])

    llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain, llm


def show_chat_history(memory):
    """Display chat history from memory."""
    for message in memory.chat_memory.messages:
        role = "You" if message.type == "human" else "Bot"
        print(f"{role}: {message.content}")


def handle_user_message(agent, user_input):
    """Handle user message with the conversational agent."""
    response = agent.run(user_input)
    return response


class ChatManager:
    """Hybrid RAG + Web Search Synthesis Manager.

    Orchestrates conversation by combining internal knowledge (RAG) with
    external web search results (Tavily) and synthesizing them into a single response.
    """

    def __init__(self, llm, tavily_tool, retriever, rag_chain):
        self.llm = llm
        self.tavily_tool = tavily_tool
        self.retriever = retriever
        self.rag_chain = rag_chain

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )

        self.synthesis_prompt_template = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful medical assistant. Synthesize an answer based only on the internal (RAG) and web (Tavily) context. Do not answer it on your own.\n"
             "Also just also do this:\n"
             "Rag: Yes or No\n"
             "Tavily: Yes or No\n"
             "If non-medical, say you're not allowed to discuss it."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

    def _get_rag_context(self, user_input, chat_history):
        try:
            rag_response = self.rag_chain.invoke({"input": user_input, "chat_history": chat_history})
            return rag_response.get("answer", "Sorry, no RAG answer found.")
        except Exception as e:
            return f"Error retrieving RAG context: {e}"

    def _get_tavily_context(self, user_input):
        try:
            search_results = self.tavily_tool.invoke({"query": user_input})
            contents = [res['content'] for res in search_results if 'content' in res]
            return "\n\n".join(contents) if contents else "No relevant web search results."
        except Exception as e:
            return f"Error retrieving Tavily context: {e}"

    def handle_user_message(self, user_input):
        chat_history = self.memory.chat_memory.messages
        rag_context = self._get_rag_context(user_input, chat_history)
        tavily_context = self._get_tavily_context(user_input)

        input_for_synthesis = {
            "input": user_input,
            "chat_history": chat_history,
            "rag_context": rag_context,
            "tavily_context": tavily_context
        }

        synthesis_chain = self.synthesis_prompt_template | self.llm
        response = synthesis_chain.invoke(input_for_synthesis)
        final_answer = getattr(response, 'content', response)

        self.memory.save_context({"input": user_input}, {"answer": final_answer})

        return final_answer.strip()

    def clear_history(self):
        self.memory.clear()

    def get_history(self):
        return self.memory.chat_memory.messages


def create_rag_tool(rag_chain):
    """Create a RAG context retriever tool for the agent."""
    def rag_context_func(user_input):
        try:
            rag_response = rag_chain.invoke({"input": user_input})
            return rag_response.get("answer", "Sorry, no RAG answer found.")
        except Exception as e:
            return f"Error retrieving RAG context: {e}"

    return Tool(
        name="RAG Context Retriever",
        func=rag_context_func,
        description="Retrieves relevant RAG context given user input and chat history."
    )


def create_medical_agent(llm, tools, memory):
    """Create a medical AI agent with RAG and Tavily tools."""
    prompt = PromptTemplate(
        input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
        template=AGENT_TEMPLATE
    )

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        output_key="output",
        handle_parsing_errors=True
    )

    return agent_executor


def test_conversational_retrieval_chain(llm, retriever):
    """Test ConversationalRetrievalChain with follow-up questions."""
    print("\n--- Testing ConversationalRetrievalChain ---")

    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    chat_history = []

    # First question
    question = "what is optical nerve damage"
    ai_msg_1 = rag_chain.invoke({"question": question, "chat_history": chat_history})
    chat_history.append((question, ai_msg_1["answer"]))
    print(f"Q1: {question}")
    print(f"A1: {ai_msg_1['answer']}\n")

    # Second question
    second_question = "What is the treatment for it?"
    ai_msg_2 = rag_chain.invoke({"question": second_question, "chat_history": chat_history})
    print(f"Q2: {second_question}")
    print(f"A2: {ai_msg_2['answer']}\n")


def test_tavily_agent(llm, tavily_tool):
    """Test conversational agent with Tavily web search."""
    print("\n--- Testing Tavily Conversational Agent ---")

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = initialize_agent(
        tools=[tavily_tool],
        llm=llm,
        memory=memory,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION
    )

    # Test queries
    queries = [
        "what are some events in new york for this weekend?",
        "What is bird flu?",
    ]

    for query in queries:
        print(f"Q: {query}")
        response = agent.run(query)
        print(f"A: {response}\n")

    memory.clear()


def test_chat_manager(llm, tavily_tool, retriever, rag_chain):
    """Test the ChatManager with hybrid RAG + web search."""
    print("\n--- Testing ChatManager (Hybrid RAG + Web Search) ---")

    chat = ChatManager(llm, tavily_tool, retriever, rag_chain)

    queries = [
        "what is optical nerve damage",
        "what are some treatments for diabetes",
    ]

    for query in queries:
        print(f"Q: {query}")
        response = chat.handle_user_message(query)
        print(f"A: {response}\n")

    chat.clear_history()


def test_medical_agent(agent_executor):
    """Test the medical AI agent with various queries."""
    print("\n--- Testing Medical AI Agent ---")

    queries = [
        "what is optical nerve damage treatment",
        "how to reduce weight",
        "what is osteoporosis",
    ]

    for query in queries:
        print(f"\nQ: {query}")
        response = agent_executor.invoke({"input": query}, handle_parsing_errors=True)
        print(f"A: {response['output']}")


if __name__ == "__main__":
    # Initialize API keys
    init_api_keys()

    # Load dataset and initialize vector store
    df = load_dataset()
    vectorstore, retriever = init_vector_store(df)

    # Sanity check - embedding dimensions
    print(f"\nVector store contains {vectorstore._collection.count()} documents")
    temp_embeddings_q = np.array((vectorstore.embeddings.embed_query("What is nerve damage?")))
    print(f"Embedding dimensions: {temp_embeddings_q.shape}")
    print(f"Sample embedding: {temp_embeddings_q[:5]}...")

    # Build RAG chain
    rag_chain, llm = build_rag_chain(retriever)

    # Initialize Tavily tool
    tavily_tool = TavilySearchResults()

    # Test ConversationalRetrievalChain
    test_conversational_retrieval_chain(llm, retriever)

    # Test Tavily agent
    test_tavily_agent(llm, tavily_tool)

    # Test ChatManager
    test_chat_manager(llm, tavily_tool, retriever, rag_chain)

    # Create and test medical AI agent
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    rag_tool = create_rag_tool(rag_chain)
    tools = [tavily_tool, rag_tool]
    agent_executor = create_medical_agent(llm, tools, memory)
    test_medical_agent(agent_executor)
