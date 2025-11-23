#!/usr/bin/env -S uv run --with-requirements requirements.txt
# -*- coding: utf-8 -*-
"""Simple LLM Powered Chatbot using Google's Gemini."""

import os
from pathlib import Path

from dotenv import load_dotenv
import google.generativeai as genai

# Load API keys from .env
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not set. Add it to .env before running the script.")

# Configure the Generative AI client
genai.configure(api_key=GOOGLE_API_KEY)

# Load the Gemini model for chat
model = genai.GenerativeModel("gemini-2.0-flash")

# Start a chat session (enables memory across turns)
chat = model.start_chat()


def chat_with_gemini(user_input: str) -> str:
    """Send a message to Gemini and return the response."""
    response = chat.send_message(user_input)
    return response.text


def main():
    """Run the interactive chatbot loop."""
    print("Gemini Chatbot Ready! Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Bot: Goodbye!")
            break
        reply = chat_with_gemini(user_input)
        print("Bot:", reply)


if __name__ == "__main__":
    main()
