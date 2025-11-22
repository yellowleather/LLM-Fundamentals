#!/usr/bin/env -S uv run --with-requirements requirements.txt
# -*- coding: utf-8 -*-


import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent


# init OpenAI client
load_dotenv(BASE_DIR.parent / ".env")
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env in the project root before running the script.")
OpenAI()



embeddings = OpenAIEmbeddings() # default model is text-embedding-ada-002
vector = embeddings.embed_query("hello")
print(len(vector))
print(vector[:3])


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)
vector = embeddings.embed_query("hello")
print(len(vector))
print(vector[:3])
