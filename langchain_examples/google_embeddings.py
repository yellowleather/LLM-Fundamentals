#!/usr/bin/env -S uv run --with-requirements requirements.txt
# -*- coding: utf-8 -*-


import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings

BASE_DIR = Path(__file__).resolve().parent


# init Google client
load_dotenv(BASE_DIR.parent / ".env")
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise RuntimeError("GOOGLE_API_KEY is not set. Add it to .env in the project root before running the script.")


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector = embeddings.embed_query("hello")
print(len(vector))
print(vector[:3])


embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vector = embeddings.embed_query("hello")
print(len(vector))
print(vector[:3])
