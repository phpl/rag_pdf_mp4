#!/bin/zsh
python3 rag_chatbot.py \
  --pdfs \
    "rag/RAG Intro.pdf" \
    "rag/Productized & Enterprise RAG.pdf" \
    "rag/Databases for GenAI.pdf" \
    "rag/Architecture & Design Patterns.pdf" \
  --videos \
    "rag/1 part. RAG Intro.mp4" \
    "rag/1st Part_Productized Enterprise RAG.mp4" \
    "rag/2 part Databases for GenAI.mp4" \
    "rag/2nd Part_Architecture & Design Patterns.mp4"
