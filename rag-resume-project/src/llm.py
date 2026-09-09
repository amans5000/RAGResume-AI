import os

from dotenv import load_dotenv
from groq import Groq


# Load .env from multiple possible locations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv()


# Get API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. "
        "Check your .env file."
    )


# Create Groq client
client = Groq(
    api_key=api_key
)


# Groq model (can be configured via GROQ_MODEL in .env)
MODEL_NAME = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")


def generate_answer(query, context):

    system_prompt = """
You are a resume question-answering assistant.

Answer the user's question using ONLY the
provided resume context.

Rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not present in the
   context, say that the information is
   not available in the provided resume.
4. Keep the answer concise and relevant.
5. Use bullet points when appropriate.
"""


    user_prompt = f"""
RESUME CONTEXT
==============

{context}


USER QUESTION
=============

{query}


Answer the user's question using only
the resume context above.
"""


    response = client.chat.completions.create(
        model=MODEL_NAME,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        temperature=0.2,
        max_tokens=500
    )


    return response.choices[0].message.content