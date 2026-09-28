import os
from unittest import result

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from ingest import response

# ================= load env properties from .env
load_dotenv()

# ================ get the actual api key from .env file
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# =========== create openai client
client = OpenAI(api_key=openai_api_key)

# =========== create pinecone client
pc = Pinecone(api_key=pinecone_api_key)

# ===== pinecone index configuration
index_name = "company-policy-docs"
dimensions = 1536
index = pc.Index(index_name)


def ask_question(question):
    # create embedding for the question
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=question,
    )
    question_embedding = response.data[0].embedding

    # with question embedding search pinecone
    results = index.query(
        vector=question_embedding,
        top_k=3,
        include_metadata=True
    )

    print("Results: ", results)
    documents = []
    for match in results["matches"]:
        document = match["metadata"]["text"]
        documents.append(document)

    print("Documents :", documents)

    # combine list objects into context
    context = "\n\n".join(documents)
    print(context)

    # create RAG prompt (Augmentation)
    prompt = f"""

    Answer the only question using the context below:

    Context: {context}

    Question: {question}

    """

    response = client.responses.create(
        model="gpt-6-astra",
        input=prompt,
    )

    return response.output_text