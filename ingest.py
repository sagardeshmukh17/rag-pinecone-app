import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

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

# ================================================
# Create Index if it doesn't exist
# ================================================

if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        dimension = dimensions,
        metric = "cosine",
        spec = ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# =========== connect to index
index  = pc.Index(index_name)

# ====== read file data
with open("documents/company.txt", "r", encoding="utf-8") as file:
    document = file.read()

    # === split document into multiple chunks
    chunks = document.split("\n\n")

    # === generate embeddings and store them
    vectors = []

    for index_number, chunk in enumerate(chunks):
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )

        print("Response :: ", response)

        embedding = response.data[0].embedding

        ## Create Pinecone vector

        vectors.append({
            "id": f"chunk-{index_number}",
            "values": embedding,
            "metadata": {
                "text": chunk
            }
        })

        # upload vectors to pinecone
        index.upsert(vectors=vectors)

        print(f"Chunk {index} stored successfully")