from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage,HumanMessage
from pydantic import BaseModel
from collections import defaultdict
import hashlib

from dotenv import load_dotenv

load_dotenv()


#Initialize the embeddding model
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

persist_directory='db/chroma_basic_db'


# load the vectorstore
db = Chroma(embedding_function=embedding, persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"})


# define a pydantic model
class QueryFormat(BaseModel):
    query: list[str]


#search for the relevant documents
query = "In what year does Tesla roadster production started?"
print(f"Actual user query: {query}")

# Step-1 - Generate 3 different similary question using llm inorder to pull the relevant chunks
prompt_text = f"""Generate 3 different meaningful 3 varaitions of user questio{query}"""

# invoke llm
llm = ChatOpenAI(model= "gpt-5-nano") 
llm_structured = llm.with_structured_output(QueryFormat)
variations = llm_structured.invoke(prompt_text)
variations_query = variations.query
print("Generated query variations: ")
for i, variation in enumerate(variations_query, 1):
    print(f"{i}.{variation}")

# Step 2 - Retrieve the chunks using each variations and store it
retriever = db.as_retriever(search_kwargs={"k": 5})

all_retrieved_results = []

for i, variation in enumerate(variations_query, 1):
    print(f"Retrieval results of variation query {i}")
    vector_store = retriever.invoke(variation)
    all_retrieved_results.append(vector_store)

    print(f"Retrieved {len(vector_store)} documents")
    for j , doc in enumerate(vector_store, 1):
        print(f"Document {j}:")
        print(doc.page_content[:100])
            
    print("-" * 50)
print("\n" + "=" * 60)
print("Multi-Query Retrieval Completed")

# Step 3 - Apply Reciprocal Rank Funsion
def reciprocal_rank_fusion(chunks_list, k=60, verbose=True):
    """Implement Reciprocal rank fusin for multiple queries"""

    #initialize empty variabes
    scores = {}
    chunk_map = {}

    for query_idx, chunks in enumerate(chunks_list, 1):
        print(f"Processing Query {query_idx} results")

        for position, chunk in enumerate(chunks, 1):
            #use hash for content as a unique identifier
            chunk_hash = hash(chunk.page_content)
            
            # calculate the score for this position
            score = 1/(k+position)
            
            # identfying the duplicate chunk content
            if chunk_hash not in chunk_map:
                chunk_map[chunk_hash] = chunk
            
            # if duplicate chunk add the old score and new score
            if chunk_hash in scores:
                scores[chunk_hash] += score
            else:
                scores[chunk_hash] = score
    
    results = []
    for chunk_hash, score in scores.items():
        chunk = chunk_map[chunk_hash]
        results.append((chunk, score)) # append as a tuple
    # sample result o/p
    # Document(id='11037e80-186e-4769-8177-a246505e86fc', metadata={'source': 'docs/Tesla.txt'}, page_content="2008.[21] In August 2007, Michael Markssla began production of the ark.[25][26] By January 2009, Tesla had raised\n$187\xa0million and delivered 147 cars. Musk had contributed
    #  of his money to\nthe company.[27]\n\nTesla Motors insignia as \u20092010"), 0.04918032786885246

    #sort the score by descending order
    #syntax of sort list.sort(key=None, reverse=False)
    results.sort(key=lambda x:x[1], reverse=True) # x[0] is content and x[1] is score in tuple and we need to sort score
    
    print(f"✅ RRF Complete! Processed {len(results)} unique chunks from {len(chunks_list)} queries.")

    return results
    

if __name__ == "__main__":
   fused_results = reciprocal_rank_fusion(chunks_list=all_retrieved_results)

   # Step 4  - Display the fused results
   print("\n" + "="*60)
   print("FINAL RRF RANKING")
   print("="*60)
   
   print(f"\nTop {len(fused_results)} documents after RRF fusion:\n")

   for rank, (content,score) in enumerate(fused_results, 1):
    print(f"🏆 RANK {rank} (RRF Score: {score:.4f})")
    print(f"{content.page_content[:400]}...")
    print("-" * 50)
