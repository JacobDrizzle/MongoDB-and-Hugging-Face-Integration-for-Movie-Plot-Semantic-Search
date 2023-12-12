import pymongo
from dotenv import load_dotenv
import os
import requests

load_dotenv()

MONGOKEY = os.getenv('MONGODB')
client = pymongo.MongoClient(MONGOKEY)

db = client.sample_mflix
collection = db.movies

hf_token = os.getenv('HFTOKEN')
embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

# FUNCTION TO GENERATE EMBEDDINGS

def generate_embedding(text: str) -> list[float]:
    
   response = requests.post(
       embedding_url,
       headers={"Authorization": f"Bearer {hf_token}"},
       json={"inputs": text})
    
   if response.status_code != 200:
       raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")
   return response.json()

# END OF FUNCTION TO GENERATE EMBEDDINGS

# Function to create embedings for 50 documents in the movieDB with the field 'plot'

#for doc in collection.find({'plot':{"$exists":True}}).limit(50):
    
    # Generate an embedding for each plot and add to a new field in the db
#    doc['plot_embedding_hf'] = generate_embedding(doc['plot'])
    
    # Update the document in the db with new information
#   collection.replace_one({'_id':doc['_id']}, doc)

# End of Function to Create vector embedings for 50 documents in the movieDB with the field 'plot'

query = "imiganiary characters from outer space at war"

results = collection.aggregate([
    {"$vectorSearch": {
        # Generate an embedding for the query and search for similary embeddings
        "queryVector": generate_embedding(query),
        # Field to check the query embedding against
        "path": "plot_embedding_hf",
        # Amount of candidates to check against in mongoDB Higher = Better but more compute heavy
        "numCandidates": 100,
        # Number of results returned 
        "limit": 4,
        # Specifies the name of the index to search i.e the one we created 
        "index": "PlotSemanticSearch",
    }}
])

for document in results:
    print(f'Movie Name: { document["title"]}, \nMovie Plot: {document["plot"]}\n')