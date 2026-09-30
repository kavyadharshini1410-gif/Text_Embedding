from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Read sentences
with open("sentences.txt", "r", encoding="utf-8") as file:
    texts = [line.strip() for line in file if line.strip()]

print("Number of Texts:", len(texts))

# Stop if no text is found
if len(texts) == 0:
    print("ERROR: sentences.txt is empty or no sentences were read.")
    exit()

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
embeddings = model.encode(texts)

print("Embeddings shape:", embeddings.shape)

# Save embeddings
np.save("embeddings.npy", embeddings)

print("Embeddings saved successfully!")

# Calculate cosine similarity
similarity_matrix = cosine_similarity(embeddings)

print("\nCosine Similarity Matrix:")
print(similarity_matrix)

# Save similarity matrix
np.save("similarity_matrix.npy", similarity_matrix)

print("\nSimilarity matrix saved successfully!")