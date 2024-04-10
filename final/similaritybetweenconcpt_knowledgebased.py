import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import seaborn as sns
# Initialize PubMedBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

# Define function to get embedding
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].squeeze().detach().numpy()
    return embeddings

# Define function to compute cosine similarity
def compute_similarity(embedding1, embedding2):
    embedding1 = embedding1.flatten()
    embedding2 = embedding2.flatten()
    return 1 - cosine(embedding1, embedding2)

# Define function to find the most similar concept
def find_most_similar_concept(parent_concept, similar_concepts_str):
    if pd.isna(similar_concepts_str) or similar_concepts_str == "":
        return np.nan, np.nan  # Skip computation if "Similar Concepts" is NaN or empty

    similar_concepts = similar_concepts_str.split(', ')
    parent_embedding = get_embedding(parent_concept)

    max_similarity = -np.inf
    most_similar_concept = None
    similarities = []

    for concept in similar_concepts:
        concept_embedding = get_embedding(concept)
        similarity = compute_similarity(parent_embedding, concept_embedding)
        similarities.append(similarity)

        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_concept = concept

    return ', '.join([str(sim) for sim in similarities]), most_similar_concept

# Load the DataFrame
df = pd.read_csv('modified_file_neighborhoodfamilysocial.csv')

# Apply function only to rows with non-empty "Similar Concepts"
results = df.apply(lambda row: find_most_similar_concept(row['Parent Concept'], row['Similar Concepts']) if pd.notnull(row['Similar Concepts']) else (np.nan, np.nan), axis=1, result_type="expand")
#results = df.apply(lambda row: find_most_similar_concept(row['Parent Concept'], row['Similar Concepts']) if pd.notnull(row['Similar Concepts']) else (np.nan, np.nan), axis=1, result_type="expand")

# Assign results to new columns
df[['Similarities', 'Most Similar Concept']] = results

# Convert the 'Similarities' column to a list of floats, flattening all the lists
all_similarities = [float(sim) for sublist in df['Similarities'].dropna().str.split(', ').tolist() for sim in sublist]
all_similarities = [sim for sim in all_similarities if 0 <= sim <= 1]
# Generate the density plot using Seaborn
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.kdeplot(all_similarities, shade=True)
plt.title('Density Plot of Knoweldge based similarity ',fontsize=14, fontweight='bold')
plt.xlabel('Cosine Similarity Score',fontsize=12)
plt.xlim(0, 1)
plt.ylabel('Kernel Density',fontsize=14)
plt.show()
# Assign results to new columns
df[['Similarities', 'Most Similar Concept']] = results

# Save the updated DataFrame
df.to_csv('modified_file_with_similarity_cosine_neighborhoodfamilysocial.csv', index=False)
