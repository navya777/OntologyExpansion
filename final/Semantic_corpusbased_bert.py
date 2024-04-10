from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def get_bert_embedding(text):
    """Returns BERT embedding for the given text."""
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()


# Load your CSV file
df = pd.read_csv("extracted_rdf_triples_economicstability.csv", header=None)
df.columns = ['PMID', 'Child Concept', 'Relationship', 'Parent Concept']

# Assuming 'gold_terms' contains the gold standard terms loaded from a text file

with open("leaflevel4_economicinstability", "r") as file:
    gold_terms = [line.strip().lower() for line in file.readlines()]
# Calculate BERT embeddings for gold terms and parent concepts
gold_terms_embeddings = np.vstack([get_bert_embedding(term) for term in gold_terms])
parent_concepts_embeddings = np.vstack([get_bert_embedding(concept) for concept in df['Parent Concept']])

# New columns for the most similar term based on BERT embeddings
df['Most Similar Term (BERT)'] = ''
df['Cosine Similarity (BERT)'] = 0.0


similarity_threshold = 0.5  # Adjust this value as needed

for i, concept_embedding in enumerate(parent_concepts_embeddings):
    # Calculate cosine similarity
    similarities = cosine_similarity([concept_embedding], gold_terms_embeddings)[0]

    # Find the index of the highest similarity score
    max_index = similarities.argmax()
    max_similarity = similarities[max_index]

    # Update DataFrame only if the similarity score is above the threshold
    if max_similarity >= similarity_threshold:
        df.at[i, 'Most Similar Term (BERT)'] = gold_terms[max_index]
        df.at[i, 'Cosine Similarity (BERT)'] = max_similarity
    else:
        # If no term meets the threshold, you might choose to leave the fields empty or fill them with a placeholder
        df.at[i, 'Most Similar Term (BERT)'] = None  # or "No match above threshold"
        df.at[i, 'Cosine Similarity (BERT)'] = None  # or 0, or any indicator of your choice


# Save updated DataFrame to CSV
df.to_csv("economicstability_level4_with_bert_similarity.csv", index=False)
# Ensure only non-null similarity scores are used for the plot
valid_similarities = df['Cosine Similarity (BERT)'].dropna()

# Set the style for the plot
sns.set(style="whitegrid")

# Create the density plot
plt.figure(figsize=(10, 6))
sns.kdeplot(valid_similarities, shade=True)
plt.title('Kernel density estimate Semantic Similarity (PubMeD BERT+Cosine)',fontsize=14, fontweight='bold')
plt.xlabel('Cosine Similarity Score',fontsize=12)
plt.ylabel('Kernel Density',fontsize=12)

# Save the plot
plt.savefig('cosine_similarity_density_plot_economicstability_level4.png', dpi=300)

# Show the plot
plt.show()
