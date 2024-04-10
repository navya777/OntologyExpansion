import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import CountVectorizer
import Levenshtein as lev

# Load gold standard terms and CSV data
with open("neighborhoodsocialfamily_level2", "r") as file:
    gold_terms = [line.strip().lower() for line in file.readlines()]

df = pd.read_csv("extracted_rdf_triples_neighborhoodfamilysocial.csv", header=None)
parent_concepts = df.iloc[:, 3].apply(lambda x: x.lower()).tolist()

vectorizer = CountVectorizer().fit(gold_terms + parent_concepts)
gold_terms_vec = vectorizer.transform(gold_terms).toarray()
parent_concepts_vec = vectorizer.transform(parent_concepts).toarray()

# Initialize columns for distances, most matching term, and Levenshtein match
df['Min Jaccard Distance'] = np.nan
df['Min Euclidean Distance'] = np.nan
df['Most Matching Term by Jaccard'] = None
df['Most Matching Term by Euclidean'] = None
df['Most Matching Term by Levenshtein'] = None
df['Min Levenshtein Distance'] = np.nan

# Define thresholds
jaccard_threshold = 0.7  # Adjust as needed
euclidean_threshold = 1.0  # Adjust as needed

# Calculate distances and identify most matching terms
for i, concept_vec in enumerate(parent_concepts_vec):
    jaccard_dists = []
    euclidean_dists = []
    levenshtein_dists = []
    for j, term_vec in enumerate(gold_terms_vec):
        # Calculate Jaccard distance
        intersection = np.logical_and(concept_vec, term_vec).sum()
        union = np.logical_or(concept_vec, term_vec).sum()
        jaccard_dist = 1 - intersection / union if union != 0 else 1
        jaccard_dists.append(jaccard_dist)

        # Calculate Euclidean distance
        euclidean_dist = np.linalg.norm(concept_vec - term_vec)
        euclidean_dists.append(euclidean_dist)

        # Calculate Levenshtein distance
        levenshtein_dist = lev.distance(parent_concepts[i], gold_terms[j])
        levenshtein_dists.append(levenshtein_dist)

    # Apply thresholds and find minimum distances
    min_jaccard_idx, min_euclidean_idx, min_levenshtein_idx = np.argmin(jaccard_dists), np.argmin(
        euclidean_dists), np.argmin(levenshtein_dists)
    min_jaccard_dist, min_euclidean_dist, min_levenshtein_dist = jaccard_dists[min_jaccard_idx], euclidean_dists[
        min_euclidean_idx], levenshtein_dists[min_levenshtein_idx]

    if min_jaccard_dist <= jaccard_threshold:
        df.at[i, 'Min Jaccard Distance'] = min_jaccard_dist
        df.at[i, 'Most Matching Term by Jaccard'] = gold_terms[min_jaccard_idx]
    if min_euclidean_dist <= euclidean_threshold:
        df.at[i, 'Min Euclidean Distance'] = min_euclidean_dist
        df.at[i, 'Most Matching Term by Euclidean'] = gold_terms[min_euclidean_idx]
    df.at[i, 'Most Matching Term by Levenshtein'] = gold_terms[min_levenshtein_idx]
    df.at[i, 'Min Levenshtein Distance'] = min_levenshtein_dist

# Save the updated DataFrame
df.to_csv("lexicalsimilariy_neighborhoodfamilysocial_level2.csv", index=False)

# Visualization (optional and based on specific needs)
sns.heatmap(df[['Min Euclidean Distance']], annot=True)
plt.title('Euclidean Distance Heatmap')
plt.show()

# Filter for good matches based on thresholds
good_matches = df[(df['Min Jaccard Distance'] <= jaccard_threshold) |
                  (df['Min Euclidean Distance'] <= euclidean_threshold) |
                  (df['Min Levenshtein Distance'] < len(df['Most Matching Term by Levenshtein']) * 0.5)]  # Example condition for Levenshtein

# Save the filtered DataFrame
good_matches.to_csv("good_matches_socialcommunity_level3_lexical.csv", index=False)

