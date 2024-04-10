import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import CountVectorizer
import Levenshtein as lev
import os
# Load gold standard terms and CSV data
with open("leaflevel4_economicinstability", "r") as file:
    gold_terms = [line.strip().lower() for line in file.readlines()]

df = pd.read_csv("extracted_rdf_triples_neighborhoodfamilysocial.csv", header=None)
parent_concepts = df.iloc[:, 3].apply(lambda x: x.lower()).tolist()
def longest_common_substring_length(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    length_max = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                length_max = max(length_max, dp[i][j])
            else:
                dp[i][j] = 0
    return length_max

vectorizer = CountVectorizer().fit(gold_terms + parent_concepts)
gold_terms_vec = vectorizer.transform(gold_terms).toarray()
parent_concepts_vec = vectorizer.transform(parent_concepts).toarray()
# Initialize columns for distances, most matching term, LCS Length, and Levenshtein match
df['Min Jaccard Distance'] = np.nan
df['Most Matching Term by Jaccard'] = None
df['Min LCS Length'] = np.nan  # New column for LCS Length
df['Most Matching Term by Levenshtein'] = None
df['Min Levenshtein Distance'] = np.nan
jaccard_threshold = 0.7  # Adjust as needed
# Adjusted loop to include LCS Length calculation
for i, concept_vec in enumerate(parent_concepts_vec):
    jaccard_dists = []
    lcs_lengths = []  # New list for LCS lengths
    levenshtein_dists = []
    for j, term_vec in enumerate(gold_terms_vec):
        # Jaccard distance calculation remains the same
        intersection = np.logical_and(concept_vec, term_vec).sum()
        union = np.logical_or(concept_vec, term_vec).sum()
        jaccard_dist = 1 - intersection / union if union != 0 else 1
        jaccard_dists.append(jaccard_dist)

        # Calculate LCS length
        lcs_length = longest_common_substring_length(parent_concepts[i], gold_terms[j])
        lcs_lengths.append(lcs_length)

        # Levenshtein distance calculation remains the same
        levenshtein_dist = lev.distance(parent_concepts[i], gold_terms[j])
        levenshtein_dists.append(levenshtein_dist)

    # Find the index of the gold term with the maximum LCS length for each parent concept
    max_lcs_idx = np.argmax(lcs_lengths)
    max_lcs_length = lcs_lengths[max_lcs_idx]

    # Apply Jaccard threshold and update DataFrame
    if min(jaccard_dists) <= jaccard_threshold:
        min_jaccard_idx = np.argmin(jaccard_dists)
        df.at[i, 'Min Jaccard Distance'] = jaccard_dists[min_jaccard_idx]
        df.at[i, 'Most Matching Term by Jaccard'] = gold_terms[min_jaccard_idx]

    # Update DataFrame with LCS information
    df.at[i, 'Min LCS Length'] = max_lcs_length
    df.at[i, 'Most Matching Term by Levenshtein'] = gold_terms[np.argmin(levenshtein_dists)]
    df.at[i, 'Min Levenshtein Distance'] = min(levenshtein_dists)


for i, concept_vec in enumerate(parent_concepts_vec):
    jaccard_dists = []
    lcs_lengths = []  # New list for LCS lengths
    levenshtein_dists = []
    for j, term_vec in enumerate(gold_terms_vec):
        # Jaccard distance calculation remains the same
        intersection = np.logical_and(concept_vec, term_vec).sum()
        union = np.logical_or(concept_vec, term_vec).sum()
        jaccard_dist = 1 - intersection / union if union != 0 else 1
        jaccard_dists.append(jaccard_dist)

        # Calculate LCS length
        lcs_length = longest_common_substring_length(parent_concepts[i], gold_terms[j])
        lcs_lengths.append(lcs_length)

        # Levenshtein distance calculation remains the same
        levenshtein_dist = lev.distance(parent_concepts[i], gold_terms[j])
        levenshtein_dists.append(levenshtein_dist)

    # Find the index of the gold term with the maximum LCS length for each parent concept
    max_lcs_idx = np.argmax(lcs_lengths)
    max_lcs_length = lcs_lengths[max_lcs_idx]

    # Apply Jaccard threshold and update DataFrame
    if min(jaccard_dists) <= jaccard_threshold:
        min_jaccard_idx = np.argmin(jaccard_dists)
        df.at[i, 'Min Jaccard Distance'] = jaccard_dists[min_jaccard_idx]
        df.at[i, 'Most Matching Term by Jaccard'] = gold_terms[min_jaccard_idx]

    # Update DataFrame with LCS information
    df.at[i, 'Min LCS Length'] = max_lcs_length
    df.at[i, 'Most Matching Term by Levenshtein'] = gold_terms[np.argmin(levenshtein_dists)]
    df.at[i, 'Min Levenshtein Distance'] = min(levenshtein_dists)
# New columns for composite score and most similar parent concept
df['Best Composite Score'] = np.nan
df['Most Similar Parent Concept'] = None
# Function to calculate composite score (simplified version)
def calculate_composite_score(jaccard_sim, levenshtein_dist, lcs_length, max_len):
    # Inverse Levenshtein distance: higher is better, normalized by max length of compared strings
    inv_levenshtein_score = 1 - levenshtein_dist / max_len
    # Directly use Jaccard similarity and LCS length, assuming they've been normalized or are on a comparable scale
    composite_score = (jaccard_sim + inv_levenshtein_score + lcs_length / max_len) / 3
    return composite_score

for i, concept_vec in enumerate(parent_concepts_vec):
    composite_scores = []

    for j, term_vec in enumerate(gold_terms_vec):
        # Continue with Jaccard, LCS, and Levenshtein calculations

        # Calculate the composite score for each parent concept
        max_len = max(len(parent_concepts[i]), len(gold_terms[j]))
        composite_score = calculate_composite_score(
            1 - jaccard_dists[j],  # Jaccard similarity as 1 - distance
            levenshtein_dists[j],
            lcs_lengths[j],
            max_len
        )
        composite_scores.append(composite_score)

    # Identify the parent concept with the best (highest) composite score
    best_idx = np.argmax(composite_scores)
    df.at[i, 'Best Composite Score'] = composite_scores[best_idx]
    df.at[i, 'Most Similar Parent Concept'] = gold_terms[best_idx]



# Proceed to save the DataFrame and any desired visualizations
df.to_csv("updated_with_distances_and_lcs_economicstability_similarityterm_lexicallevel4.csv", index=False)

# Set the style of the visualization
sns.set(style="whitegrid")
figures_directory = "figures"
if not os.path.exists(figures_directory):
    os.makedirs(figures_directory)
# Plot KDE of Min Jaccard Distance
plt.figure(figsize=(10, 6))
sns.kdeplot(df['Min Jaccard Distance'], shade=True, label='Min Jaccard Distance')
plt.title('Kernel Density Estimate of Jaccard Distance',fontsize=14, fontweight='bold')
plt.xlabel('Jaccard Distance',fontsize=12)
plt.ylabel('Kernel Density',fontsize=12)
plt.legend()
plt.show()
plt.savefig(os.path.join(figures_directory, 'Min_Jaccard_Distance_KDE.png'), dpi=300)
plt.close()
# Plot KDE of Min LCS Length
plt.figure(figsize=(10, 6))
sns.kdeplot(df['Min LCS Length'], shade=True, label='Min LCS Length')
plt.title('Kernel Density Estimate of LCS Similarity',fontsize=14, fontweight='bold')
plt.xlabel('LCS Length',fontsize=12)
plt.ylabel('Kernel Density',fontsize=12)
plt.legend()
plt.show()

# Plot KDE of Min Levenshtein Distance
plt.figure(figsize=(10, 6))
plt.xlim(0, 10)  # This sets the x-axis to range from 0 to 20
sns.kdeplot(df['Min Levenshtein Distance'], shade=True, label='Min Levenshtein Distance')
plt.title('Kernel Density Estimate of Levenshtein Distance',fontsize=14, fontweight='bold')
plt.xlabel('Levenshtein Distance',fontsize=12)
plt.ylabel('Kernel Density',fontsize=12)
plt.legend()
plt.show()


# Assuming Best Composite Score has been calculated and added to the DataFrame
plt.figure(figsize=(10, 6))
sns.kdeplot(df['Best Composite Score'], shade=True, label='Best Composite Score')
plt.title('Kernel Density Estimate of Best Composite Score')
plt.xlabel('Best Composite Score')
plt.ylabel('Density')
plt.legend()
plt.show()

# Set the style of the visualization
sns.set(style="whitegrid")

# KDE Plot of Longest LCS Lengths
plt.figure(figsize=(10, 6))
plt.xlim(0, 20)  # This sets the x-axis to range from 0 to 20
sns.kdeplot(df['Min LCS Length'], shade=True, label='Longest LCS Length')
plt.title('Kernel Density Estimate of Longest LCS Length')
plt.xlabel('Longest LCS Length')
plt.ylabel('Density')
plt.legend()
plt.show()

# Optionally, save the plot
plt.savefig(os.path.join(figures_directory, 'Longest_LCS_Length_KDE.png'), dpi=300)
plt.close()  # Close the plot to free resources