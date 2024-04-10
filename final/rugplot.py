import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('modified_file_with_similarity_cosine.csv')
# Assuming `df` is your DataFrame
# Example conversion if 'Similarities' is a string that represents lists of floats
df['Similarities'] = df['Similarities'].apply(lambda x: np.fromstring(x, sep=',') if pd.notnull(x) else np.nan)

# Extract the maximum similarity score for each row
df['Max Similarity Score'] = df['Similarities'].apply(lambda x: np.max(x) if isinstance(x, np.ndarray) else np.nan)

# Ensure there's no row with NaN in 'Max Similarity Score' for t-SNE
df_filtered = df.dropna(subset=['Max Similarity Score'])
# Assuming df_filtered['Max Similarity Score'] contains your data
scores = df_filtered['Max Similarity Score'].dropna()

plt.figure(figsize=(10, 6))
sns.kdeplot(scores, shade=True, color="r", alpha=0.5)
sns.rugplot(scores, color="g")

plt.title('Density and Rug Plot of Maximum Similarity Scores')
plt.xlabel('Similarity Score')
plt.ylabel('Density')
plt.show()