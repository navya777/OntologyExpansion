import pandas as pd

# Step 1: Read the text file
with open('leaflevel4_economicinstability', 'r') as file:
    terms = [line.strip().strip("'") for line in file.readlines()]

# Step 2: Read the CSV file
df = pd.read_csv('postprocessed_similarity_cosine.csv')  # Replace 'your_file.csv' with your actual file path

# Step 3: Process the Terms (Optional, depending on your matching needs)
# For this example, assuming direct matching is sufficient

# Step 4: Find Matches
# Assuming the column in your CSV file you're checking against is named 'most_similar_concepts'
matches = df[df['Most Similar Concept'].isin(terms)]

# Step 5: Extract Matching Rows
print(matches)
print(f"Number of matching terms found: {len(matches)}")

# Optionally, save the extracted rows to a new CSV file
matches.to_csv('matching_rows_postprocessing.csv', index=False)
