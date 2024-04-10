import pandas as pd

# Path to your input .csv file
input_csv_file = 'modified_file_with_similarity_cosine.csv'

# Read the data into a pandas DataFrame
df = pd.read_csv(input_csv_file)

# Filter rows where either Child Concept, Parent Concept, or Most Similar Concept contains 'Economic stability'
filtered_df = df[df.apply(lambda x: 'Education' in x.tolist(), axis=1)]

# Path for the new .csv file to save filtered data
output_csv_file = 'knoweldge_based_education.csv'

# Save the filtered DataFrame to a .csv file
filtered_df.to_csv(output_csv_file, index=False)
