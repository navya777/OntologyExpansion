import pandas as pd

df=pd.read_csv('modified_file_with_similarity_cosine_neighborhoodfamilysocial.csv')
df_working = df.drop(['Similar Concepts','Similarities'], axis=1)
df_cleaned = df_working.dropna(subset=['Most Similar Concept'])
df_cleaned.to_csv('postprocessed_similarity_cosine_neighborhoodfamilysocial.csv', index=False)