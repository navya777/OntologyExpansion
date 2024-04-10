import csv
import re

# Function to parse numbered RDF triples from the "Response" column
def extract_rdf_triples(response_text):
    # Regex to match the numbered RDF triples
    rdf_triple_pattern = re.compile(r'\d+\.\s+\(([^,]+),\s*([^,]+),\s*([^)]+)\)')
    rdf_triples = rdf_triple_pattern.findall(response_text)
    return rdf_triples

# Path to the input CSV file
csv_file_path = 'rdftriples_results_pubmed_neighborhoodfamilysocial_v1.csv'
# Path to the output CSV file
output_csv_file_path = 'extracted_rdf_triples_neighborhoodfamilysocial.csv'

# Prepare to collect data for the new CSV
data_to_write = []

# Reading the input CSV file
with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        # Extracting numbered RDF triples from the "Response" column
        rdf_triples = extract_rdf_triples(row['Response'])
        for child_concept, relationship, parent_concept in rdf_triples:
            # Cleaning up the extracted values
            child_concept = child_concept.strip()
            relationship = relationship.strip()
            parent_concept = parent_concept.strip()
            # Appending the PMID and extracted RDF triple components to the data list
            data_to_write.append({
                'PMID': row['PMID'],
                'Child Concept': child_concept,
                'Relationship': relationship,
                'Parent Concept': parent_concept
            })

# Writing the collected RDF triples into a new CSV file
with open(output_csv_file_path, mode='a', newline='', encoding='utf-8') as new_csv_file:
    fieldnames = ['PMID', 'Child Concept', 'Relationship', 'Parent Concept']
    writer = csv.DictWriter(new_csv_file, fieldnames=fieldnames)

    # Writing the header
    writer.writeheader()

    # Writing the data rows
    for data_row in data_to_write:
        writer.writerow(data_row)

print(f"RDF triples have been extracted and saved to '{output_csv_file_path}'.")
