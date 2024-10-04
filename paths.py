import os
# Paths
pdf_directory = 'data'
vector_database_path = "vector_database"


# Create directories if they don't exist
os.makedirs(pdf_directory, exist_ok=True)
os.makedirs(vector_database_path, exist_ok=True)
