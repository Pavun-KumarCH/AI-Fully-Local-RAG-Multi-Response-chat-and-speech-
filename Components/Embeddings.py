import ollama

# Function to generate embeddings
def generate_embeddings(text_chunks, model_name='nomic-embed-text'):
    embeddings = [ollama.embeddings(model_name, prompt=chunk) for chunk in text_chunks]
    return embeddings