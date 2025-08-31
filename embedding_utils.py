import os
import numpy as np
import google.generativeai as genai

def get_google_embeddings(texts, batch_size=32, api_key=None):
    """Generates Google embeddings for a list of texts with batching and normalization.

    Args:
        texts (list[str]): A list of strings to embed.
        batch_size (int, optional): The batch size for embedding requests. Defaults to 32.
        api_key (str, optional): The Gemini API key. Defaults to None.

    Returns:
        np.ndarray: A numpy array of normalized embeddings.
    """
    if api_key:
        genai.configure(api_key=api_key)
    else:
        print("GEMINI_API_KEY not provided to get_google_embeddings.")
        return np.array([]) # Return empty array if no API key

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        try:
            result = genai.embed_content(model="models/embedding-001",
                                           content=batch_texts,
                                           task_type="retrieval_document")
            embeddings = result['embedding']
            # Normalize embeddings to unit vectors
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized_embeddings = embeddings / norms
            all_embeddings.extend(normalized_embeddings)
        except Exception as e:
            print(f"An error occurred during embedding: {e}")
            # Handle the error as needed, e.g., by appending zero vectors
            all_embeddings.extend(np.zeros((len(batch_texts), 768)))

    return np.array(all_embeddings)

if __name__ == '__main__':
    # Example usage:
    # Make sure to set your GEMINI_API_KEY environment variable before running
    sample_texts = [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?"
    ]

    # For example usage, we still rely on environment variable or direct pass
    api_key_example = os.environ.get("GEMINI_API_KEY")
    if not api_key_example:
        print("Please set the GEMINI_API_KEY environment variable to run the example.")
    else:
        embeddings = get_google_embeddings(sample_texts, api_key=api_key_example)
        print("Embeddings shape:", embeddings.shape)
        # Example of finding similarity:
        # Similarity between the first and fourth text
        similarity = np.dot(embeddings[0], embeddings[3])
        print(f"Similarity between first and fourth text: {similarity:.4f}")