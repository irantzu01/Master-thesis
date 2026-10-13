import numpy as np
import io

def load_vectors(fname, max_words=50000):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array([float(x) for x in tokens[1:]])
        if len(data) >= max_words:
            break
    return data

# 1. Load the mapped vectors exported by your MUSE job
print("Loading vectors...")
es_vectors = load_vectors("dumped/muse_job_13674359/debug/fcub8bnxn3/vectors-es.txt")
eu_vectors = load_vectors("dumped/muse_job_13674359/debug/fcub8bnxn3/vectors-eu.txt")

# Extract matrix for target language (Basque)
eu_words = list(eu_vectors.keys())
eu_matrix = np.array([eu_vectors[w] for w in eu_words])
# Normalize for cosine similarity calculation
eu_matrix = eu_matrix / np.linalg.norm(eu_matrix, axis=1, keepdims=True)

def get_closest_basque_words(es_word, top_k=5):
    if es_word not in es_vectors:
        return f"Word '{es_word}' not found in Spanish vocabulary."
    
    # Normalize source vector
    v_es = es_vectors[es_word] / np.linalg.norm(es_vectors[es_word])
    
    # Calculate cosine similarities across the entire Basque space
    similarities = np.dot(eu_matrix, v_es)
    
    # Get top indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    print(f"\nSpanish Word: '{es_word}'")
    for idx in top_indices:
        print(f" -> Basque: {eu_words[idx]} (Cosine Similarity: {similarities[idx]:.4f})")

# 2. Test your specific target adjectives!
test_adjectives = ["grande", "pequeño", "rojo", "difícil", "interesante"]
for adj in test_adjectives:
    get_closest_basque_words(adj)