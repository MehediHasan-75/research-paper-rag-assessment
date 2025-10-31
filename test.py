from src.services.embedding_service import embedding_service

# Single text test
text = "Machine learning is fascinating."
embedding = embedding_service.encode_single(text)
print("✅ Embedding shape:", embedding.shape)
print("✅ First 5 values:", embedding[:5])
print("✅ Dimension from config:", embedding_service.get_dimension())
print("✅ Model name:", embedding_service.get_model_name())

# Batch test
texts = ["AI is the future.", "I love deep learning.", "Sentence transformers are powerful."]
embeddings = embedding_service.encode_batch(texts)
print("✅ Batch shape:", embeddings.shape)
