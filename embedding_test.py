from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def main():
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    documents = [
        "RAG, dokümanlardan ilgili bilgiyi bulup LLM'e context olarak verir.",
        "SQLite, yerel ve hafif bir veritabanıdır.",
        "Foundry Local, cihaz üzerinde local LLM çalıştırmayı sağlar."
    ]

    question = "RAG ne işe yarar?"

    document_embeddings = model.encode(documents)
    question_embedding = model.encode([question])

    similarities = cosine_similarity(question_embedding, document_embeddings)[0]

    for index, score in enumerate(similarities):
        print("Doküman:", documents[index])
        print("Benzerlik skoru:", score)
        print("----")

    best_index = similarities.argmax()

    print("En alakalı doküman:")
    print(documents[best_index])


if __name__ == "__main__":
    main()