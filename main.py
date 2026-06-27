from app.retrieval import get_top_chunks


def main():
    while True:
        question = input("Soru sor: ")

        if question.lower() in ["q", "quit", "exit"]:
            break

        results = get_top_chunks(question, top_k=3)

        print("\nEn alakalı chunklar:")

        for result in results:
            print("----")
            print("ID:", result["id"])
            print("Kaynak:", result["source_name"])
            print("Skor:", result["score"])
            print("Metin:", result["chunk_text"])

        print()


if __name__ == "__main__":
    main()