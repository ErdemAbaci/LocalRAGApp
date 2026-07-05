import openai
from foundry_local import FoundryLocalManager


MODEL_ALIAS = "phi-3.5-mini"


def main():
    print("Foundry Local başlatılıyor...")

    manager = FoundryLocalManager()

    print("Model yükleniyor...")
    model_info = manager.load_model(MODEL_ALIAS)

    print("Model yüklendi:")
    print(model_info)

    client = openai.OpenAI(
        base_url=manager.endpoint,
        api_key=manager.api_key
    )

    response = client.chat.completions.create(
        model=model_info.id,
        messages=[
            {
                "role": "system",
                "content": "Sen kısa, net ve Türkçe cevap veren bir asistansın."
            },
            {
                "role": "user",
                "content": "RAG nedir? Tek cümleyle açıkla."
            }
        ],
        temperature=0.2
    )

    print("\nModel cevabı:")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()