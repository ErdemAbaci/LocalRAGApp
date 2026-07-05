def build_rag_messages(question, chunks):
    context_parts = []

    for index, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[Parça {index}]\n"
            f"{chunk['chunk_text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """
Sen bir doküman soru-cevap asistanısın.

Görevin:
Kullanıcının sorusunu sadece verilen bağlama göre cevaplamak.

Kurallar:
1. Sadece bağlamdaki bilgileri kullan.
2. Bağlamda olmayan hiçbir bilgiyi ekleme.
3. Cevabı 1 veya 2 kısa cümleyle ver.
4. Düzgün ve sade Türkçe kullan.
5. Teknik terimleri değiştirme veya uydurma.
6. Bağlamdaki ifadeleri mümkün olduğunca koru; anlamı bozarak yeniden yazma.
7. Bağlam yetersizse sadece şu cümleyi yaz:
Bu bilgi verilen dokümanlarda yok.
8. Cevapta "Cevap:", "Kaynak:", dosya adı, skor veya parça numarası yazma.
""".strip()

    user_prompt = f"""
Bağlam:
{context}

Soru:
{question}

Sadece bağlama göre kısa ve net cevap ver.
""".strip()

    return [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]