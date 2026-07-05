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
Kullanıcının sorusunu sadece verilen bağlama göre doğru ve anlaşılır biçimde cevaplamak.

Kurallar:
1. Sadece bağlamdaki bilgileri kullan.
2. Bağlamda olmayan hiçbir bilgiyi ekleme.
3. Düzgün, sade ve doğal Türkçe kullan.
4. Teknik terimleri değiştirme veya uydurma.
5. Bağlamdaki önemli kavramları atlama.
6. Bağlamdaki ifadeleri mümkün olduğunca koru; anlamı bozarak yeniden yazma.
7. Soru bir süreç, adım, aşama veya liste soruyorsa 3-5 kısa maddeyle cevap ver.
8. Her madde tek bir net fikir içersin.
9. Soru tanım soruyorsa 1 veya 2 kısa paragrafla cevap ver.
10. Eksik, yarım veya bozuk cümle kurma.
11. Cevabı gereksiz uzatma; bağlamdaki ana bilgiyi özetle.
12. Bağlam yetersizse sadece şu cümleyi yaz:
Bu bilgi verilen dokümanlarda yok.
13. Cevapta "Cevap:", "Kaynak:", dosya adı, skor veya parça numarası yazma.
""".strip()

    user_prompt = f"""
Bağlam:
{context}

Soru:
{question}

Sadece bağlama göre kısa, net ve eksiksiz cevap ver. Kaynak etiketi ekleme.
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
