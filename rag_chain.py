from openai import OpenAI
from vector_store import VectorStore

CHAT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are an expert car buying assistant for the Indian and global automotive market.
You have access to the following data sources:
  • cardata.csv         - 301 Indian used car sales records
  • car_price_prediction.csv - 2500 global car price records
  • cardekho.csv        - 8128 CarDekho India used car listings
  • used_cars.csv       - 4009 US market used car listings
  • newswift_pricelist.pdf  - Maruti Suzuki New Swift Chennai on-road prices (2025)
  • hyundai_pricelist.pdf   - KUN Hyundai Chennai on-road prices (March 2024)

Guidelines:
- Always ground your answers in the provided context passages.
- For price questions, clearly state the currency (INR or USD) and whether it's ex-showroom or on-road.
- If comparing cars, use the data from the relevant datasets.
- If information isn't in the context, say so honestly — don't guess prices.
- Keep answers structured: use bullet points for comparisons or lists of variants.
- When discussing new car prices, mention the source dealer and date of the price list.
- Be helpful, concise, and accurate.
"""


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[Source {i}] {c['source']} | {c['page']} | Relevance: {c.get('score', 0):.2f}\n"
            f"{c['content']}"
        )
    return "\n\n---\n\n".join(parts)


def answer(
    query: str,
    store: VectorStore,
    client: OpenAI,
    chat_history: list[dict],
    top_k: int = 6,
) -> tuple[str, list[dict]]:
    retrieved = store.search(query, top_k=top_k)
    context = build_context(retrieved)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in chat_history[-8:]:
        messages.append(turn)

    messages.append({
        "role": "user",
        "content": f"Context from car datasets:\n\n{context}\n\nQuestion: {query}",
    })

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.15,
        max_tokens=1200,
    )
    return resp.choices[0].message.content, retrieved