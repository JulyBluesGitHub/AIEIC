
import os
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

load_dotenv()
sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY"))
client = OpenAI()
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
CHAT_MODEL  = os.getenv("CHAT_MODEL",  "gpt-5-thinking")

SYSTEM_JP = (
"あなたは日本語会話のパートナーです。ユーザーのレベルは {level} です。"
"返答は必ず {register} で行い、発話は短く自然に。事実は渡されたコンテキストの範囲で話し、"
"根拠が無い場合は推測せず確認してください。誤りがあれば最多２点を最小限に訂正し、"
"やさしい説明と１つの練習例を示し、最後に質問で締めます。"
)

def embed(text):
    return client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding

def retrieve(query_vec, topk=5, jlpt_max="N3", topic=None):
    # Simple JLPT ceiling filter
    filters = []
    if topic:
        filters.append(f"(metadata->>'topic') = '{topic}'")
    # No numeric compare for N-level; keep simple for MVP
    where = " and ".join(filters) if filters else "true"
    sql = f"""
    select id, content, metadata
    from docs
    where {where}
    order by embedding <-> to_vector(%s)
    limit {topk};
    """
    # Supabase RPC alternative is to use PostgREST; use a raw query via REST if available in your plan,
    # or use a Supabase SQL function. For quick MVP, use the 'rpc' with a SQL function you define.
    # To keep it simple here, assume you've created a Postgres function 'match_docs' and call it instead.
    raise NotImplementedError("For production, add a Postgres function to perform the vector search.")

def simple_template(context, user_input, level="N4", register="丁寧形"):
    sys = SYSTEM_JP.format(level=level, register=register)
    ctx = "\n\n---\n".join([f"\n{c['content']}" for c in context])
    messages = [
        {"role": "system", "content": sys},
        {"role": "system", "content": f"参考コンテキスト：\n{ctx}"},
        {"role": "user", "content": user_input}
    ]
    return messages

def chat(user_input, level="N4", register="丁寧形"):
    qv = embed(user_input)
    # ⚠ For a quick start without SQL function, you can do the similarity search in Python by fetching all embeddings once and caching.
    # But best practice is a DB-side <-> operator. See README for adding a SQL function to expose vector search via RPC.
    # Below we mock retrieved context for clarity:
    context = [{"id": "demo", "content": "て形：動詞の連用形。例：食べて、行って、読んで。", "metadata": {}}]
    messages = simple_template(context, user_input, level, register)
    resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.3)
    print(resp.choices[0].message.content)

if __name__ == "__main__":
    chat("て形の練習をしたいです。どう使いますか？", level="N5", register="丁寧形")
