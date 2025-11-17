
import os, hashlib, json, glob
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

load_dotenv()
sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY"))
client = OpenAI()
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")

def md(file_path):
    # Simple metadata heuristic from path
    name = os.path.basename(file_path)
    jlpt = "N5" if "n5" in name.lower() else ("N4" if "n4" in name.lower() else "N3")
    topic = "レストラン" if "restaurant" in name else ("旅行" if "travel" in name else "一般")
    kind = "dialog" if "dialog" in file_path else ("grammar" if "grammar" in file_path else "errors")
    return {"jlpt": jlpt, "topic": topic, "kind": kind, "source": name}

def chunks(text, size=500, overlap=100):
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i+size])
        i += size - overlap
    return out

def upsert_row(id, content, metadata, embedding):
    sb.table("docs").upsert({
        "id": id,
        "content": content,
        "metadata": metadata,
        "embedding": embedding
    }).execute()

def main():
    files = glob.glob("data/**/*.txt", recursive=True)
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            text = f.read().strip()
        for i, ch in enumerate(chunks(text)):
            uid = hashlib.sha1(f"{fp}:{i}".encode()).hexdigest()
            meta = md(fp)
            emb = client.embeddings.create(model=EMBED_MODEL, input=ch).data[0].embedding
            upsert_row(uid, ch, meta, emb)
            print("Upserted:", meta["source"], i)
    print("Done.")

if __name__ == "__main__":
    main()
