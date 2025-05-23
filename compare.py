import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from time import sleep
from dotenv import load_dotenv
import openai

# === Load environment variables ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# === Paths ===
DATA_FILE = "academic_bible_comparison.json"
PLOT_DIR = "plots"
OUTPUT_JSON = "embedding_results.json"

# === Create plot directory ===
os.makedirs(PLOT_DIR, exist_ok=True)

# === Load verses ===
with open(DATA_FILE, "r", encoding="utf-8") as f:
    verses = json.load(f)

# === Get embedding from OpenAI ===
def get_embedding(text, model="text-embedding-3-large"):
    response = openai.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding).tolist()

# === Initialize results dict ===
embedding_results = {}

# === Process each verse ===
for verse_ref, data in verses.items():
    print(f"Processing {verse_ref}...")

    verse_result = {
        "source_text": data["source"]["text"],
        "source_embedding": get_embedding(data["source"]["text"]),
        "translations": {}
    }

    sleep(1)  # Respect rate limits

    for version, translation in data["translations"].items():
        try:
            translation_embedding = get_embedding(translation)
            similarity = 1 - cosine(verse_result["source_embedding"], translation_embedding)

            verse_result["translations"][version] = {
                "text": translation,
                "embedding": translation_embedding,
                "similarity_to_source": similarity
            }

            sleep(1)

        except Exception as e:
            print(f"Error with {version} in {verse_ref}: {e}")
            verse_result["translations"][version] = {
                "text": translation,
                "embedding": None,
                "similarity_to_source": None,
                "error": str(e)
            }

    embedding_results[verse_ref] = verse_result

    # Plot similarities
    versions = []
    scores = []
    for v, vdata in verse_result["translations"].items():
        if vdata["similarity_to_source"] is not None:
            versions.append(v)
            scores.append(vdata["similarity_to_source"])

    plt.figure(figsize=(8, 5))
    plt.bar(versions, scores)
    plt.ylim(0, 1)
    plt.title(f"Semantic Similarity to Source Text\n{verse_ref}")
    plt.ylabel("Cosine Similarity")
    plt.xlabel("Bible Translation")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    filename = verse_ref.replace(" ", "_").replace(":", "-") + ".png"
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()

# === Save all embeddings and results ===
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(embedding_results, f, ensure_ascii=False, indent=2)

print(f"\n✅ All plots saved to: {PLOT_DIR}/")
print(f"✅ Full embedding results saved to: {OUTPUT_JSON}")
