# Bible Translation Assessor

This repository contains a json file containing verses from the KJV, ESV, NIV, and NLT along with the source text from which they were translated. Sourcing was done through ChatGPT and while they have been cross-reference through biblehub.com, I don't know enough about biblical manuscripts to assert sufficient authenticity exists. Nonetheless embeddings were created from each verse and cosine similarity measured and plotted. 

The goal is to use OpenAI text embeddings to assess the semantic similarity between the source text and its English translations. This analysis can help visualize how closely each translation aligns, semantically, with the source verse.

## Embeddings and Cosine Similarity 
Embeddings were generated using OpenAI’s text-embedding-3-large embedding model to capture the semantic content of each verse in both the translated texts and the source text. Cosine similarity was then computed between the source text embedding and each translated text embeddings. These similarity scores were plotted for each verse to visualize how semantically aligned each English translation is to the original source text.

## Further Work
* More embedding models should be used and their results analysed and compared. 

* Expand dataset with more verses 

* Source text accuracy should be verified by a biblical linguist. 

* Compare semantic similarity scores with scholarly rankings of translation quality to evaluate potential correlations.

## Files
* `verses.json:` Contains each verse in source text + four translations

* `embed_and_plot.py`: Script to generate embeddings, compute cosine * similarity, and save plots

* `embeddings.json:` Stores all embeddings for reproducibility

* `plots/:` Saved cosine similarity graphs per verse