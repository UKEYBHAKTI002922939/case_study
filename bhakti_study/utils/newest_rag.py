import os
import re
import string
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from tabulate import tabulate
from transformers import pipeline

# Initialize a free text-generation pipeline using a Hugging Face model.
# (You can try a larger model if you have the resources.)
generator = pipeline("text2text-generation", model="google/flan-t5-base", tokenizer="google/flan-t5-base")

def load_data():
    """
    Load the CSV files:
      - content_df: contains 'title' and 'slug'
      - labels_df: contains mislabeled columns that we fix and then merge with content_df
      - test_df: contains 'searchTerm'
    """
    content_df = pd.read_csv("data/raw/content_data_MASTER.csv")  # Contains 'title', 'slug'
    labels_df = pd.read_csv("data/raw/labels_MASTER.csv")           # Has misnamed columns
    test_df = pd.read_csv("data/raw/test_MASTER.csv")               # Contains 'searchTerm'

    # Rename columns in labels_df to our expected names
    labels_df.rename(columns={
        '#4 #14 connector': 'searchTerm', 
        'types-of-pipe-fittings': 'slug', 
        'RELEVANT': 'label'
    }, inplace=True)

    # Merge labels with content to obtain titles for the labeled rows
    labels_df = labels_df.merge(content_df, on="slug", how="left")
    labels_df.dropna(subset=["title"], inplace=True)
    labels_df.drop_duplicates(inplace=True)
    labels_df.reset_index(drop=True, inplace=True)
    return content_df, labels_df, test_df

def preprocess_text(text):
    """
    Lowercase text, remove punctuation, and extra spaces.
    """
    text = str(text).lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_data(labels_df):
    """
    Apply text preprocessing to both the search terms and titles.
    """
    labels_df["processed_search"] = labels_df["searchTerm"].apply(preprocess_text)
    labels_df["processed_title"] = labels_df["title"].apply(preprocess_text)
    return labels_df

def fit_sentence_embeddings(labels_df):
    """
    Generate sentence embeddings for article titles and normalize them 
    for cosine similarity.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentences = labels_df["processed_title"].tolist()
    sentence_embeddings = model.encode(sentences, convert_to_numpy=True)
    # Normalize embeddings to unit length for cosine similarity.
    norms = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
    sentence_embeddings = sentence_embeddings / norms
    return model, sentence_embeddings

def build_faiss_index(embeddings):
    """
    Build a FAISS index using inner product (which is equivalent to cosine similarity
    if the vectors are normalized).
    """
    dimension = embeddings.shape[1]
    embeddings = embeddings.astype(np.float32)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index

def retrieve_articles_faiss(search_term, model, index, k=5):
    """
    For a given search term, compute its embedding (and normalize it) and retrieve 
    the top-k articles from the FAISS index. Returns cosine similarity scores and indices.
    """
    processed_search = preprocess_text(search_term)
    search_embedding = model.encode([processed_search], convert_to_numpy=True)
    search_embedding = search_embedding / np.linalg.norm(search_embedding, axis=1, keepdims=True)
    similarities, indices = index.search(search_embedding.astype(np.float32), k)
    return similarities[0], indices[0]

def generate_llm_recommendation(search_term, candidate_articles):
    """
    Build a prompt using the candidate articles and have the free LLM generate
    a natural language recommendation.
    """
    prompt = f"You are an expert content recommendation system. A user searched for '{search_term}'.\n"
    if candidate_articles:
        prompt += "Below are some candidate articles:\n"
        for idx, article in enumerate(candidate_articles, start=1):
            prompt += f"{idx}. Title: {article['title']}\n   Slug: {article['slug']}\n"
        prompt += ("\nPlease rank these articles by their relevance to the search term and provide "
                   "a natural language recommendation with a brief explanation for each ranking.")
    else:
        prompt += "No relevant articles were found."
    
    try:
        result = generator(prompt, max_length=300, do_sample=True, temperature=0.7)
        recommendation = result[0]['generated_text']
    except Exception as e:
        recommendation = f"Error generating recommendation: {e}"
    return recommendation

def recommend_articles_with_rag(test_df, labels_df, model, faiss_index, top_k=5, threshold_similarity=0.5):
    """
    For each search term in the test set, use the FAISS index to retrieve candidate articles.
    If the best candidate's cosine similarity is below the threshold, consider that no relevant 
    articles were found. Then use the free LLM to generate a recommendation.
    """
    recommendations = {}
    for search_term in test_df["searchTerm"]:
        similarities, indices = retrieve_articles_faiss(search_term, model, faiss_index, k=top_k)
        candidate_articles = []
        seen_slugs = set()  # Use this set to avoid duplicate articles based on slug

        # If the top candidate's similarity is below the threshold, treat it as no relevant match.
        if len(similarities) == 0 or similarities[0] < threshold_similarity:
            candidate_articles = []
        else:
            for idx, sim in zip(indices, similarities):
                article = labels_df.iloc[idx]
                # Normalize the slug for deduplication (strip whitespace and lowercase)
                slug_norm = str(article["slug"]).strip().lower()
                if slug_norm not in seen_slugs:
                    candidate_articles.append({
                        "slug": article["slug"],
                        "title": article["title"],
                        "processed_title": article["processed_title"],
                        "similarity": sim
                    })
                    seen_slugs.add(slug_norm)
        llm_recommendation = generate_llm_recommendation(search_term, candidate_articles)
        recommendations[search_term] = {
            "candidates": candidate_articles,
            "llm_recommendation": llm_recommendation
        }
    return recommendations

def format_llm_recommendations(recommendations):
    """
    Format the output by printing a header for each search term, a table of candidate articles,
    and the LLM-generated recommendation.
    """
    formatted_output = []
    for search_term, rec_data in recommendations.items():
        candidates = rec_data["candidates"]
        llm_rec = rec_data["llm_recommendation"]
        formatted_output.append(f"\nSearch Term: **{search_term}**\n")
        if candidates:
            formatted_output.append("Candidate Articles:")
            table_data = []
            for rank, article in enumerate(candidates, start=1):
                table_data.append([rank, article["slug"], article["title"], f"{article['similarity']:.3f}"])
            formatted_output.append(tabulate(table_data, headers=["Rank", "Slug", "Title", "Cosine Similarity"], tablefmt="grid"))
        else:
            formatted_output.append("No relevant articles found.")
        formatted_output.append("\nLLM Recommendation:")
        formatted_output.append(llm_rec)
        formatted_output.append("\n" + "="*80 + "\n")
    return "\n".join(formatted_output)

def main():
    # Load and preprocess the data.
    content_df, labels_df, test_df = load_data()
    labels_df = preprocess_data(labels_df)
    
    # Generate normalized sentence embeddings.
    model, sentence_embeddings = fit_sentence_embeddings(labels_df)
    
    # Build the FAISS index.
    faiss_index = build_faiss_index(sentence_embeddings)
    
    # Retrieve articles and generate recommendations.
    recommendations = recommend_articles_with_rag(test_df, labels_df, model, faiss_index, top_k=5, threshold_similarity=0.5)
    
    # Format and print the results.
    output = format_llm_recommendations(recommendations)
    print(output)

if __name__ == "__main__":
    main()
