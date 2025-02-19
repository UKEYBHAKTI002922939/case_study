import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bhakti_study.components.common import Evaluator, Recommender

class HybridModel:
    """
    Model class for a hybrid approach: 
    Weighted combination of TF-IDF similarity and Sentence Transformer similarity.
    """
    def __init__(self, labels_df):
        self.labels_df = labels_df
        # TF-IDF
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.labels_df["processed_title"])
        # Sentence Transformer
        self.st_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.sentence_embeddings = self.st_model.encode(
            self.labels_df["processed_title"].tolist(), 
            convert_to_numpy=True
        )

    def similarity_func(self, search_term):
        # TF-IDF similarity
        search_tfidf = self.vectorizer.transform([search_term])
        tfidf_similarities = cosine_similarity(search_tfidf, self.tfidf_matrix).flatten()

        # Sentence Transformer similarity
        search_embedding = self.st_model.encode([search_term], convert_to_numpy=True)
        embedding_similarities = cosine_similarity(search_embedding, self.sentence_embeddings).flatten()

        # Weighted combination
        # Adjust weights as needed
        hybrid_scores = 0.2 * tfidf_similarities + 0.5 * embedding_similarities
        return hybrid_scores

    def run(self, test_df):
        evaluator = Evaluator()
        best_threshold, best_f1 = evaluator.find_best_threshold(self.labels_df, self.similarity_func)
        print(f"Best threshold: {best_threshold} with macro F1: {best_f1}")

        evaluator.evaluate_model(self.labels_df, self.similarity_func, best_threshold)

        recommender = Recommender()
        recommendations = recommender.recommend_articles_for_test(
            test_df, self.labels_df, self.similarity_func, best_threshold, top_k=5
        )
        print(recommender.format_recommendations(recommendations))
