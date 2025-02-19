import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bhakti_study.components.common import Evaluator, Recommender

class SentenceTransformerModel:
    """Model class for Sentence Transformer based similarity."""
    def __init__(self, labels_df):
        self.labels_df = labels_df
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.sentence_embeddings = self.model.encode(
            self.labels_df["processed_title"].tolist(), 
            convert_to_numpy=True
        )

    def similarity_func(self, search_term):
        search_embedding = self.model.encode([search_term], convert_to_numpy=True)
        return cosine_similarity(search_embedding, self.sentence_embeddings).flatten()

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
