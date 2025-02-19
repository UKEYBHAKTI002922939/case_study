import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bhakti_study.components.common import Evaluator, Recommender

class TfidfModel:
    """Model class for TF-IDF based similarity."""
    def __init__(self, labels_df):
        self.labels_df = labels_df
        self.vectorizer = TfidfVectorizer()
        # Fit on the processed_title column
        self.tfidf_matrix = self.vectorizer.fit_transform(self.labels_df["processed_title"])

    def similarity_func(self, search_term):
        # Transform the incoming query
        search_tfidf = self.vectorizer.transform([search_term])
        return cosine_similarity(search_tfidf, self.tfidf_matrix).flatten()

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
