import pandas as pd
import numpy as np
import re
import string
import logging
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

# Configure a module-level logger (this will use the configuration from logger/__init__.py)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Loads and merges data from CSV files.
    """
    def __init__(self, content_path, labels_path, test_path):
        self.content_path = content_path
        self.labels_path = labels_path
        self.test_path = test_path

    def load_data(self):
        logger.info("Loading content data from %s", self.content_path)
        content_df = pd.read_csv(self.content_path)
        
        logger.info("Loading labels data from %s", self.labels_path)
        labels_df = pd.read_csv(self.labels_path)
        
        logger.info("Loading test data from %s", self.test_path)
        test_df = pd.read_csv(self.test_path)

        logger.info("Renaming columns in labels data")
        labels_df.rename(columns={
            '#4 #14 connector': 'searchTerm', 
            'types-of-pipe-fittings': 'slug', 
            'RELEVANT': 'label'
        }, inplace=True)

        logger.info("Merging content and labels data")
        labels_df = labels_df.merge(content_df, on="slug", how="left")

        logger.info("Dropping rows with missing titles, duplicates and resetting index")
        labels_df.dropna(subset=["title"], inplace=True)
        labels_df.drop_duplicates(inplace=True)
        labels_df.reset_index(drop=True, inplace=True)
        logger.info("Data loaded successfully")
        return content_df, labels_df, test_df

class TextPreprocessor:
    """
    Handles text preprocessing for search terms, titles, and slugs.
    """
    def __init__(self):
        pass

    def preprocess_text(self, text):
        text = str(text).lower()
        text = re.sub(f"[{string.punctuation}]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def preprocess_data(self, labels_df):
        logger.info("Starting text preprocessing for labels data")
        labels_df["processed_search"] = labels_df["searchTerm"].apply(self.preprocess_text)
        labels_df["processed_title"] = labels_df["title"].apply(self.preprocess_text)
        labels_df["processed_slug"] = labels_df["slug"].apply(self.preprocess_text)
        logger.info("Text preprocessing complete")
        return labels_df

class Evaluator:
    """
    Contains methods for threshold tuning and model evaluation.
    """
    def __init__(self):
        pass

    def find_best_threshold(self, labels_df, similarity_func):
        logger.info("Finding best threshold for classifying documents")
        thresholds = np.linspace(0, 1, 101)
        best_threshold = None
        best_f1 = 0

        X_train, X_test, y_train, y_test = train_test_split(
            labels_df["processed_search"], labels_df["label"], test_size=0.2, random_state=42
        )
        label_mapping = {"NOT RELEVANT": 0, "RELEVANT": 1}
        y_test_num = [label_mapping[label] for label in y_test]

        for t in thresholds:
            preds = []
            for search_term in X_test:
                scores = similarity_func(search_term)
                pred = "RELEVANT" if max(scores) > t else "NOT RELEVANT"
                preds.append(pred)
            y_pred_num = [label_mapping[p] for p in preds]
            current_f1 = f1_score(y_test_num, y_pred_num, average="macro")
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = t

        logger.info("Best threshold found: %s with macro F1: %s", best_threshold, best_f1)
        return best_threshold, best_f1

    def evaluate_model(self, labels_df, similarity_func, threshold):
        logger.info("Evaluating model performance using threshold: %s", threshold)
        X_train, X_test, y_train, y_test = train_test_split(
            labels_df["processed_search"], labels_df["label"], test_size=0.2, random_state=42
        )
        predictions = []
        for search_term in X_test:
            scores = similarity_func(search_term)
            predicted_label = "RELEVANT" if max(scores) > threshold else "NOT RELEVANT"
            predictions.append(predicted_label)

        report = classification_report(y_test, predictions)
        logger.info("Classification Report:\n%s", report)
        return report

class Recommender:
    """
    Generates article recommendations for the test set and formats the output.
    """
    def __init__(self):
        pass

    def recommend_articles_for_test(self, test_df, labels_df, similarity_func, threshold, top_k=5):
        logger.info("Generating recommendations for test data")
        recommendations = {}
        for search_term in test_df["searchTerm"]:
            scores = similarity_func(search_term)
            top_indices = np.argsort(scores)[::-1][:top_k]
            top_scores = scores[top_indices]

            if len(top_scores) == 0 or max(top_scores) < threshold:
                recommendations[search_term] = "Result not found"
            else:
                candidate_rows = labels_df.iloc[top_indices][["slug", "title", "processed_title"]].values.tolist()
                seen_slugs = set()
                unique_candidates = []
                for slug, title, _ in candidate_rows:
                    if slug not in seen_slugs:
                        unique_candidates.append([slug, title])
                        seen_slugs.add(slug)
                recommendations[search_term] = unique_candidates

        logger.info("Recommendations generated successfully")
        return recommendations

    def format_recommendations(self, recommendations):
        formatted_output = []
        for search_term, articles in recommendations.items():
            formatted_output.append(f"\nSearch Term: **{search_term}**\n")
            if articles == "Result not found":
                formatted_output.append("Result not found")
            else:
                table_data = []
                for slug, title in articles:
                    table_data.append([slug, title])
                formatted_output.append(tabulate(table_data, headers=["Slug", "Recommended Title"], tablefmt="grid"))
        return "\n".join(formatted_output)
