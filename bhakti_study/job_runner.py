import argparse
import logging
from bhakti_study.config.configuration import CONTENT_DATA_FILE_PATH, LABELS_DATA_FILE_PATH, TEST_DATA_FILE_PATH
from bhakti_study.components.common import DataLoader, TextPreprocessor
from bhakti_study.utils.tfidf_model import TfidfModel
from bhakti_study.utils.sentence_transformer_model import SentenceTransformerModel
from bhakti_study.utils.hybrid_model import HybridModel
from bhakti_study.exception import CustomException
import sys

logger = logging.getLogger(__name__)

def main():
    try:
        parser = argparse.ArgumentParser(description="Run a specified model.")
        parser.add_argument(
            "--model",
            type=str,
            choices=["tfidf", "sentence", "hybrid"],
            required=True,
            help="Which model to run: 'tfidf', 'sentence', or 'hybrid'."
        )
        args = parser.parse_args()

        logging.info("Starting job_runner...")

        # 1. Load Data
        loader = DataLoader(
            content_path=CONTENT_DATA_FILE_PATH,
            labels_path=LABELS_DATA_FILE_PATH,
            test_path=TEST_DATA_FILE_PATH
        )
        content_df, labels_df, test_df = loader.load_data()
        logging.info("Data loaded successfully.")

        # 2. Preprocess Data
        preprocessor = TextPreprocessor()
        labels_df = preprocessor.preprocess_data(labels_df)
        test_df["processed_search"] = test_df["searchTerm"].apply(preprocessor.preprocess_text)
        logging.info("Data preprocessing complete.")

        # 3. Choose and Initialize Model
        if args.model == "tfidf":
            logging.info("Initializing TF-IDF model...")
            model = TfidfModel(labels_df)
        elif args.model == "sentence":
            logging.info("Initializing Sentence Transformer model...")
            model = SentenceTransformerModel(labels_df)
        else:  # "hybrid"
            logging.info("Initializing Hybrid model...")
            model = HybridModel(labels_df)

        # 4. Run Model
        logging.info("Running %s model...", args.model)
        model.run(test_df)
        logging.info("%s model finished.", args.model)

    except Exception as e:
        # Catch any exceptions not handled elsewhere and wrap in CustomException
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()
