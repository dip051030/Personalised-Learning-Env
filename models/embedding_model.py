import logging

from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class EmbeddingModel:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
        if self._model is None:
            logging.info("Initializing SentenceTransformer model: Shashwat13333/bge-base-en-v1.5_v4")
            self._model = SentenceTransformer('Shashwat13333/bge-base-en-v1.5_v4')
            logging.info("SentenceTransformer model initialized.")

    def get_model(self):
        return self._model


# Create a singleton instance
embedding_model = EmbeddingModel().get_model()
