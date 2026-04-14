import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'fraud-detect-dev-secret-2024'

    SQLALCHEMY_DATABASE_URI = (
        os.environ.get('DATABASE_URL')
        or 'sqlite:///' + os.path.join(BASE_DIR, 'instance', 'fraud_cases.db')
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Directory paths
    DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
    DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')

    # Upload
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500 MB
    ALLOWED_EXTENSIONS = {'csv'}

    # ML settings
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
