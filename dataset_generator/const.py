import os

PROJECT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)))
DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')
GENERATOR_FOLDER = os.path.join(PROJECT_FOLDER, 'dataset_generator')
NEWS_FOLDER = os.path.join(DATA_FOLDER, 'news')