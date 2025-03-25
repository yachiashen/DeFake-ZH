import os

def __parent_folder(src_path, level):
    if level == 0:
        return src_path
    elif level > 0:
        return __parent_folder(os.path.dirname(src_path), level - 1)
    else:
        raise ValueError(f"\"level < 0\" is prohibited")

PROJECT_FOLDER = __parent_folder(src_path = __file__, level = 2)
NEWS_SOURCE_FOLDER = os.path.join(PROJECT_FOLDER, 'news_source')
TRUTH_SOURCE_FOLDER = os.path.join(PROJECT_FOLDER, 'truth_source')


