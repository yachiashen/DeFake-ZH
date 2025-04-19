import json
import os 
import datetime
import pandas as pd
from const import DATA_FOLDER

NEWS_SOURCES = [
    'cna', 
    'cts',
    'ftv',
    'ltn',
    'mirrormedia',
    'pts',
    'setn',
    'ttv', 
    'tvbs',
    'udn'
]

NEWS_CLASSIFICATION = [
    'entertain & sports',
    'international',
    'local & society',
    'politics',
    'technology & life'
]

# -----------------------------------------------------------------------------------------------------------------------

def date_add_month_or_year(date_obj: datetime.date, delta_year: int, delta_month: int):
    
    total_months = (date_obj.year * 12 + date_obj.month) + (delta_year * 12 + delta_month)
    new_year, new_month = total_months // 12, total_months % 12

    if new_month == 0:
        new_year -= 1
        new_month = 12

    return datetime.date(
        year  = new_year,
        month = new_month,
        day = date_obj.day
    )    

def load_json_file(json_file_path: str) -> dict:

    # default json file 
    ret_json_file:dict = {
        'news_source': None,
        'year-month' : None,
        'news'       : dict()
    }
    if os.path.exists(json_file_path) and os.path.isfile(json_file_path):
        try:
            with open(json_file_path, 'r', encoding = 'utf-8') as f:
                ret_json_file = json.load(f)
        except: raise ValueError(f'{json_file_path} is not a json file')
    return ret_json_file

def save_json_file(data: dict , json_file_path: str) -> None:
    with open(json_file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii = False, indent = 4)
    return

def clear_all_json(folder_path: str):
    """
    ** function purpose **
    (1) Delete JSON files: Delete all JSON files under the specified folder_path, including files in subdirectories (recursively).
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError
    elif not os.path.isdir(folder_path):
        raise ValueError

    for p in os.listdir(folder_path):
        sub_path = os.path.join(folder_path, p)
        if os.path.isdir(sub_path):
            clear_all_json(sub_path)
        elif os.path.isfile(sub_path):
            _, extension = os.path.splitext(sub_path)
            if extension == ".json":
                os.remove(os.path.abspath(sub_path))
    return 

def convert_news_data_format(news_folder_path: str, start_date: datetime.date, end_date: datetime.date) -> None:
    """
    ** function purpose **
    (1) Convert news format: Transform news from CSV format to JSON format.
    (2) Merge news articles: Combine news articles from the same source (news_source), published in the same year-month (year-month), but belonging to different classifications (classification).
    
    ** directory structure **
    [news_folder_path]/
    │── csv/
    │   ├── news/
    │   │   ├── [classification]/
    │   │   │   ├── [year]/
    │   │   │   │   ├── [month]/
    │   │   │   │   │   ├── [news_source.csv]
    │
    │── json/
    │   ├── [year-month]/
    │   │   ├── [news_source.json]
    """
    CSV_NEWS_FOLDER = os.path.join(news_folder_path, 'csv', 'news')
    JSON_FOLDER = os.path.join(news_folder_path, 'json')

    if not os.path.exists(JSON_FOLDER):
        os.makedirs(JSON_FOLDER, exist_ok = True)
    elif not os.path.isdir(JSON_FOLDER):
        raise FileExistsError(f"{JSON_FOLDER} exists, but it is not a folder")

    if not os.path.exists(CSV_NEWS_FOLDER):
        raise FileNotFoundError(f"{CSV_NEWS_FOLDER} is not found")
    elif not os.path.isdir(CSV_NEWS_FOLDER):
        raise FileExistsError(f"{CSV_NEWS_FOLDER} exists, but it is not a folder")
    

    time_range = (start_date, end_date)
    t = time_range[0]

    while t <= time_range[1]:
        date_folder = os.path.join(JSON_FOLDER, f'{t.year:4d}-{t.month:02d}')
        os.makedirs(date_folder, exist_ok = True)

        for source in NEWS_SOURCES:
            json_file_path = os.path.join(date_folder, f"{source}.json")
            json_data = load_json_file(json_file_path)

            if json_data['news_source'] is None and json_data['year-month'] is None:
                json_data['news_source'] = source
                json_data['year-month'] = f'{t.year:4d}-{t.month:02d}'
            
            for clsf in NEWS_CLASSIFICATION:
                df = pd.read_csv(os.path.join(CSV_NEWS_FOLDER, clsf, str(t.year), str(t.month), f"{source}.csv"))

                for i in df.index:
                    try:
                        if len(str(df.loc[i, 'Title']).strip()) == 0 or \
                        len(str(df.loc[i, 'Content']).strip()) == 0: continue 
                        
                        json_data['news'][df.loc[i, 'Title'].strip()] = {
                            'Title': df.loc[i, 'Title'].strip(),
                            'Content': df.loc[i, 'Content'].strip(),
                            'Date': df.loc[i, 'Date'],
                            'Url': df.loc[i, 'Url'],
                            'Classification': df.loc[i, 'Classification']
                        }
                    except:
                        pass

            save_json_file(json_data, json_file_path)

        t = date_add_month_or_year(t, delta_year = 0, delta_month = +1)
    return

# -----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    convert_news_data_format(os.path.join(DATA_FOLDER, 'news'), start_date = datetime.date(2025, 1, 1), end_date = datetime.date(2025, 3, 1))
