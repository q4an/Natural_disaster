import pandas as pd 
from sqlalchemy import create_engine
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_PATH = PROJECT_ROOT / "data" / "disaster_db"

disaster_cat_path =(
    DATA_DIR / "disaster_categories.csv"
)

disaster_mes_path = (
    DATA_DIR / "disaster_messages.csv"
)

def load_data():
    categories = pd.read_csv(disaster_cat_path)
    messages =  pd.read_csv(disaster_mes_path)
    return categories,messages

def transform_data(categories,messages):
    df = messages.merge(categories, on="id")

    categories = df["categories"].str.split(";",expand=True)
    category_column_names = categories.iloc[0].apply(lambda x: x.split("-")[0])
    categories.columns = category_column_names
    for column in categories:
        categories[column]=categories[column].str[-1].astype(int)

    categories['related'] = categories['related'].clip(0,1)

    df.drop("categories",axis=1,inplace=True)
    df=pd.concat([df,categories],axis=1)
    return df


def cleaning_data(df):
    df.drop_duplicates(keep="first",inplace=True)
    df.dropna(subset=["message"],axis=0,inplace=True)
    df.drop(["id", "original"], axis=1, inplace=True)
    return df


def save_database(df):
    database_url = (
        f"sqlite:///{DATABASE_PATH.as_posix()}"
    )
    engine = create_engine(database_url)
    df.to_sql("messages",engine,if_exists="replace",index=False)



if __name__ == "__main__":
    categories, messages = load_data()
    df = transform_data(categories, messages)
    df = cleaning_data(df)
    save_database(df)
