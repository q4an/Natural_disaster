from procces_data import load_data,transform_data,cleaning_data,save_database ,create_engine




from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV,GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multioutput import MultiOutputClassifier
import pandas as pd
import joblib
from text_processing import tokenize
import logging
from pathlib import Path
from sklearn.model_selection import ParameterGrid
from time import perf_counter



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

logger = logging.getLogger(__name__)


def format_duration(total_seconds):
    minutes,seconds = divmod(
        total_seconds,
        60
    
    )

    return f"{minutes} dakika {seconds} saniye"




def split_data(df):
    X = df["message"].values
    Y=df.drop(columns=["message","genre","child_alone"])
    category_names = Y.columns
    gss = GroupShuffleSplit(n_splits=1, 
        test_size=0.2, 
        random_state=42
    )
    train_idx,test_idx = next(gss.split(X,Y,groups=X))

    X_train=X[train_idx]
    X_test = X[test_idx]

    Y_train= Y.iloc[train_idx]
    Y_test = Y.iloc[test_idx]
        
    print(len(set(X_train)& set(X_test)))
    return X_train,X_test,Y_train,Y_test,category_names





def build_model():
    """Builds classification model """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize,token_pattern=None)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(XGBClassifier(learning_rate=0.1,n_jobs=1,random_state=42)))
    ])

    parameters = {
        "clf__estimator__max_depth": [8, 16],
        "clf__estimator__colsample_bytree":[0.5, 0.75]
    }
    number_of_combinations = len(
    list(ParameterGrid(parameters))
)

    number_of_cv_folds = 3

    total_cv_jobs = (
    number_of_combinations
    * number_of_cv_folds
)

    logger.info(
    "%s parametre kombinasyonu bulundu.",
    number_of_combinations
)

    logger.info(
    "Toplam %s çapraz doğrulama eğitimi yapılacak.",
    total_cv_jobs
)
    grou_cv = GroupKFold(n_splits=3)
    cv = GridSearchCV(pipeline, cv=grou_cv, param_grid=parameters, n_jobs=-1, scoring="f1_micro",verbose=2)
    return cv





def evaluate_model(model, X_test, Y_test, category_names):

    y_preds = model.predict(X_test)

    print(classification_report(
        Y_test.values,
        y_preds,
        target_names=category_names,
        zero_division=0
        ))
    # collect accuracy scores in a dict
    category_name_2_accuracy_score = {}

    for i in range(len(category_names)):
       category_name_2_accuracy_score[Y_test.columns[i]] = accuracy_score(Y_test.values[:,i],y_preds[:,i])
    print(pd.Series(category_name_2_accuracy_score))






def save_model(model, category_names, filename):
    model_bundle = {
        "model": model.best_estimator_,
        "category_names": list(category_names)
    }

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = filename.with_suffix(".tmp")

    joblib.dump(model_bundle, temporary_path)
    temporary_path.replace(filename)



def main():
    logger.info("Veri hazırlama başladı.")

    categories, messages = load_data()

    df = transform_data(
        categories,
        messages
    )

    df = cleaning_data(df)

    save_database(df)

    logger.info(
        "Veri hazırlama tamamlandı: %s mesaj.",
        len(df)
    )

    X_train, X_test, Y_train, Y_test, category_names = (
        split_data(df)
    )

    logger.info(
        "Eğitim verisi hazırlandı: %s mesaj.",
        len(X_train)
    )

    logger.info(
        "Test verisi hazırlandı: %s mesaj.",
        len(X_test)
    )

    logger.info(
        "Kategori sayısı: %s.",
        len(category_names)
    )

    model = build_model()

    logger.info(
        "Model eğitilmeye başlandı."
    )

    training_start_time = perf_counter()

    try:
        model.fit(
            X_train,
            Y_train,
            groups=X_train
        )

    except Exception:
        logger.exception(
            "Eğitim sırasında hata oluştu."
        )
        raise

    training_duration = (
        perf_counter()
        - training_start_time
    )

    logger.info(
        "Model eğitimi tamamlandı. Süre: %s",
        format_duration(training_duration)
    )

    logger.info(
        "En iyi parametreler: %s",
        model.best_params_
    )

    logger.info(
        "En iyi CV F1 micro skoru: %.4f",
        model.best_score_
    )

    model_path = (
        Path(__file__).resolve().parent.parent
        / "models"
        / "disaster_model_bundle.pkl"
    )

    logger.info("Model paketi kaydediliyor: %s", model_path)

    save_model(
        model=model,
        category_names=category_names,
        filename=model_path
    )

    logger.info("Model ve kategori isimleri kaydedildi.")

    logger.info(
        "Model değerlendirmesi başladı."
    )

    evaluate_model(
        model,
        X_test,
        Y_test,
        category_names
    )

    logger.info(
        "Model değerlendirmesi tamamlandı."
    )


if __name__ == "__main__":
   main()


