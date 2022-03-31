import pickle
import re
import sys
import warnings

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

warnings.filterwarnings("ignore")
nltk.download(["punkt", "stopwords"])


def load_data(database_filepath: str) -> tuple:
    """
    Loads data from the sqlite database and splits into X and y.
    Here, the database name and the table name must be the same.

    :param database_filepath: name of the database
    :return: numpy array with X and y and list with classification categories
    """
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table("DisasterResponse", engine)

    category_names = df.columns[4:].tolist()
    
    # set all category columns to 0 or 1 (a few values, e.g. in "related", are higher than 1)
    df[category_names] = df[category_names].astype(bool).astype(int)
    
    X = df["message"].values
    y = df[category_names].values
    return X, y, category_names


def tokenize(text: str) -> list:
    """
    Preprocesses a text sequence for a prediction model by
    setting lower case, removing punctuation, removing stop words
    and stemming the words in the text sequence.

    :param text: input text sequence
    :return: tokenized text sequence
    """
    # normalize text (set lowercase and remove punctuation)
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower().strip())

    # tokenize text (split messages into words)
    tokens = word_tokenize(text)

    # remove stop words from text
    tokens = [t for t in tokens if t not in stopwords.words("english")]

    # lemmatize text
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(token, pos="v") for token in tokens]
        
    return clean_tokens


def build_model() -> GridSearchCV:
    """
    Builds a prediction model including grid search in scikit-learn.

    :return: prediction model
    """
    pipeline = Pipeline(
        [
            (
                "text_pipeline",
                Pipeline(
                    [
                        ("vect", CountVectorizer(tokenizer=tokenize)),
                        ("tfidf", TfidfTransformer()),
                    ]
                ),
            ),
            ("clf", MultiOutputClassifier(RandomForestClassifier(n_estimators=5))),
        ]
    )

    parameters = {
        "clf__estimator__min_samples_split": [2, 3],
        "clf__estimator__n_estimators": [5, 10],
    }

    cv = GridSearchCV(
        estimator=pipeline,
        param_grid=parameters,
        n_jobs=-1,  # do the possible computation in parallel
        verbose=3,  # show real-time training progress
        cv=2,  # limit number of CV folds
    )
    return cv


def evaluate_model(
        model: GridSearchCV, X_test: np.array, Y_test: np.array, category_names: list
) -> None:
    """
    Predicts values for the test set, and displays the overall model accuracy
    and the classification report for each category.

    :param model: trained model
    :param X_test: features to use for predicting
    :param Y_test: true labels
    :param category_names: classification category names
    """
    Y_pred = model.predict(X_test)
    
    print(classification_report(Y_test, Y_pred, target_names=category_names))
    print(f"Accuracy: {np.mean(Y_test == Y_pred)}")


def save_model(model: GridSearchCV, model_filepath: str) -> None:
    """
    Saves the prediction model to pickle.

    :param model: model to save to pickle
    :param model_filepath: name of the pickle output
    """
    with open(model_filepath, "wb") as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()