import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


punctuation_regex = re.compile(r"[^\w\s]")
stop_words = stopwords.words("english")
wordnet_lemmatizer = WordNetLemmatizer()
pos_tags_to_lemmatize = ["n", "v"]


def tokenize(text):
    text = punctuation_regex.sub(
        " ",
        text.lower()
    ).strip()

    tokens = word_tokenize(text)

    for pos_tag in pos_tags_to_lemmatize:
        tokens = [
            wordnet_lemmatizer.lemmatize(
                token,
                pos=pos_tag
            )
            for token in tokens
        ]

    tokens = [
        token
        for token in tokens
        if token not in stop_words
    ]

    return tokens


