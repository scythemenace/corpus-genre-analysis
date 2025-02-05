import nltk
import string
from nltk.corpus import wordnet

# Download necessary NLTK data
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger")


def normalize_text(
    content,
    word_counts,
    stem=False,
    lemmatize=False,
    lowercase=False,
    remove_stopwords=False,
    remove_punctuation=False,
):
    # Tokenize the content
    tokens = nltk.word_tokenize(content)

    # Apply preprocessing steps based on the flags
    if lowercase:
        tokens = [token.lower() for token in tokens]

    if stem:
        stemmer = nltk.stem.PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]

    if lemmatize:
        lemmatizer = nltk.stem.WordNetLemmatizer()

        def get_wordnet_pos(tag):
            if tag.startswith("J"):
                return wordnet.ADJ
            elif tag.startswith("V"):
                return wordnet.VERB
            elif tag.startswith("N"):
                return wordnet.NOUN
            elif tag.startswith("R"):
                return wordnet.ADV
            else:
                return None

        pos_tags = nltk.pos_tag(tokens)
        tokens = [
            lemmatizer.lemmatize(token, pos=pos) if pos else token
            for token, tag in pos_tags
            for pos in [get_wordnet_pos(tag)]
        ]

    if remove_stopwords:
        stopwords = nltk.corpus.stopwords.words("english")
        tokens = [token for token in tokens if token.lower() not in stopwords]

    if remove_punctuation:
        extended_punctuation = string.punctuation + ",.;“’--”!*:?...."
        tokens = [token for token in tokens if token not in extended_punctuation]

    # Update word counts
    for token in tokens:
        word_counts[token] += 1