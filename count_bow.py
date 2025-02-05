import os
import glob
import normalize_text_module
from collections import defaultdict
import math
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim
import numpy as np

# Creating two different docs for the two categories
romance_docs = []
crime_docs = []

def open_txt_files_in_directory(directory, array):

    # A search pattern to match all .txt files in the directory
    search_pattern = os.path.join(directory, '*.txt')
    
    # glob finds all .txt files in the directory
    txt_files = glob.glob(search_pattern)
    
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as file:
                contents = file.read()
                array.append(contents)

        except Exception as e:
            print(f"Error opening file {txt_file}: {e}")

# Opening all chapters for each romance based novel
open_txt_files_in_directory('./text_data/romance/stella_rosevelt', romance_docs)
open_txt_files_in_directory('./text_data/romance/thrice_wedded', romance_docs)
open_txt_files_in_directory('./text_data/romance/under_the_desert_stars', romance_docs)
open_txt_files_in_directory('./text_data/romance/wooing_of_leola', romance_docs)

# Opening all chapters for each crime based novel
open_txt_files_in_directory('./text_data/crime_mystery/black_twenty_two', crime_docs)
open_txt_files_in_directory('./text_data/crime_mystery/draycott', crime_docs)
open_txt_files_in_directory('./text_data/crime_mystery/hound_of_baskerville', crime_docs)
open_txt_files_in_directory('./text_data/crime_mystery/murder_on_the_links', crime_docs)
open_txt_files_in_directory('./text_data/crime_mystery/sherlock_holmes', crime_docs)

# Creating two different dicts for creating bag-of-words
romance_bog = defaultdict(int)
crime_bog = defaultdict(int)

def count_words(documents, bog, **kwargs):
    tokenized_docs = []
    for doc in documents:
        tokens = normalize_text_module.normalize_text(doc, bog, **kwargs)  # kwargs just helps me to avoid writing default values for each option here
        tokenized_docs.append(tokens)
    return tokenized_docs

# Preprocess documents for each category
romance_docs_tk = count_words(
    romance_docs,
    romance_bog,
    lemmatize=True,
    lowercase=True,
    remove_stopwords=True,
    remove_punctuation=True
)

crime_docs_tk = count_words(
    crime_docs,
    crime_bog,
    lemmatize=True,
    lowercase=True,
    remove_stopwords=True,
    remove_punctuation=True
)

categories = {"romance": romance_bog, "crime": crime_bog}

# Here we just unpack the values and store the total_words for each category
total_words = {c: sum(bog.values()) for c, bog in categories.items()}


def calculate_probability(word, word_counts, total_words, vocab_size, alpha=1):
    return (word_counts.get(word, 0) + alpha) / (total_words + alpha * vocab_size)

# Vocab
vocab = set(list(romance_bog.keys()) + list(crime_bog.keys()))
vocab_size = len(vocab)

P_w_romance = {word: calculate_probability(word, romance_bog, total_words["romance"], vocab_size) for word in vocab}
P_w_crime = {word: calculate_probability(word, crime_bog, total_words["crime"], vocab_size) for word in vocab}

LLR_romance = {word: math.log(P_w_romance[word]) - math.log(P_w_crime[word]) for word in vocab}
LLR_crime = {word: math.log(P_w_crime[word]) - math.log(P_w_romance[word]) for word in vocab}

top_words_romance = sorted(LLR_romance.items(), key=lambda x: x[1], reverse=True)[:10]
top_words_crime = sorted(LLR_crime.items(), key=lambda x: x[1], reverse=True)[:10]

print("Top 10 words for Romance:", top_words_romance)
print("Top 10 words for Crime:", top_words_crime)

documents = romance_docs_tk + crime_docs_tk
dictionary = corpora.Dictionary(documents)
corpus = [dictionary.doc2bow(doc) for doc in documents]

num_topics = 4

lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, random_state = 100, update_every=1, chunksize=100, passes=10, alpha="auto")

topics = lda_model.print_topics(num_words=25)
for topic in topics:
    print(topic)

vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary, mds="mmds", R=30)
pyLDAvis.save_html(vis,"lda_vis_bow.html")

def topic_distribution(lda_model, romance_docs, crime_docs, dictionary):

    romance_corpus = [dictionary.doc2bow(text) for text in romance_docs]
    crime_corpus = [dictionary.doc2bow(text) for text in crime_docs]


    romance_topic_dist = [lda_model.get_document_topics(doc) for doc in romance_corpus]
    crime_topic_dist = [lda_model.get_document_topics(doc) for doc in crime_corpus]

    # Sum topic probabilities for each category
    romance_topic_sum = {}
    crime_topic_sum = {}

    for doc_topics in romance_topic_dist:
        for topic, prob in doc_topics:
            if topic not in romance_topic_sum:
                romance_topic_sum[topic] = prob
            else:
                romance_topic_sum[topic] += prob

    for doc_topics in crime_topic_dist:
        for topic, prob in doc_topics:
            if topic not in crime_topic_sum:
                crime_topic_sum[topic] = prob
            else:
                crime_topic_sum[topic] += prob

    # Compute average topic probabilities by dividing by the number of documents in each category
    romance_avg_dist = {topic: prob / len(romance_docs) for topic, prob in romance_topic_sum.items()}
    crime_avg_dist = {topic: prob / len(crime_docs) for topic, prob in crime_topic_sum.items()}

    return romance_avg_dist, crime_avg_dist

romance_avg_dist, crime_avg_dist = topic_distribution(lda_model, romance_docs_tk, crime_docs_tk, dictionary)

print("Category 1 Average Topic Distribution:")
for topic, prob in sorted(romance_avg_dist.items(), key=lambda x: x[1], reverse=True):
    print(f"Topic {topic}: {prob:.4f}")


print("\nCategory 2 Average Topic Distribution:")
for topic, prob in sorted(crime_avg_dist.items(), key=lambda x: x[1], reverse=True):
    print(f"Topic {topic}: {prob:.4f}")