import os
import glob
from collections import defaultdict
import math
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim
import normalize_text_module

# Function to open text files
def open_txt_files_in_directory(directory, array):
    search_pattern = os.path.join(directory, '*.txt')
    txt_files = glob.glob(search_pattern)
    
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as file:
                contents = file.read()
                array.append(contents)
        except Exception as e:
            print(f"Error opening file {txt_file}: {e}")

# Load documents
romance_docs, crime_docs = [], []

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

# Preprocess documents
romance_bog = defaultdict(int)
crime_bog = defaultdict(int)

romance_docs_tk = [normalize_text_module.normalize_text(doc, romance_bog, binary=True) for doc in romance_docs]
crime_docs_tk = [normalize_text_module.normalize_text(doc, crime_bog, binary=True) for doc in crime_docs]

# Probability Calculations
categories = {"romance": romance_bog, "crime": crime_bog}
total_words = {c: sum(bog.values()) for c, bog in categories.items()}
vocab = set(list(romance_bog.keys()) + list(crime_bog.keys()))
vocab_size = len(vocab)

# Laplace smoothing
def calculate_probability(word, word_counts, total_words, vocab_size, alpha=1):
    return (word_counts.get(word, 0) + alpha) / (total_words + alpha * vocab_size)

P_w_romance = {word: calculate_probability(word, romance_bog, total_words["romance"], vocab_size) for word in vocab}
P_w_crime = {word: calculate_probability(word, crime_bog, total_words["crime"], vocab_size) for word in vocab}

LLR_romance = {word: math.log(P_w_romance[word] / P_w_crime[word]) for word in vocab}
LLR_crime = {word: math.log(P_w_crime[word] / P_w_romance[word]) for word in vocab}

# Top words
print(sorted(LLR_romance.items(), key=lambda x: x[1], reverse=True)[:10])
print(sorted(LLR_crime.items(), key=lambda x: x[1], reverse=True)[:10])

# LDA Modeling
documents = romance_docs_tk + crime_docs_tk
dictionary = corpora.Dictionary(documents)
corpus = [dictionary.doc2bow(doc) for doc in documents]

lda_model = models.LdaModel(corpus, num_topics=4, id2word=dictionary, passes=10)

topics = lda_model.print_topics(num_words=10)
for topic in topics:
    print(topic)


vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary, mds="mmds", R=30)
pyLDAvis.save_html(vis,"lda_vis_bin.html")


def topic_distribution(lda_model, romance_docs, crime_docs, dictionary):
    # Convert documents to bag-of-words format
    romance_corpus = [dictionary.doc2bow(text) for text in romance_docs_tk]
    crime_corpus = [dictionary.doc2bow(text) for text in crime_docs_tk]

    # Get topic distributions for each document
    romance_topic_dist = [lda_model.get_document_topics(doc) for doc in romance_corpus]
    crime_topic_dist = [lda_model.get_document_topics(doc) for doc in crime_corpus]

    # Sum topic probabilities for each category
    romance_topic_sum = {}
    crime_topic_sum = {}

    for doc_topics in romance_topic_dist:
        for topic, prob in doc_topics:
            romance_topic_sum[topic] = romance_topic_sum.get(topic, 0) + prob

    for doc_topics in crime_topic_dist:
        for topic, prob in doc_topics:
            crime_topic_sum[topic] = crime_topic_sum.get(topic, 0) + prob

    # Compute average topic probabilities
    romance_avg_dist = {topic: prob / len(romance_docs) for topic, prob in romance_topic_sum.items()}
    crime_avg_dist = {topic: prob / len(crime_docs) for topic, prob in crime_topic_sum.items()}

    return romance_avg_dist, crime_avg_dist

# Call the function to get average topic distributions
romance_avg_dist, crime_avg_dist = topic_distribution(lda_model, romance_docs, crime_docs, dictionary)

# Display the results
print("Average Topic Distribution for Romance Documents:")
for topic, prob in romance_avg_dist.items():
    print(f"Topic {topic}: {prob:.4f}")

print("\nAverage Topic Distribution for Crime Documents:")
for topic, prob in crime_avg_dist.items():
    print(f"Topic {topic}: {prob:.4f}")
