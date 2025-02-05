import os
import glob
import normalize_text_module
from collections import defaultdict

# Creating two different docs for the two categories
romance_docs = []
crime_docs = []

def open_txt_files_in_directory(directory, array):
    # Construct the search pattern to match all .txt files in the directory
    search_pattern = os.path.join(directory, '*.txt')
    
    # Use glob to find all .txt files in the directory
    txt_files = glob.glob(search_pattern)
    
    # Loop through each .txt file and open it
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
    for doc in documents:
        normalize_text_module.normalize_text(doc, bog, **kwargs)  # Applying preprocessing, kwargs just avoids me to write default values for each option here

# Preprocess documents for each category
count_words(
    romance_docs,
    romance_bog,
    lemmatize=True,
    lowercase=True,
    remove_stopwords=True,
    remove_punctuation=True
)

count_words(
    crime_docs,
    crime_bog,
    lemmatize=True,
    lowercase=True,
    remove_stopwords=True,
    remove_punctuation=True
)

# Calculate total words for each class
total_romance_words = sum(romance_bog.values())
total_crime_words = sum(crime_bog.values())

# probability calculation (support laplace smoothing)
def calculate_probability(word, word_counts, total_words, vocab_size, alpha=1):
    return (word_counts[word] + alpha) / (total_words + alpha * vocab_size)

# Example calculation
word = "say"
vocab = set(list(romance_bog.keys()) + list(crime_bog.keys()))
vocab_size = len(vocab)
p_say_romance = calculate_probability(word, romance_bog, total_romance_words, vocab_size)
p_say_crime = calculate_probability(word, crime_bog, total_crime_words, vocab_size)

# Print results
print(f"The conditional probability of the word '{word}' given the class 'romance' is: {p_say_romance:.2f}")
print(f"The conditional probability of the word '{word}' given the class 'crime' is: {p_say_crime:.2f}")