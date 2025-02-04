import os
import glob

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
open_txt_files_in_directory('./text_data/romance/black_twenty_two', crime_docs)
open_txt_files_in_directory('./text_data/romance/draycott', crime_docs)
open_txt_files_in_directory('./text_data/romance/hound_of_baskerville', crime_docs)
open_txt_files_in_directory('./text_data/romance/murder_on_the_links', crime_docs)
open_txt_files_in_directory('./text_data/romance/sherlock_holmes', crime_docs)