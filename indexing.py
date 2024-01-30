import json
from collections import defaultdict
from nltk.stem import PorterStemmer
# we create the reverse index here. We wil be using it alot in the application
# it is notably used for making the boolean calculations possible and is used in boolean_retrieval
 
def load_processed_data(filename='processed_papers_data.json'):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

def create_inverted_index(papers):
    stemmer = PorterStemmer()
    index = defaultdict(set)
    for paper in papers:
        doc_id = paper['title']
        for word in paper['processed_summary']:
            stemmed_word = stemmer.stem(word)
            index[stemmed_word].add(doc_id) 
    
    # saving in Json format 
    for word in index:
        index[word] = list(index[word])

    return index

papers = load_processed_data()
inverted_index = create_inverted_index(papers)

with open('inverted_index.json', 'w', encoding='utf-8') as file:
    json.dump({k: list(v) for k, v in inverted_index.items()}, file, ensure_ascii=False, indent=4)