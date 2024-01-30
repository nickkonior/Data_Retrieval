import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
import re
# this app is processing the data fetched from crawler.py 
# stemming, lowercase, stop words removal, deleting no-word characters
# logic must be the same as query processing in the main  app

def process_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    processed_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return processed_tokens

def load_data(filename='papers_data.json'):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_processed_data(data, filename='processed_papers_data.json'):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

papers = load_data()
for paper in papers:
    paper['processed_summary'] = process_text(paper['summary'])

# final save
save_processed_data(papers)