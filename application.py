import json
import re
import tkinter as tk
from tkinter import messagebox
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
from algorithms import boolean_retrieval, vector_space_model, bm25_retrieval
from tkinter import simpledialog
from rank_bm25 import BM25Okapi

search_results = []  
papers_data = []     
filters = {'author': '', 'year': ''} 

def preprocess_query(query):
    query = query.lower()
    # remove signs
    query = re.sub(r'[^\w\s]', '', query)
    # tokenization
    tokens = word_tokenize(query)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Stemming !!!!!!!!!!!
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    return tokens


def load_processed_papers():
    with open('processed_papers_data.json', 'r', encoding='utf-8') as file:
        processed_papers = json.load(file)
    return processed_papers

def save_processed_data(data, filename='processed_papers_data.json'):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def load_inverted_index():
    with open('inverted_index.json', 'r', encoding='utf-8') as file:
        inverted_index = json.load(file)
        # lists -> sets
        inverted_index = {k: set(v) for k, v in inverted_index.items()}
    return inverted_index


processed_papers = load_processed_papers()
documents = [paper['processed_summary'] for paper in load_processed_papers()]


inverted_index = load_inverted_index()


def display_results(results, papers):
    results_text_widget.delete("1.0", tk.END)  # clear

    results_text = "No results found."
    if results:
        if isinstance(results, set):
            results_text = "\n".join(f"• {title.strip()}" for title in sorted(results))
        elif isinstance(results, list):
            titles = [papers[i]['title'].strip() for i in results if isinstance(i, int) and i < len(papers)]
            results_text = "\n".join(f"• {title}" for title in titles) if titles else "No results found."
    
    results_text_widget.insert("1.0", results_text)


def apply_filters(search_results, filters, papers_data):
    author_filter = filters['author'].lower().strip()
    year_filter = filters['year'].strip()

    filtered_results = []

    for paper_title in search_results:
        # Find the paper dictionary in papers_data by title
        paper = next((p for p in papers_data if p['title'] == paper_title), None)
        if paper:
            # filter for authors if there are matches
            if author_filter and all(author_filter not in author.lower() for author in paper['authors']):
                continue  # if author doesn't match, skip 
            if year_filter and year_filter not in paper['publication_date']:
                continue  # if year doesn't match, skip
            filtered_results.append(paper_title) 

    return filtered_results


def handle_search():
    global search_results, papers_data
    query = entry.get()
    processed_query = preprocess_query(query)
    selected_method = search_method.get()
    selected_boolean_operator = boolean_operation.get()

    author_filter = filters['author']
    year_filter = filters['year']

    if not papers_data:
        papers_data = load_processed_papers()

    if selected_method == 'boolean':
        search_results = boolean_retrieval(processed_query, inverted_index, papers_data, selected_boolean_operator, filters['author'], filters['year'])
    elif selected_method == 'vsm':
        search_results = vector_space_model(processed_query, papers_data, author_filter, year_filter)
    elif selected_method == 'bm25':
        search_results = bm25_retrieval(processed_query, papers_data, filters['author'], filters['year'])

    display_results(search_results, papers_data)


def open_filter_dialog():
    global filters
    
    dialog = tk.Toplevel(root)
    dialog.title("Filter Options")

    tk.Label(dialog, text="Author:").pack(side="left")
    author_entry = tk.Entry(dialog)
    author_entry.pack(side="left")
    author_entry.insert(0, filters['author'])  # Pre-fill with current filter

    tk.Label(dialog, text="Year:").pack(side="left")
    year_entry = tk.Entry(dialog)
    year_entry.pack(side="left")
    year_entry.insert(0, filters['year'])  # Pre-fill with current filter

    # Save button
    save_button = tk.Button(dialog, text="Save", command=lambda: save_filters(author_entry.get(), year_entry.get(), dialog))
    save_button.pack(side="left")


def save_filters(author, year, dialog):
    global filters
    filters['author'] = author
    filters['year'] = year
    dialog.destroy()


# main application
root = tk.Tk()
root.title("Search Application")
root.geometry("1200x1200")

root.grid_rowconfigure(1, weight=1)
for col in range(3):
    root.grid_columnconfigure(col, weight=1)

# Search method dropdown
search_method = tk.StringVar(root)
search_methods = ('boolean', 'vsm', 'bm25')
search_method.set(search_methods[0])
search_method_menu = tk.OptionMenu(root, search_method, *search_methods)
search_method_menu.grid(row=0, column=0, padx=10, pady=10, sticky='ew')

# AND OR NOT button 
boolean_operation = tk.StringVar(root)
boolean_operations = ('OR', 'AND', 'NOT')
boolean_operation.set(boolean_operations[0])  # Default to OR
boolean_operation_menu = tk.OptionMenu(root, boolean_operation, *boolean_operations)
boolean_operation_menu.grid(row=0, column=5, padx=5, pady=10, sticky='w')

# Search bar 
label = tk.Label(root, text="Enter your search query:")
label.grid(row=0, column=1, padx=10, pady=10, sticky='e')

# Search button
entry = tk.Entry(root)
entry.grid(row=0, column=2, sticky='ew', padx=10, pady=10)
search_button = tk.Button(root, text="Search", command=handle_search)  # No lambda or arguments needed
search_button.grid(row=0, column=3, padx=10, pady=10, sticky='ew')

filter_button = tk.Button(root, text="Filters", command=open_filter_dialog)
filter_button.grid(row=0, column=4, padx=10, pady=10, sticky='ew')

# Results
results_text_widget = tk.Text(root, height=25, width=80)
results_text_widget.grid(row=1, column=0, columnspan=5, padx=10, pady=10, sticky='nsew')
scrollbar = tk.Scrollbar(root, command=results_text_widget.yview)
scrollbar.grid(row=1, column=5, sticky='ns')
results_text_widget['yscrollcommand'] = scrollbar.set


root.mainloop()