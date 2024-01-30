from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def boolean_retrieval(query_terms, inverted_index, papers_data, boolean_operator, author_filter=None, year_filter=None):
    # split terms if it's a string
    if isinstance(query_terms, str):
        query_terms = query_terms.split()

    results = set()
    all_titles = set(paper['title'] for paper in papers_data) 

    if boolean_operator == 'NOT':
        # Special handling for NOT operator, assuming a single term in query_terms
        exclude_term = query_terms[0]  # Assuming single term for NOT operation
        exclude_results = {paper['title'] for paper in papers_data if exclude_term in paper['processed_summary']}
        results = all_titles - exclude_results
    else:
        for term in query_terms:
            term_results = {paper['title'] for paper in papers_data if term in paper['processed_summary']}
            if boolean_operator == 'AND':
                results = term_results if not results else results & term_results
            elif boolean_operator == 'OR':
                results |= term_results

    # filter results !!!!!!!
    if author_filter:
        results = {title for title in results if any(author_filter.lower() in author.lower() for author in next((paper['authors'] for paper in papers_data if paper['title'] == title), []))}
    if year_filter:
        results = {title for title in results if year_filter in next((paper['publication_date'] for paper in papers_data if paper['title'] == title), '')}

    return results


def vector_space_model(query_terms, papers_data, author_filter=None, year_filter=None):
    # filter papers
    filtered_papers = [paper for paper in papers_data if 
                       (not author_filter or author_filter.lower() in [author.lower() for author in paper['authors']]) and
                       (not year_filter or year_filter in paper['publication_date'])]

    # convert filtered papers to summaries
    summaries = [' '.join(paper['processed_summary']) for paper in filtered_papers]

    # vectorizer, document transformation
    vectorizer = TfidfVectorizer()
    try:
        doc_vectors = vectorizer.fit_transform(summaries)
    except ValueError:
        # return null if empty 
        return []

    query_vector = vectorizer.transform([' '.join(query_terms)])

    # similarity between query and documents 
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()

    scored_papers = [(score, papers_data[i]) for i, score in enumerate(similarities)]
    scored_papers.sort(key=lambda x: x[0], reverse=True)  # score of the documents matching
    for score, paper in scored_papers:
        print(f"Title: {paper['title']}, Similarity Score: {score}")

    # filter and rank documents based on similarities
    relevant_docs_indices = [i for i, score in enumerate(similarities) if score > 0]
    relevant_docs_indices.sort(key=lambda i: similarities[i], reverse=True)  # Sort by similarity in descending order

    return relevant_docs_indices


def bm25_retrieval(query_terms, papers_data, author_filter=None, year_filter=None):
    # filters on papers_data
    filtered_papers = [paper for paper in papers_data if 
                       (not author_filter or author_filter.lower() in [author.lower() for author in paper['authors']]) and
                       (not year_filter or year_filter in paper['publication_date'])]

    # tokenized summaries
    tokenized_summaries = [paper['processed_summary'] for paper in filtered_papers]

    # Initialize BM25 with the tokenized summaries
    bm25 = BM25Okapi(tokenized_summaries)

    # BM25 scores for tokenized query
    doc_scores = bm25.get_scores(query_terms)

    scored_papers = [(score, paper) for score, paper in zip(doc_scores, filtered_papers)]
    scored_papers.sort(key=lambda x: x[0], reverse=True)  # Sort by score in descending order

    for score, paper in scored_papers:
        print(f"Title: {paper['title']}, BM25 Score: {score}")

    # filter and rank documents based on scores
    relevant_docs_indices = [i for i, score in enumerate(doc_scores) if score > 0]
    relevant_docs_indices.sort(key=lambda i: doc_scores[i], reverse=True)  # Sort by score in descending order

    return relevant_docs_indices



