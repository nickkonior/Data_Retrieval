import requests
import xml.etree.ElementTree as ET
import json
# this crawler.py targets arXiv and completes data fetcing of 100 documents
#downloads papar's summaries, authors, titles and releas dates

def fetch_arxiv_data(query, max_results=100):
    base_url = 'http://export.arxiv.org/api/query?'
    query_params = f'search_query={query}&start=0&max_results={max_results}'
    response = requests.get(base_url + query_params)
    return response.text

def parse_xml(xml_data):
    root = ET.fromstring(xml_data)
    papers = []
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
        authors = [author.find('{http://www.w3.org/2005/Atom}name').text for author in entry.findall('{http://www.w3.org/2005/Atom}author')]
        summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()
        publication_date = entry.find('{http://www.w3.org/2005/Atom}published').text.strip()

        papers.append({
            'title': title,
            'authors': authors,
            'summary': summary,
            'publication_date': publication_date
        })
    return papers

def save_data_to_json(data, filename='papers_data.json'):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def main():
    query = 'Economics'  
    xml_data = fetch_arxiv_data(query)
    papers = parse_xml(xml_data)
    save_data_to_json(papers)

if __name__ == '__main__':
    
    main()
