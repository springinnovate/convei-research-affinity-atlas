"""Convert input text into numerical vectors that can be understood by LLM."""
import os
import pickle
import hashlib
import chardet
import numpy
import logging
import glob
import torch
import faiss


from sentence_transformers import SentenceTransformer
import tiktoken
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.INFO)
logging.getLogger('sentence_transformers').setLevel(logging.WARN)
logging.getLogger('httpx').setLevel(logging.WARN)
LOGGER = logging.getLogger(__name__)

GPT_MODEL = 'gpt-4o' # 'gpt-3.5-turbo'
ENCODING = tiktoken.encoding_for_model(GPT_MODEL)
TOP_K = 10

BODY_TAG = 'body'
CITATION_TAG = 'citation'

DATA_STORE = [
    r"data/web_of_science_query/abstracts_2023_11_02/*.ris",
    r"data/scopus/*/*.bib"
    ]

CACHE_DIR = 'llm_cache'
for dirpath in [CACHE_DIR]:
    os.makedirs(dirpath, exist_ok=True)


def parse_file_bib(bib_file_path):
    def generate_citation(record):
        # Generate a citation string from the record dictionary
        authors = ' and '.join(record.get('author', []))
        title = record.get('title', 'N/A')
        journal = record.get('journal', 'N/A')
        year = record.get('year', 'N/A')
        volume = record.get('volume', 'N/A')
        issue = record.get('number', 'N/A')
        pages = record.get('pages', 'N/A')
        doi = record.get('doi', 'N/A')
        return f"{authors} ({year}). {title}. {journal}, {volume}({issue}), {pages}. DOI: {doi}"

    with open(bib_file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']

    record_list = []
    with open(bib_file_path, 'r', encoding=encoding) as file:
        content = file.read()
        entries = content.split('@ARTICLE')[1:]  # Skip the initial split part before the first @ARTICLE
        for entry in entries:
            entry = entry.strip()
            record = {}
            for line in entry.split('\n'):
                line = line.strip()
                if line.endswith(','):
                    line = line[:-1]
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('{}').strip('"')
                    if key == 'author':
                        record[key] = [author.strip() for author in value.split(' and ')]
                    else:
                        record[key] = value
            if 'abstract' in record:
                record_list.append({
                    BODY_TAG: record['abstract'],
                    CITATION_TAG: generate_citation(record)
                })
        return record_list


# Example function to concatenate document contents and generate a hash
def generate_hash(documents):
    concatenated = ''.join(documents)
    hash_object = hashlib.sha256(concatenated.encode('utf-8'))
    hash_hex = hash_object.hexdigest()
    return hash_hex


def parse_file_ris(ris_file_path):
    def generate_citation(record):
        authors = ' and '.join(record.get('Authors', []))
        title = record.get('Title', 'N/A')
        journal = record.get('Journal', 'N/A')
        year = record.get('Year', 'N/A')
        volume = record.get('Volume', 'N/A')
        issue = record.get('Issue', 'N/A')
        start_page = record.get('StartPage', 'N/A')
        end_page = record.get('EndPage', 'N/A')
        doi = record.get('DOI', 'N/A')
        citation = f'{authors} ({year}). {title}. {journal}, {volume}({issue}), {start_page}-{end_page}. DOI: {doi}'
        return citation

    print(ris_file_path)
    with open(ris_file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']

    with open(ris_file_path, 'r', encoding=encoding) as file:
        record = {}
        record_list = []
        for line in file:
            line = line.strip()
            if line == '':
                if 'Abstract' in record:
                    record_list.append({
                        BODY_TAG: record['Abstract'],
                        CITATION_TAG: generate_citation(record)})
                record = {}
            else:
                payload = line.split('  - ')
                if len(payload) == 2:
                    index, body = payload
                else:
                    index = payload[0]
                    body = None
                record[index] = body

                if index == 'TY':
                    record['Type'] = body
                elif index == 'AU':
                    if 'Authors' not in record:
                        record['Authors'] = []
                    record['Authors'].append(body)
                elif index == 'TI':
                    record['Title'] = body
                elif index == 'T2':
                    record['Journal'] = body
                elif index == 'AB':
                    record['Abstract'] = body
                elif index == 'DA':
                    record['Date'] = body
                elif index == 'PY':
                    record['Year'] = body
                elif index == 'VL':
                    record['Volume'] = body
                elif index == 'IS':
                    record['Issue'] = body
                elif index == 'SP':
                    record['StartPage'] = body
                elif index == 'EP':
                    record['EndPage'] = body
                elif index == 'DO':
                    record['DOI'] = body

        return record_list


def parse_file(file_path):
    if file_path.endswith('.ris'):
        return parse_file_ris(file_path)
    elif file_path.endswith('.bib'):
        return parse_file_bib(file_path)


def save_embeddings(documents, model, filename):
    # Encode the documents
    document_embeddings = model.encode(documents, convert_to_tensor=True)

    # Convert to numpy array if it's a PyTorch tensor
    document_embeddings_np = document_embeddings.cpu().numpy()

    # Save to a .npy file
    numpy.save(filename, document_embeddings_np)


def load_embeddings(filename):
    # Load from the .npy file
    document_embeddings_np = numpy.load(filename)

    # Convert back to a PyTorch tensor
    document_embeddings = torch.tensor(document_embeddings_np)

    return document_embeddings


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)

    file_paths = [
        file_path
        for file_pattern in DATA_STORE
        for file_path in glob.glob(file_pattern)]

    file_hash = generate_hash(file_paths)
    article_list_pkl_path = os.path.join(CACHE_DIR, f'{file_hash}.pkl')
    fiass_path = os.path.join(CACHE_DIR, f'{file_hash}.faiss')

    if os.path.exists(article_list_pkl_path):
        with open(article_list_pkl_path, 'rb') as file:
            article_list = pickle.load(file)
        documents = [article[BODY_TAG] for article in article_list]
        citations = [article[CITATION_TAG] for article in article_list]
        index = faiss.read_index(fiass_path)
    else:
        article_list = []
        for file_path in file_paths:
            article_list += parse_file(file_path)
        with open(article_list_pkl_path, 'wb') as file:
            pickle.dump(article_list, file)

        documents = [article[BODY_TAG] for article in article_list]
        citations = [article[CITATION_TAG] for article in article_list]

        LOGGER.debug('embedding')
        document_embeddings = embedding_model.encode(
            documents, convert_to_tensor=True)
        # Index the embeddings using FAISS
        LOGGER.debug('indexing')
        index = faiss.IndexFlatL2(document_embeddings.shape[1])
        index.add(document_embeddings.cpu().numpy())
        faiss.write_index(index, fiass_path)

    def answer_question_with_gpt(question, documents, citations, index):
        # Encode the question
        from openai import OpenAI
        client = OpenAI()

        question_embedding = embedding_model.encode(question, convert_to_tensor=True).cpu().numpy()

        # Ensure the question_embedding is 2D
        if len(question_embedding.shape) == 1:
            question_embedding = question_embedding.reshape(1, -1)

        # Retrieve the most similar documents
        distances, indices = index.search(question_embedding, TOP_K)
        retrieved_docs = [documents[idx] for idx in indices[0]]
        relevant_citations = [citations[idx] for idx in indices[0]]

        # Concatenate the retrieved documents to form the context
        context = " ".join(retrieved_docs)

        # Call OpenAI API with the context and question
        LOGGER.debug('setting up stream')
        stream = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are given a set of research paper abstracts which the question will be about. You should respond to the question with relevant information from the abstracts and any connections you may make from them. If you cannot answer the question, state why and what more context you need, do not make up any information."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\nAnswer:"}
            ],
            stream=True,
            max_tokens=300  # Adjust the number of tokens to get a longer answer
        )
        print('Answer: ',)
        response = ''
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content
                print(chunk.choices[0].delta.content, end="")
        return response, relevant_citations

    while True:
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        if question.strip() == '':
            continue
        LOGGER.debug(f'Asking question: {question}')
        answer, relevant_citations = answer_question_with_gpt(
            question, documents, citations, index)
        print("\n\nRELEVANT CITATIONS:\n    * " + '\n    * '.join(relevant_citations))
        print('\n')


if __name__ == '__main__':
    load_dotenv()
    main()
