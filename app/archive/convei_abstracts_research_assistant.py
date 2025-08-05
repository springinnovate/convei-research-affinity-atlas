"""Convert input text into numerical vectors that can be understood by LLM."""
import chardet
import datetime
import faiss
import glob
import hashlib
import logging
import numpy
import os
import pickle
import re
import torch

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import openai
import textwrap
import tiktoken

logging.basicConfig(
    level=logging.INFO,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.INFO)
logging.getLogger('sentence_transformers').setLevel(logging.WARN)
logging.getLogger('httpx').setLevel(logging.WARN)
LOGGER = logging.getLogger(__name__)

#GPT_MODEL = 'gpt-4o'
#GPT_MODEL, MAX_TOKENS, MAX_RESPONSE_TOKENS = 'gpt-3.5-turbo', 16000, 4000
GPT_MODEL, MAX_TOKENS, MAX_RESPONSE_TOKENS = 'gpt-4o', 20000, 4000
ENCODING = tiktoken.encoding_for_model(GPT_MODEL)
TOP_K = 100

SYSTEM_CONTEXT = "You will be asked a synthesis question about a set of research abstracts identified by their reference index. Respond to the question with relevant information from those abstracts and synthesis connections you make from them. Relevant information in your answer from the abstract should be cited as `(reference index: {index})` If you cannot answer the question, state why and do not make up any information."

BODY_TAG = 'body'
CITATION_TAG = 'citation'

DATA_STORE = [
    r"data/web_of_science_query/abstracts_2023_11_02/*.ris",
    r"data/scopus/*/*.bib"
]

LOG_DIR = 'convei_research_assistant_log'
CACHE_DIR = 'llm_cache'
for dirpath in [CACHE_DIR, LOG_DIR]:
    os.makedirs(dirpath, exist_ok=True)


def token_count(context):
    return len(ENCODING.encode(context))


def trim_context(context, max_tokens):
    tokens = ENCODING.encode(context)
    tokens = tokens[:max_tokens]
    trimmed_context = ENCODING.decode(tokens)
    return trimmed_context


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

    current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    streaming_log_path = os.path.join(LOG_DIR, f'{current_time}.txt')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)

    file_paths = [
        file_path
        for file_pattern in DATA_STORE
        for file_path in glob.glob(file_pattern)]

    file_hash = generate_hash(file_paths)
    body_citation_list_pkl_path = os.path.join(CACHE_DIR, f'{file_hash}.pkl')
    fiass_path = os.path.join(CACHE_DIR, f'{file_hash}.faiss')

    if not os.path.exists(body_citation_list_pkl_path):
        article_list = []
        for file_path in file_paths:
            article_list += parse_file(file_path)

        LOGGER.debug(f'detecting duplicates in {len(article_list)} articles')
        document_list = []
        citation_list = []
        seen_documents = set()
        for document, citation in zip(
                [article[BODY_TAG] for article in article_list],
                [article[CITATION_TAG] for article in article_list]):
            hash_object = hashlib.sha256(document.encode('utf-8'))
            hash_hex = hash_object.hexdigest()
            if hash_hex not in seen_documents:
                document_list.append(document)
                citation_list.append(citation)
                seen_documents.add(hash_hex)

        LOGGER.debug(
            f'pickling cleaned document/citation list '
            f'{body_citation_list_pkl_path}')
        with open(body_citation_list_pkl_path, 'wb') as file:
            pickle.dump((document_list, citation_list), file)

        LOGGER.debug('embedding for indexing')
        document_embeddings = embedding_model.encode(
            document_list, convert_to_tensor=True)
        LOGGER.debug('indexing')
        index = faiss.IndexFlatL2(document_embeddings.shape[1])
        index.add(document_embeddings.cpu().numpy())
        faiss.write_index(index, fiass_path)
    else:
        with open(body_citation_list_pkl_path, 'rb') as file:
            document_list, citation_list = pickle.load(file)
        index = faiss.read_index(fiass_path)

    def answer_question_with_gpt(
            question, abstract_list, citation_list, index):
        try:
            stream = OPENAI_CLIENT.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": "Given the following question, restructure it so it's a statement or just a set of words that will better match a research paper abstract in a dense vector index that could answer that question."},
                    {"role": "user", "content": f"nQuestion: {question}"}
                ],
                stream=True,
                max_tokens=token_count(question)*2
            )
            search_phrase = ''
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    search_phrase += chunk.choices[0].delta.content

            search_phrase_embedding = embedding_model.encode(
                question, convert_to_tensor=True).cpu().numpy()
            if len(search_phrase_embedding.shape) == 1:
                search_phrase_embedding = search_phrase_embedding.reshape(
                    1, -1)
            distances, indices = index.search(search_phrase_embedding, TOP_K)
            relevant_abstracts = [abstract_list[idx] for idx in indices[0]]
            relevant_citations = [citation_list[idx] for idx in indices[0]]

            context = "\n\n".join([
                f'(Reference index: {index}), Abstract: {context}'
                for index, context in enumerate(relevant_abstracts)])
            remaining_tokens = (
                MAX_TOKENS -
                MAX_RESPONSE_TOKENS -
                token_count(SYSTEM_CONTEXT) -
                token_count(question))
            context = trim_context(context, max_tokens=remaining_tokens)
            context_counts = context.count("Reference index: ")

            stream = OPENAI_CLIENT.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_CONTEXT},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\nAnswer:"}
                ],
                stream=True,
                max_tokens=MAX_RESPONSE_TOKENS
            )
            response = (
                f'\n(Analyzing the top {context_counts} relevant abstracts found using the modified search phrase: "{search_phrase}"")\n\n')
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    response += chunk.choices[0].delta.content
            # Find all matches
            pattern = (
                r'(references? )?(index(es)?)?: ?'
                r'(\{\d+(,? ?\d+)*\}|\d+|,| )+')
            matches = re.finditer(pattern, response, re.IGNORECASE)

            # Iterate over matches and collect numbers
            reference_indexes = set()
            for match in matches:
                # This gets the entire matched text, then we need to further extract reference_indexes
                full_match_text = match.group(0)
                # Extract reference_indexes from the matched text
                reference_indexes |= set([int(x) for x in re.findall(r'\d+', full_match_text)])

            processed_indexes = set()
            referenced_citations = ''
            for index in sorted(reference_indexes):
                index = int(index)
                if index in processed_indexes:
                    continue
                processed_indexes.add(index)
                referenced_citations += (
                    f'\t{index}.  {relevant_citations[index]}\n'
                    f'\t\t{relevant_abstracts[index]}\n\n')
            return response, referenced_citations
        except openai.APIError as e:
            print(
                f'There was an error, try your question again.\nError: {e}')

    while True:
        question = input("\n\n\nPLEASE ASK A QUESTION (or type 'exit' to exit): ")
        if question.lower() in ['exit', 'quit', 'q']:
            break
        if question.strip() == '':
            continue
        answer, relevant_references = answer_question_with_gpt(
            question, document_list, citation_list, index)

        print(f'\n{answer}\n')
        print(f'See {streaming_log_path} for reference index citations')

        with open(streaming_log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(
                '************************\n'
                f'Question: {question}\n\nAnswer: {answer}\n\n'
                f'References:\n{relevant_references}\n')


if __name__ == '__main__':
    load_dotenv()
    OPENAI_CLIENT = openai.OpenAI()

    # for log_file_path in glob.glob(os.path.join(LOG_DIR, '*.txt')):
    #     print(log_file_path)
    #     with open(log_file_path, 'r') as file:
    #         for line in file:
    #             if not line.startswith('Question: '):
    #                 continue
    #             question = line.split('Question: ')[1]
    #             print(f'Original question: {question}')
    #             stream = OPENAI_CLIENT.chat.completions.create(
    #                 model=GPT_MODEL,
    #                 messages=[
    #                     {"role": "system", "content": "Given the following question, restructure it so it's a statement or just a set of words that will better match a research paper abstract in a dense vector index that could answer that question."},
    #                     {"role": "user", "content": f"nQuestion: {question}"}
    #                 ],
    #                 stream=True,
    #                 max_tokens=token_count(question)*2
    #             )
    #             response = ''
    #             for chunk in stream:
    #                 if chunk.choices[0].delta.content is not None:
    #                     response += chunk.choices[0].delta.content
    #             print(f'Rephrased question: {response}\n\n')
    #             break
    #     break
    main()
