"""NLP Topic mapping given abstract files"""
import argparse
import glob
import logging
import pickle
import os

import numpy
from gensim.models import EnsembleLda
from gensim.models import Phrases
from gensim.models import phrases
from gensim import corpora
from gensim.models import LdaModel
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)

ARTICLES_TO_DROP = ['breast', 'melanoma']


class ChunkerLdaModel(LdaModel):
    def __init__(self, *args, **kwargs):
        # Set a custom chunk size
        kwargs['chunksize'] = 100000
        super().__init__(*args, **kwargs)

def main():
    parser = argparse.ArgumentParser(description='Parse and process abstract files')
    parser.add_argument(
        'abstract_file_pattern', nargs='+', help=(
            'Path(s) or pattern(s) to abstract files of the form:\n'
            '0KEY1 - VALUE1\n'
            '0KEY2 - VALUE2\n'
            '0KEY3 - VALUE3\n'
            '0KEY4 - VALUE3\n'
            '<blank line>\n'
            '0KEY1 - VALUE1\n'
            '0KEY2 - VALUE2\n'
            '...'))
    # 2) the natural habitat eo characteristics in and out of polygon
    # 3) proportion of area outside of polygon
    args = parser.parse_args()

    ensamble_path = 'ensamble.pkl'
    if not os.path.exists(ensamble_path):
        file_list = [
            path
            for pattern in args.abstract_file_pattern
            for path in glob.glob(pattern)]
        LOGGER.debug(file_list)

        abstract_list = []
        abstract_key = 'AB'
        for file_path in file_list:
            with open(file_path, 'rb') as file:
                LOGGER.info(f'processing {file_path}')
                for line in file:
                    line = line.decode('UTF-8')
                    if line.startswith(abstract_key):
                        line = line.split('-')[1]
                        if not any(word in line for word in ARTICLES_TO_DROP):
                            abstract_list.append(line)
                        else:
                            pass
                LOGGER.info(f'{len(abstract_list)} records so far')

        # Tokenize, remove stopwords and lemmatize the documents.
        ensemble = topic_map(abstract_list)
        with open(ensamble_path, 'wb') as file:
            pickle.dump(ensemble, file)
    else:
        with open(ensamble_path, 'rb') as file:
            ensemble = pickle.load(file)

    shape = ensemble.asymmetric_distance_matrix.shape
    without_diagonal = ensemble.asymmetric_distance_matrix[~numpy.eye(shape[0], dtype=bool)].reshape(shape[0], -1)

    ensemble.recluster(eps=0.3, min_samples=2, min_cores=2)
    top_words_per_topic = get_top_words_for_each_topic(ensemble, num_words=10)

    # Print the top words for each topic
    print(without_diagonal.min(), without_diagonal.mean(), without_diagonal.max())
    print()
    for topic_num, top_words in top_words_per_topic:
        quoted_top_words = [f"'{word}'" for word in top_words]
        print(f"Topic {topic_num}: {', '.join(quoted_top_words)}")


def get_top_words_for_each_topic(ensemble, num_words=10):
    top_words_per_topic = []
    for topic_idx, word_distribution in enumerate(ensemble.get_topics()):
        top_words = [ensemble.id2word[i] for i in sorted(range(len(word_distribution)), key=lambda i: -word_distribution[i])[:num_words]]
        top_words_per_topic.append((topic_idx, top_words))

    for topic_idx, word_distribution in enumerate(ensemble.get_topics()):
        # Sort words in topic based on their probability and select top words
        sorted_words = sorted(enumerate(word_distribution), key=lambda x: -x[1])[:num_words]

        # Retrieve the word and its probability for the top words
        top_words_with_probability = [
            (ensemble.id2word[word_idx], prob)
            for word_idx, prob in sorted_words]

    return top_words_per_topic


def topic_map(docs):
    # Split the documents into tokens.
    # NLTK Stop words
    stop_words = set(stopwords.words('english'))
    # Add any custom stopwords
    custom_stopwords = {
        'this', 'study', 'paper', 'this_work', 'recent', 'have', 'le',
        'recently'}
    stop_words = stop_words.union(custom_stopwords)

    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(docs)):
        docs[idx] = docs[idx].lower()  # Convert to lowercase.
        docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.
        docs[idx] = [token for token in docs[idx] if token not in stop_words]

    # Remove numbers, but not words that contain numbers.
    docs = [[token for token in doc if not token.isnumeric()] for doc in docs]
    # Remove words that are too short
    docs = [[token for token in doc if len(token) > 3] for doc in docs]

    lemmatizer = WordNetLemmatizer()
    docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

    # Add bigrams and trigrams to docs (only ones that appear 20 times or more).
    bigram = Phrases(
        docs, min_count=20, connector_words=phrases.ENGLISH_CONNECTOR_WORDS)
    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)

    # Create a dictionary representation of the documents.
    dictionary = corpora.Dictionary(docs)
    dictionary.filter_extremes(no_below=20, no_above=0.5)  # This is optional, but it helps in refining the dictionary
    corpus = [dictionary.doc2bow(text) for text in docs]

    LOGGER.info('Number of unique tokens: %d' % len(dictionary))
    LOGGER.info('Number of documents: %d' % len(corpus))

    # Set training parameters.

    ensemble_workers = 4
    num_models = 8
    num_topics = 100
    passes = 20
    distance_workers = 4

    # Make an index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.

    ensemble = EnsembleLda(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        num_models=num_models,
        ensemble_workers=ensemble_workers,
        distance_workers=distance_workers,
        topic_model_class=ChunkerLdaModel
    )

    return ensemble


if __name__ == '__main__':
    main()
