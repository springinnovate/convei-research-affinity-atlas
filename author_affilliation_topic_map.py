"""NLP Topic mapping given abstract files"""
import argparse
import glob
import logging
import pickle
import os

from scipy.optimize import differential_evolution
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


class ChunkerLdaModel(LdaModel):
    def __init__(self, *args, **kwargs):
        # Set a custom chunk size
        kwargs['chunksize'] = 100000
        super().__init__(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description='Parse and process affiliation data')
    parser.add_argument(
        'affiliation_file', help='Path to "type0: string..etc, file')
    parser.add_argument('topic_type', help='university, or research...')
    parser.add_argument(
        '--eps', type=float)
    parser.add_argument(
        '--min_samples', type=int)
    parser.add_argument(
        '--min_cores', type=int)
    parser.add_argument(
        '--num_words', type=int)
    parser.add_argument(
        '--differential_evolution', action='store_true')
    # 2) the natural habitat eo characteristics in and out of polygon
    # 3) proportion of area outside of polygon
    args = parser.parse_args()

    ensamble_path = f'{args.topic_type}_affiliation_ensamble.pkl'
    if not os.path.exists(ensamble_path):
        affiliation_list = []
        with open(args.affiliation_file, 'r', encoding='utf-8') as file:
            for line in file:
                topic_type = line.split(':')[0]
                if topic_type == args.topic_type:
                    affiliation = ':'.join(line.split(':')[1:]).rstrip().lstrip()
                    affiliation_list.append(affiliation)

        # Tokenize, remove stopwords and lemmatize the documents.
        ensemble = topic_map(affiliation_list)
        with open(ensamble_path, 'wb') as file:
            pickle.dump(ensemble, file)
    else:
        with open(ensamble_path, 'rb') as file:
            ensemble = pickle.load(file)

    shape = ensemble.asymmetric_distance_matrix.shape
    without_diagonal = ensemble.asymmetric_distance_matrix[~numpy.eye(shape[0], dtype=bool)].reshape(shape[0], -1)

    bounds = [
        (0.0, 1.0), # eps
        (1, 10), # min samples
        (1, 10), # num cores
        ]

    if args.differential_evolution:
        result = differential_evolution(
            recluster,
            bounds,
            (ensemble,),
            strategy='best1bin',
            maxiter=1000,
            popsize=15,
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.7,
            workers=-1)
        print(result)
    else:
        # Print the top words for each topic
        ensemble.recluster(
            eps=args.eps,
            min_samples=int(args.min_samples),
            min_cores=int(args.min_cores))
        top_words_with_probability = get_top_words_for_each_topic(ensemble, num_words=args.num_words)
        print(without_diagonal.min(), without_diagonal.mean(), without_diagonal.max())
        print(top_words_with_probability)
        csv_file = open(f'{args.topic_type}_topics.csv', 'w')
        for topic_num, top_words in enumerate(top_words_with_probability):
            prob_array = [prob for _, prob in top_words]
            percentile_value = numpy.percentile(prob_array, 90)
            quoted_top_words = [
                f"{word}: {probability}" for (word, probability) in top_words
                if probability >= percentile_value]
            print(f"Topic {topic_num}: {', '.join(quoted_top_words)}")
            csv_file.write(
                f'{quoted_top_words[0].split(":")[0]},' +
                ','.join(quoted_top_words))
            csv_file.write('\n')
        csv_file.close()


def recluster(x, ensemble):
    eps, min_samples, min_cores = x
    ensemble.recluster(
        eps=eps,
        min_samples=int(min_samples),
        min_cores=int(min_cores))

    top_words_with_probability = get_top_words_for_each_topic(ensemble, num_words=10)
    return -len(top_words_with_probability)


def get_top_words_for_each_topic(ensemble, num_words=10):
    top_words_per_topic = []
    # for topic_idx, word_distribution in enumerate(ensemble.get_topics()):
    #     top_words = [ensemble.id2word[i] for i in sorted(range(len(word_distribution)), key=lambda i: -word_distribution[i])[:num_words]]
    #     top_words_per_topic.append((topic_idx, top_words))

    for topic_idx, word_distribution in enumerate(ensemble.get_topics()):
        # Sort words in topic based on their probability and select top words
        sorted_words = sorted(enumerate(word_distribution), key=lambda x: -x[1])[:num_words]

        # Retrieve the word and its probability for the top words
        top_words_with_probability = [
            (ensemble.id2word[word_idx], prob)
            for word_idx, prob in sorted_words]
        top_words_per_topic.append(top_words_with_probability)

    return top_words_per_topic


def scrub_docs(docs, min_length):
    # Split the documents into tokens.
    # NLTK Stop words
    docs_copy = docs.copy()
    stop_words = set(stopwords.words('english'))
    # Add any custom stopwords
    custom_stopwords = {
        'this', 'study', 'paper', 'this_work', 'recent', 'have', 'le',
        'recently'}

    stop_words = stop_words.union(custom_stopwords)

    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(docs_copy)):
        docs_copy[idx] = docs_copy[idx].lower()  # Convert to lowercase.
        docs_copy[idx] = tokenizer.tokenize(docs_copy[idx])  # Split into words.
        docs_copy[idx] = [token for token in docs_copy[idx] if token not in stop_words]

    # Remove numbers, but not words that contain numbers.
    docs_copy = [[token for token in doc if not token.isnumeric()] for doc in docs_copy]
    # Remove words that are too short
    docs_copy = [[token for token in doc if len(token) >= min_length] for doc in docs_copy]

    lemmatizer = WordNetLemmatizer()
    docs_copy = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs_copy]
    return docs_copy


def make_bigram(docs):
    # Add bigrams and trigrams to docs (only ones that appear 20 times or more).
    bigram = Phrases(
        docs, min_count=20, connector_words=phrases.ENGLISH_CONNECTOR_WORDS)
    return bigram
    # for idx in range(len(docs)):
    #     for token in bigram[docs[idx]]:
    #         if '_' in token:
    #             # Token is a bigram, add to document.
    #             docs[idx].append(token)


def topic_map(docs):
    # Split the documents into tokens.
    # NLTK Stop words

    docs = scrub_docs(docs, 3)

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
