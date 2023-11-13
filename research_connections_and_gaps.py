"""NLP Topic mapping given abstract files"""
import argparse
import glob
import logging
import pickle
import os

from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas
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
from topic_map import ARTICLES_TO_DROP
from topic_map import scrub_docs

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.WARN)
logging.getLogger('PIL').setLevel(logging.WARN)

def main():
    parser = argparse.ArgumentParser(description='Parse and process abstract files')
    parser.add_argument('abstract_file_pattern', help='pth to abstracts')
    parser.add_argument('topic_table', help='path to topic table')
    # 2) the natural habitat eo characteristics in and out of polygon
    # 3) proportion of area outside of polygon
    args = parser.parse_args()
    topic_table = pandas.read_csv(args.topic_table, header=None)
    print(topic_table)

    global_topics = []
    topic_to_word_map = {}
    for _, row in topic_table.iterrows():
        if not isinstance(row[0], str):
            continue
        topic = row[0]
        if topic == '':
            continue
        global_topics.append(topic)

        words = [
            (v.split(':')[0], float(v.split(':')[1][:-1]))
            for v in row[1:] if isinstance(v, str)]
        topic_to_word_map[topic] = words
    file_list = [
        path
        for path in glob.glob(args.abstract_file_pattern)]
    file_list = file_list[0:1]
    print(file_list)
    print(global_topics)

    abstract_list = []
    abstract_key = 'AB'
    for file_path in file_list:
        with open(file_path, 'rb') as file:
            LOGGER.info(f'processing {file_path}')
            for line in file:
                line = line.decode('UTF-8')
                if line.startswith(abstract_key):
                    line = '-'.join(line.split('-')[1:])
                    if not any(word in line for word in ARTICLES_TO_DROP):
                        abstract_list.append(line)
                    else:
                        pass
            LOGGER.info(f'{len(abstract_list)} records so far')

    scrubbed_docs = scrub_docs(abstract_list)

    correlation_matrix = []
    for abstract, doc in zip(abstract_list, scrubbed_docs):
        prob_vector = []
        for topic in global_topics:
            running_sum = 0.0
            for word, prob in topic_to_word_map[topic]:
                running_sum += doc.count(word)*prob
            prob_vector.append(running_sum)
        correlation_matrix.append(prob_vector)
        # for debugging to see what the top topics ard
        # print(abstract)
        # print(
        #     list(
        #         sorted([
        #             f'{word}: {prob}'
        #             for word, prob in zip(global_topics, prob_vector) if prob > 0.0],
        #             key=lambda x: -float(x.split(':')[1]))))

    correlation_matrix = numpy.array(correlation_matrix)
    print(correlation_matrix)
    linkage_matrix = linkage(correlation_matrix.transpose(), method='ward')
    print(linkage_matrix.shape)
    # Plot a dendrogram to visualize the clustering
    dendrogram(linkage_matrix, labels=global_topics, orientation='right', color_threshold=5)
    plt.show()


if __name__ == '__main__':
    main()
