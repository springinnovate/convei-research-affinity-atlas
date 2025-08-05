"""NLP Topic mapping given abstract files"""
import argparse
import collections
import glob
import logging

from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas
import numpy
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
    topic_to_word_map = collections.defaultdict(lambda: collections.defaultdict(float))
    topic_count = collections.defaultdict(int)
    topic_prob_sum = collections.defaultdict(float)
    for _, row in topic_table.iterrows():
        if not isinstance(row[0], str):
            continue
        topic = row[0]
        if topic == '':
            continue
        if topic_count[topic] == 0:
            global_topics.append(topic)
        topic_count[topic] += 1
        for v in row[1:]:
            if not isinstance(v, str):
                continue
            word = v.split(':')[0]
            prob = float(v.split(':')[1])

            topic_to_word_map[topic][word] += prob
            topic_prob_sum[topic] += prob
    for topic, word_map in topic_to_word_map.items():
        for word in word_map:
            word_map[word] /= topic_prob_sum[topic]
    file_list = [
        path
        for path in glob.glob(args.abstract_file_pattern)]
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

    scrubbed_docs = scrub_docs(abstract_list, 4)

    correlation_matrix = []
    for abstract, doc in zip(abstract_list, scrubbed_docs):
        prob_vector = []
        for topic in global_topics:
            running_sum = 0.0
            for word, prob in topic_to_word_map[topic].items():
                norm_prob = prob/topic_count[topic]
                running_sum += doc.count(word)*norm_prob
            prob_vector.append(running_sum)
        correlation_matrix.append(prob_vector)


    correlation_matrix = numpy.array(correlation_matrix)
    print(correlation_matrix)
    linkage_matrix = linkage(correlation_matrix.transpose(), method='ward')
    print(linkage_matrix.shape)
    # Plot a dendrogram to visualize the clustering
    # Example usage
    print_dendrogram(linkage_matrix, global_topics)
    dendrogram(linkage_matrix, labels=global_topics, orientation='right', color_threshold=5)
    plt.show()


def print_dendrogram(linkage_matrix, labels, level=0, index=-1):
    print(index)
    if index == -1:
        index = linkage_matrix.shape[0] + linkage_matrix.shape[1] - 2

    if index < len(labels):
        print('    ' * level + '- ' + labels[index])
    else:
        left = int(linkage_matrix[index - len(labels), 0])
        right = int(linkage_matrix[index - len(labels), 1])

        print('    ' * level + '+ Cluster ' + str(index))

        print_dendrogram(linkage_matrix, labels, level + 1, left)
        print_dendrogram(linkage_matrix, labels, level + 1, right)


if __name__ == '__main__':
    main()
