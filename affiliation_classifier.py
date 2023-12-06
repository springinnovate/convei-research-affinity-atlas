import argparse
import pickle
import logging

from transformers import pipeline


logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Affiliation classifier')
    parser.add_argument('affiliation_pickle_list', help='path to affiliation pickle list')
    args = parser.parse_args()

    print('load affilation_list')
    with open(args.affiliation_pickle_list, 'rb') as file:
        affilation_list = pickle.load(file)

    print('load candidate_labels')
    with open('data/candidate_labels_lig.txt', 'r') as file:
        candidate_labels = ','.join([
            v for v in file.read().split('\n')
            if len(v) > 0])
    print(candidate_labels)

    classifier = pipeline(
        "zero-shot-classification", model="facebook/bart-large-mnli")
    affiliation = next(iter(affilation_list))
    print(f'classify {affiliation}')
    result = classifier(affiliation, candidate_labels)
    print(result)


if __name__ == '__main__':
    main()
