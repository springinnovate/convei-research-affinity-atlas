import argparse
import glob
import logging
import os
import pickle
import random

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
    affilation_set = set()
    for affiliation_pickle_file in glob.glob(args.affiliation_pickle_list):
        with open(args.affiliation_pickle_list, 'r', encoding='utf-8') as file:
            for line in file:
                affilation_set.add(line.rstrip().lstrip())
    affilation_list = list(affilation_set)
    random.shuffle(affilation_list)
    print('load candidate_labels')
    with open('data/affiliation_tags.txt', 'r') as file:
        candidate_labels = ', '.join([
            v for v in file.read().split('\n')
            if len(v) > 0])
    print(candidate_labels)

    classifier = pipeline(
        "zero-shot-classification", model="facebook/bart-large-mnli")

    with open('%s_classified%s' % os.path.splitext(args.affiliation_pickle_list), 'w', encoding='utf-8') as file:
        for affiliation in affilation_list:
            print(f'processing {affiliation}')
            file.write(f'{affiliation}\n')
            result = classifier(
                affiliation, candidate_labels, multi_label=True)
            score_sum_threshold = 0.9*sum(result['scores'])

            for label, score in zip(result['labels'], result['scores']):
                score_sum_threshold -= score
                file.write(f'{label}: {score}\n')
                if score_sum_threshold < 0:
                    break
            file.flush()


if __name__ == '__main__':
    main()
