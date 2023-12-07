import argparse
import glob
import logging
import os
import pickle
import random

from transformers import pipeline
from flair.data import Sentence
from flair.models import SequenceTagger


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
    tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")

    with open('%s_classified%s' % os.path.splitext(args.affiliation_pickle_list), 'w', encoding='utf-8') as file:
        for affiliation in affilation_list:
            print(f'processing {affiliation}')
            tagged_affiliation = Sentence(affiliation)
            tagger.predict(tagged_affiliation)
            org_components = ''
            for entity in tagged_affiliation.get_spans('ner'):
                if len(entity.text) > 5 and entity.get_label().value == 'ORG':
                    org_components += f'{entity.text} '
            file.write(f'{affiliation}\n')
            file.write(f'{org_components}\n')
            if len(org_components) == 0:
                org_components = affiliation
            result = classifier(
                org_components, candidate_labels, multi_label=True)
            for label, score in zip(result['labels'], result['scores']):
                if score < 0.8:
                    break
                file.write(f'{label}: {score}\n')
            file.write('\n')
            file.flush()


if __name__ == '__main__':
    main()
