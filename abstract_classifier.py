import argparse
import glob
import logging
import os
import pickle
import random
import time
import re

from transformers import pipeline
from flair.data import Sentence
from flair.models import SequenceTagger

from transformers import AutoTokenizer, AutoModelForTokenClassification


logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Affiliation classifier')
    parser.add_argument('bib_file', help='path to bibliography list')
    parser.add_argument('abstract_tag_file', help='path to abstract tags')
    parser.add_argument('--target_path', help='target classified table')
    args = parser.parse_args()

    print('load bib_file')
    affilation_set = set()
    with open(args.bib_file, 'r', encoding='utf-8') as file:
        affiliation_str = None
        abstract_str = None
        article_id = None
        for line in file:
            if '@ARTICLE{' in line:
                if abstract_str is not None or affiliation_str is not None:
                    print(f'WARNING: "{article_id}" had these were left over', abstract_str, affiliation_str)
                article_id = re.search('@ARTICLE{(.*),', line).group(1)
                affiliation_str = None
                abstract_str = None
                continue
            elif 'abstract =' in line:
                abstract_str = re.search('{(.*)}', line).group(1)
            elif 'affiliations =' in line:
                affiliation_str = re.search('{(.*)}', line).group(1)
            if abstract_str and affiliation_str:
                affilation_set.add((article_id, affiliation_str, abstract_str))
                article_id = None
                affiliation_str = None
                abstract_str = None

    print('load candidate_labels')
    print(len(affilation_set))
    with open(args.abstract_tag_file, 'r') as file:
        candidate_labels = ', '.join([
            v for v in file.read().split('\n')
            if len(v) > 0])
    print(candidate_labels)

    classifier = pipeline(
        "zero-shot-classification", model="facebook/bart-large-mnli")

    target_path = args.target_path
    if target_path is None:
        target_path = '%s_classified%s' % os.path.splitext(args.affiliation_pickle_list)

    with open(target_path, 'w', encoding='utf-8') as file:
        for article_id, affiliation_str, abstract_str in affilation_set:
            start_time = time.time()
            print(f'processing {article_id}')

            file.write(f'{article_id}\n{affiliation_str}\n{abstract_str}\n')
            result = classifier(
                abstract_str, candidate_labels, multi_label=True)
            for label, score in zip(result['labels'], result['scores']):
                file.write(f'{label}: {score}\n')
            file.write('\n')
            file.flush()
            print(f'took {time.time()-start_time}s to tag')


if __name__ == '__main__':
    main()
