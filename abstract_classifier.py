import argparse
import glob
import logging
import os
import pickle
import random
import time
import torch
import re
import warnings

#from optimum.pipelines import pipeline
from transformers import pipeline
from flair.data import Sentence
from flair.models import SequenceTagger

from transformers import AutoTokenizer, AutoModelForTokenClassification


warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data")

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)


def main():
    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser(description='Affiliation classifier')
    parser.add_argument('bib_file', help='path to bibliography list')
    parser.add_argument('abstract_tag_file', help='path to abstract tags')
    parser.add_argument('--target_path', help='target classified table')
    args = parser.parse_args()

    print('load bib_file')
    affiliation_set = set()
    with open(args.bib_file, 'r', encoding='utf-8') as file:
        affiliation_str = None
        abstract_str = None
        article_id = None
        for line in file:
            try:
                #if abstract_str is not None or affiliation_str is not None:
                #    print(f'WARNING: "{article_id}" had these were left over', abstract_str, affiliation_str)
                article_id = re.search('@[^{]+{(.*),', line).group(1)
                if article_id is None:
                    print(f'ERROR: {line}')
                affiliation_str = None
                abstract_str = None
                continue
            except:
                pass
            if 'abstract =' in line:
                abstract_str = re.search('{(.*)}', line).group(1)
            elif 'affiliations =' in line:
                affiliation_str = re.search('{(.*)}', line).group(1)
            if abstract_str and affiliation_str:
                if article_id is None:
                    print(f'ERROR: {abstract_str}')
                affiliation_set.add((article_id, affiliation_str, abstract_str))
                article_id = None
                affiliation_str = None
                abstract_str = None

    print('load candidate_labels')
    print(len(affiliation_set))
    with open(args.abstract_tag_file, 'r') as file:
        candidate_labels = ', '.join([
            v for v in file.read().split('\n')
            if len(v) > 0])
    print(candidate_labels)

    batch_size = 10
    classifier = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        device=0, batch_size=batch_size, truncation=True)
    print(classifier.model.device)

    target_path = args.target_path
    if target_path is None:
        target_path = '%s_classified%s' % os.path.splitext(args.affiliation_pickle_list)


    total_time = 0
    events = 0
    with open(target_path, 'w', encoding='utf-8') as file:
        print(f'opening {target_path} for writing {len(affiliation_set)} affiliations')
        start_time = time.time()
        def affiliation_generator():
            for _, affiliation_str, _ in affiliation_set:
                yield affiliation_str
        index = 1
        for (article_id, affiliation_str, abstract_str), result in zip(
                affiliation_set,
                classifier(affiliation_generator(), candidate_labels, multi_label=True)):
            file.write(f'{article_id}\n{affiliation_str}\n{abstract_str}\n')
            for label, score in zip(result['labels'], result['scores']):
                file.write(f'{label}: {score}\n')
            file.write('\n')
            file.flush()
            current_time = (time.time()-start_time)
            events += 1
            total_time += current_time
            print(f'({index}/{len(affiliation_set)} took {current_time}s to tag {article_id} (time left) {total_time/events*(len(affiliation_set)-index)}')
            start_time = time.time()
            index += 1


if __name__ == '__main__':
    main()
