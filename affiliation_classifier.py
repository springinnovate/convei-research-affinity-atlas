import argparse
import glob
import logging
import re
import time

from transformers import pipeline
import torch

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Affiliation classifier')
    parser.add_argument('bib_file_pattern', help='path to SCOPUS bib file')
    parser.add_argument('affiliation_tag_path', help='Path to file with affiliation tags.')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')
    parser.add_argument('--target_path', help='Path to target output file.')
    args = parser.parse_args()
    device = None
    if args.cuda and not torch.cuda.is_available():
        raise ValueError('CUDA not supported')
    else:
        device = 0

    print('load affilation_list')
    affiliation_set = set()
    for bib_file in glob.glob(args.bib_file_pattern):
        with open(bib_file, 'r', encoding='utf-8') as file:
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
                    affiliation_set |= set([x.strip() for x in affiliation_str.split(';')])


    with open(args.affiliation_tag_path, 'r', encoding='utf-8') as file:
        candidate_labels = ', '.join([x.strip() for x in file.read().split('\n') if x.strip() != ''])

    batch_size = 10
    classifier = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        device=device, batch_size=batch_size, truncation=True)

    events = 0
    total_time = 0
    with open(args.target_path, 'w', encoding='utf-8') as file:
        start_time = time.time()
        def affiliation_generator():
            for _, affiliation_str, _ in affiliation_set:
                yield affiliation_str
        index = 1
        for affiliation_str, result in zip(
                affiliation_set,
                classifier(affiliation_generator(), candidate_labels, multi_label=True)):
            file.write(f'{affiliation_str}\n')
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
