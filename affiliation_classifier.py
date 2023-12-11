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
    batch_size = None
    if args.cuda:
        if not torch.cuda.is_available():
            raise ValueError('CUDA not supported')
        else:
            device = 0
            batch_size = 10

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
                    affiliation_set |= set([
                        x.strip() for x in affiliation_str.split(';')
                        if x.strip() != ''])

    def affiliation_generator():
        for affiliation_str in affiliation_set:
            print(f'affilation: {affiliation_str}')
            yield affiliation_str

    token_tagger = pipeline(
        "ner", model='dslim/bert-base-NER', device=device, batch_size=batch_size)
    scrubbed_affilliation_list = []
    for affiliation_str, result in zip(affiliation_set, token_tagger(affiliation_generator())):
        org_components = ''
        for entity in result:
            if entity['entity'][2:] in ['ORG', 'MIS']:
                word = entity['word']
                print(f'WORD: {word}')
                if not word.startswith('##'):
                    space = ' '
                else:
                    word = word[2:]
                org_components += f'{space}{word}'
        if len(org_components) == 0:
            org_components = affiliation_str
        scrubbed_affilliation_list.append((affiliation_str.strip(), org_components.strip()))
        print(scrubbed_affilliation_list[-1])

    def affiliation_generator():
        for _, affiliation_str in scrubbed_affilliation_list:
            print(f'affilation: {affiliation_str}')
            yield affiliation_str

    with open(args.affiliation_tag_path, 'r', encoding='utf-8') as file:
        candidate_labels = ', '.join([x.strip() for x in file.read().split('\n') if x.strip() != ''])

    classifier = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        device=device, batch_size=batch_size, truncation=True)

    events = 0
    total_time = 0
    print(f'affillition labels: {candidate_labels}')
    with open(args.target_path, 'w', encoding='utf-8') as file:
        start_time = time.time()

        index = 1
        for (affiliation_str, _), result in zip(
                scrubbed_affilliation_list,
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
