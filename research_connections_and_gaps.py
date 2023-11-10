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
from topic_map import ARTICLES_TO_DROP
from topic_map import scrub_docs

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Parse and process abstract files')
    parser.add_argument('abstract_file_pattern', help='pth to abstracts')
    parser.add_argument('topic_table', help='path to topic table')
    # 2) the natural habitat eo characteristics in and out of polygon
    # 3) proportion of area outside of polygon
    args = parser.parse_args()

    file_list = [
        path
        for path in glob.glob(args.abstract_file_pattern)]
    print(file_list)
    abstract_list = []
    abstract_key = 'AB'
    for file_path in file_list:
        with open(file_path, 'rb') as file:
            LOGGER.info(f'processing {file_path}')
            for line in file:
                line = line.decode('UTF-8')
                if line.startswith(abstract_key):
                    line = line.split('-')[1]
                    if not any(word in line for word in ARTICLES_TO_DROP):
                        abstract_list.append(line)
                    else:
                        pass
            LOGGER.info(f'{len(abstract_list)} records so far')

    scrubbed_docs = scrub_docs(abstract_list)
    print(scrubbed_docs[0])


if __name__ == '__main__':
    main()
