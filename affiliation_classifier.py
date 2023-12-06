import argparse
import pickle

from transformers import pipeline


def main():
    parser = argparse.ArgumentParser(description='Affiliation classifier')
    parser.add_argument('affiliation_pickle_list', help='path to affiliation pickle list')
    args = parser.parse_args()

    print('load affilation_list')
    with open(args.affiliation_pickle_list, 'rb') as file:
        affilation_list = pickle.load(file)

    print('load candidate_labels')
    with open('data/candidate_labels.txt', 'r') as file:
        candidate_labels = [
            v for v in file.read().split('\n')
            if len(v) > 0]

    classifier = pipeline(
        "zero-shot-classification", model="facebook/bart-large-mnli")
    affiliation = next(iter(affilation_list))
    print(f'classify {affiliation}')
    result = classifier(affiliation, candidate_labels)
    print(result)


if __name__ == '__main__':
    main()
