"""
what are the most to least common affiliation tags?
which affiliation tags are working with which other ones? (like when is water working with disasters when the person with water in their affiliation doesn't have disasters in their affiliation and vice versa) what are the strength of those connections?
"""
import argparse
import collections


def main():
    parser = argparse.ArgumentParser(description='Parse and process affiliation data')
    parser.add_argument(
        'classified_affiliation_file',
        help='Path file with affilliation, name, and tags')
    args = parser.parse_args()

    affiliation_map = {}
    tag_to_affiliation_map = collections.defaultdict(list)
    with open(args.classified_affiliation_file, 'rb', encoding='utf-8') as file:
        while True:
            affiliation_dict = {}
            affiliation = file.readline()
            affiliation_dict['affiliation'] = affiliation
            if affiliation == '':
                break
            short_affiliation = file.readline()
            affiliation_dict['tags'] = {}
            while True:
                try:
                    tag, prob = file.readline().split(':')
                    affiliation_dict['tags'][tag] = prob
                    tag_to_affiliation_map[tag].append(affiliation)
                except ValueError:
                    # can't split, so blank line, next section
                    break
                affiliation_map[affiliation] = affiliation_dict
    for tag, affiliation_list in tag_to_affiliation_map.items():
        print(f'{tag}: {len(affiliation_list)}')


if __name__ == '__main__':
    main()
