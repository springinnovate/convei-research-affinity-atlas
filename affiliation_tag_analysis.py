"""
what are the most to least common affiliation tags?
which affiliation tags are working with which other ones? (like when is water working with disasters when the person with water in their affiliation doesn't have disasters in their affiliation and vice versa) what are the strength of those connections?
"""
import argparse
import collections
import re

def main():
    parser = argparse.ArgumentParser(description='Parse and process affiliation data')
    parser.add_argument(
        'classified_affiliation_file',
        help='Path file with affiliation, name, and tags')
    parser.add_argument(
        'bib_file', help='path to raw bibilography file')
    parser.add_argument('--filter', nargs='+')
    parser.add_argument('--show_names', action='store_true')
    parser.add_argument('--confidence_threshold', type=float)
    parser.add_argument('--show_related', action='store_true')
    args = parser.parse_args()

    # give an affiliation and see all the others associated with it
    affiliation_sets = collections.defaultdict(list)
    with open(args.bib_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.lstrip()
            if not line.startswith('affiliations'):
                continue
            affiliation_list = set([
                x.lstrip().rstrip() for x in
                re.search(r'\{(.*)\}', line).group(1).split(';')])
            for affiliation in affiliation_list:
                affiliation_sets[affiliation].append(affiliation_list)

    affiliation_map = {}
    tag_to_affiliation_map = collections.defaultdict(list)
    with open(args.classified_affiliation_file, 'r', encoding='utf-8') as file:
        while True:
            affiliation_dict = {}
            affiliation = file.readline().lstrip().rstrip()
            affiliation_dict['affiliation'] = affiliation
            if affiliation == '':
                break
            short_affiliation = file.readline()
            affiliation_dict['tags'] = {}
            while True:
                try:
                    tag, prob = file.readline().split(':')
                    if args.confidence_threshold and float(prob) < args.confidence_threshold:
                        continue
                    affiliation_dict['tags'][tag] = prob
                    tag_to_affiliation_map[tag].append(affiliation)
                except ValueError:
                    # can't split, so blank line, next section
                    break
                affiliation_map[affiliation] = affiliation_dict

    valid_affiliation_set = set()
    if not args.filter:
        for tag, affiliation_list in sorted(
                tag_to_affiliation_map.items(),
                key=lambda x: -len(x[1])):
            print(f'{tag}: {len(affiliation_list)}')
    else:
        tag_count = collections.defaultdict(int)
        for affiliation_dict in affiliation_map.values():
            valid = True
            for required_tag in args.filter:
                if required_tag.startswith('not_'):
                    if required_tag[4:] in affiliation_dict['tags']:
                        valid = False
                        break
                elif required_tag not in affiliation_dict['tags']:
                    valid = False
                    break
            if not valid:
                continue
            valid_affiliation_set.add(affiliation_dict['affiliation'])
            for tag_id in affiliation_dict['tags']:
                tag_count[tag_id] += 1
        for tag, count in sorted(tag_count.items(), key=lambda x: -x[1]):
            print(f'{tag}: {count}')

    related_affiliation_set = set()
    for affiliation in valid_affiliation_set:
        for local_affiliation_set in affiliation_sets[affiliation]:
            for local_affiliation in local_affiliation_set:
                if local_affiliation in valid_affiliation_set:
                    continue
                related_affiliation_set.add(local_affiliation)

    if args.show_related:
        print('\nrelated affillations topic count:')
        tag_count = collections.defaultdict(int)
        for affiliation, affiliation_dict in affiliation_map.items():
            if affiliation not in related_affiliation_set:
                continue
            for tag_id in affiliation_dict['tags']:
                tag_count[tag_id] += 1
        for tag, count in sorted(tag_count.items(), key=lambda x: -x[1]):
            print(f'{tag}: {count}')

    if args.show_names:
        print('\nAFFILIATIONS:')
        for affiliation in sorted(valid_affiliation_set):
            print(affiliation.lstrip().rstrip())

        if args.show_related:
            print('\nRELATED AFFILATIONS:')
            for related_affiliation in sorted(related_affiliation_set):
                print(related_affiliation)


if __name__ == '__main__':
    main()
