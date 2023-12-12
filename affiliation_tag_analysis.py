"""
x what are the most to least common affiliation tags?
* which affiliation tags are working with which other ones? (like when is water
  working with disasters when the person with water in their affiliation doesn't
  have disasters in their affiliation and vice versa) what are the strength of
  those connections?
* what are the most to least common topic tags (from the abstracts)?
* which affiliations are working on which topics together, and what are the
  strongest interactions?
"""
import argparse
import collections
import glob
import re

import pandas as pd
import numpy
import umap
import matplotlib.pyplot as plt

from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource

output_notebook()

# subset affiliations based on abstracts
def cluster_map():

    print('fit the reducer')
    reducer = umap.UMAP()
    topic_weight_array = numpy.array(topic_weight_array)
    reducer.fit(topic_weight_array)
    print('transform the data')
    embedding = reducer.transform(topic_weight_array)

    plt.scatter(embedding[:, 0], embedding[:, 1], s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of affiliation topics', fontsize=24)
    plt.show()

    topics_df = pd.DataFrame(embedding, columns=('x', 'y'))
    topics_df['weights'] = weights_as_str

    datasource = ColumnDataSource(topics_df)

    plot_figure = figure(
        title='UMAP projection of the Digits dataset',
        plot_width=600,
        plot_height=600,
        tools=('pan, wheel_zoom, reset')
    )

    plot_figure.add_tools(HoverTool(tooltips="""
    <div>
        @weights
    </div>
    """))

    plot_figure.circle(
        'x',
        'y',
        source=datasource,
        line_alpha=0.6,
        fill_alpha=0.6,
        size=4
    )
    show(plot_figure)


def dump_topic_tags(target_csv_file, affiliation_subset, affiliation_map, affiliation_topic_list):
    print(affiliation_subset)
    print(affiliation_map)
    topic_weight_sum = numpy.zeros(len(affiliation_topic_list))
    count = 0
    for affiliation in affiliation_subset:
        if affiliation not in affiliation_map:
            continue
        topic_weights = affiliation_map[affiliation]
        topic_weight_sum += topic_weights
        count += 1
    with open(target_csv_file, 'w') as file:
        file.write(f'{count}\n')
        for weight, topic in sorted(
                zip(topic_weight_sum, affiliation_topic_list),
                reverse=True):
            file.write(f'{topic},{weight}\n')


def main():
    parser = argparse.ArgumentParser(description='Parse and process affiliation data')
    parser.add_argument(
        'classified_affiliation_file',
        help='Path file with affiliation, name, and tags')
    parser.add_argument(
        'abstract_classification_file',
        help='Path to abstract classification file')
    parser.add_argument(
        'bib_path_pattern',
        help='Path to bibliography file')
    parser.add_argument('--filter', nargs='+')
    parser.add_argument('--show_names', action='store_true')
    parser.add_argument('--confidence_threshold', type=float)
    parser.add_argument('--show_related', action='store_true')
    parser.add_argument('--max_afl', type=int, default=-1)
    args = parser.parse_args()

    affiliation_map = {}
    affiliation_topic_to_index = collections.defaultdict(
        lambda: len(affiliation_topic_to_index))

    print('parse affillations')
    with open(args.classified_affiliation_file, 'r', encoding='utf-8') as file:
        # build up topic map
        file.readline()  # skip the first line
        while True:
            topic = file.readline().strip().split(':')[0]
            if topic == '':
                file.seek(0)
                break
            affiliation_topic_to_index[topic]  # trigger the index

        affiliation_topic_list = [
            topic for topic, index in
            sorted(affiliation_topic_to_index.items(), key=lambda x: x[1])]

        max_to_get = args.max_afl
        while True:
            if max_to_get == 0:
                break
            max_to_get -= 1
            affiliation = file.readline().strip()
            if affiliation == '':
                break

            topic_weights = numpy.zeros(len(affiliation_topic_to_index))
            while True:
                try:
                    topic, prob = file.readline().strip().split(':')
                    topic_weights[affiliation_topic_to_index[topic]] = prob
                except ValueError:
                    # can't split, so blank line, next section
                    break
            affiliation_map[affiliation] = topic_weights

    abstract_list = []
    abstract_topic_to_index = collections.defaultdict(
        lambda: len(abstract_topic_to_index))

    print('what are the most to least common affiliation tags')
    topic_weight_sum = numpy.zeros(len(affiliation_topic_to_index))

    topic_weight_array = []
    weights_as_str = []
    for topic_weights in affiliation_map.values():
        topic_weight_sum += topic_weights
        topic_weight_array.append(topic_weights)

        x = ','.join([f'{topic}' for weight, topic in sorted(
                  zip(topic_weights, affiliation_topic_list),
                  key=lambda x: -x[0])
                  if weight >= .9])
        weights_as_str.append(x)

    with open('affiliation_topic_ranks.csv', 'w') as file:
        for topic, weight in sorted(zip(affiliation_topic_list, topic_weight_sum), key=lambda x: x[1]):
            file.write(f'{topic},{weight}\n')

    print('parse bibs')
    bib_map_by_affiliation = collections.defaultdict(list)
    bib_map_by_abstract = collections.defaultdict(list)

    for bib_file in glob.glob(args.bib_path_pattern):
        print(f'processing bib file: {bib_file}')
        with open(bib_file, 'r', encoding='utf-8') as file:
            affilation_str = None
            abstract_str = None
            for line in file:
                if line.startswith('@'):
                    abstract_id = re.search('{(.*),', line).group(1)
                    affilation_str = None
                    abstract_str = None
                elif line.startswith('\taffiliations'):
                    affilation_str = re.search('{(.*)},', line).group(1).strip()
                elif line.startswith('\tabstract'):
                    abstract_str = re.search('{(.*)},', line).group(1).strip()
                if affilation_str and abstract_str:
                    affiliation_list = [
                        x.strip() for x in affilation_str.split(';')]
                    bib_entry = {
                        'affiliation_list': affiliation_list,
                        'abstract': abstract_str,
                        'abstract_id': abstract_id
                    }
                    for affiliation in affiliation_list:
                        if affiliation not in affiliation_map:
                            # okay, either we skipped or there's a weird case
                            # we already checked and found this rarely happens
                            # i.e. 25 out of 100000 times
                            continue
                        bib_map_by_affiliation[affiliation].append(bib_entry)
                    bib_map_by_abstract[abstract_str] = {
                        'affiliation_list': affiliation_list
                        }
                    affilation_str = None
                    abstract_str = None
    print(f'done mapping bibs {len(bib_map_by_affiliation)} {len(bib_map_by_abstract)}')

    print('parse abstracts')
    with open(args.abstract_classification_file, 'r', encoding='utf-8') as file:
        # build up topic map
        file.readline()  # skip the first 3 lines
        file.readline()
        file.readline()

        while True:
            topic = file.readline().strip().split(':')[0]
            if topic == '':
                file.seek(0)
                break
            abstract_topic_to_index[topic]  # trigger the index

        abstract_topic_list = [
            topic for topic, index in
            sorted(abstract_topic_to_index.items(), key=lambda x: x[1])]

        count = 0
        while True:
            abstract_id = file.readline().strip()
            if abstract_id == '':
                break
            affiliation_list = [
                v.strip() for v in file.readline().strip().split(';')
                if len(v.strip()) > 0]
            abstract = file.readline().strip()
            if abstract not in bib_map_by_abstract:
                count += 1
            else:
                bib_map_by_abstract[abstract]['topic_weights'] = topic_weights

            topic_weights = numpy.zeros(len(abstract_topic_to_index))
            while True:
                try:
                    topic, prob = file.readline().strip().split(':')
                    topic_weights[abstract_topic_to_index[topic]] = prob
                except ValueError:
                    # can't split, so blank line, next section
                    break
            abstract_list.append({
                'abstract_id': abstract_id,
                'affiliation_list': affiliation_list,
                'abstract': abstract,
                'topic_weights': topic_weights,
                })
        print(f'unknown abstract count {count}')

    print('what are the most to least common abstract tags')
    topic_weight_sum = numpy.zeros(len(abstract_topic_to_index))

    topic_weight_array = []
    weights_as_str = []
    for abstract in abstract_list:
        topic_weight_sum += abstract['topic_weights']
        topic_weight_array.append(abstract['topic_weights'])
        x = ','.join([f'{topic}' for weight, topic in sorted(
                  zip(topic_weights, abstract_topic_list),
                  key=lambda x: -x[0])
                  if weight >= .9])
        weights_as_str.append(x)

    with open('abstract_topics_rank.csv', 'w') as file:
        for topic, weight in sorted(zip(abstract_topic_list, topic_weight_sum), key=lambda x: x[1]):
            file.write(f'{topic},{weight}\n')

    # create a lookup of abstract by topic
    prob_threshold = 0.9
    abstract_topic_to_abstract_and_weights = collections.defaultdict(list)
    for abstract_topic_index, abstract_topic in enumerate(abstract_topic_list):
        for abstract in abstract_list:
            if abstract['topic_weights'][abstract_topic_index] >= prob_threshold:
                abstract_topic_to_abstract_and_weights[abstract_topic].append(
                    (abstract['affiliation_list'],
                     abstract['abstract'],
                     abstract['topic_weights']))

    # Subset the abstracts with these topics, then do the afilliations in the
    # subset
    abstract_topic_subsets = [
        ['agriculture', 'food'],
        ['business',  'financial'],
        ['climate',],
        ['mining', 'transportation', 'building', 'urban'],
        ['disaster'],
        ['energy'],
        ['environment', 'pollution', 'conservation', 'wildlife'],
        ['fire'],
        ['social equity', 'poverty', 'environmental justice'],
        ['people', 'population'],
        ['public health',  'disease', 'air quality'],
        ['ocean', 'coastal'],
        ['tourism'],
        ['water'],
    ]

    for abstract_topic_subset in abstract_topic_subsets:
        abstract_subset = []
        for abstract_topic in abstract_topic_subset:
            abstract_subset.extend(abstract_topic_to_abstract_and_weights[abstract_topic])
        affiliation_subset = set()
        for abstract in abstract_subset:
            # 0 is the index of the affiliation list
            # 1 is the abstract
            # 2 is the topic weights
            affiliation_subset |= set(abstract[0])
        print(f'what kinds of organizations publish on {abstract_topic_subset}')
        dump_topic_tags(
            '_'.join(abstract_topic_subset) + '.csv',
            affiliation_subset,
            affiliation_map,
            affiliation_topic_list)
    #     subset_abstracts

    return

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
