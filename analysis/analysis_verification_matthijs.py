import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import itertools
from tqdm import tqdm

from functools import reduce
from operator import mul
from math import factorial as fac
import krippendorff
import seaborn as sns

from analysis import load_results

import random
import statistics

from scipy.stats import spearmanr

import math

random.seed(12345)
np.random.seed(129387)
pd.set_option('display.max_colwidth', 180)
pd.set_option('expand_frame_repr', False)
sns.set(font_scale=1.2)


ADEQUACY_THRESHOLD = .5   # adequacy of entry name (and, in one condition, alternative names) should be at least...
CANONICAL_THRESHOLD = .5   # weight of canonical cluster should be at least ...
NUM_SAMPLES_FOR_STABILITY = 30
SUBSAMPLE = False       # for debugging

VERIFIED = False

def basic_stats():

    csvfile = '../proc_data_phase0/verification/all_responses_round0-3_verified.csv'
    df = load_results.load_cleaned_results(csvfile)

    df['entry_name'] = df['spellchecked_min2'].apply(lambda x: x.most_common(1)[0][0])
    for i, row in df.iterrows():
        df.at[i,'entry_name_verified'] = row['entry_name'] if row['verified'][row['entry_name']]['adequacy'] > ADEQUACY_THRESHOLD else np.nan

    if VERIFIED:   # restrict to verified cases
        df = df.loc[~(df['entry_name_verified'].isna())]
        df['entry_name'] = df['entry_name_verified']

    all_entry_names_statuses = {n: [] for n in df['entry_name'].unique()}

    same_cluster = []

    count_pairs = 0
    for i, row in df.iterrows():
        n0 = row['entry_name']
        # if clear winner:
        most_common = row['spellchecked_min2'].most_common()
        if len(most_common) == 1 or .5*most_common[0][1] > most_common[1][1]:
            all_entry_names_statuses[n0].append(True)
            for n1 in row['spellchecked_min2']:
                if VERIFIED:
                    if n1 in all_entry_names_statuses and n1 != n0 and row['verified'][n1]['adequacy'] > ADEQUACY_THRESHOLD and row['verified'][n0]['cluster'] == row['verified'][n1]['cluster']:
                        count_pairs += 1
                        all_entry_names_statuses[n1].append(False)
                        same_cluster.append(True)
                    elif n1 in all_entry_names_statuses and n1 != n0 and row['verified'][n1]['adequacy'] > ADEQUACY_THRESHOLD:
                        same_cluster.append(False)

                else:
                    if n1 in all_entry_names_statuses and n1 != n0 and .5*row['spellchecked_min2'][n0] > row['spellchecked_min2'][n1]:
                        count_pairs += 1
                        all_entry_names_statuses[n1].append(False)

    all_entry_names_proportions = {}
    for n in all_entry_names_statuses:
        if len(all_entry_names_statuses[n]) > 0:
            all_entry_names_proportions[n] = sum(all_entry_names_statuses[n])/len(all_entry_names_statuses[n])
        else:
            all_entry_names_proportions[n] = np.nan

    proportions = pd.DataFrame(all_entry_names_proportions.items(), columns=['name', 'proportion']).sort_values(by='proportion')
    proportions = proportions.loc[proportions['proportion'] != 0]

    print(proportions['proportion'].mean(), proportions['proportion'].std())

    print(proportions.to_string())
    proportions.hist()
    plt.show()


    print(sum(same_cluster)/len(same_cluster))


    # NOT VERIFIED:  0.4595991801966485          SD    0.3417786915948539
    # VERIFIED: 0.6606025985622637 0.348298392688703    (clustering: 0.9424675756568008)

    # NO ADEQUACY THRESHOLD:
    # non-verified: 0.37012184268554943
    # non-verified, big dif: 0.37925039030764296
    # non-verified, .85 dif, adequate: 0.5137965701789666
    # non-verified, .5 dif, adequate: 0.5590255848385242
    # ... same cluster: 0.6308093518377098
    # verified, plain: 0.39042199768877683
    # verified, big dif: 0.39983132517813524
    # verified, .85 dif, adequate: 0.5271917340163332
    # verified, .5 dif, adequate: 0.5705043951474471
    # ... same cluster: 0.6285499228510919 63%      SD: 0.34936729280912676

    # genuinely not verified, .5 dif: 0.41272882720253123   SD: 0.32621766242442995

    # proportion same cluster: .939431704885344.

    quit()

    errors = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        for n in row['spellchecked']:
            if n in row['verified']:
                errors.append([row['cat'],
                               n,
                               row['spellchecked_min2'][n],
                               row['verified'][n]['adequacy'],
                               row['verified'][n]['inadequacy_type'],
                               int(round(row['verified'][n]['adequacy']))
                               ])
            elif row['spellchecked'][n] == 1:
                errors.append([row['cat'],
                               n,
                               1,
                               0,
                               'noise',
                               0,
                               ])


    errors = pd.DataFrame(errors, columns=['cat', 'name', 'count', 'adequacy', 'inadequacy_type', 'adequacy_bin'])
    errors['inadequacy_type_level'] = errors['inadequacy_type'].apply(str) + errors['adequacy_bin'].apply(str)

    counts = errors['inadequacy_type_level'].value_counts()

    counts = counts[['None1', 'None0', 'bounding box1', 'bounding box0', 'visual1', 'visual0', 'linguistic1', 'linguistic0', 'other1', 'other0', 'noise0']]

    print(counts)
    print(counts / counts.sum())

    counts.plot.pie()
    plt.show()

    quit()

    pass



def basic_stats_old():
    """
    Originally from merge_and_clean_csv.py
    :return:
    """

    annotations_all = pd.read_csv("name_annotations_filtered_ANON.csv")

    annotations_targets = annotations_all.loc[
        annotations_all['control_type'].isna() | (annotations_all['control_type'] == 'vg_majority') | (
            annotations_all['control_type'].apply(lambda x: isinstance(x, str) and x.startswith('syn')))]

    print("\n-------------")
    print(annotations_all[:5].to_string())
    print('name_annotations:', len(annotations_all))
    print('targets:', len(annotations_targets))
    print('unique images:', len(annotations_all['image'].unique()), "/", len(annotations_targets['image'].unique()))
    print('unique names:', len(annotations_all['name'].unique()), "/", len(annotations_targets['name'].unique()))
    print("-------------")

    print("\n~~~~~~~~~~~~~~")
    print("Inter-annotator agreement:")
    scores_per_name = annotations_targets.groupby(['image', 'object', 'name']).agg({'adequacy': lambda x: x.tolist(), 'inadequacy_type': lambda x: x.tolist(), 'name_cluster': lambda x: x.tolist(), 'same_object': lambda x: x.tolist()}).reset_index()
    # Remove points with <=1 annotator to avoid division by zero
    scores_per_name = scores_per_name.loc[scores_per_name['adequacy'].apply(lambda x: len(x) > 1)]
    # Are these stats biased towards images with more names? They are counted more times... Then again, there are more names there to agree or not agree on.
    scores_per_name['adequacy_bin'] = scores_per_name['adequacy'].apply(lambda x: [0 if a == 0 else 1 for a in x])
    scores_per_name['inadequacy_type'] = scores_per_name['inadequacy_type'].apply(lambda x: [str(t) for t in x])
    scores_per_name['adequacy_agreement_cat'] = scores_per_name['adequacy'].apply(lambda x: sum([(x.count(score) * (x.count(score) - 1)) / (len(x) * (len(x) - 1)) for score in [0, 1, 2]]))
    scores_per_name['adequacy_agreement_bin'] = scores_per_name['adequacy_bin'].apply(lambda x: sum([(x.count(score) * (x.count(score) - 1)) / (len(x) * (len(x) - 1)) for score in [0, 1]]))
    scores_per_name['adequacy_agreement_std'] = scores_per_name['adequacy'].apply(np.std)
    scores_per_name['inadequacy_type_agreement'] = scores_per_name['inadequacy_type'].apply(lambda x: sum([(x.count(score) * (x.count(score) - 1)) / (len(x) * (len(x) - 1)) for score in['nan', 'linguistic', 'bounding box', 'visual', 'other']]))
    scores_per_name['same_object'] = scores_per_name['same_object'].apply(lambda x: {key: [d[key] for d in x] for key in x[0]})
    scores_per_name['same_object_agreement'] = scores_per_name['same_object'].apply(lambda x: [sum([(x[k].count(score) * (x[k].count(score) - 1)) / (len(x[k]) * (len(x[k]) - 1)) for score in [0, 1]]) for k in x])
    scores_per_name['same_object_agreement'] = scores_per_name['same_object_agreement'].apply(lambda x: sum(x) / len(x))
    print(scores_per_name[['adequacy_agreement_cat', 'adequacy_agreement_bin', 'adequacy_agreement_std', 'inadequacy_type_agreement', 'same_object_agreement']].mean())
    print("~~~~~~~~~~~~~~")
    print()

    ## Aggregate multiple annotations for a image+name:
    # adequacy: mean
    # inadequacy_type: majority, otherwise 'unknown'
    # same_object: majority, otherwise 'unknown'
    scores_per_name['adequacy_mean'] = scores_per_name['adequacy'].apply(lambda x: sum(x)/len(x))
    scores_per_name['inadequacy_type_majority'] = scores_per_name['inadequacy_type'].apply(lambda x: Counter(x).most_common(1)[0][0]) # TODO What if no option has majority?
    scores_per_name['same_object_majority'] = scores_per_name['same_object'].apply(lambda x: {k: Counter(x[k]).most_common(1)[0][0] for k in x})
    scores_per_name['name_cluster_majority'] = scores_per_name['same_object_majority'].apply(lambda x: tuple(sorted(k for k in x if x[k] == 1)))
    # print(scores_per_name[:20].to_string())

    print("Number of names total:", len(scores_per_name))
    print("Number of adequate names total:", len(scores_per_name.loc[scores_per_name['adequacy_mean'] < 1]))
    print("Adequate names per image (<1):", scores_per_name.loc[scores_per_name['adequacy_mean'] < 1].groupby('image').agg({'name': 'count'}).mean().values[0])
    print("Clusters (majority) per image:", scores_per_name.groupby('image').agg({'name_cluster_majority': 'nunique'}).mean().values[0])
    print("Adequate clusters (majority) per image:", scores_per_name.loc[scores_per_name['adequacy_mean'] < 1].groupby('image').agg({'name_cluster_majority': 'nunique'}).mean().values[0])
    cluster_sizes = scores_per_name.groupby('image').agg({'name_cluster_majority': lambda x: [len(t) for t in x.unique()]})
    cluster_sizes_flat = [x for y in cluster_sizes['name_cluster_majority'].tolist() for x in y]
    print("Cluster size (majority):", sum(cluster_sizes_flat)/len(cluster_sizes_flat))
    cluster_sizes = scores_per_name.loc[scores_per_name['adequacy_mean'] < 1].groupby('image').agg({'name_cluster_majority': lambda x: [len(t) for t in x.unique()]})
    cluster_sizes_flat = [x for y in cluster_sizes['name_cluster_majority'].tolist() for x in y]
    print("Adequate cluster size (majority):", sum(cluster_sizes_flat)/len(cluster_sizes_flat))


def compute_stability_analytic(auxfile):

    settings = ['all_names', 'entry_cluster_names']

    recompute = not os.path.exists(auxfile) or input("Overwrite existing auxiliary file (recompute the sampling)? y/N").lower().startswith("y")

    if recompute:

        csvfile = '../proc_data_phase0/verification/all_responses_round0-3_verified.csv'
        df = load_results.load_cleaned_results(csvfile)

        if SUBSAMPLE:
            print("WARNING: LOOKING ONLY AT PART OF THE DATA FOR DEBUGGING")
            df = df.sample(n=SUBSAMPLE)

        df['probabilities_all_names'] = df['spellchecked'].apply(lambda x: sorted([x[n] / 36 for n in x], reverse=True))
        df['probabilities_entry_cluster_names'] = [[] for _ in range(len(df))]
        for i, row in df.iterrows():
            df.at[i, 'probabilities_entry_cluster_names'] = sorted([row['spellchecked_min2'][n] / 36 for n in row['spellchecked_min2'] if row['verified'][n]['cluster_id'] == 0], reverse=True)

        for i, row in tqdm(df.iterrows(), total=len(df)):
            df.at[i, 'names_needed_95_all_names'] = n_names_for_prob_majority(row['probabilities_all_names'][:4])
            df.at[i, 'names_needed_95_entry_cluster_names'] = n_names_for_prob_majority(row['probabilities_entry_cluster_names'][:5])

        df.to_csv(auxfile)

    df = pd.read_csv(auxfile)
    print("Loaded", auxfile)

    # for i, row in df.iterrows():
    #     domains = {}
    #     for r in range(4):
    #         domains.update(eval(row['responses_domains_r{}'.format(r)]))
    #     entry_name = eval(row['spellchecked_min2']).most_common(1)[0][0]
    #     if entry_name in domains:
    #         df.at[i, 'domain'] = domains[entry_name]
    #     else:
    #         print(entry_name, 'not in', domains)

    print(df.columns)
    print(df[:10].to_string())

    df.columns = [col.replace('names_needed_95_', '') for col in df.columns]

    stacked = df[['cat', 'all_names', 'entry_cluster_names']].melt(id_vars=['cat']).reset_index()

    stacked.rename(columns={'variable': 'setting', 'value': 'names_needed_95'}, inplace=True)

    print(stacked[:10].to_string())

    counted = stacked.groupby(['setting', 'names_needed_95']).count().reset_index().rename(
        columns={'index': 'count'})
    counted['percentage'] = counted.groupby(['setting'])['count'].apply(lambda x: 100 * x / float(x.sum()))
    counted['cumulative %'] = counted.groupby(['setting'])['percentage'].cumsum()
    counted = counted.loc[counted['names_needed_95'] <= 36]

    # counted = stacked.groupby(['cat', 'setting', 'names_needed_95']).count().reset_index().rename(columns={'index': 'count'})
    # counted['percentage'] = counted.groupby(['cat', 'setting'])['count'].apply(lambda x: 100 * x / float(x.sum()))
    # counted['cumulative %'] = counted.groupby(['cat', 'setting'])['percentage'].cumsum()
    # counted = counted.loc[counted['names_needed_95'] <= 36]

    counted.rename(columns={'names_needed_95': 'number of names gathered', 'cumulative %': '% of entry names identified'}, inplace=True)

    ax = sns.lineplot(x='number of names gathered', y='% of entry names identified', hue='setting', data=counted, hue_order=settings)
    # ax = sns.lineplot(x='number of names gathered', y='% of entry names identified', hue='cat', data=counted.loc[counted['setting'] == 'entry_cluster_names'])
    # ax = sns.lineplot(x='number of names gathered', y='% of entry names identified', hue='cat', data=counted.loc[counted['setting'] == 'all_names'])
    plt.ylim((0, 100))
    plt.xlim((0, 36))
    ax.legend().texts[0].set_text("identification among:")
    plt.show()


def compute_stability_stochastic(auxfile):

    settings = ['all names', 'adequate names', 'canonical object names', 'adequate canonical names']

    SUBSETS_TO_PLOT = ['all names', 'canonical object names']

    recompute = not os.path.exists(auxfile) or input(
        "Overwrite existing auxiliary file (recompute the sampling)? y/N").lower().startswith("y")

    if recompute:

        csvfile = '../proc_data_phase0/verification/all_responses_round0-3_verified.csv'
        df = load_results.load_cleaned_results(csvfile)

        # # For manual inspection...
        # r = ['responses_r0', 'responses_r1', 'responses_r2', 'responses_r3']
        # t = ['top_name_r0', 'top_name_r1', 'top_name_r2', 'top_name_r3']
        # for rcolumn, tcolumn in zip(r, t):
        #     df[rcolumn] = df[rcolumn].apply(eval)
        #     df[tcolumn] = df[rcolumn].apply(lambda x: max(x, key=x.get))
        #
        # for i, row in df.iterrows():
        #     print(row[['top_name_r0', 'top_name_r1', 'top_name_r2', 'top_name_r3']])

        if SUBSAMPLE:
            print("WARNING: LOOKING ONLY AT PART OF THE DATA FOR DEBUGGING")
            df = df.sample(n=SUBSAMPLE)

        print("Total objects: ", len(df))
        print(df.columns)

        df['entry_name'] = df['spellchecked_min2'].apply(lambda x: x.most_common(1)[0][0])

        # create some object-type columns in advance to avoid error when df.at-ing them in
        for set in settings:
            if False:  # old way
                df[set] = [[] for _ in range(len(df))]
            else:  # new way
                df[set] = [{} for _ in range(len(df))]
        for set in settings:
            df['stabilities_' + set] = [[] for _ in range(len(df))]

        # compute lists of names of different kinds
        for i, row in tqdm(df.iterrows(), total=len(df)):
            entry_name_info = row['verified'][row['entry_name']]
            if entry_name_info['adequacy'] >= ADEQUACY_THRESHOLD and entry_name_info['cluster_id'] == 0 and \
                            entry_name_info['cluster_weight'] >= CANONICAL_THRESHOLD:
                if False:  # old way
                    all_names = list(itertools.chain(
                        *[[name] * row['spellchecked_min2'][name] for name in row['spellchecked_min2']]))
                    # df.at[i, 'all names'] = all_names
                    df.at[i, 'adequate names'] = [n for n in all_names if
                                                  row['verified'][n]['adequacy'] >= ADEQUACY_THRESHOLD]
                    df.at[i, 'canonical object names'] = [n for n in all_names if
                                                          row['verified'][n]['cluster_id'] == 0]
                    df.at[i, 'adequate canonical names'] = [n for n in all_names if
                                                            row['verified'][n]['cluster_id'] == 0 and
                                                            row['verified'][n]['adequacy'] >= ADEQUACY_THRESHOLD]
                    ## Using non-spellchecked names makes a 5%pt difference:
                    df.at[i, 'all names'] = list(
                        itertools.chain(*[[name] * row['spellchecked'][name] for name in row['spellchecked']]))
                else:  # new way
                    df.at[i, 'all names'] = {name: row['spellchecked'][name] for name in row['spellchecked']}
                    df.at[i, 'adequate names'] = {name: row['spellchecked_min2'][name] for name in
                                                  row['spellchecked_min2'] if
                                                  row['verified'][name]['adequacy'] >= ADEQUACY_THRESHOLD}
                    df.at[i, 'canonical object names'] = {name: row['spellchecked_min2'][name] for name in
                                                          row['spellchecked_min2'] if
                                                          row['verified'][name]['cluster_id'] == 0}
                    df.at[i, 'adequate canonical names'] = {name: row['spellchecked_min2'][name] for name in
                                                            row['spellchecked_min2'] if row['verified'][name][
                                                                'adequacy'] >= ADEQUACY_THRESHOLD and
                                                            row['verified'][name]['cluster_id'] == 0}

        for i, row in tqdm(df.iterrows(), total=len(df)):
            for set in settings:
                names = list(row[set].keys())
                probabilities = [row[set][name] for name in names]
                probabilities = [p / sum(probabilities) for p in probabilities]
                if len(names) > 0:
                    earliest_stabilities = []
                    for _ in range(NUM_SAMPLES_FOR_STABILITY):
                        names_list = np.random.choice(names, size=40, p=probabilities, replace=True)
                        # Walk backwards through shuffled list of names until entry_name is NOT a majority anymore
                        for j in range(len(names_list) - 1, -1, -1):
                            if not row['entry_name'] in [t[0] for t in Counter(names_list[:j + 1]).most_common(1)]:
                                break
                        earliest_stabilities.append(j + 1)
                    earliest_stability_avg = (sum(earliest_stabilities) / len(earliest_stabilities))
                    df.at[i, 'stabilities_' + set] = earliest_stabilities
                    df.at[i, 'stability_' + set] = earliest_stability_avg
                    df.at[i, 'stability_' + set + '_std'] = statistics.stdev(earliest_stabilities)

        df.to_csv(auxfile)

    df = pd.read_csv(auxfile,
                     converters={'stabilities_' + set: lambda x: eval(x.replace('nan', 'np.nan')) for set in
                                 settings})

    for set in settings:
        del df[set]

    print("df:", len(df))
    print(df[:10].to_string())

    # Barplot based on full results per image
    df.columns = [col.replace('stabilities_', '') for col in df.columns]
    stacked = df[
        ['all names', 'adequate names', 'canonical object names', 'adequate canonical names']].stack().reset_index()
    stacked.rename(columns={'level_1': 'setting', 0: 'stabilities'}, inplace=True)

    expanded = []
    for i, row in stacked.iterrows():
        for k, j in enumerate(row['stabilities']):
            expanded.append([row['setting'], k, j])
    stacked = pd.DataFrame(expanded, columns=['setting', 'sample', 'stability']).reset_index()

    counted = stacked.groupby(['setting', 'sample', 'stability']).count().reset_index().rename(
        columns={'level_0': 'count', 'index': 'count'})
    counted['percentage'] = counted.groupby(['setting', 'sample'])['count'].apply(
        lambda x: 100 * x / float(x.sum()))
    counted['cumulative %'] = counted.groupby(['setting', 'sample'])['percentage'].cumsum()

    # totals = counted.groupby['setting'].agg({'count': sum})
    # counted['count'] = 100 * counted['count'] / counted['count'].sum()

    print(counted.loc[counted['stability'] % 9 == 0].to_string())

    counted = counted.loc[counted['setting'].isin(SUBSETS_TO_PLOT)]

    if False:
        # stacked.hist(column='stability', by='setting', sharex=True, sharey=True)
        sns.barplot(x='stability', y='percentage', hue='setting', data=counted, hue_order=SUBSETS_TO_PLOT, ci="sd")
        plt.show()

    else:
        counted.replace(to_replace={'setting': {'canonical object names': 'entry-level cluster names'}},
                        inplace=True)
        ax = sns.lineplot(x='stability', y='cumulative %', hue='setting', data=counted,
                          hue_order=['all names', 'entry-level cluster names'], ci="sd")
        # sns.scatterplot(x='stability', y='cum_percentage', hue='setting', data=counted, hue_order=SUBSETS_TO_PLOT, ax=ax)
        plt.ylim((50, 100))
        plt.xlim((0, 36))
        plt.xlabel('Number of names gathered')
        plt.ylabel('% of entry names identified')
        ax.legend().texts[0].set_text("Identification among:")
        # ax.legend().texts[2].set_text("entry-level object names")
        plt.show()

        # for set in settings:
        #     sns.distplot(stacked.loc[stacked['setting'] == ('stability_'+set)]['stability'], label=set)
        # plt.legend()

        # also plot entry_freq ?

#
def majorities(n_elements, total_sum):
    candidates = [[n] for n in range(total_sum + 1, 0, -1)]
    solutions = []
    while len(candidates) > 0:
        c = candidates.pop(0)
        if len(c) >= n_elements:
            if sum(c) == total_sum:
                solutions.append(c)
        else:
            for o in range(min(total_sum-sum(c), c[0]-1), -1, -1):
                candidates.append(c + [o])
    return solutions

# probability of majority
def prob_majority(probs, n_names):
    probs = [p/sum(probs) for p in probs]   # normalize
    prob = 0.0
    for counts in majorities(len(probs), n_names):
        prob += reduce(mul, [prob**count for prob,count in zip(probs,counts)], 1) * fac(n_names) / reduce(mul, [fac(count) for count in counts], 1)
    return prob

def n_names_for_prob_majority(probs, threshold=.9, max_n_names=38, increment=2):
    # time-saving shortcut: if prob[0] and prob[1] aren't so different, then even with 36 names only .85 can be reached.
    if threshold > .85 and len(probs) > 1 and probs[1] > .66 * probs[0]:
        return 99
    for n in list(range(1,10,1)) + list(range(10,max_n_names+1,increment)):
        if prob_majority(probs, n) > threshold:
            return n
    return 99


def adequacy_against_frequency():
    csvfile = '../proc_data_phase0/verification/all_responses_round0-3_verified.csv'
    df = load_results.load_cleaned_results(csvfile)

    datapoints = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        freqs = row['spellchecked_min2'].values()
        total = sum(freqs)
        max_freq = max(freqs) / total
        for name in row['verified']:
            freq = row['spellchecked_min2'][name] / total
            adeq = row['verified'][name]['adequacy']
            top = "top" if freq == max_freq else "other"

            # freq = math.ceil(10*(row['spellchecked_min2'][name]/total))/10
            # adeq = math.ceil(6*row['verified'][name]['adequacy'])/6
            # Merge the first into a bin
            # freq = max(.1, freq)
            # adeq = max(1/6, adeq)

            if freq != 0:
                datapoints.append([freq, adeq, top, 1])

    datapoints = pd.DataFrame(datapoints, columns=["frequency", "adequacy", "name", "count"])

    # Heatmap attempt 1:
    if False:
        datapoints = datapoints.groupby(["frequency", "adequacy"]).count().unstack(level=-1)['count']

        ax = sns.heatmap(datapoints, square=True)
        ax.set_ylim(0, 10)  # to fix bug due to matplotlib
        ax.set_yticklabels(ax.get_yticklabels())  # Reset to remove rotation
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_horizontalalignment('left')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_verticalalignment('top')

        plt.show()

    # Heatmap attempt 2:
    if False:
        plt.hist2d(data=datapoints, x="adequacy", y="frequency", bins=[7, 36])
        plt.show()

    print(spearmanr(datapoints['frequency'], datapoints['adequacy']))

    # Scatterplot:
    if True:
        sns.lmplot(data=datapoints, x='frequency', y='adequacy', hue='name', x_jitter=.03, y_jitter=.07, truncate=True,
                   scatter_kws={'alpha': .05})
        plt.show()


def inter_rater_agreement():
    verification_data_path = '../verification_phase0/1_crowdsourced/results/merged_1-2-3-4-5-6-7-8_redone/name_annotations_filtered_ANON.csv'
    verifications = pd.read_csv(verification_data_path, converters={'name_cluster-nofillers': eval, 'same_object-nofillers': eval})
    del verifications['batch']
    del verifications['source']
    del verifications['hitid']
    del verifications['assignmentid']
    del verifications['name_cluster']
    del verifications['same_object']

    # Remove all manipulated control items
    # ad hoc fix for weird bug: elizabethville is not in fact a control item
    verifications = verifications.loc[(verifications['name'] == 'elizabethville') |
                                      verifications['control_type'].isna() |
                                      (verifications['control_type'] == 'vg_majority') |
                                      (verifications['control_type'].apply(
                                          lambda x: isinstance(x, str) and x.startswith('syn')))]
    verifications.rename(columns={'name_cluster-nofillers': 'name_cluster', 'same_object-nofillers': 'same_object'},
                         inplace=True)

    verifications['inadequacy_type'] = verifications['inadequacy_type'].apply(lambda x: "none" if pd.isna(x) else x)

    replacements = {'none': 0, 'linguistic': 1, 'bounding box': 2, 'visual': 3, 'other': 4}
    verifications['inadequacy_type_num'] = verifications['inadequacy_type'].apply(lambda x: replacements[x])

    # TODO I don't understand why there are duplicates...
    verifications = verifications[~verifications.duplicated(['workerid', 'image', 'object', 'name'], keep='first')]

    print(verifications[:5].to_string())

    # # Computing adequacy agreement
    verifications['unit'] = verifications.apply(lambda x: f"{x['image']}-{x['object']}-{x['name']}", axis=1)
    # reliability_data = verifications[['workerid', 'unit', 'adequacy']].pivot(index='workerid', columns='unit')
    # print("Adequacy: Krippendorff's alpha (ordinal) =", krippendorff.alpha(reliability_data, level_of_measurement='ordinal'))
    #
    # # computing inadequacy type agreement
    # reliability_data = verifications[['workerid', 'unit', 'inadequacy_type_num']].pivot(index='workerid', columns='unit')
    # print("Inadequacy type: Krippendorff's alpha (nominal) =", krippendorff.alpha(reliability_data, level_of_measurement='nominal'))

    # computing same-object agreement
    new_rows = []
    for i, row in tqdm(verifications.iterrows()):
        for key in row['same_object']:
            new_rows.append([row['workerid'], row['unit'] + '-' + key, row['same_object'][key]])
    verifications_per_name_pair = pd.DataFrame(new_rows, columns=["workerid", "unit", "same_object"])
    reliability_data = verifications_per_name_pair[['workerid', 'unit', 'same_object']].pivot(index='workerid', columns='unit')
    print("Same_object: Krippendorff's alpha (nominal) =", krippendorff.alpha(reliability_data, level_of_measurement='nominal'))


if __name__ == "__main__":

    if False:
        basic_stats()

    if True:
        inter_rater_agreement()
        # Adequacy: Krippendorff's alpha (ordinal) = 0.7067100617783595
        # Inadequacy type: Krippendorff's alpha (nominal) = 0.6734359534740075
        # Same-object: Krippendorff's alpha (nominal) = 0.8332111041592695

    if False:
        adequacy_against_frequency()

    if False:
        do_analytic = not input("Analytic or stochastic? A/s").lower().startswith('s')

        auxfile = 'aux/stability_samples_{}_{}_{}_{}{}'.format(ADEQUACY_THRESHOLD, CANONICAL_THRESHOLD, NUM_SAMPLES_FOR_STABILITY, SUBSAMPLE, "_analytic" if do_analytic else "")
        os.makedirs('aux', exist_ok=True)

        if do_analytic:
            compute_stability_analytic(auxfile)
        else:
            compute_stability_stochastic(auxfile)
