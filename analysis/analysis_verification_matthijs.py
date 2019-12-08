import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import itertools
from tqdm import tqdm

from functools import reduce
from operator import mul
from math import factorial as fac, floor

import seaborn as sns

from analysis import load_results

import random
import statistics

random.seed(12345)
np.random.seed(129387)
pd.set_option('display.max_colwidth', 180)
pd.set_option('expand_frame_repr', False)
sns.set(font_scale=1.2)


ADEQUACY_THRESHOLD = .5   # adequacy of entry name (and, in one condition, alternative names) should be at least...
CANONICAL_THRESHOLD = .5   # weight of canonical cluster should be at least ...
NUM_SAMPLES_FOR_STABILITY = 30
SUBSAMPLE = False       # for debugging

def analytic(auxfile):

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

    df.columns = [col.replace('names_needed_95_', '') for col in df.columns]

    stacked = df[['all_names', 'entry_cluster_names']].stack().reset_index()
    stacked.rename(columns={'level_1': 'setting', 0: 'names_needed_95'}, inplace=True)

    counted = stacked.groupby(['setting', 'names_needed_95']).count().reset_index().rename(
        columns={'level_0': 'count'})
    counted['percentage'] = counted.groupby(['setting'])['count'].apply(lambda x: 100 * x / float(x.sum()))
    counted['cumulative %'] = counted.groupby(['setting'])['percentage'].cumsum()
    counted = counted.loc[counted['names_needed_95'] <= 36]

    counted.rename(columns={'names_needed_95': 'number of names gathered', 'cumulative %': '% of entry names identified'}, inplace=True)

    ax = sns.lineplot(x='number of names gathered', y='% of entry names identified', hue='setting', data=counted, hue_order=settings)
    plt.ylim((0, 100))
    plt.xlim((0, 36))
    ax.legend().texts[0].set_text("identification among:")
    plt.show()


def stochastic(auxfile):

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
    for n in range(1,max_n_names+1,increment):
        if prob_majority(probs, n) > threshold:
            return n
    return 99


if __name__ == "__main__":

    do_analytic = not input("Analytic or stochastic? A/s").lower().startswith('s')

    auxfile = 'aux/stability_samples_{}_{}_{}_{}{}'.format(ADEQUACY_THRESHOLD, CANONICAL_THRESHOLD, NUM_SAMPLES_FOR_STABILITY, SUBSAMPLE, "_analytic" if do_analytic else "")
    os.makedirs('aux', exist_ok=True)

    if do_analytic:
        analytic(auxfile)
    else:
        stochastic(auxfile)
