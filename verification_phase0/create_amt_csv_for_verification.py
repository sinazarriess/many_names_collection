import os
import sys
from collections import Counter
import math

import pandas as pd

import numpy as np
from tensorflow.contrib.slim.python.slim.nets.vgg import vgg_16

from analysis import load_results
from tqdm import tqdm
import random

import json



PHASE = "pre-pilot" # "pilot", "main"
MAX_IMAGES_PER_HIT = 8
MAX_NAMES_PER_IMAGE = 15

def main():

    np.random.seed(12345)
    random.seed(12345)

    relevant_cols = ['vg_img_id', 'cat', 'synset', 'vg_obj_name', 'vg_domain', 'vg_object_id',
                     'url', 'opt-outs', 'spellchecked', 'all_responses']

    manual_annotations = '../raw_data_phase0/verification_pilot/verif_annos_pilot.csv'
    many_names = "../proc_data_phase0/spellchecking/all_responses_round0-3_cleaned.csv"

    df = preprocess_responses(many_names, relevant_cols)


    # Collect all names in a list
    df['names_list'] = df['spellchecked_min2'].apply(lambda x: [n for c,n in sorted([(x[n], n) for n in x], reverse=True)])
    # And always incude vg name:
    for i, row in df.iterrows():
        if row['vg_obj_name'] not in row['names_list']: # add vg name to the back
            df.at[i,'names_list'] += [row['vg_obj_name']]
        elif row['vg_is_max']:  # bring vg name to the front
            df.at[i, 'names_list'].insert(0, row['names_list'].pop(row['names_list'].index(row['vg_obj_name'])))
    df['n_names'] = df['names_list'].apply(len)
    # Restrict to multiple names:
    df = df.loc[df['n_names'] > 1]

    print(df.head().to_string())

    # Create quality control items
    # TODO read original vg data for that
    with open('../dataset_creation/add_data/vgenome/objects.json') as file:
        vg_data = json.load(file)
    vg_dict = {img['image_id']: img for img in vg_data}

    vg_names = list(set([name for img in vg_data for object in img['objects'] for name in object['names']]))

    df['quality_control_dict'] = [{} for _ in range(len(df))]
    df['n_fillers'] = [0 for _ in range(len(df))]

    for i, row in tqdm(df.iterrows()):
        num_typos = round(len(row['names_list'])/8)
        num_random = round(len(row['names_list'])/9)
        num_alts = round(len(row['names_list'])/5)

        while num_typos + num_random + num_alts < .1 * len(row['names_list']):
            choice = random.choices(['typo', 'random', 'alt'], [2,1,5], k=1)
            if choice == 'typo':
                num_typos += 1
            elif choice == 'random':
                num_random += 1
            else:
                num_alts += 1

        typonames = []
        # randomly generate name variants with spelling error
        while len(typonames) < num_typos:
            name_idx = 0 if random.random() > .1 else random.randint(1, len(row['names_list'])-1)   # 9/10 of cases it's the first name
            name = row['names_list'][name_idx]
            typo = random.choices(['skip', 'double', 'reverse', 'miss'], [2, 5, 4 if len(name) > 2 else 0, 1 if len(name)>4 else 0], k=1)
            char = ' '
            while char == ' ':
                char_id = random.randint(1, len(name)-2 if len(name) > 2 else len(name)-1)
                char = name[char_id]
            if typo == 'skip':
                newname = name[:char_id, char_id+1:]
            elif typo == 'double':
                newname = name[:char_id+1, char_id:]
            elif typo == 'reverse':
                newname = name[:char_id] + name[char_id+1] + name[char_id] + name[char_id+2:]
            else:
                newname = name[:char_id] + miss_char(name[char_id]) + name[char_id+1:]
            if newname not in typonames:
                typonames.append(newname)

        # insert vg name for another object "alt" in the image
        altnames = []
        abort = 0
        while len(altnames) < num_alts and abort < 20:
            object = random.sample(vg_dict[row['vg_img_id']]['objects'], 1)[0]
            newname = random.sample(object['names'], 1)[0]
            if newname not in row['spellchecked'] and ' ' not in newname and newname.isalpha() and newname not in typonames:
                altnames.append(newname)
            else:
                abort += 1


        # insert completely random vg names
        randomnames = []
        while len(randomnames) < num_random:
            newname = random.sample(vg_names, 1)[0]
            if newname not in row['spellchecked'] and ' ' not in newname and newname.isalpha() and newname not in altnames + typonames:
                randomnames.append(newname)

        # add to names in dataframe
        df.at[i, 'n_fillers'] = len(typonames) + len(randomnames) + len(altnames)
        df.at[i, 'names_list'] += typonames + randomnames + altnames

        # store meta-info too
        if row['vg_is_max']:
            df.at[i, 'quality_control_dict'].update({row['names_list'][0]: 'pos'})
        df.at[i, 'quality_control_dict'].update({name: 'typo' for name in typonames})
        df.at[i, 'quality_control_dict'].update({name: 'rand' for name in randomnames})
        df.at[i, 'quality_control_dict'].update({name: 'alt' for name in altnames})

    # update
    df['n_names'] = df['names_list'].apply(len)


    # Print some stats
    print("Total:")
    print(" Number of images:", len(df))
    print(" Min n_names: {} (of which fillers: {})".format(df['n_names'].min(), df['n_fillers'].min()))
    print(" Max n_names: {} (of which fillers: {})".format(df['n_names'].max(), df['n_fillers'].max()))
    print(" Mean n_names: {} (of which fillers: {})".format(df['n_names'].mean(), df['n_fillers'].mean()))
    print()

    # For pilot, restrict to only already annotated images
    if PHASE in ["pilot", "pre-pilot"]:
        # # for initial, internal pilot by Carina:
        # sample_df = sample_objects("proc_data_phase0/spellchecking/all_responses_round0-3_cleaned.csv", 30, relevant_cols)

        print("Restricting dataset for {}.".format(PHASE))

        # now, simply reuse previously annotated images:
        df_annotated = pd.read_csv(manual_annotations, sep="\t")
        annotated_urls = df_annotated['url'].unique()
        df = df.loc[df['url'].isin(annotated_urls)]

    if PHASE in ["pre-pilot"]:
        df = df.sample(frac=.3)


    if PHASE in ["pilot", "pre-pilot"]:
        print(" Number of images:", len(df))
        print(" Min n_names: {} (of which fillers: {})".format(df['n_names'].min(), df['n_fillers'].min()))
        print(" Max n_names: {} (of which fillers: {})".format(df['n_names'].max(), df['n_fillers'].max()))
        print(" Mean n_names: {} (of which fillers: {})".format(df['n_names'].mean(), df['n_fillers'].mean()))


    # Create roughly equal-sized HITs:
    n_bins = math.ceil(len(df) / MAX_IMAGES_PER_HIT)
    bins = [[] for _ in range(n_bins)]
    bin_weights = [0] * n_bins
    full_bins = []
    full_bin_weights = []
    df_shuffled = df.sample(frac=1).sort_values(by='n_names', ascending=False)
    for i, row in tqdm(df_shuffled.iterrows()):
        min_idx = min(list(range(len(bin_weights))), key=lambda x: bin_weights[x])
        bins[min_idx].append(i)
        bin_weights[min_idx] += row['n_names']
        if len(bins[min_idx]) == MAX_IMAGES_PER_HIT:
            full_bins.append(bins.pop(min_idx))
            full_bin_weights.append(bin_weights.pop(min_idx))

    bins = full_bins + bins
    bin_weights = full_bin_weights + bin_weights

    print("Bins: {} (min: {}, max: {}, mean: {})".format(len(full_bins), min(bin_weights), max(bin_weights), sum(bin_weights)/len(bin_weights)))

    # Now turn bins into rows of the AMT csv:
    header = [["image_url_{}".format(i)] + ["name_{}_{}".format(i,n) for n in range(MAX_NAMES_PER_IMAGE)] + ["quality_control"] for i in range(MAX_IMAGES_PER_HIT)]
    header = [e for l in header for e in l]
    rows = []
    for bin in bins:
        row = [[""] + ["" for _ in range(MAX_NAMES_PER_IMAGE)] + ["{}"] for _ in range(MAX_IMAGES_PER_HIT)]
        for i, idx in enumerate(bin[::-1]):
            row[i][0] = df.at[idx, 'url']
            # Shuffle names_list (already contains fillers)
            names_list = df.at[idx, 'names_list'].copy()
            random.shuffle(names_list)
            for j, name in enumerate(names_list):
                row[i][j+1] = name
            row[i][-1] = str(df.at[idx, 'quality_control_dict'])
        row = [e for l in row for e in l]
        rows.append(row)

    df_amt = pd.DataFrame(rows, columns=header)

    print(df_amt[:20].to_string())

    with open('1_pre-pilot/verification_{}2_amt.csv'.format(PHASE), 'w+') as outfile:
        df_amt.to_csv(outfile, index=False)
        print("Csv written to", outfile.name)



def miss_char(c):
    keyboard = [list('qwertyuiop'),
                list('asdfghjkl'),
                list('zxcvbnm')]
    index = None
    for row in range(len(keyboard)):
        for col in range(len(keyboard[row])):
            if keyboard[row][col] == c:
                index = (row,col)
    if index is None:
        # character not found, setting to 'f'
        index = (1,3)

    newindex = index
    while index == newindex:
        newrow = index[0] + random.randint(-1,1)
        newcol = index[1] + round(random.random() > .8) * random.randint(-1,1)
        if newrow in [0,1,2]:
            if newcol >= 0 and newcol < len(keyboard[newrow]):
                newindex = (newrow, newcol)

    newchar = keyboard[newindex[0]][newindex[1]]

    return newchar


def preprocess_responses(filename, relevant_cols):
    resdf = load_results.load_cleaned_results(filename)

    new_df = pd.DataFrame(resdf[relevant_cols])
    new_df['spellchecked_min2'] = new_df['spellchecked'].apply(lambda x: Counter({k: x[k] for k in x if x[k] > 1}))
    # vocab_counter = Counter()
    vg_is_common = []
    ntypes = []
    ntypes_notvg = []

    for ix, row in new_df.iterrows():
        # vocab_counter += row['spellchecked']
        max_name = row['spellchecked'].most_common(1)[0][0]
        vg_is_common.append(int(max_name == row['vg_obj_name']))
        resp_ntypes = len(row['spellchecked_min2'].keys())
        ntypes.append(resp_ntypes)
        ntypes_notvg.append(resp_ntypes - (row['vg_obj_name'] in row["spellchecked_min2"]))

    new_df['vg_is_max'] = vg_is_common
    new_df['n_types'] = ntypes
    new_df['n_types_notvg'] = ntypes_notvg
    return new_df


if __name__ == "__main__":
    main()
