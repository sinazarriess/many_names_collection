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

from nltk.corpus import wordnet as wn


PHASE = "round0" # "pre-pilot" # "pilot", "phase0", "phase1", None
IMAGES_PER_HIT = 6

MAX_IMAGE_PARAMS_PER_HIT = 11
MAX_NAME_PARAMS_PER_IMAGE = 16

REMOVE_SYNONYMS = False


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

    print(df[:5].to_string())


    ## Identify and, if required, remove WordNet synonyms
    if REMOVE_SYNONYMS:
        n_images = len(df)
        n_names = df['n_names'].sum()
        print("Prior to synonym filter: {} images, {} names".format(n_images, n_names))

    df['synonym_clusters'] = [[] for _ in range(len(df))]
    for i, row in tqdm(df.iterrows()):
        # organize names into synset clusters:
        synset_to_names = {}
        name_to_synsets = {}
        for name in row['names_list']:
            name_to_synsets[name] = [synset.name() for synset in wn.synsets(name)]
            for synset in name_to_synsets[name]:
                if not synset in synset_to_names:
                    synset_to_names[synset] = [name]
                elif not name in synset_to_names[synset]:
                    synset_to_names[synset].append(name)

        # choose the biggest synset for each name (greedy)
        synsets_list = []
        for name in row['names_list']:
            synsets = sorted([synset for synset in name_to_synsets[name]], key=lambda x: len(synset_to_names[x]))
            if len(synsets) == 0:
                synsets_list.append(None)
            else:
                synsets_list.append(synsets[-1])
        df.at[i, 'synonym_clusters'] = [synset_to_names[synset] for synset in synsets_list if synset is not None]

        if REMOVE_SYNONYMS:
            names_list_no_synonyms = []
            synsets_covered = []
            for n, name in enumerate(row['names_list']):
                if synsets_list[n] is None or synsets_list[n] not in synsets_covered or sum([name in synset_to_names[ss] for ss in set(synsets_list) if ss is not None]) > 1:
                    names_list_no_synonyms.append(name)
                    synsets_covered.append(synsets_list[n])
                else:
                    pass
            df.at[i, 'names_list'] = names_list_no_synonyms


    # update
    df['n_names'] = df['names_list'].apply(len)
    # Restrict to multiple names:
    df = df.loc[df['n_names'] > 1]

    if REMOVE_SYNONYMS:
        n_images_new = len(df)
        n_names_new = df['n_names'].sum()
        print("After synonym filter: {} images (-{}%), {} names (-{}%)".format(n_images_new, int(round(100*(n_images - n_images_new) / n_images)), n_names_new, int(round(100*(n_names - n_names_new) / n_names)),))



    # Create quality control items
    with open('../dataset_creation/add_data/vgenome/objects.json') as file:
        vg_data = json.load(file)
    vg_dict = {img['image_id']: img for img in vg_data}

    vg_names = list(set([name for img in vg_data for object in img['objects'] for name in object['names']]))

    df['quality_control_dict'] = [{} for _ in range(len(df))]
    df['n_fillers'] = [0 for _ in range(len(df))]

    for i, row in tqdm(df.iterrows()):

        newnames = []

        outer_abort = 0
        while len(newnames) < .15 * len(row['names_list']) and outer_abort < 10:    # TODO improve these hideous nested while loops

            choice = random.choices(['typo', 'alt', 'random'], [2,3,2], k=1)[0]

            # randomly generate name variants with spelling error
            if choice == 'typo':
                abort = 0
                while abort < 20:
                    name_idx = 0 if random.random() > .3 else random.randint(1, len(row['names_list'])-1)   # 7/10 of cases it's the first name
                    name = row['names_list'][name_idx]
                    if len(name) < 6 and abort < 10:
                        newname = None
                        abort += 1
                    elif len(name) < 4:
                        newname = None
                        abort += 1
                    else:
                        typo = random.choices(['skip', 'double', 'reverse', 'miss'], [2, 1, 2, 1], k=1)[0]
                        char = ' '
                        while char == ' ':
                            char_id = random.randint(1, len(name)-2 if len(name) > 2 else len(name)-1)
                            char = name[char_id]
                        if typo == 'skip':
                            newname = name[:char_id] + name[char_id+1:]
                        elif typo == 'double':
                            newname = name[:char_id+1] + name[char_id:]
                        elif typo == 'reverse':
                            newname = name[:char_id] + name[char_id+1] + name[char_id] + name[char_id+2:]
                        else:
                            newname = name[:char_id] + miss_char(name[char_id]) + name[char_id+1:]
                        if newname in vg_names or newname in newnames:
                            newname = None
                            abort += 1
                        else:
                            df.at[i, 'quality_control_dict'].update({newname: 'typo-{}'.format(name)})
                            newnames.append(newname)
                            break

            # insert vg name for another object "alt" in the image
            elif choice == 'alt':
                abort = 0
                while abort < 20:
                    object = random.sample(vg_dict[row['vg_img_id']]['objects'], 1)[0]
                    original_object = [obj for obj in vg_dict[row['vg_img_id']]['objects'] if int(obj['object_id']) == int(row['vg_object_id'])][0]
                    newname = random.sample(object['names'], 1)[0]
                    if int(object['object_id']) == int(row['vg_object_id']): # not the same object
                        abort += 1
                        newname = None
                    elif abs(original_object['x'] - object['x']) < 20 or abs(original_object['y'] - object['y']) < 20: # bounding box not too close
                        abort += 1
                        newname = None
                    # elif object['w'] < 1.4*original_object['w'] and object['w'] > 0.6*original_object['w']: # bounding box not too wide or narrow
                    #     abort += 1
                    # elif object['h'] < 1.4*original_object['h'] and object['h'] > 0.6*original_object['h']: # bounding box not too high or low
                    #     abort += 1
                    elif newname in row['spellchecked'] or newname == row['vg_obj_name'] or newname in newnames: # not an existing name
                        abort += 1
                        newname = None
                    elif ' ' in newname or not newname.isalpha():   # no non-alphabetical stuff
                        abort += 1
                        newname = None
                    else:   # only then it's a good name for an alternative object
                        df.at[i, 'quality_control_dict'].update({newname: 'alt'})
                        newnames.append(newname)
                        break

            elif choice == 'random':
                # insert completely random vg names
                abort = 0
                while abort < 20:
                    newname = random.sample(vg_names, 1)[0]
                    if len(newname) > 7 or newname in row['spellchecked'] or ' ' in newname or not newname.isalpha() or newname in newnames: # proper unused name
                        abort += 1
                        newname = None
                    elif newname in [name for object in vg_dict[row['vg_img_id']]['objects'] for name in object['names']]: # not vg name for object in current img
                        abort += 1
                        newname = None
                    else: # only then it's a good random name
                        df.at[i, 'quality_control_dict'].update({newname: 'rand'})
                        newnames.append(newname)
                        break

            if abort == 20:
                outer_abort += 1

        # add to names in dataframe
        df.at[i, 'n_fillers'] = len(newnames)
        df.at[i, 'names_list'] += newnames

        # positive item:
        if row['vg_is_max']:
            df.at[i, 'quality_control_dict'].update({row['names_list'][0]: 'pos'})

        # synonym items:
        clusters = [cluster for cluster in row['synonym_clusters'] if len(cluster) > 1]
        if len(clusters) > 0:
            synonyms = random.sample(clusters, 1)[0]
            synonyms_non_pos = [n for n in synonyms if not n in row['quality_control_dict']]
            target = random.sample(synonyms_non_pos, 1)[0]
            df.at[i, 'quality_control_dict'].update({target: 'syn-{}'.format(','.join([n for n in synonyms if n != target]))})

            # TODO add quality control to javascript

    # update
    df['n_names'] = df['names_list'].apply(len)


    # Summarize data
    print("Total:")
    print(" Number of images:", len(df))
    print(" Min n_names: {} (of which fillers: {})".format(df['n_names'].min(), df['n_fillers'].min()))
    print(" Max n_names: {} (of which fillers: {})".format(df['n_names'].max(), df['n_fillers'].max()))
    print(" Mean n_names: {} (of which fillers: {})".format(df['n_names'].mean(), df['n_fillers'].mean()))
    quality_controls = ["typo" if x.startswith("typo") else "syn" if x.startswith("syn") else x for y in df['quality_control_dict'].apply(lambda x: list(x.values())).tolist() for x in y]
    print("  Pos: {}, typo: {}, alt: {}, rand: {}, syn: {}".format(sum([x == "pos" for x in quality_controls]), sum([x == "typo" for x in quality_controls]), sum([x == "alt" for x in quality_controls]), sum([x == "rand" for x in quality_controls]), sum([x == "syn" for x in quality_controls])))

    print()

    # Data restrictions depending on round
    if PHASE is not None:
        print("Restricting dataset for {}.".format(PHASE))

        if PHASE in ["round0", "round1"]:
            if PHASE == "round0":
                with open('test_imgids/bottomup.nottopMN.imgids') as imgids:
                    imgids = [int(s.strip()) for s in imgids]
            elif PHASE == "round1":
                with open('test_imgids/bottomup.nottrain.imgids') as imgids:
                    imgids = [int(s.strip()) for s in imgids]
            df = df.loc[df['vg_img_id'].isin(imgids)]

        # For pilot, restrict to only already annotated images
        elif PHASE in ["pilot", "pre-pilot"]:

            # now, simply reuse previously annotated images:
            df_annotated = pd.read_csv(manual_annotations, sep="\t")
            annotated_urls = df_annotated['url'].unique()
            df = df.loc[df['url'].isin(annotated_urls)]

            # For pre-pilot, restrict further
            if PHASE in ["pre-pilot"]:
                df = df.sample(frac=.3)

     # either way, summarize new data:
        print(" Number of images:", len(df))
        print(" Min n_names: {} (of which fillers: {})".format(df['n_names'].min(), df['n_fillers'].min()))
        print(" Max n_names: {} (of which fillers: {})".format(df['n_names'].max(), df['n_fillers'].max()))
        print(" Mean n_names: {} (of which fillers: {})".format(df['n_names'].mean(), df['n_fillers'].mean()))
        quality_controls = ["typo" if x.startswith("typo") else "syn" if x.startswith("syn") else x for y in df['quality_control_dict'].apply(lambda x: list(x.values())).tolist() for x in y]
        print("  Pos: {}, typo: {}, alt: {}, rand: {}, syn: {}".format(sum([x == "pos" for x in quality_controls]), sum([x == "typo" for x in quality_controls]), sum([x == "alt" for x in quality_controls]), sum([x == "rand" for x in quality_controls]), sum([x == "syn" for x in quality_controls])))


    # Create roughly equal-sized HITs:
    n_bins = math.ceil(len(df) / IMAGES_PER_HIT)
    bins = [[] for _ in range(n_bins)]
    bin_weights = [0] * n_bins
    full_bins = []
    full_bin_weights = []
    df_shuffled = df.sample(frac=1).sort_values(by='n_names', ascending=False)
    for i, row in tqdm(df_shuffled.iterrows()):
        min_idx = min(list(range(len(bin_weights))), key=lambda x: bin_weights[x])
        bins[min_idx].append(i)
        bin_weights[min_idx] += row['n_names']
        if len(bins[min_idx]) == IMAGES_PER_HIT:
            full_bins.append(bins.pop(min_idx))
            full_bin_weights.append(bin_weights.pop(min_idx))

    bins = full_bins + bins
    bin_weights = full_bin_weights + bin_weights

    print("Bins: {} (min: {}, max: {}, mean: {})".format(len(full_bins), min(bin_weights), max(bin_weights), sum(bin_weights)/len(bin_weights)))



    # Now turn bins into rows of the AMT csv:
    header = [["image_url_{}".format(i)] + ["name_{}_{}".format(i,n) for n in range(MAX_NAME_PARAMS_PER_IMAGE)] + ["quality_control_{}".format(i)] for i in range(MAX_IMAGE_PARAMS_PER_HIT)]
    header = [e for l in header for e in l]
    rows = []
    n_controls = []
    n_controls_pos = []
    for bin in bins:
        row = [[""] + ["" for _ in range(MAX_NAME_PARAMS_PER_IMAGE)] + ["{}".encode('utf-8').hex()] for _ in range(MAX_IMAGE_PARAMS_PER_HIT)]
        controls = 0
        controls_pos = 0
        n_names = 0
        for i, idx in enumerate(bin[::-1]):
            row[i][0] = df.at[idx, 'url']
            # Shuffle names_list (already contains fillers)
            names_list = df.at[idx, 'names_list'].copy()
            random.shuffle(names_list)
            for j, name in enumerate(names_list):
                row[i][j+1] = name
                n_names += 1
            # Obfuscate with font labels: TODO Move the obfuscation down; do more globally, separately.
            for key in df.at[idx, 'quality_control_dict']:
                controls += 1
                if df.at[idx, 'quality_control_dict'][key] == "pos":
                    df.at[idx, 'quality_control_dict'][key] = "arial"
                    controls_pos += 1
                elif df.at[idx, 'quality_control_dict'][key].startswith("typo"):
                    df.at[idx, 'quality_control_dict'][key] = "sans" + df.at[idx, 'quality_control_dict'][key][4:]
                elif df.at[idx, 'quality_control_dict'][key] == "alt":
                    df.at[idx, 'quality_control_dict'][key] = "serif"
                elif df.at[idx, 'quality_control_dict'][key] == "rand":
                    df.at[idx, 'quality_control_dict'][key] = "courier"
                elif df.at[idx, 'quality_control_dict'][key].startswith("syn"):
                    df.at[idx, 'quality_control_dict'][key] = "bold" + df.at[idx, 'quality_control_dict'][key][3:]
            # Obfuscate further with hex encoding
            row[i][-1] = str(df.at[idx, 'quality_control_dict']).replace("'", '"').encode('utf-8').hex()
        row = [e for l in row for e in l]
        rows.append(row)
        n_controls.append(controls/n_names)
        n_controls_pos.append(controls_pos/n_names)

    df_amt = pd.DataFrame(rows, columns=header)

    print("Controls:", sum(n_controls)/len(n_controls), min(n_controls), max(n_controls), "of which pos:", sum(n_controls_pos)/len(n_controls_pos), min(n_controls_pos), max(n_controls_pos))

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
