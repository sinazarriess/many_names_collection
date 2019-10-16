import os
import sys
from collections import Counter
import math

import pandas as pd

import numpy as np

from analysis import load_results
from tqdm import tqdm

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


def sample_objects(filename, k, relevant_cols):
    """
    For each domain:
        - sample k objects, uniformly:
          number of types from [1, ..., M] with M==max number of types in domain; 1 name type !=VG name

    Args:
        k: the number of objects per domain to be samped
    """
    df = preprocess_responses(filename, relevant_cols)
    samples = pd.DataFrame(columns=relevant_cols)

    domains = df["vg_domain"].unique().tolist()
    for domain in domains:
        domaindf = df[df["vg_domain"] == domain]

        denom = domaindf["n_types_notvg"] != 0
        max_ntypes = max(domaindf["n_types_notvg"])
        mean_ntypes = int(sum(domaindf["n_types_notvg"]) / len(domaindf[domaindf["n_types_notvg"] != 0]))
        samples_per_ntype = int(k / max(domaindf["n_types_notvg"]))
        add_samples_from_mean_ntypes = k - (max_ntypes * samples_per_ntype)

        for i in range(1, max_ntypes + 1):
            if i == mean_ntypes:
                # sample objects from set with mean number of types leater, when filling up remaining objects
                continue
            ki = min(samples_per_ntype, sum(domaindf["n_types_notvg"] == i))
            samples = samples.append(domaindf[domaindf["n_types_notvg"] == i].sample(n=ki))
            add_samples_from_mean_ntypes += (samples_per_ntype - ki)

        # sample remaining objects from set with mean number of types
        samples_from_mean_ntypes = add_samples_from_mean_ntypes + samples_per_ntype
        samples = samples.append(domaindf[domaindf["n_types_notvg"] == mean_ntypes].sample(n=samples_from_mean_ntypes))

    return samples


# def _single_name_not_vg(resdf):
#    return resdf[resdf["vg_is_max"] + resdf["n_types"] == 1]

def write_amt_csv(sample_df, csv_fname, max_num_hits=500):
    num_hits = 0
    part = 0
    fname_imgObjIds = open(csv_fname.replace('.csv','') + ".imgobjids", "w")
    for _, sample in sample_df.iterrows():
        names = list(set(list(sample["spellchecked_min2"]) + [sample["vg_obj_name"]]))
        names += [""] * (10 - len(names))
        if num_hits == 0:
            fout = open(csv_fname.replace(".csv", "part%d.csv" % part), "w")
            fout.write("image_url_1,name_0,name_1,name_2,name_3,name_4,name_5,name_6,name_7,name_8,name_9\n")
        fout.write("{},{}\n".format(sample['url'], ','.join(names)))
        fname_imgObjIds.write("{0[vg_img_id]}\t{0[vg_object_id]}\n".format(sample))
        num_hits += 1
        if num_hits >= max_num_hits:
            fout.close()
            num_hits = 0
            part += 1

    if num_hits > 0:
        fout.close()
    fname_imgObjIds.close()


PILOT = True;
MAX_IMAGES_PER_HIT = 5;
MAX_NAMES_PER_IMAGE = 11;

np.random.seed(12345)

relevant_cols = ['vg_img_id', 'cat', 'synset', 'vg_obj_name', 'vg_domain', 'vg_object_id',
                 'url', 'opt-outs', 'spellchecked', 'all_responses']

df = preprocess_responses("proc_data_phase0/spellchecking/all_responses_round0-3_cleaned.csv", relevant_cols)



# Collect all names in a list
df['names_list'] = df['spellchecked_min2'].apply(lambda x: [n for c,n in sorted([(x[n], n) for n in x], reverse=True)])
# And always incude vg name:
for i, row in df.iterrows():
    if row['vg_obj_name'] not in row['names_list']:
        df.at[i,'names_list'] += [row['vg_obj_name']]
df['n_names'] = df['names_list'].apply(len)
# Restrict to multiple names:
df = df.loc[df['n_names'] > 1]

print(df.head().to_string())

# Print some stats
print("Number of images:", len(df))
print("Min n_names:", df['n_names'].min())
print("Max n_names:", df['n_names'].max())
print("Mean n_names:", df['n_names'].mean())

# For pilot, restrict to only already annotated images
if PILOT:
    # # for initial, internal pilot by Carina:
    # sample_df = sample_objects("proc_data_phase0/spellchecking/all_responses_round0-3_cleaned.csv", 30, relevant_cols)

    print("Restricting dataset for pilot.")

    # now, simply reuse previously annotated images:
    df_annotated = pd.read_csv('raw_data_phase0/verification_pilot/verif_annos_pilot.csv', sep="\t")
    annotated_urls = df_annotated['url'].unique()
    df = df.loc[df['url'].isin(annotated_urls)]

    print("Number of images:", len(df))
    print("Min n_names:", df['n_names'].min())
    print("Max n_names:", df['n_names'].max())
    print("Mean n_names:", df['n_names'].mean())

# Create roughly equal-sized HITs:
n_bins = math.ceil(len(df)/5)
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
header = [["image_url_{}".format(i)] + ["name_{}_{}".format(i,n) for n in range(MAX_NAMES_PER_IMAGE)] for i in range(MAX_IMAGES_PER_HIT)]
header = [e for l in header for e in l]
rows = []
for bin in bins:
    row = [[""] + ["" for _ in range(MAX_NAMES_PER_IMAGE)] for _ in range(MAX_IMAGES_PER_HIT)]
    for i, idx in enumerate(bin):
        row[i][0] = df.at[idx, 'url']
        for j, name in enumerate(df.at[idx, 'names_list']):
            row[i][j+1] = name
    row = [e for l in row for e in l]
    rows.append(row)

df_amt = pd.DataFrame(rows, columns=header)

print(df_amt[:20].to_string())

with open('verification_pilot_amt.csv', 'w+') as outfile:
    df_amt.to_csv(outfile, index=False)


# write rows to csv


# write_amt_csv(df, "test_amt_csv.csv")
