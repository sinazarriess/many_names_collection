# global variables
ADEQUACY_THRESHOLD = 0.5
CANONICAL_THRESHOLD = 0.5
NUM_SAMPLES_FOR_STABILITY = 5

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ast import literal_eval
from collections import Counter
import itertools
from tqdm import tqdm

from analysis import load_results

import random
random.seed(12345)

pd.set_option('display.max_colwidth', 180)
pd.set_option('expand_frame_repr', False)


#csvfile='../proc_data_phase0/verification/all_responses_round0-3_verified.csv'
#csvfile='kk.csv'
csvfile = '../proc_data_phase0/verification/all_responses_round0-3_verified.csv'
df = load_results.load_cleaned_results(csvfile)

# df = pd.read_csv(csvfile,
#     converters={'responses_r0': eval, 'responses_r1': eval, 'responses_r2': eval, 'responses_r3': eval, 'verified': eval},
#     sep="\t")
# df['spellchecked_min2'] = df['spellchecked_min2'].apply(eval)


total_objects=len(df)
print("Total objects: ", total_objects)
print(df.columns)



if True:
    print("WARNING: LOOKING ONLY AT PART OF THE DATA FOR DEBUGGING")
    df = df[:500]


# get necessary data from csv file
adequacies=[] # number of adequate names (total)
tot_names=0; tot_ad=0 # total n of names; n of -adequate names (see threshold above)
list_n_ad_names=[] # number of adequate names per canonical object
list_entry_names=[] # entry names for each canonical object (cluster 0)
list_freqs_entry_name=[] # entry names for each canonical object (cluster 0)
no_entry_lev=0
for row in df.itertuples():
    name_freqs=row.spellchecked_min2
    #print(name_freqs)
    # find entry-level name; this should be done for the canonical object,
    # but I can't recover the info from the verified column (name data in a dictionary),
    # so I use the one from all data and check if the entry level name form those data is in cluster 0
    # there are only 700 imgs where the most frequent name is not in cluster 0; TODO: check these cases; mend this
    entry_name,entry_freq=name_freqs.most_common(1)[0] # returns tuple, e.g. ('man',11) # problem: what if there's a tie of most common names? (happens, but rarely)
    d = row.verified # dictionary with verification data
    n_ad_names_img=0
    freqs_canon_names=1 # *** SMOOTHING FOR THE CASE WHERE THERE ARE 0 CANONICAL NAMES; TO MEND IN FUTURE ***
    entry_name_in_cluster0=False
    for name in d.keys(): # each name for this object
        #print("\t",name)
        tot_names+=1
        values_dict = d[name]
        ad=values_dict['adequacy']
        adequacies.append(ad);
        if ad > ADEQUACY_THRESHOLD:
            tot_ad += 1
            if values_dict['cluster_id']==0: # labels canonical object
                n_ad_names_img += 1 # number of adequate names for that image
                freqs_canon_names += name_freqs[name] # total frequency of canonical names
                if name==entry_name: entry_name_in_cluster0=True

#    print("\t",freqs_canon_names)
    if entry_name_in_cluster0==False: entry_name,entry_freq=None,0 ### TO BE MENDED IN THE FUTURE (see comment 'find entry-level name' above)
    list_n_ad_names.append(n_ad_names_img)
    list_entry_names.append(entry_name)
    freq_entry_name=(entry_freq+1)/freqs_canon_names # *** SMOOTHING -- TO MEND ***
    list_freqs_entry_name.append(freq_entry_name)

df['n_canonical_names']=list_n_ad_names
df['entry_name']=list_entry_names
df['entry_freq']=list_freqs_entry_name


# find rows where entry name is adequate and in frequent enough cluster;
# for each row, construct list of names with repetitions, with or without inadequate names, and within or without cluster

settings = ['all', 'adequate', 'all_canonical', 'adequate_canonical']
for set in settings:
    df['names_'+set] = [[] for _ in range(len(df))]

for i, row in tqdm(df.iterrows(), total=len(df)):
    if row['entry_name'] is not None:
        entry_name_info = row['verified'][row['entry_name']]
        if entry_name_info['adequacy'] > ADEQUACY_THRESHOLD and entry_name_info['cluster_id'] == 0 and entry_name_info['cluster_weight'] > CANONICAL_THRESHOLD:
            all_names = list(itertools.chain(*[[name] * row['spellchecked_min2'][name] for name in row['spellchecked_min2']]))
            df.at[i, 'names_all'] = all_names
            df.at[i, 'names_adequate'] = [n for n in all_names if row['verified'][n]['adequacy'] > ADEQUACY_THRESHOLD]
            df.at[i, 'names_all_canonical'] = [n for n in all_names if row['verified'][n]['cluster_id'] == 0]
            df.at[i, 'names_adequate_canonical'] = [n for n in all_names if row['verified'][n]['cluster_id'] == 0 and row['verified'][n]['adequacy'] > ADEQUACY_THRESHOLD]

for i, row in tqdm(df.iterrows(), total=len(df)):
    for set in settings:
        names = row['names_'+set]
        if len(names) > 0:
            earliest_stabilities = []
            for _ in range(NUM_SAMPLES_FOR_STABILITY):
                random.shuffle(names)
                # Walk backwards through shuffled list of names until entry_name is NOT a majority anymore
                for j in range(len(names)-1, -1, -1):
                    if not row['entry_name'] in [t[0] for t in Counter(names[:j+1]).most_common()]:
                        j += 1  # back up
                        break
                earliest_stabilities.append(j)
            earliest_stability_avg = (sum(earliest_stabilities)/len(earliest_stabilities))/len(names)
            df.at[i, 'stability_'+set] = earliest_stability_avg

for set in settings:
    del df['names_' + set]

print("df:", len(df))
print(df[:10].to_string())

stacked = df[['stability_all','stability_adequate','stability_all_canonical','stability_adequate_canonical']].stack().reset_index()
stacked.rename(columns={'level_1': 'setting', 0: 'stability'}, inplace=True)
print(stacked[:10].to_string())
stacked.hist(column='stability', by='setting')
plt.show()

# TODO Now plot the distribution:   stability_all  stability_adequate  stability_all_canonical  stability_adequate_canonical (along with entry_freq?)