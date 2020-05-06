# -*- coding: utf-8 -*-

# created gbt Sun Feb 24 2019
# modified gbt Dec 4 2019

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ast import literal_eval
from collections import Counter

pd.set_option('display.max_colwidth', 180)
pd.set_option('expand_frame_repr', False)

def unpack_verified_column(df):
    # get necessary data from csv file
    adequacies=[] # number of adequate names (total)
    tot_names=0; tot_ad=0 # total n of names; n of -adequate names (see threshold above)
    list_n_ad_names=[] # number of adequate names per canonical object
    list_entry_names=[] # entry names for each canonical object (cluster 0)
    list_freqs_entry_name=[] # entry names for each canonical object (cluster 0)
    no_entry_lev=0
    for row in df.itertuples():
        name_freqs=eval(row.spellchecked_min2)
        #print(name_freqs)
        # find entry-level name; this should be done for the canonical object,
        # but I can't recover the info from the verified column (name data in a dictionary),
        # so I use the one from all data and check if the entry level name form those data is in cluster 0
        # there are only 700 imgs where the most frequent name is not in cluster 0; TODO: check these cases; mend this
        entry_name,entry_freq=name_freqs.most_common(1)[0] # returns tuple, e.g. ('man',11) # problem: what if there's a tie of most common names? (happens, but rarely)
        d = literal_eval(row.verified) # dictionary with verification data
        n_ad_names_img=0
        freqs_canon_names=1 # *** SMOOTHING FOR THE CASE WHERE THERE ARE 0 CANONICAL NAMES; TO MEND IN FUTURE ***
        entry_name_in_cluster0=False
        for name in d.keys(): # each name for this object
            #print("\t",name)
            tot_names+=1
            values_dict = d[name]
            ad=values_dict['adequacy']
            adequacies.append(ad); 
            if ad > adequacy_threshold:
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

def get_stats_stability_top_response(df):
     # TBDONE...
# get necessary data from csv file
for row in df.itertuples():
    name_freqs=eval(row.responses_r0)
    scores_per_name['adequacy_bin'] = scores_per_name['adequacy'].apply(lambda x: [0 if a == 0 else 1 for a in x])

#        {'batter': 3, 'baseball player': 1, 'helmet': 1, 'man': 1, 'boy': 1, 'uniform': 1, 'player': 1}
    print(name_freqs)
    # find top response
    entry_name,_=name_freqs.most_common(1)[0] # returns tuple, e.g. ('man',11) # problem: what if there's a tie of most common names? (happens, but rarely)

    for name in d.keys(): # each name for this object
        #print("\t",name)
        tot_names+=1
        values_dict = d[name]
        ad=values_dict['adequacy']
        adequacies.append(ad); 
        if ad > adequacy_threshold:
            tot_ad += 1
            if values_dict['cluster_id']==0: # labels canonical object
                n_ad_names_img += 1 # number of adequate names for that image
                freqs_canon_names += name_freqs[name] # total frequency of canonical names
                if name==entry_name: entry_name_in_cluster0=True

#    print("\t",freqs_canon_names)
    if entry_name_in_cluster0==False: entry_name,entry_freq=None,0 ### TO BE MENDED IN THE FUTURE (see comment 'find entry-level name' above)
    list_n_ad_names.append(n_ad_names_img)
    
#    df['entry_freq']=list_freqs_entry_name
        
csvfile = '../proc_data_phase0/verification/all_responses_round0-3_verified_entry-level-focus.csv'
#df = pd.read_csv(csvfile, 
#    converters={'responses_r0': eval, 'responses_r1': eval, 'responses_r2': eval, 'responses_r3': eval},
#    sep="\t")
#total_objects=len(df)

#'all_responses'
#name_freqs=eval(row.spellchecked_min2)
##print(name_freqs)
#entry_name,entry_freq=name_freqs.most_common(1)[0] # returns tuple, e.g. ('man',11) # problem: what if there's a tie of most common names? (happens, but rarely)
#
#for rcolumn, tcolumn in zip(r,t):
#    df[tcolumn]=df[rcolumn].apply(lambda x: max(x, key=x.get))

