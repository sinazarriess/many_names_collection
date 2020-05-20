#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
from collections import Counter, defaultdict
import re
import sys

import numpy as np

import load_results
from agreement_table import snodgrass_agreement

COL_ORDER = ['vg_image_id', 
             'vg_object_id', 
             'url', 
             'topname', 
             'domain',
             'responses', 
             'N', 
             'perc_top', 
             'H', 
             'singletons', 
             'vg_obj_name', 
             'vg_domain', 
             'vg_synset', 
             #'vg_cat'
             ]

COL_MAP = {
        'vg_img_id': 'vg_image_id', 
        'object_id': 'vg_object_id', 
        #'cat': 'vg_cat',
        'synset': 'vg_synset',
        'spellchecked': 'responses',       
        'mn_topname': 'topname',
        'mn_domain': 'domain'
        }
        
#### Streamline column names for MN v1.0 and v2.0
def streamline_columns(mn_df):
    mn_df.rename(columns=COL_MAP, inplace=True)
    mn_df.sort_values(by=["vg_image_id"], inplace=True)

#### Functions for preprocessing MN version 0 and 1  ####
def nm2domain_map(df):
    d = dict(df[["vg_obj_name", "vg_domain"]].values)
    d["hotdog"] = "food"
    return d

def add_mn_domains(df):
    df.rename(columns={"spellchecked_min2": "responses_min2"}, inplace=True)
    is_mn_v2 = "vg_same_object" in df.columns
    relevant_columns = ["responses_min2", "vg_domain", "vg_obj_name"]
    if is_mn_v2:
        relevant_columns.append(["vg_same_object"])
    
    nm2domain = nm2domain_map(df)
    preferred_names = []
    mn_domains = []
    for (idx, responses) in df[relevant_columns].iterrows():
        top_name = responses["responses_min2"].most_common(1)[0][0]
        vg_domain = responses["vg_domain"]
        mn_domain = np.nan
        if top_name == responses["vg_obj_name"]:
            mn_domain = vg_domain
        elif is_mn_v2 and "vg_same_object" in top_name not in nm2domain and responses["vg_same_object"][top_name]>0.7: # names refer to same object
            mn_domain = vg_domain
            #print("  retrieved from VG name: ", top_name, vg_domain)
        else:
            # do majority voting using all MN names
            nm2do = defaultdict(int)
            for (nm,cnt) in responses["responses_min2"].items():
                nm2do[nm2domain.get(nm, "X")] += cnt
            pot_mn_domain =Counter(nm2do).most_common(1)[0][0]
            mn_domain = pot_mn_domain if pot_mn_domain != "X" else vg_domain
            #print("  retrieved from majority voting or VG name: ", top_name, "VG:", vg_domain, "MN:", mn_domain, "pot:", pot_mn_domain)
            
        mn_domains.append(mn_domain)
        
    return mn_domains

def relfrequ(responses):
    total = sum(responses.values())
    return {a[0]:(a[1]/total) for a in responses.items()}

#### Process ManyNames version 0.0 --> version 1.0
def convert_manynames_v0_to_v1(manynames_v0):
    # add the following information: responses_min2, mn_topname, mn_domain, N, %top, H
    manynames_v0["responses_min2"] = manynames_v0.apply(lambda a: Counter({nm:cnt for (nm,cnt) in a["spellchecked"].items() if cnt>1}), axis=1)
    manynames_v0["singletons"] = manynames_v0.apply(lambda a: {nm:cnt for (nm,cnt) in a["spellchecked"].items() if cnt<=1}, axis=1)
    manynames_v0["mn_topname"] = manynames_v0.apply(lambda a: a["spellchecked"].most_common(1)[0][0], axis=1)
    manynames_v0["mn_domain"] = add_mn_domains(manynames_v0)
    manynames_v0["N"] = manynames_v0.apply(lambda a: len(a["responses_min2"]), axis=1)
    manynames_v0["perc_top"] = manynames_v0.apply(
        lambda a: round(100*a["responses_min2"].most_common(1)[0][1]/sum(a["responses_min2"].values()),1), axis=1)
    manynames_v0['H'] = manynames_v0['responses_min2'].apply(lambda a: round(snodgrass_agreement(a, {}, True),2))
    
    # the MN responses contain only names with count>1:
    manynames_v0['responses'] = manynames_v0['responses_min2']
    manynames_v0.drop("responses_min2", axis=1, inplace=True)
    manynames_v0.drop("spellchecked", axis=1, inplace=True)    

if __name__=="__main__":
    target_dir = os.path.join("..", "proc_data_phase0", "mn_v1.0/")
    out_fname = os.path.join(target_dir, "manynames_v1.0_public.tsv")

    manynamesv0_path = os.path.join("..", "proc_data_phase0", "spellchecking", "all_responses_round0-3_cleaned.csv")
    manynames_v0 = load_results.load_cleaned_results(manynamesv0_path)
 
    manynames_v0 = manynames_v0[['vg_img_id', 'synset', 'vg_obj_name', 'vg_domain', 'url', 'vg_object_id', 'spellchecked']]

    convert_manynames_v0_to_v1(manynames_v0)
    streamline_columns(manynames_v0)
       
    if out_fname is  not None:
        manynames_v0[COL_ORDER].to_csv(out_fname, sep="\t", index=False)
        sys.stdout.write("Manynames_v1.0 written to %s.\n" % out_fname)
    else:
        manynames_v0 = manynames_v0[COL_ORDER]
    
    
   
    
    
    
    
    
    
    