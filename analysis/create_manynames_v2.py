#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
from collections import Counter, defaultdict
import re
import sys

import numpy as np
from numpy import nan
from itertools import chain, combinations

from analysis import load_results


#### Functions for preprocessing MN+verifications  ####
def nm2domain_map(df):
    d = dict(df[["vg_obj_name", "vg_domain"]].values)
    d["hotdog"] = "food"
    return d

def add_mn_domains(df):
    nm2domain = nm2domain_map(df)
    preferred_names = []
    mn_domains = []
    for (idx, responses) in df[["spellchecked_min2", "vg_domain", "vg_obj_name", "vg_same_object"]].iterrows():
        top_name = responses["spellchecked_min2"].most_common(1)[0][0]
        vg_domain = responses["vg_domain"]
        mn_domain = np.nan
        if top_name == responses["vg_obj_name"]:
            mn_domain = vg_domain
        elif top_name not in nm2domain and responses["vg_same_object"][top_name]>0.7: # names refer to same object
            mn_domain = vg_domain
            #print("  retrieved from VG name: ", top_name, vg_domain)
        else:
            # do majority voting using all MN names
            nm2do = defaultdict(int)
            for (nm,cnt) in responses["spellchecked_min2"].items():
                nm2do[nm2domain.get(nm, "X")] += cnt
            pot_mn_domain = Counter(nm2do).most_common(1)[0][0]
            mn_domain = pot_mn_domain if pot_mn_domain != "X" else vg_domain
            #print("  retrieved from majority voting or VG name: ", top_name, "VG:", vg_domain, "MN:", mn_domain, "pot:", pot_mn_domain)
            
        mn_domains.append(mn_domain)
        
    return mn_domains

def relfrequ(responses):
    total = sum(responses.values())
    return {a[0]:(a[1]/total) for a in responses.items()}


#### Functions for creating new, valid MN  ####
#### (based on adequacy and same_object(.|n_1) judgements)

def get_name_df(df):
    """
    Extract all relevant information to filter names by their adequacy and same_object(.|n_1) judgements.
    """
    image_id = []
    url = []
    vg_obj_name = []
    mn_obj_name = []
    isTopName = [] # binary
    top_name = []
    mn_domain = []
    name_prob = []
    adequacy_mean = []
    same_object = []
    inadequ_type = []

    for (_, row) in df[['vg_image_id', "url", 'vg_obj_name', 'spellchecked_min2', 'adequacy_mean', 'inadequacy_type', 'same_object', 'mn_domain']].iterrows():
        mn_nameProbs = relfrequ(row['spellchecked_min2'])
        topMN = row['spellchecked_min2'].most_common()[0][0]
        for (name,prob) in mn_nameProbs.items():
            top_name.append(topMN)
            inadequ_type.append(row['inadequacy_type'][name])
            image_id.append(row["vg_image_id"])
            url.append(row["url"])
            vg_obj_name.append(row["vg_obj_name"])
            mn_obj_name.append(name)
            if name == topMN:
                isTopName.append(True)
            else:
                isTopName.append(False)
            mn_domain.append(row["mn_domain"]) 
            name_prob.append(prob) 
            adequacy_mean.append(row['adequacy_mean'][name])
            same_object.append(row['same_object'][topMN][name])
    
    name_df = pd.DataFrame(columns=['vg_image_id', 'vg_obj_name', 'mn_obj_name', "topMN", 'is_top_name', 'name_prob', 'inadequacy_type', 'adequacy_mean', 'same_object', 'mn_domain', "url"])
    name_df['vg_image_id'] = image_id
    name_df['vg_obj_name'] = vg_obj_name
    name_df['mn_obj_name'] = mn_obj_name
    name_df['topMN'] = top_name
    name_df['is_top_name'] = isTopName
    name_df['mn_domain'] = mn_domain
    name_df['name_prob'] = name_prob
    name_df['adequacy_mean'] = adequacy_mean
    name_df['inadequacy_type'] = inadequ_type
    name_df['same_object'] = same_object
    name_df['url'] = url
    return name_df

def make_filtered_df_publish(df, invalid_names_df, min_count=2):
    """
    Columns to be changed: 
        ['spellchecked', 'spellchecked_min2', 'adequacy_mean', 'inadequacy_type', 'same_object']
    Columns to be discarded:
        ['sample_type', 'spellchecked_min2']
    invalid_names (pandas DataFrame):    img_id | invalid_names 
    """
    instances_changed = 0
    all_cols = ['vg_image_id', 'url', 'vg_object_id', 'vg_obj_name', 'synset', 'cat', 'vg_domain', 
                'vg_adequacy_mean', 'vg_inadequacy_type', 'vg_same_object', 
                'mn_topname', 'mn_domain', 'spellchecked',  
                'adequacy_mean', 'inadequacy_type', 'same_object',
                'incorrect'] # incorrect: new column (dict): {name --> {count (int), adequacy_mean (float), 
                             #                                inadequacy_type (dict), same_object (dict)}, ...}
    valid_data = defaultdict(list)
    for (_, row) in df.iterrows():
        image_id = row['vg_image_id']
        inval_names = set(invalid_names_df[invalid_names_df["vg_image_id"]==image_id]["mn_obj_name"].values)
        inval_names_cnt = set([nm for (nm,cnt) in row['spellchecked'].items() if cnt < min_count])
        inval_names = inval_names.union(inval_names_cnt)
        #inval_names = invalid_names.get(image_id, None)
        topMN = row['spellchecked'].most_common()[0][0]
        incorrect_col = dict()
        
        if len(inval_names) == 0:
            for col in all_cols:
                if col == 'mn_topname':
                    valid_data['mn_topname'].append(topMN)
                elif col == 'incorrect':
                    valid_data[col].append(incorrect_col)
                else:
                    valid_data[col].append(row[col])
            continue
            
        instances_changed += 1
        # copy data columns which are not changed
        for dtype in ['vg_image_id', 'url', 'vg_object_id', 'vg_obj_name', 'synset', 'cat', 'vg_domain', 
                      'vg_adequacy_mean', 'vg_inadequacy_type', 'vg_same_object', 'mn_topname', 'mn_domain']:
            if dtype == 'mn_topname':
                valid_data['mn_topname'].append(topMN)
            elif dtype == 'vg_same_object':
                new_dict = {}
                for (mn_nm, val) in row[dtype].items():
                    if row['vg_obj_name'] != mn_nm and val > 0.0:
                        new_dict[mn_nm] = val
                valid_data[dtype].append(new_dict)        
            else:
                valid_data[dtype].append(row[dtype])
            
        # copy the parts of the data which are not deleted (simple dict)
        for col2change in ['spellchecked', 'adequacy_mean', 'inadequacy_type']:
            old_dict = row[col2change]
            new_dict = type(old_dict)()
            for (nm, val) in old_dict.items():
                nm_cnt = row['spellchecked'][nm]
                if nm not in inval_names and nm_cnt >= min_count: # remove also names with count < 2
                    new_dict[nm] = val
                else:
                    incorrect_col.setdefault(nm, dict())
                    incorrect_col[nm][col2change.replace('spellchecked', 'count')] = val
            valid_data[col2change].append(new_dict)
        
        # copy the same_object annotations (keys and keys of nested dict) for the valid names (nested dict)
        new_same_obj = defaultdict(dict)
        for (nm, oth_nms) in row['same_object'].items():
            nm_cnt = row['spellchecked'][nm]
            if nm not in inval_names and nm_cnt >= min_count: # remove also names with count < 2
                new_same_obj[nm] = dict()
                for (oth_nm, val) in oth_nms.items():
                    oth_cnt = row['spellchecked'][oth_nm]
                    if oth_nm not in inval_names and oth_nm != nm and oth_cnt >= min_count: # remove also names with count < 2
                        new_same_obj[nm].update({oth_nm: val})
            else:
                incorrect_col[nm]['same_object'] = oth_nms
        valid_data['same_object'].append(dict(new_same_obj))
        valid_data['incorrect'].append(incorrect_col)
        
    return dict(valid_data)

if __name__=="__main__":
    many_names_path = "../proc_data_phase0/verification/all_responses_round0-3_verified_new.csv"
    manynames = load_results.load_cleaned_results(many_names_path)
    manynames["mn_domain"] = add_mn_domains(manynames)   

    # Set cutoff to adequacy_mean<=X
    adequ_threshold = 0.4
    same_obj_threshold = 0.0
    min_count = 2
    mnv2_path = "../proc_data_phase0/mn_v2.0/manynames-v2.0_valid_responses_ad%.2f_cnt%d.csv" % (adequ_threshold, min_count)


    # Create new MN DataFrame

    name_df = get_name_df(manynames)
    other_names = name_df[name_df["is_top_name"]==False]

    indices_inadequate = other_names[other_names["adequacy_mean"]<=adequ_threshold].index.values
    indices_other_object = other_names[other_names["same_object"]<=same_obj_threshold].index.values
    inds_invalid = set(indices_inadequate).union(set(indices_other_object))
    invalid_names_df = name_df.iloc[list(inds_invalid)][["vg_image_id", "mn_obj_name"]]

    # Save new DataFrame
    cols_ordered = ['vg_image_id', 'vg_object_id', 'url', 
                    'mn_topname', 'mn_domain', 'responses', 'same_object', 'adequacy_mean', 'inadequacy_type', 'incorrect', 
                    'vg_obj_name', 'vg_domain', 'vg_synset', 'vg_cat', 'vg_same_object', 'vg_adequacy_mean', 'vg_inadequacy_type']
    colname_map = {'spellchecked': 'responses', 'synset': 'vg_synset', 'cat': 'vg_cat'}
    new_valid_data = make_filtered_df_publish(manynames, invalid_names_df, min_count)
    new_df = pd.DataFrame.from_dict(new_valid_data)
    new_df.rename(columns=colname_map, inplace=True)
    new_df[cols_ordered].to_csv(mnv2_path, sep="\t", index=False)

