#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 17:27:28 2019

@author: Carina Silberer
"""
import operator
import os
import re
import sys
from collections import defaultdict

from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

import load_results
import add_younameit_categories as cat_mapper

BASE_DIR = os.path.dirname(os.getcwd())

def get_name2domain_map(df):
    # get object name to category mapping, to augment collected names with categories (as far as available)
    name_cat_map = df[["vg_domain", "vg_obj_name"]].to_dict()
    name_cat_map = {name_cat_map["vg_obj_name"][idx]:name_cat_map["vg_domain"][idx] \
                    for (idx, cat) in enumerate(name_cat_map["vg_domain"])}
    return name_cat_map

def name2domain(name, mapping):
    return mapping.get(name, "$"+name)

def _streamline_name(obj_name):
    return re.sub("\s+", " ", obj_name)

if __name__=="__main__":
    if len(sys.argv) < 2:
        sys.stderr.write("Please provide a results file or directory.\n")
        sys.exit()
        
    results_path = os.path.dirname(sys.argv[1])
    if os.path.isdir(results_path):
        out_results_file = os.path.join(results_path, "all_results_files_preprocessed_%s.csv" % (os.path.basename(results_path)))
    else:
        out_results_file = re.sub("(.+?)(\.(csv|tsv))?$", r"\1_preprocessed.csv", sys.argv[1])
        
    df = load_results.load_resultsdata_csv(sys.argv[1])

    # add category mapping (WordNet --> YouNameIt) ==> column "domain"
    df = cat_mapper._add_younameit_cats(df)
    df.rename(columns={"domain": "vg_domain"}, inplace=True)

    df.rename(columns={"obj_name": "vg_obj_name"}, inplace=True)
    analysis_cat = "vg_domain" # "cat"

    name_domain_map = get_name2domain_map(df)
    domains = set(name_domain_map.values())
                  
    ## Post-processing of df of collected data
    # separate opt-outs and object names
    df["opt-outs"] = df["responses"].apply(
            lambda resp: {n:c for (n,c) in resp.items() if n.startswith("#")})
    df["responses"] = df["responses"].apply(
            lambda resp: {_streamline_name(n):c for (n,c) in resp.items() if not n.startswith("#")})

    # add responses over domains
    df["responses_domains"] = df["responses"].apply(
            lambda resp: [(name2domain(nm, name_domain_map),cnt) \
                          for (nm,cnt) in {n:c for (n,c) in resp.items() \
                              if not n.startswith("#")}.items()])

    # sum counts over domains (e.g., girl:1, woman:3 --> person:4)
    col_domain_counts = []
    for (idx, row) in df.iterrows():
        domain_counts = defaultdict(list)
        total_count = 0
        for (domain, count) in row.responses_domains:
            domain_counts[domain].append(count)
            total_count += count
        weighted_domain_counts = {n:round((sum(c)/total_count),3) \
                                  for (n,c) in domain_counts.items()}
        col_domain_counts.append(weighted_domain_counts)

    df["responses_domains"] = col_domain_counts

    #TODO
    df["top_response_domain"] = df.responses_domains.apply(
            lambda a: max(a.items(), key=operator.itemgetter(1))[0] \
            if len(a) > 0 else "--")

    out_cols = ['vg_img_id', 'vg_object_id', 'cat', 'synset', 
                'vg_obj_name', 'responses', 'opt-outs', 
                'vg_domain', 'top_response_domain', 'responses_domains',                
                'url', 'sample_type']
    df[out_cols].to_csv(out_results_file, sep="\t")
    sys.stderr.write("Preprocessed response file(s) written to %s\n" % out_results_file)
