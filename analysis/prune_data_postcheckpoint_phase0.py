#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11

@author: Carina Silberer
"""
import os
import sys

from collections import Counter, defaultdict

import pandas

#sys.path.append(os.path.join(os.getcwd().split("amt_phase0")[0], "analysis"))
import load_results 

def domain_distribution(data_df):
    domain_distr = defaultdict(int)
    for (idx, row) in data_df.iterrows():
        domain_distr[row["vg_domain"]] += 1
    
    print(domain_distr)
    return domain_distr

if __name__=="__main__":
    if len(sys.argv) < 2:
        sys.stderr.write("Please provide a checkpoint file (csv/tsv format).\n\
                         If not created yet, do so with amt_phase0/scripts/run_checkpoint.py.")
        sys.exit()
        
    checkpoint_df = pandas.read_csv(sys.argv[1], sep="\t")
    
    print("All data:")
    domain_distr = domain_distribution(checkpoint_df)
    print("total: ", len(checkpoint_df))
    prune_df = pandas.DataFrame.from_dict(data=domain_distr, orient="index", columns=["ALL"])

    MAX_OCCL = 0
    MAX_BBOX = 2
    MAX_PLURALS = 0
    param_str ="occl-max%d_AND_bbox-max%d_AND_plurals-max%d-OR-food" % (
                    MAX_OCCL, MAX_BBOX, MAX_PLURALS)
    print("pruning parameters: ", param_str)
    pruned_df = checkpoint_df[(checkpoint_df["#occl"] <= MAX_OCCL) & (checkpoint_df["#bbox"] <= MAX_BBOX) & ((checkpoint_df["plurals"]<=MAX_PLURALS) | (checkpoint_df["vg_domain"] == "food"))]
    domain_distr = domain_distribution(pruned_df)
    prune_df[param_str] = prune_df.index.map(domain_distr)
    print("total: ", len(pruned_df))
    pruned_df.to_csv(
            "results/pruned_df_occl-max%d_AND_bbox-max%d_AND_plurals-max%d-OR-domain-food" % (
                    MAX_OCCL, MAX_BBOX, MAX_PLURALS),
                    sep="\t", float_format='%.2f')
    MAX_OCCL = 1
    MAX_BBOX = 2
    MAX_PLURALS = 0
    param_str ="occl-max%d_AND_bbox-max%d_AND_plurals-max%d-OR-food" % (
                    MAX_OCCL, MAX_BBOX, MAX_PLURALS)
    print("pruning parameters: ", param_str)
    pruned_df = checkpoint_df[(checkpoint_df["#occl"] <= MAX_OCCL) & (checkpoint_df["#bbox"] <= MAX_BBOX) & ((checkpoint_df["plurals"]<=MAX_PLURALS) | (checkpoint_df["vg_domain"] == "food"))]
    domain_distr = domain_distribution(pruned_df)

    prune_df[param_str] = prune_df.index.map(domain_distr)
    print("total: ", len(pruned_df))
    pruned_df.to_csv(
            "results/pruned_df_occl-max%d_AND_bbox-max%d_AND_plurals-max%d-OR-domain-food" % (
                    MAX_OCCL, MAX_BBOX, MAX_PLURALS),
                    sep="\t", float_format='%.2f')             
    
    prune_df.loc["SUM"] = prune_df.cumsum().iloc[-1].values
    prune_df.to_csv("results/overview_pruned_data.csv", sep="\t", float_format='%.2f')