#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11

@author: Carina Silberer
"""
import sys
from collections import Counter, defaultdict

import pandas

import visualise

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

    ## Option 1
    MAX_OCCL = 0
    MAX_BBOX = 2
    MAX_PLURALS = 0.17
    param_str ="occl-max%d_AND_bbox-max%d_AND_plurals-max%.2f-OR-food" % (
                    MAX_OCCL, MAX_BBOX, MAX_PLURALS)
    print("pruning parameters: ", param_str)
    pruned_df = checkpoint_df[(checkpoint_df["#occl"] <= MAX_OCCL) & (checkpoint_df["#bbox"] <= MAX_BBOX) & ((checkpoint_df["plurals"]<=MAX_PLURALS) | (checkpoint_df["vg_domain"] == "food"))]
    domain_distr = domain_distribution(pruned_df)
    prune_df[param_str] = prune_df.index.map(domain_distr)
    print("total: ", len(pruned_df))
    pruned_df.to_csv("results/pruned_df_%s.csv" % (param_str),
                    sep="\t", float_format='%.2f')
    
    ## Option 2
    MAX_OCCL = 0
    MAX_BBOX = 2
    MAX_PLURALS = 0.17
    DOMAIN_MATCH = True
    param_str ="domain-match%d_AND_occl-max%d_AND_bbox-max%d_AND_plurals-max%.2f-OR-food" % (
                    DOMAIN_MATCH, MAX_OCCL, MAX_BBOX, MAX_PLURALS)
    print("pruning parameters: ", param_str)
    pruned_df = checkpoint_df[(checkpoint_df["#occl"] <= MAX_OCCL) & (checkpoint_df["#bbox"] <= MAX_BBOX) & \
        ((checkpoint_df["plurals"]<=MAX_PLURALS) | (checkpoint_df["vg_domain"] == "food")) & \
        (checkpoint_df["domain_match"]==DOMAIN_MATCH)]
    domain_distr = domain_distribution(pruned_df)

    prune_df[param_str] = prune_df.index.map(domain_distr)
    print("total: ", len(pruned_df))
    pruned_df.to_csv("results/pruned_df_%s.csv" % (param_str),
                    sep="\t", float_format='%.2f')
    
    ## SELECTED: Option 2b (apply plural filter also to food)
    MAX_OCCL = 0
    MAX_BBOX = 2
    MAX_PLURALS = 0.17
    DOMAIN_MATCH = True
    param_str ="domain-match%d_AND_occl-max%d_AND_bbox-max%d_AND_plurals-max%.2f" % (
                    DOMAIN_MATCH, MAX_OCCL, MAX_BBOX, MAX_PLURALS)
    print("pruning parameters: ", param_str)
    criteria_filter = (checkpoint_df["#occl"] <= MAX_OCCL) & (checkpoint_df["#bbox"] <= MAX_BBOX) & \
        (checkpoint_df["plurals"]<=MAX_PLURALS) & \
        (checkpoint_df["domain_match"]==DOMAIN_MATCH)
    pruned_df = checkpoint_df[criteria_filter]
    domain_distr = domain_distribution(pruned_df)

    prune_df[param_str] = prune_df.index.map(domain_distr)
    print("total: ", len(pruned_df))
    pruned_df.to_csv("results/pruned_df_%s.csv" % (param_str),
                    sep="\t", float_format='%.2f')
    visualise.write_html_table(pruned_df, "kept_images_%s.html" % (param_str))
    
    ## Visualise criteria in html    
    removed_df = checkpoint_df[((checkpoint_df["#occl"] <= MAX_OCCL) & (checkpoint_df["#bbox"] <= MAX_BBOX) & \
        (checkpoint_df["plurals"]<=MAX_PLURALS) & \
        (checkpoint_df["domain_match"]==DOMAIN_MATCH)) == False]
    removed_df = checkpoint_df[criteria_filter == False]
    visualise.write_html_table(removed_df, "removed_images_%s.html" % (param_str))
    
    ## Option 3
    MAX_OCCL = 1
    MAX_BBOX = 2
    MAX_PLURALS = 0.17
    param_str ="occl-max%d_AND_bbox-max%d_AND_plurals-max%.2f-OR-food" % (
                    MAX_OCCL, MAX_BBOX, MAX_PLURALS)
    print("pruning parameters: ", param_str)
    pruned_df = checkpoint_df[(checkpoint_df["#occl"] <= MAX_OCCL) & (checkpoint_df["#bbox"] <= MAX_BBOX) & ((checkpoint_df["plurals"]<=MAX_PLURALS) | (checkpoint_df["vg_domain"] == "food"))]
    domain_distr = domain_distribution(pruned_df)
    
    prune_df[param_str] = prune_df.index.map(domain_distr)
    print("total: ", len(pruned_df))
    pruned_df.to_csv("results/pruned_df_%s.csv" % (param_str),
                    sep="\t", float_format='%.2f')     
    
    ## Option 4
    MAX_OCCL = 1
    MAX_BBOX = 2
    MAX_PLURALS = 0.17
    DOMAIN_MATCH = True
    param_str ="domain-match%d_AND_occl-max%d_AND_bbox-max%d_AND_plurals-max%.2f-OR-food" % (
                    DOMAIN_MATCH, MAX_OCCL, MAX_BBOX, MAX_PLURALS)
    print("pruning parameters: ", param_str)
    pruned_df = checkpoint_df[(checkpoint_df["#occl"] <= MAX_OCCL) & (checkpoint_df["#bbox"] <= MAX_BBOX) & \
        ((checkpoint_df["plurals"]<=MAX_PLURALS) | (checkpoint_df["vg_domain"] == "food")) & \
        (checkpoint_df["domain_match"]==DOMAIN_MATCH)]
    domain_distr = domain_distribution(pruned_df)

    prune_df[param_str] = prune_df.index.map(domain_distr)
    print("total: ", len(pruned_df))
    pruned_df.to_csv("results/pruned_df_%s.csv" % (param_str),
                    sep="\t", float_format='%.2f')

    # Option 4b (apply plural filter also to food)
    MAX_OCCL = 1
    MAX_BBOX = 2
    MAX_PLURALS = 0.17
    DOMAIN_MATCH = True
    param_str ="domain-match%d_AND_occl-max%d_AND_bbox-max%d_AND_plurals-max%.2f" % (
                    DOMAIN_MATCH, MAX_OCCL, MAX_BBOX, MAX_PLURALS)
    print("pruning parameters: ", param_str)
    pruned_df = checkpoint_df[(checkpoint_df["#occl"] <= MAX_OCCL) & (checkpoint_df["#bbox"] <= MAX_BBOX) & \
        (checkpoint_df["plurals"]<=MAX_PLURALS) & \
        (checkpoint_df["domain_match"]==DOMAIN_MATCH)]
    domain_distr = domain_distribution(pruned_df)

    prune_df[param_str] = prune_df.index.map(domain_distr)
    print("total: ", len(pruned_df))
    pruned_df.to_csv("results/pruned_df_%s.csv" % (param_str),
                    sep="\t", float_format='%.2f')          

    ## GEMMA: Option 5a
    MAX_OCCL = 1
    MAX_BBOX = 1
    MAX_PLURALS = 0.17
    DOMAIN_MATCH = True
    param_str ="domain-match%d_AND_occl-max%d_AND_bbox-max%d_AND_plurals-max%.2f" % (
                    DOMAIN_MATCH, MAX_OCCL, MAX_BBOX, MAX_PLURALS)
    print("pruning parameters: ", param_str)
    criteria_filter = (checkpoint_df["#occl"] <= MAX_OCCL) & (checkpoint_df["#bbox"] <= MAX_BBOX) & \
        (checkpoint_df["plurals"]<=MAX_PLURALS) & \
        (checkpoint_df["domain_match"]==DOMAIN_MATCH)
    pruned_df = checkpoint_df[criteria_filter]
    domain_distr = domain_distribution(pruned_df)

    prune_df[param_str] = prune_df.index.map(domain_distr)
    print("total: ", len(pruned_df))
    pruned_df.to_csv("results/pruned_df_%s.csv" % (param_str),
                    sep="\t", float_format='%.2f')
    visualise.write_html_table(pruned_df, "kept_images_%s.html" % (param_str))
    
    removed_df = checkpoint_df[criteria_filter == False]
    visualise.write_html_table(removed_df, "removed_images_%s.html" % (param_str))
    
    ## GEMMA: Option 5b
    MAX_OCCL = 2
    MAX_BBOX = 1
    MAX_PLURALS = 0.17
    DOMAIN_MATCH = True
    param_str ="domain-match%d_AND_occl-max%d_AND_bbox-max%d_AND_plurals-max%.2f" % (
                    DOMAIN_MATCH, MAX_OCCL, MAX_BBOX, MAX_PLURALS)
    print("pruning parameters: ", param_str)
    criteria_filter = (checkpoint_df["#occl"] <= MAX_OCCL) & (checkpoint_df["#bbox"] <= MAX_BBOX) & \
        (checkpoint_df["plurals"]<=MAX_PLURALS) & \
        (checkpoint_df["domain_match"]==DOMAIN_MATCH)
    pruned_df = checkpoint_df[criteria_filter]
    domain_distr = domain_distribution(pruned_df)

    prune_df[param_str] = prune_df.index.map(domain_distr)
    print("total: ", len(pruned_df))
    pruned_df.to_csv("results/pruned_df_%s.csv" % (param_str),
                    sep="\t", float_format='%.2f')
    visualise.write_html_table(pruned_df, "kept_images_%s.html" % (param_str))
    
    removed_df = checkpoint_df[criteria_filter == False]
    visualise.write_html_table(removed_df, "removed_images_%s.html" % (param_str))
    
    ## Visualise individual criteria in html    
    param_str ="domain-match%d" % (DOMAIN_MATCH)
    removed_df = checkpoint_df[(checkpoint_df["domain_match"]==DOMAIN_MATCH) == False]
    visualise.write_html_table(removed_df, "removed_images_%s.html" % (param_str))

    param_str ="plurals-max%.2f" % (MAX_PLURALS)
    removed_df = checkpoint_df[(checkpoint_df["plurals"]<=MAX_PLURALS) == False]
    removed_df = removed_df.sort_values(by=["plurals"], ascending=[False])
    visualise.write_html_table(removed_df, "removed_images_%s.html" % (param_str))
    
    MAX_OCCL = [1, 2]
    MAX_BBOX = 1
    param_str ="occl-btw%d-%d_AND_bbox-max%d" % (MAX_OCCL[0], MAX_OCCL[1], MAX_BBOX)
    criteria_filter = ((checkpoint_df["#occl"] >= MAX_OCCL[0]) & (checkpoint_df["#occl"] <= MAX_OCCL[1])) & \
                                      (checkpoint_df["#bbox"] <= MAX_BBOX)
    removed_df = checkpoint_df[criteria_filter]
    removed_df = removed_df.sort_values(by=["#occl"], ascending=[False])
    visualise.write_html_table(removed_df, "removed_images_%s.html" % (param_str))
    
    # save overview of all options
    prune_df.loc["SUM"] = prune_df.cumsum().iloc[-1].values
    prune_df.to_csv("results/overview_pruned_data.csv", sep="\t", float_format='%.2f')
