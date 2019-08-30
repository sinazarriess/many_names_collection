import os
import re
import sys
from collections import Counter, defaultdict

import numpy
import pandas

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

import load_results
# TODO: improve mapping
from  preprocess_results import get_name2domain_map, name2domain

BASE_DIR = os.path.dirname(os.getcwd())


if __name__=="__main__":
    if len(sys.argv) < 2:
        sys.stderr.write("Please provide results files to be merged.\n")
        sys.exit()
        
    """
    cols = ['vg_img_id', 'vg_object_id', 'cat', 'synset', 
            'vg_obj_name', 'responses', 'opt-outs', 
            'vg_domain', 'top_response_domain', 'responses_domains',                
            'url', 'sample_type']
    """
    df_list = []
    #"/home/u148188/object_naming/amt_phase0/result_files_rounds0-2/"
    if os.path.isdir(sys.argv[1]):
        result_files = []
        for res_dir in sys.argv[1:]:
            result_files.extend([os.path.join(res_dir, fname) for fname in os.listdir(res_dir) if (not os.path.isdir(os.path.join(res_dir, fname)) and "_final" in fname)])
    else:
        result_files = sys.argv[1:]

    round_idx = 0
    round_inds = []
    for res_csvs in result_files:
        print(res_csvs)
        round_idx = int(re.search("_round([0-9]+)\.csv", res_csvs).group(1))
        round_inds.append(round_idx)
        df = load_results.load_preprocessed_results_csv(res_csvs)
        
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], axis=1, inplace=True)
        for col_name in df.columns:
            if "response" in col_name:
                new_col_name = col_name + "_r%d" % round_idx
                df.rename(columns={col_name: new_col_name}, inplace=True)
                
        df.set_index(["vg_object_id"], inplace=True)
        df_list.append(df)

    cols_to_use = []
    merged_df = df_list[0]
    for df in df_list[1:]:
        cols_to_use = df.columns.difference(df_list[0].columns)
        merged_df = pandas.concat([merged_df, df[cols_to_use]], axis=1, join='inner')
        
    merged_df["vg_object_id"] = merged_df.index
    merged_df.to_csv("../data_phase0/all_results_files_preprocessed_rounds%d-%d.csv" % (numpy.min(round_inds), numpy.max(round_inds)), index=False, sep="\t")
    
    
    
