#import glob
#import json
import os
import re
import sys

from collections import Counter, defaultdict

#import xmltodict

import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', 5)

from load_results import load_cleaned_results

COLS_NAME_CORRECT = ['Answer.incorrect.on', 'Answer.not_really_correct.on', 
                     'Answer.correct_not_sure.on', 
                     'Answer.quite correct.on', 'Answer.correct.on']
# Set from [-2,2] to make direction clear: -2 incorrect, 2 correct
CORRECT_NAME_KEYS = {ans:idx-2 for (idx, ans) in enumerate(COLS_NAME_CORRECT)}
CORRECT_NAME_KEYS[-3] = np.nan

COLS_BBOX_AMBIGUOUS = ['Answer.ambiguous_box.on',  'Answer.ambiguous_box_notsure.on', 'Answer.no_ambiguous_box.on']
BBOX_AMBIGUOUS_KEYS = {idx:["Yes", "Not_sure", "no"][idx] for idx,_ in enumerate(COLS_BBOX_AMBIGUOUS)}

def preprocess_annos(verif_df, outfname=None):
    """
        relevant_cols = ['HITId', 'AssignmentId', 
        'WorkerId', 'WorkTimeInSeconds', 
        'Input.image_url_1', 'Input.vg_name_1', 'Input.response_1_1',  
        'Answer.identification.label', 
        'Answer.ambiguous_box.on', 'Answer.ambiguous_box_notsure.on', 'Answer.no_ambiguous_box.on', 
        'Answer.comments_1',
        'Answer.correct.on', 'Answer.correct_not_sure.on', 'Answer.incorrect.on',  'Answer.not_really_correct.on', 'Answer.quite correct.on']
    """
    verif_df.rename(columns={
        'Answer.identification.label':'same_object', 
        'Answer.comments_1': 'comments',
        'Input.image_url_1': 'url',
        'Input.vg_name_1': 'vg_obj_name',
        'Input.response_1_1': 'mn_obj_name'}, inplace=True)
    
    # worker id to index in list of annotations per HIT
    wtoi = {w:idx for (idx, w) in enumerate(verif_df['WorkerId'].unique().tolist())}
    
    #for (idx, row) in verif_df.iterrows():
        
    # map annotations of CORRECT NAME to scale: [-2,2], with -2 is incorrect and 2 is correct; -3 is "NA"
    verif_df['correct_name'] = verif_df[COLS_NAME_CORRECT].apply(lambda a: np.where(a.values==True)[0][0]-2  if True in a.values else -3, axis=1)

    verif_df['comments'] = verif_df[['comments']].replace(np.nan, '')
    
    # join annotations of AMBIGUOUS BBOX in one column
    verif_df['ambiguous_bbox'] = verif_df[COLS_BBOX_AMBIGUOUS].apply(lambda a: BBOX_AMBIGUOUS_KEYS[np.where(a.values==True)[0][0]]  if True in a.values else "", axis=1)
    
    rel_cols = ['HITId', 'AssignmentId', 
                     'WorkerId', 'WorkTimeInSeconds', 
                     'url', 'vg_obj_name', 'mn_obj_name', 
                     'comments', 'same_object', 'correct_name', 'ambiguous_bbox']
    if outfname is not None:
        verif_df[rel_cols].to_csv(outfname, sep="\t")
        
    return verif_df[rel_cols]



if __name__=="__main__":
    # paths to be adapted to new structure in github repo
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "../data_phase0/"
    res_df = load_cleaned_results(os.path.join(data_dir, "all_responses_round0-3_cleaned.csv"))
    
    verifanno_fnames = ["Batch_245785_batch_results_part1.csv", "Batch_245947_batch_results_part2.csv"]
    verif_annos = []
    for verif_f in verifanno_fnames:
        verif_annos.append(pd.read_csv(os.path.join(data_dir, "verification_pilot", verif_f)))
    verif_df = pd.concat(verif_annos).reset_index(drop=True)
    
    verif_df = preprocess_annos(verif_df,
                                outfname=os.path.join(data_dir, "verification_pilot", "verif_annos_pilot.csv"))

    
