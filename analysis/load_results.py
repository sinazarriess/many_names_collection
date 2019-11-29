from collections import Counter, defaultdict
import glob
import json
import os
import re
import sys

#import xmltodict

import numpy as np
import pandas as pd
#from spellchecker import SpellChecker
#from nltk.stem.wordnet import WordNetLemmatizer

def load_verified_results(datafile_csv="../proc_data_phase0/spellchecking/all_responses_round0-3_cleaned.csv",
                          veriffile_csv="../many_names_collection/verification_phase0/1_crowdsourced/results/merged_1-2-3-4-5-6-7-8/name_annotations_ANON.csv",
                          min_adequacy=1,
                          clustered=False):
    """
    Adds a column to the MN dataframe with response sets for only verified (mean adequacy>=min_adequacy) names. 
    (TODO: make more efficient)
    Args:
        clustered (bool)    Loads response set with all "adequate" names, irrespective of the clustering
    """
    mn_df = load_cleaned_results(datafile_csv)    
    verif_df = pd.read_csv(veriffile_csv, sep=",")
    groups = verif_df[["image","object","name","adequacy"]].groupby(by=["image","name"])
    names_per_imgid = defaultdict(list)
    for gr in groups:
        imgid,objname = gr[0]
        adequacy = gr[1]["adequacy"].mean()
        if adequacy <= min_adequacy:
            names_per_imgid[imgid].append(objname)
    
    verified = [None]*len(mn_df)
    for row in mn_df.iterrows():
        imgid = row[1]["vg_img_id"]
        idx = row[0]
        verified_responses = Counter(dict([(nm,cnt) for (nm,cnt) in row[1]["spellchecked"].most_common() if nm in names_per_imgid[imgid]]))
        # assumption: img not in verif_df ==> take all responses with cnt > 1, because these are highly frequent <-> valid [@MW: double-check that this is what has been done]
        if len(verified_responses) == 0:
            verified_responses = Counter(dict([(nm,cnt) for (nm,cnt) in row[1]["spellchecked"].most_common() if cnt>1]))
        verified[idx] = verified_responses
        
    mn_df["verified"] = verified
    return mn_df

def load_resultsdata_csv(datafile_csv):
    if os.path.isdir(datafile_csv):
        return load_all_results_csv(datafile_csv)
    fulldf = pd.read_csv(datafile_csv, sep="\t")
    fulldf[["responses"]] = fulldf["responses"].apply(lambda a: eval(a))
    return fulldf
    
def load_all_results_csv(data_dir):
    res_patt =  re.compile("results-created.+?_final\.csv")
    csv_files = [f for f in os.listdir(data_dir) if res_patt.match(f)]
    all_dfs = []
    for csv_file in csv_files:
        all_dfs.append(load_resultsdata_csv(os.path.join(data_dir, csv_file)))
    return pd.concat(all_dfs, ignore_index=True, sort=False)

def load_preprocessed_results_csv(datafile_csv):
    """
    cols = ['vg_img_id', 'vg_object_id', 'cat', 'synset', 
            'vg_obj_name', 'responses', 'opt-outs', 
            'vg_domain', 'top_resonse_domain', 'responses_domains',                
            'url', 'sample_type']
    """
    df = pd.read_csv(datafile_csv, sep="\t")
    for dict_col in ['responses', 'opt-outs', 'responses_domains']:
        if dict_col in df.columns:
            df[[dict_col]] = df[dict_col].apply(lambda a: eval(a))
    return df

def load_cleaned_results(filename, sep="\t"):
    resdf = pd.read_csv(filename, sep=sep)
    resdf['spellchecked'] = resdf['spellchecked'].apply(lambda x: Counter(eval(x)))
    resdf['clean'] = resdf['clean'].apply(lambda x: Counter(eval(x)))
    resdf['canon'] = resdf['canon'].apply(lambda x: Counter(eval(x)))
    return resdf

def _img2objname_map(imgdf):
    # meta file used to publish the hits
    img2names_map = dict()
    for (idx, row) in imgdf.iterrows():
        img2names_map[str(row.image_id)+"_"+str(row.object_id)] = (row.obj_names, row.category)
    return img2names_map

def add_objnames_categories(fulldf, img2names_map):
    fulldf["sample_type"] = fulldf["cat"]
    obj_names = []
    obj_cats = []
    for (idx, row) in fulldf.iterrows():
        name_cat = img2names_map.get(str(row.vg_img_id)+"_"+str(row.vg_object_id), ["UNK", "UNK"])
        obj_names.append(name_cat[0])
        obj_cats.append(name_cat[1])
    fulldf["obj_name"] = obj_names
    fulldf["cat"] = obj_cats
    return fulldf


def process_answers(resultfile, hit2images, with_opt_outs=False):
    with open(resultfile, 'r') as handle:
        resultlist = json.load(handle)
    
    obj2answers = {}
    obj2info = {}
    
    for assignm in resultlist:
        hitid = assignm['HITId']
        #print(hitid)
        for a in assignm['Assignments']:
            skip_assignment = False
            #print(a)
            workerId = a['WorkerId']
            answers = a['Answers']
            for ix in range(10):
                object_id = hit2images[hitid][str(ix)][1]
                name = answers[str(ix)]['inputbox_objname-'+str(ix)]
                if object_id not in obj2answers:
                    obj2answers[object_id] = Counter()
                    obj2info[object_id] = hit2images[hitid][str(ix)]

                if name:
                    name = name.lower()
                    obj2answers[object_id][name] += 1
                elif with_opt_outs:
                    try:
                        opt_out = "#"+answers[str(ix)]['optout-'+str(ix)]
                        if answers[str(ix)]['other_reasons-'+str(ix)]:
                            opt_out += "_" + answers[str(ix)]['other_reasons-'+str(ix)].lower()
                        obj2answers[object_id][opt_out] += 1
                    except KeyError:
                        # THIS SHOULD NOT HAPPEN!!
                        sys.stderr.write("No input given for object %d (assignmentId %s)\n" % (ix, a['AssignmentId']))
                        skip_assignment = True
                        break
            if skip_assignment:
                sys.stderr.write("Skipping assignment with Id %s\n" % (a['AssignmentId']))
                continue
                    
    ## optional: spell checking
    ## commented this out, as it didn't really seem to work
    ## obj2clean_answers = {objid:check_spelling(obj2answers[objid]) for objid in obj2answers}

    allrows = [] 
    for objid in obj2answers:
        row = obj2info[objid]
        row.append(obj2answers[objid])
        allrows.append(row)
    
    if len(allrows[0]) == 8:
        # new hits file (json), with category and object name
        cols = ["vg_img_id", "vg_object_id", "sample_type", "synset", "obj_name", "cat", "url", "responses"]
    else:
        # old hits file (up to run 0f)
        cols = ['vg_img_id','vg_object_id','cat','synset','url','responses']
    fulldf = pd.DataFrame(allrows,columns=cols)
    return fulldf
                    
def process_hits(hitfile):
    with open(hitfile, 'r') as handle:
        hitlist = json.load(handle)
        
    hit2images = {}
    for hit in hitlist:
        hit2images[hit['HIT']['HITId']] = hit['Images']
    return hit2images
