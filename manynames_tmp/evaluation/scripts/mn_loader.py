from collections import Counter, defaultdict
#import glob
#import json
import operator
import os
import re
import sys

#import xmltodict

import numpy as np
import pandas as pd
from numpy import nan


"""
== Columns in manynames v2.0: ==
vg_image_id	url	vg_object_id	vg_obj_name	synset	cat	vg_domain	vg_adequacy_mean	vg_inadequacy_type	vg_same_object	mn_topname	mn_domain	responses	adequacy_mean	inadequacy_type	same_object	incorrect


== Columns in manynames verified (all data): ==
['vg_image_id', 'cat', 'synset', 'vg_obj_name', 'responses_r0',
'opt-outs', 'vg_domain', 'top_response_domain_r0',
'responses_domains_r0', 'url', 'sample_type', 'responses_domains_r1',
'responses_r1', 'top_response_domain_r1', 'responses_domains_r2',
'responses_r2', 'top_response_domain_r2', 'responses_domains_r3',
'responses_r3', 'top_response_domain_r3', 'vg_object_id',
'all_responses', 'clean', 'canon', 'spellchecked', 'spellchecked_min2',
'adequacy_mean', 'inadequacy_type', 'same_object', 'vg_adequacy_mean',
'vg_inadequacy_type', 'vg_same_object']
"""

def load_manynames(filename, sep="\t", index_col=None):
    """ 
    Load ManyNames v2.0:
        ['vg_image_id', 'vg_object_id', 'url', 
         'mn_topname', 'mn_domain', 'responses', 'same_object', 'adequacy_mean', 'inadequacy_type', 'incorrect', 
         'vg_obj_name', 'vg_domain', 'vg_synset', 'vg_cat', 'vg_same_object', 'vg_adequacy_mean', 'vg_inadequacy_type']
    """
    resdf = pd.read_csv(filename, sep=sep, index_col=index_col)
    
    if 'spellchecked' in resdf.columns:
        return load_manynames_all(filename, sep, index_col)

    # remove any old index columns
    columns = [col for col in resdf.columns if not col.startswith("Unnamed")]
    resdf = resdf[columns]

    for col in ['responses', 'same_object', 'adequacy_mean',
                'inadequacy_type', 'vg_same_object',
                'vg_inadequacy_type']:
        resdf[col] = resdf[col].apply(lambda x: eval(x))
    
    if 'incorrect' in resdf: # MNv2.0
        resdf['incorrect'] = resdf['incorrect'].apply(lambda x: eval(x))
    if 'singletons' in resdf: # MNv1.0 + v2.0
        resdf['singletons'] = resdf['singletons'].apply(lambda x: eval(x))

    return resdf
    
def load_manynames_all(filename, sep="\t", index_col=None):
    sys.stderr.write("\nLoading *all* ManyNames verification data from %s. For ManyNames v2.0, use the corresponding csv file.\n"%filename)
    resdf = pd.read_csv(filename, sep=sep, index_col=index_col)
    
    for col in ['spellchecked', 'spellchecked_min2', 'clean', 'canon']:
        if col in resdf:
            resdf[col] = resdf[col].apply(lambda x: Counter(eval(x)))

    # remove any old index columns
    columns = [col for col in resdf.columns if not col.startswith("Unnamed")]
    resdf = resdf[columns]

    # eval verified column if present
    """ columns: adequacy_mean	inadequacy_type	same_object	vg_adequacy_mean	vg_inadequacy_type	vg_same_object """
    if 'adequacy_mean' in resdf:
        for verif_type in ['adequacy_mean', 'inadequacy_type', 'same_object',  'vg_inadequacy_type', 'vg_same_object']:
            resdf[verif_type] = resdf[verif_type].apply(lambda x: eval(x))

    return resdf
    
def extract_all_object_names(filename, 
                         vg_alias_mapping=None, 
                         min_count=1,
                         out_fname=None):
    #out_fname = "mn_data/mn_vocab_tuple.tsv"
    df = load_manynames(filename)
    all_names = Counter()
    for responses in df["responses"].values:
        all_names.update(responses)
    # add singletons
    for singletons in df["singletons"].values:
        all_names.update(singletons)
    # add names judged as invalid (inadequate or other object)
    for invalid in df["incorrect"].values:
        nms_counts = {nm:annos["count"] for (nm, annos) in invalid.items()}
        all_names.update(nms_counts)
    
    all_names = sorted(all_names.items(), key=operator.itemgetter(1), reverse=True)
    if vg_alias_mapping is not None:
        # all_names = [(vg_alias_mapping.get(nm[0], nm[0]), nm[1]) for nm in all_names if nm[1]>=min_count]
        mapped_names = []
        if out_fname is not None:
            with open(out_fname, "w") as f:
                f.write("mn_part\tcount\tvg_name\n")
                for nm in all_names:
                    if nm[1]>=min_count:
                        new = vg_alias_mapping.get(nm[0], nm[0])
                        mapped_names.append((new, nm[1]))
                        f.write("%s\t%d\t%s\n" % (nm[0], nm[1], new))
                f.close()
        all_names = mapped_names
    return all_names
    
if __name__=="__main__":
    with open(os.path.join("vg_data", "vg_name2aliases.tsv")) as f:
        vg_alias_mapping = dict([line.strip().split("\t") for line in f])
        
    all_names = extract_all_object_names(
                    os.path.join('../', 'proc_data_phase0', 'mn_v2.0', 'manynames-v2.0.tsv'),
                    vg_alias_mapping=vg_alias_mapping,
                    out_fname="mn_data/mnAll_vocab_tuple.tsv")