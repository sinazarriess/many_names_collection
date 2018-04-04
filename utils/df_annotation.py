
import os
import re
import sys

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import annotation_utils as anno_utils

USRNAME ='u148188' # 'carina'

def load_df(fpath):
    return pd.read_json(fpath, compression='gzip', orient='split')

# Stanford PoS Tagger
def add_pos_tags(refdf, out_fpath=None):
    pos_tagger = anno_utils.load_pos_tagger()
    refdf['tagged_stnf'] = anno_utils.tag_refExp(refdf[['refexp']].values.flatten(order='K').tolist(), pos_tagger=pos_tagger)
    if out_fpath:
        refdf.to_json(out_fpath, compression='gzip', orient='split')
    return refdf

# Stanford Neural Dependency Parser
def add_dep_parses(refdf, out_fpath=None):
    dep_parser = anno_utils.load_dep_parser()    
    parses = anno_utils.parse_refEpx(
        refdf[['refexp']].values.flatten(order='K').tolist(), dep_parser)
    refdf['depparse_stnf'] = parses
    if out_fpath:
        refdf.to_json(out_fpath, compression='gzip', orient='split')
    return refdf

def add_dep_parses_from_json(json_fpath, out_fpath=None):
    # TODO: probably this can be done much better, still new to pandas ...
    parses = pd.read_json(json_fpath, compression='gzip', orient='columns')
    indices_fpath = "{0}.idx".format(
        re.sub("(.+?)(\.txt)?(\.json)?(\.gz)?", r"\1", json_fpath))
    indices = pd.read_csv(indices_fpath, sep=",", header=None, index_col=0)
    indices.rename({0: "rex_id", 1: "image_id", 2: "region_id"}, axis=1, inplace=True)

    sents = pd.DataFrame(parses["sentences"], index=parses.index)
    dep_parses = pd.DataFrame(sents.applymap(lambda x: x["parse"]), index=parses.index)
    dep_parses.rename({"sentences": "depparse_stnf"}, axis=1, inplace=True)
    parse_df = indices.join(dep_parses, how='left')
    if out_fpath:
        refdf.to_json(out_fpath, compression='gzip', orient='split')
    return parse_df
    

# Attributes and Names
def add_attrs_names(refdf, out_fpath=None):
    if 'tagged' in refdf:
        refdf['attr_name'] = refdf['tagged'].apply(lambda x: anno_utils.get_refanno(x))
    if 'tagged_stnf' in refdf:
        refdf['attr_name_stnf'] = refdf['tagged_stnf'].apply(lambda x: anno_utils.get_refanno(x))
    if out_fpath:
        refdf.to_json(out_fpath, compression='gzip', orient='split')
    return refdf
    

# WordNet
def add_synsets(refdf, out_fpath=None):
    if 'tagged' in refdf:
        refdf['wn_anno'] = refdf['tagged'].apply(lambda x: get_wn_anno(x))
    if 'tagged_stnf' in refdf:
        refdf['wn_anno_stnf'] = refdf['tagged_stnf'].apply(lambda x: get_wn_anno(x))
    if out_fpath:
        refdf.to_json(out_fpath, compression='gzip', orient='split')
    return refdf
    
def get_wn_anno(refdf_tagged):
    wn_annos = []
    for (word, tag) in refdf_tagged:
        pos = anno_utils.tag2pos(tag)
        if pos:
            synset = anno_utils.get_synset_first(word, pos=pos)
            lexfile_info = anno_utils.get_ss_lexfile_info(synset)
            wn_annos.append((anno_utils.get_synset_name(synset), lexfile_info))
        else:
            wn_annos.append((None, None))
    return wn_annos

if __name__=="__main__":
    json_fpath = "/media/%s/Carina_2017/UdS/data/flickr30k_refdf.json.gz" % (USRNAME)
    json_foutpath = "/media/%s/Carina_2017/UdS/data/flickr30k_refdf_wn.json.gz" % (USRNAME)

    if len(sys.argv) > 1:
        json_fpath = sys.argv[1]
        if len(sys.argv) > 2:
            json_foutpath = sys.argv[2]

    #refdf = load_df(json_fpath)
    #refdf = add_pos_tags(refdf) # assignment not really necessary
    #refdf = add_synsets(refdf)
    #refdf = add_attrs_names(refdf, json_foutpath)
    #refdf = add_dep_parses(refdf)
    refdf = add_dep_parses_from_json(json_fpath)
    
    
    