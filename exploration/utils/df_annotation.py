
import os
import re
import sys

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import annotation_utils as anno_utils

USRNAME = 'u148188' #'carina' # 'u148188'

def load_df(fpath):
    return pd.read_json(fpath, compression='gzip', orient='split')

# PoS Tagging
# From Stanford parses
def add_pos_tags_from_parse(parsedf, out_fpath=None):
    parsedf['tagged_parse'] = parsedf["parse"].apply(lambda x: re.findall("\(([^\)^\(]+?)\)", x))
    parsedf['tagged_parse'] = parsedf["tagged_parse"].apply(lambda x: [[a.split()[1], a.split()[0]] for a in x])
    if out_fpath:
        parsedf.to_json(out_fpath, compression='gzip', orient='split')
    return parsedf

# Stanford PoS Tagger
def add_pos_tags(refdf, out_fpath=None):
    pos_tagger = anno_utils.load_pos_tagger()
    refdf['tagged_stnf'] = anno_utils.tag_refExp(refdf[['refexp']].values.flatten(order='K').tolist(), pos_tagger=pos_tagger)
    if out_fpath:
        refdf.to_json(out_fpath, compression='gzip', orient='split')
    return refdf


# Stanford Neural Dependency Parser
def load_dep_parses_from_json(json_fpath, out_fpath=None):
    return pd.read_json("../data/%s_depdf.json.gz" % dataset, compression='gzip', orient='split')

def add_dep_parses(refdf, out_fpath=None):
    dep_parser = anno_utils.load_dep_parser()    
    parses = anno_utils.parse_refEpx(
        refdf[['refexp']].values.flatten(order='K').tolist(), dep_parser)
    refdf['depparse_stnf'] = parses
    if out_fpath:
        refdf.to_json(out_fpath, compression='gzip', orient='split')
    return refdf

def add_dep_parses_from_json(json_fpath, out_fpath=None):
    # TODO 1: probably this can be done much better, still new to pandas ...
    # TODO 2: double-check it still also works with refcoco
    parses = pd.read_json(json_fpath, #compression='gzip', 
                          orient='columns')
    indices_fpath = "{0}.idx".format(
        re.sub("(.+?)(\.txt)?(\.json)?(\.gz)?", r"\1", json_fpath))
    indices = pd.read_csv(indices_fpath, sep=",", header=None)
    indices.drop(columns=0, inplace=True)
    indices.rename({1: "rex_id", 2: "image_id", 3: "region_id"}, axis=1, inplace=True)

    sents = pd.DataFrame(parses["sentences"], index=parses.index)
    const_parses = pd.DataFrame(sents.applymap(lambda x: x["parse"]), index=parses.index)
    const_parses.rename({"sentences": "parse"}, axis=1, inplace=True)
    dep_parses = pd.DataFrame(sents.applymap(lambda x: x["basicDependencies"]), index=parses.index)
    dep_parses.rename({"sentences": "basicDependencies"}, axis=1, inplace=True)
    const_dep_parses = dep_parses.join(const_parses, how='left')
    
    parse_df = indices.join(const_dep_parses, how='left')
    if out_fpath:
        parse_df.to_json(out_fpath, compression='gzip', orient='split')
    return parse_df

def add_root_from_dep_parse(json_fpath, out_fpath=None):
    # TODO: probably this can be done much better, still new to pandas ...
    parses = pd.read_json(json_fpath, compression='gzip', orient='columns')
    indices_fpath = "{0}.idx".format(
        re.sub("(.+?)(\.txt)?(\.json)?(\.gz)?", r"\1", json_fpath))
    indices = pd.read_csv(indices_fpath, sep=",", header=None, index_col=0)
    indices.rename({0: "rex_id", 1: "image_id", 2: "region_id"}, axis=1, inplace=True)

    sents = pd.DataFrame(parses["sentences"], index=parses.index)
    dep_roots = pd.DataFrame(sents.applymap(lambda x: x["basicDependencies"][0]), index=parses.index)
    dep_roots.rename({"sentences": "deproot"}, axis=1, inplace=True)
    parse_df = indices.join(dep_parses, how='left')
    if out_fpath:
        refdf.to_json(out_fpath, compression='gzip', orient='split')
    return parse_df
    

# Attributes and Names
def add_attrs_names(refdf, out_fpath=None):
    level = 'tagged'
    ext = ''
    if 'tagged_parse' in refdf:
        ext = '_parse'
    elif 'tagged_stnf' in refdf:
        ext = '_stnf'
    level += ext
    
    if level in refdf:
        refdf['attribute%s' % ext] = refdf[level].apply(lambda x: anno_utils.get_refanno(x))
    if out_fpath:
        refdf.to_json(out_fpath, compression='gzip', orient='split')
    return refdf
    

# WordNet
def add_synsets(refdf, out_fpath=None):
    level = 'tagged'
    ext = ''
    if 'tagged_parse' in refdf:
        ext = '_parse'
    elif 'tagged_stnf' in refdf:
        ext = '_stnf'
    level += ext
    
    if level in refdf:
        refdf['wn_anno%s' % ext] = refdf[level].apply(lambda x: _get_wn_anno(x))
    
    if out_fpath:
        refdf.to_json(out_fpath, compression='gzip', orient='split')
    return refdf

def get_ss_first_name_and_lexfile(word, pos=None):
    synset = anno_utils.get_synset_first(word, pos=pos)
    lexfile_info = anno_utils.get_ss_lexfile_info(synset)
    return (anno_utils.get_synset_name(synset), lexfile_info)
   
def _get_wn_anno(refdf_tagged):
    wn_annos = []
    for (word, tag) in refdf_tagged:
        pos = anno_utils.tag2pos(tag)
        if pos:
            synset = anno_utils.get_synset_first(word, pos=pos)
            lexfile_info = anno_utils.get_ss_lexfile_info(synset)
            wn_annos.append((anno_utils.get_synset_name(synset), lexfile_info))
        else:
            wn_annos.append((word, None))
    return wn_annos

if __name__=="__main__":
    data_path = "/media/%s/Carina_2017/UPF/github/object_naming/names_in_context/data/" % USRNAME
    
    dataset = "refcoco" #"refcoco" #"flickr30k"
    
    json_fpath = os.path.join(data_path, "%s_refdf.json.gz" % (dataset))
    #json_parsefpath = "/media/%s/Carina_2017/UPF/github/object_naming/names_in_context/data/%s_depdf.json.gz" % (USRNAME, dataset)

    json_foutpath = os.path.join(data_path, "%s_%s.json.gz" % (dataset, "%s"))
    level_list = ["load_dependency_txt", "pos", "synset", "attribute"]

    if len(sys.argv) > 1:
        json_fpath = sys.argv[1]
        if len(sys.argv) > 2:
            level_list = sys.argv[2].split(",")
            if len(sys.argv) > 3:
                json_foutpath = sys.argv[3]
                
    levels = [l.lower().strip() for l in level_list]
                
    refdf = load_df(json_fpath)
    parses_avail = False
    levels_str = ""

    sys.exit()

    if "load_dependency" in levels:
        refdf = load_dep_parses_from_json(refdf)
        parses_avail = True
        levels_str += "-dep"
    elif "load_dependency_txt" in levels:
        parses_avail = True
        levels_str += "-dep"
        refdf = add_dep_parses_from_json(
            os.path.join(data_path, "%s_refexp.txt.json" % dataset), 
            json_foutpath % ("anno"+levels_str))
    elif "dependency" in levels:
        # TODO
        levels_str += "-dep"
        refdf = add_dep_parses(refdf, json_foutpath % "-dep0")
        # TODO: 
        # add_dep_parses_from_json(XXX, json_foutpath % levels_str)
        parses_avail = True

    if "pos" in levels:
        if parses_avail:
            refdf = add_pos_tags_from_parse(refdf) # assignment not really necessary
        else:
            refdf = add_pos_tags(refdf) # assignment not really necessary
        levels_str += "-pos"
    if "synset" in levels:
        refdf = add_synsets(refdf)
        levels_str += "-wn"
    if "attribute" in levels:
        refdf = add_attrs_names(refdf)
        levels_str += "-attr"

    if json_foutpath:
        levels_str = "anno"+levels_str
        refdf.to_json(json_foutpath % levels_str, compression='gzip', orient='split')
    
    
    
