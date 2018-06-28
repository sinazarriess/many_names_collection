# coding: utf-8
from collections import defaultdict
import os
import re
import sys

import json
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import df_annotation
from annotation_utils import get_synset_first, tag2pos, get_lch, get_ss_from_offset, get_path_similarity
import annotation_utils as anno_utils


dataset = "refcoco" #"flickr30k"
data_dir = "../data/"
cat_or_name = "names" #"region_cat" # "names"
cat_map_refcoco = {"stop sign": "street sign", 
                   "wine glass": "drinking glass",
                   "sports ball": "ball", 
                   "potted plant": "plant life"}

#anno_infname = "%s_anno-dep-pos-wn-attr.json.gz" % (dataset)
region_dfname = "%s_regions_distractors.json.gz" % (dataset)

#annodf = pd.read_json(os.path.join(data_dir, anno_infname), compression='gzip', orient='split')
regiondf = pd.read_json(os.path.join(data_dir, region_dfname), compression='gzip', orient='split')

synsets_ilsvrc = [get_ss_from_offset(off.strip()) for off in open("../ilsvrc_synsets.txt")]


# collect set of all possible synsets in image data

if cat_or_name == "region_cat":
    ssids1_imgdata = regiondf[cat_or_name].values.tolist()
else:
    ssids1_imgdata = [name for names in regiondf[cat_or_name] for name in names]
    
synsets_imgdata = []
no_synset_available = set()
for ssid1 in ssids1_imgdata:
    if ssid1 != None:
        if re.search("\.[nvra]\.[0-9]", ssid1):
            ss1 = anno_utils.get_synset(ssid1)
        else:
            if ssid1.startswith("RAW:"):
                ssid1 = re.sub("RAW: *", "", ssid1)
                ssid1 = cat_map_refcoco.get(ssid1, ssid1).replace(" ", "_")
            ss1 = get_synset_first(ssid1)
        if not ss1:
            no_synset_available.add(ssid1)
            continue
        synsets_imgdata.append(ss1)
        
synsets_imgdata = set(synsets_imgdata)
direct_match = synsets_imgdata.intersection(synsets_ilsvrc)

ss_img_not_covered = synsets_imgdata.difference(synsets_ilsvrc)

closest_synsets_ilvsrc = {}
ss1_hyp_of_ss2 = 0
ss2_hyp_of_ss1 = 0
for ss1 in list(ss_img_not_covered):
    min_lch = None
    best_sim = -1
    for ss2 in synsets_ilsvrc:
        path_sim = anno_utils.get_path_similarity(ss1, ss2)
        if path_sim and path_sim > best_sim:
            closest_synsets_ilvsrc[ss1] = (ss2, path_sim)
            best_sim = path_sim
            #print(ss1, ss2, path_sim)
            hypernyms2 = anno_utils.get_all_hypernyms(ss2)
            flat_hyps2 = [hyps for paths in hypernyms2 for hyps in paths]
            if ss1 in flat_hyps2:
                ss1_hyp_of_ss2 += 1
                print("img-data", ss1, "is hypernym of ilsvrc", ss2, " path sim: %.5f\n" % (best_sim))
                
            hypernyms1 = anno_utils.get_all_hypernyms(ss1)
            flat_hyps1 = [hyps for paths in hypernyms1 for hyps in paths]
            if ss2 in flat_hyps1:
                ss2_hyp_of_ss1 += 1
                print("ilsvrc", ss2, "is hypernym of", ss1, " path sim: %.5f\n" % (best_sim))
    
closest_filtered = {ss1:closest_synsets_ilvsrc[ss1] for ss1 in closest_synsets_ilvsrc if closest_synsets_ilvsrc[ss1][1]>=0.5}

print("No. synsets in image data: ", len(synsets_imgdata))
print("No. synsets in ilsvrc data: ", len(synsets_ilsvrc))
print("No. common synsets: ", len(direct_match))
print("No. synsets connected to ilsvrc through path: ", len(closest_synsets_ilvsrc))
print("No. synsets which are hypernyms of ilsvrc ss: ", ss1_hyp_of_ss2)
print("No. ilsvrc ss which are hypernyms of img-data ss: ", ss2_hyp_of_ss1)
print("No. synsets with path sim >= 0.5: ", len(closest_filtered))
not_covered_at_all = set(ss_img_not_covered).difference(closest_synsets_ilvsrc)
print("Remainig synsets in image data: ", len(not_covered_at_all))
