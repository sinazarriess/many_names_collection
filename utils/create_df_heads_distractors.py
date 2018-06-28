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
from annotation_utils import get_synset_first, tag2pos, get_lch, get_path_similarity
import annotation_utils as anno_utils


dataset = "flickr30k"
data_dir = "../data/"

anno_infname = "%s_anno-dep-pos-wn-attr.json.gz" % (dataset)

annodf = pd.read_json(os.path.join(data_dir, anno_infname), compression='gzip', orient='split')

outfname_regiondf = "%s_regions_distractors.json.gz" % (dataset)


if os.path.exists(os.path.join(data_dir, outfname_regiondf)):
    regiondf = pd.read_json(os.path.join(data_dir, outfname_regiondf), compression='gzip', orient='split')
else:
    # Get head and its position in refexp (0 is "root")
    heads = annodf["basicDependencies"].apply(lambda x:(x[0]['dependentGloss'], x[0]['dependent']-1))

    # Get XXX
    heads_wn = pd.DataFrame(heads).apply(axis=1, func=lambda x: annodf["wn_anno_parse"].iloc[x.name][heads.iloc[x.name][1]])


    images = set(annodf.image_id)
    image2region = {}
    image2cats = {}
    if "cat" in annodf:
        for im in images:
            image2region[im] = list(annodf[annodf['image_id'] == im].region_id)
            image2cats[im] = list(annodf[annodf['image_id'] == im].cat)
    else:
        for im in images:
            image2region[im] = list(annodf[annodf['image_id'] == im].region_id)
            

    # Get region to name mapping
    region2names = defaultdict(list)
    for region_id,head_wn in zip(list(annodf['region_id']),list(heads_wn)):
        region2names[region_id].append(head_wn)


    region2image = {reg:im for im in image2region for reg in image2region[im]}

    region_dict = []
    #for reg in region2image:
    for reg in annodf.region_id:
        regs_of_img = image2region.get(region2image.get(reg, 0), None)
        if regs_of_img:

            distractors = list(filter(lambda other: other!=reg, regs_of_img))

        target_cat = [name_lex[1] for name_lex in region2names[reg]]
        #target_cat = [name_lex[1].split(".")[1] for name_lex in names]
        #distr_cats = [name_lex[1].split(".")[1] for distr_reg in distractors for name_lex in region2names[distr_reg]]
        distr_cats = [name_lex[1] for distr_reg in distractors for name_lex in region2names[distr_reg]]
        distr_names = [name_lex[0] for distr_reg in distractors for name_lex in region2names[distr_reg]]
        d = {'region_id':reg,
            'image_id':region2image[reg],
            'names': [name_lex[0] for name_lex in region2names[reg]],
            'distractor_ids':distractors,
            'region_cat':target_cat,
            'distractor_cats':distr_cats,
            'distractor_names': distr_names,
            }
        region_dict.append(d)

    regiondf = pd.DataFrame(region_dict)
    regiondf.to_json(os.path.join(data_dir, outfname_regiondf), compression='gzip', orient='split')

# ANALYSIS
regiondf['ndistractors'] = regiondf.distractor_cats.apply(lambda x: len(x))
#regiondf['ndistrs_unique'] = regiondf.distractor_cats.apply(lambda x: len(set(x)))

#namings_distr_target = ["distractor_cats", "region_cat"]
namings_distr_target = ["distractor_names", "names"]

distances = []
lch_distances = []
dists_imgs = {} # {img_id -> [mean([target1_name1_dist, ..., target1_namek_dist]), ..., mean([targetn_name1_dist, ..., targetn_namek_dist])], ...}
lch_dists_imgs = {}
lch_objNames = {}
objName_lchs = {}
counts_lch = {}
for img_id in list(set(regiondf.image_id)):
    distrs_names = regiondf[lambda df: regiondf.image_id == img_id][namings_distr_target]
    
    # compare each target to each of its distractors
    for ix, row in distrs_names.iterrows():
        #lowest_common_hypernyms = []
        path_sims = []
        lch_path_sims = []
        
        # compare each name of target to each of its distractors
        for ssid1 in row[namings_distr_target[1]]:
            if ssid1 != None and not ".n" in ssid1:
                ssid1 = get_synset_first(ssid1)
                if not ssid1:
                    continue
            
            for idx, ssid_distr in enumerate(row[namings_distr_target[0]]):
                if ssid_distr != None and not ".n" in ssid_distr:
                    ssid_distr = get_synset_first(ssid_distr)
                    if not ssid_distr:
                        continue
                
                path_sim = get_path_similarity(ssid1, ssid_distr)
                if path_sim:
                    path_sims.append(path_sim)
                    
                least_common = get_lch(ssid_distr, ssid1)
                if least_common:
                    lch_path_sims.append(max(get_path_similarity(ssid1, least_common[0]),
                                             get_path_similarity(ssid_distr, least_common[0])))
                    lch_name = least_common[0].name()
                    #lowest_common_hypernyms.append(lch_name)
                    counts_lch[lch_name] = counts_lch.get(lch_name, 0) + 1
                    lch_objNames.setdefault(lch_name, set()).update({ssid1})
                    aux = objName_lchs.setdefault(ssid1, {})
                    aux[lch_name] = aux.get(lch_name, 0) + 1
                    
        if len(path_sims) > 0 and len(lch_path_sims) > 0:
            distances.append(np.mean(path_sims))
            lch_distances.append(np.mean(lch_path_sims))
        else:
            distances.append(0)
            lch_distances.append(0)
        
    dists_imgs[img_id] = distances
    lch_dists_imgs[img_id] = lch_distances

sorted_lch_counts = sorted(counts_lch.items(), key=lambda x: x[1], reverse=True)

fout = open(os.path.join(data_dir, "lch_counts"+anno_infname.replace(".json", "").replace(".gz", "")+".txt"), "w")
for lch, count in sorted_lch_counts:
    fout.write("{0}\t{1:d}\t{2}\n".format(lch, count, " ".join(lch_objNames[lch])))
    
fout.close()
print("\nlch and counts written to ", os.path.join(data_dir, "lch_counts"+anno_infname.replace(".json", "").replace(".gz", "")+".txt"))



flat_lch_distances = [d for dlist in lch_dists_imgs.values() for d in dlist]
plt.hist(flat_lch_distances)
regiondf['lch_distances'] = list(lch_dists_imgs.values())

flat_distances = [d for dlist in dists_imgs.values() for d in dlist]
plt.hist(flat_distances)
regiondf['distances'] = list(dists_imgs.values())






