import copy
import json
import os
import pathlib
import random
import re
import sys

from collections import Counter
import pandas
import numpy as np

sample_types = ['singleton_obj', 'supercat_unique', 'seed_ambiguous', 'supercat_ambiguous', "other"]

if len(sys.argv) > 1: 
    df_sampleddata_outfname = sys.argv[1]
else:
    df_sampleddata_outfname = "collected_images/sampled_data-v%d.csv" % (version)
        
sampled_data_df = pandas.read_csv(df_sampleddata_outfname, sep="\t")

seeds = sampled_data_df["synset"].unique()
supercategories = sampled_data_df["category"].unique()

# GENERAL
num_objects = len(sampled_data_df)
num_images = len(sampled_data_df["image_id"].unique())
sys.stdout.write("Total number of supercategories\t{0}\n".format(len(supercategories)))
sys.stdout.write("Total number of seeds\t{0}\n".format(len(seeds)))
sys.stdout.write("Total number of objects\t{0}\n".format(num_objects))
sys.stdout.write("Total number of images\t{0}\n".format(num_images))

for sample_type in sample_types:
    filterer_df = sampled_data_df[sampled_data_df["sample_type"]==sample_type]
    filtered_dfname = df_sampleddata_outfname.replace(".csv", "") + "-%s.csv" % (sample_type)
    filterer_df.to_csv(filtered_dfname, columns=filterer_df.columns, sep="\t", index=False)

print(sampled_data_df[["sample_type", "category", "synset"]])

sys.stdout.write("Supercategory\tNo. objects\tNo. images\n")
for scat in supercategories:
    filterer_df = sampled_data_df[sampled_data_df["category"]==scat]
    filtered_dfname = df_sampleddata_outfname.replace(".csv", "") + "-%s.csv" % (scat)
    filterer_df.to_csv(filtered_dfname, columns=filterer_df.columns, sep="\t", index=False)
    sys.stdout.write("{0}\t{1}\t{2}\n".format(scat, 
                                              len(filterer_df["image_id"]), len(filterer_df["image_id"].unique())))
    
sys.stdout.write("{0}\t{1}\t{2}\n".format("TOTAL", 
                                          len(sampled_data_df["image_id"]), len(sampled_data_df["image_id"].unique())))

# count images per seed
df_imgs_per_seed = pandas.DataFrame(columns=["category", "synset", "sample_type", "obj_names", "no_objects"])
for seed in seeds:
    objs_seed = sampled_data_df[sampled_data_df["synset"] == seed]
    if len(np.unique(objs_seed["image_id"])) != len(objs_seed):
        print(seed, len(np.unique(objs_seed["image_id"])), "!=", len(objs_seed))
    names_seed = objs_seed["obj_names"].apply(lambda a: eval(a)[0])
    
    imgid_name = names_seed.str.cat(objs_seed["image_id"].astype(str), sep="$")
    unique_names_per_img = imgid_name.unique()
    names_seed_unique = [name_imgid.split("$")[0] for name_imgid in unique_names_per_img.tolist()]
    name_distr_unique = Counter(names_seed_unique)
    name_distr = Counter(names_seed)
    category = objs_seed["category"].tolist()[0]
    for (name, count) in name_distr.items():
        new_df_item = {"category": category, "synset": seed, "obj_names": name, "no_objects": count, "no_objnms_unique_per_img": int(name_distr_unique[name])}
        df_imgs_per_seed = df_imgs_per_seed.append(new_df_item, ignore_index=True)

df_imgs_per_seed["no_objnms_unique_per_img"] = df_imgs_per_seed["no_objnms_unique_per_img"].astype(int)

counter_variant = "no_objnms_unique_per_img"

## SAMPLING: based on distribution over object names
df_imgs_per_seed.sort_values(by=[counter_variant], ascending=False)
df_imgs_per_seed["imgs_to_sample"] = 0
thresholds = [1, 501, 801, 1001, 10000000]
for idx in range(len(thresholds)-1):
    indices = df_imgs_per_seed[counter_variant].between(thresholds[idx]-1, thresholds[idx+1], inclusive=False)
    imgs_gt = sum(df_imgs_per_seed[counter_variant][indices])
    names_gt = len(df_imgs_per_seed[counter_variant][indices])
    print("Intervall ({0:d}, {1:d}) images:\tObj. names: {2:d}\tImages: {3:d}".format(
        thresholds[idx]-1, thresholds[idx+1], 
        names_gt, imgs_gt))
    df_imgs_per_seed["imgs_to_sample"][indices==True] = df_imgs_per_seed[counter_variant][indices==True]

if True:
    #df_imgs_per_seed["imgs_to_sample"][df_imgs_per_seed[counter_variant] < 500] = df_imgs_per_seed[counter_variant][df_imgs_per_seed[counter_variant] < 500]
    print("\nSampling procedure:")
    print("no_objs <= 500 --> collect all")
    print("no_objs > 500 --> sample 500 objects")
    df_imgs_per_seed["imgs_to_sample"][df_imgs_per_seed[counter_variant]>500] = 500
    print("no_objects > 800 --> sample up to 1000 objects")
    indices = df_imgs_per_seed[counter_variant].between(800, 1001, inclusive=False)
    df_imgs_per_seed["imgs_to_sample"][indices] = df_imgs_per_seed[counter_variant][indices]
    df_imgs_per_seed["imgs_to_sample"][df_imgs_per_seed[counter_variant]>1000] = 1000

print("# possible images sampled from intervall #images in (%d,%d): %d" % (thresholds[0], thresholds[-1], sum(df_imgs_per_seed["imgs_to_sample"])))

df_imgs_per_seed.sort_values(by=["category", "synset", "imgs_to_sample", ], ascending=False, inplace=True)
df_imgs_per_seed[["category", "synset", "obj_names", counter_variant,  "imgs_to_sample"]].to_csv("sampling2_max1000per_objname.csv")


## count candiate image samples by seeds
df_grouped_imgs_per_seed = pandas.DataFrame(columns=["category", "synset", "imgs_to_sample"])
grouping_synset = df_imgs_per_seed.groupby(by=["synset"]).sum()
no_objs_per_seed = grouping_synset[["imgs_to_sample"]]
for (seed, data) in no_objs_per_seed.iterrows():
    scat = sampled_data_df[sampled_data_df["synset"]==seed]["category"].iloc[0]
    count = data["imgs_to_sample"]
    new_df_item = {"category": scat, "synset": seed, "imgs_to_sample": count}
    df_grouped_imgs_per_seed = df_grouped_imgs_per_seed.append(new_df_item, ignore_index=True)
    
df_grouped_imgs_per_seed.sort_values(by=["category", "imgs_to_sample"], ascending=False, inplace=True)
print(df_grouped_imgs_per_seed)


# count images per supercategory
#df_imgs_per_scat = pandas.DataFrame(columns=["category", "synset", "obj_names", "imgs_to_sample"])
for category in supercategories:
    objs_scat = df_imgs_per_seed[df_imgs_per_seed["category"] == category]
    objs_scat.sort_values(by=["synset", "imgs_to_sample"], ascending=False)
    
grouping_scat = df_imgs_per_seed.groupby(by=["category"]).sum()
no_objs_per_scat = grouping_scat[["imgs_to_sample"]]
print(no_objs_per_scat)
print("TOTAL", sum(no_objs_per_scat["imgs_to_sample"]))
sys.exit()

##########
## group by seeds
df_grouped_imgs_per_seed = pandas.DataFrame(columns=["category", "synset", counter_variant])
grouping_synset = df_imgs_per_seed.groupby(by=["synset"]).sum()
no_objs_per_seed = grouping_synset[[counter_variant]]
for (seed, data) in no_objs_per_seed.iterrows():
    scat = sampled_data_df[sampled_data_df["synset"]==seed]["category"].iloc[0]
    count = data[counter_variant]
    new_df_item = {"category": scat, "synset": seed, counter_variant: count}
    df_grouped_imgs_per_seed = df_grouped_imgs_per_seed.append(new_df_item, ignore_index=True)


df_grouped_imgs_per_seed.sort_values(by=["category", counter_variant], ascending=False, inplace=True)

print(df_grouped_imgs_per_seed)
#df_imgs_per_seed["log_no_objs"] = pandas.Series(np.log(df_imgs_per_seed[counter_variant].tolist()))



sys.exit()
# count images per supercategory
df_imgs_per_scat = pandas.DataFrame(columns=["category", "synset", "obj_names", counter_variant, "log_no_objs"])
for category in supercategories:
    objs_scat = df_imgs_per_seed[df_imgs_per_seed["category"] == category]
    objs_scat.sort_values(by=["synset", counter_variant], ascending=False)
    


grouping_scat = df_imgs_per_seed.groupby(by=["category"]).sum()
no_objs_per_scat = grouping_scat[[counter_variant]]
