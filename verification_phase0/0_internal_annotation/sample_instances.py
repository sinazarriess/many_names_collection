import os
import sys
from collections import Counter

import pandas as pd

sys.path.append("../analysis/")
import load_results

def preprocess_responses(filename, relevant_cols):
    resdf = load_results.load_cleaned_results(filename)

    new_df = pd.DataFrame(resdf[relevant_cols])
    new_df['spellchecked_min2'] = new_df['spellchecked'].apply(lambda x: Counter({k:x[k] for k in x if x[k] > 1}))
    #vocab_counter = Counter()
    vg_is_common = []
    ntypes = []
    ntypes_notvg = []

    for ix,row in new_df.iterrows():
        #vocab_counter += row['spellchecked']
        max_name = row['spellchecked'].most_common(1)[0][0]
        vg_is_common.append(int(max_name == row['vg_obj_name']))
        resp_ntypes = len(row['spellchecked_min2'].keys())
        ntypes.append(resp_ntypes)
        ntypes_notvg.append(resp_ntypes - (row['vg_obj_name'] in row["spellchecked_min2"]))

    new_df['vg_is_max'] = vg_is_common
    new_df['n_types'] = ntypes
    new_df['n_types_notvg'] = ntypes_notvg
    return new_df

def sample_objects(filename, k, relevant_cols):
    """
    For each domain:
        - sample k objects, uniformly:
          number of types from [1, ..., M] with M==max number of types in domain; 1 name type !=VG name
          
    Args:
        k: the number of objects per domain to be samped
    """
    df = preprocess_responses(filename, relevant_cols)
    samples = pd.DataFrame(columns=relevant_cols)

    domains = df["vg_domain"].unique().tolist()
    for domain in domains:
        domaindf = df[df["vg_domain"]==domain]
        
        denom = domaindf["n_types_notvg"] != 0
        max_ntypes = max(domaindf["n_types_notvg"])
        mean_ntypes = int(sum(domaindf["n_types_notvg"])/len(domaindf[domaindf["n_types_notvg"] != 0]))
        samples_per_ntype = int(k/max(domaindf["n_types_notvg"]))
        add_samples_from_mean_ntypes = k - (max_ntypes * samples_per_ntype)
        
        for i in range(1, max_ntypes+1):
            if i == mean_ntypes:
                # sample objects from set with mean number of types leater, when filling up remaining objects 
                continue
            ki = min(samples_per_ntype, sum(domaindf["n_types_notvg"]==i))
            samples = samples.append(domaindf[domaindf["n_types_notvg"]==i].sample(n=ki))
            add_samples_from_mean_ntypes += (samples_per_ntype - ki)
        
        # sample remaining objects from set with mean number of types
        samples_from_mean_ntypes = add_samples_from_mean_ntypes + samples_per_ntype
        samples = samples.append(domaindf[domaindf["n_types_notvg"]==mean_ntypes].sample(n=samples_from_mean_ntypes))
    
    return samples

#def _single_name_not_vg(resdf):
#    return resdf[resdf["vg_is_max"] + resdf["n_types"] == 1]

def write_amt_csv(sample_df, csv_fname, max_num_hits=500):
    num_hits = 0
    part = 0
    fname_imgObjIds = open(csv_fname.replace(".csv", "")+".imgobjids", "w")
    for sample in sample_df.iterrows():
        coll_names = set(sample[1]["spellchecked_min2"]).difference([sample[1]["vg_obj_name"]])
        for nm in coll_names:
            if num_hits == 0:
                fout = open(csv_fname.replace(".csv", "part%d.csv" % part), "w")
                fout.write("image_url_1,vg_name_1,response_1_1\n")
            fout.write("{0[url]},{0[vg_obj_name]},{1}\n".format(sample[1], nm))
            fname_imgObjIds.write("{0[vg_img_id]}\t{0[vg_object_id]}\n".format(sample[1]))
            num_hits += 1
            if num_hits >= max_num_hits:
                fout.close()
                num_hits = 0
                part += 1
                
                
    if num_hits > 0:
        fout.close()
    fname_imgObjIds.close()
            
    
relevant_cols = ['vg_img_id', 'cat', 'synset', 'vg_obj_name', 'vg_domain','vg_object_id',
                     'url', 'opt-outs', 'spellchecked', 'all_responses']
sample_df = sample_objects("../data_phase0/all_responses_round0-3_cleaned.csv", 30, relevant_cols)
write_amt_csv(sample_df, "test_amt_csv.csv")
