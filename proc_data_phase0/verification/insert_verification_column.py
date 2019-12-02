from analysis import load_results
import os
import pandas as pd
from collections import Counter
from tqdm import tqdm
import numpy as np

verification_data_path = '../../verification_phase0/1_crowdsourced/results/merged_1-2-3-4-5-6-7-8_redone/name_annotations_filtered_ANON.csv'
many_names_path = "../spellchecking/all_responses_round0-3_cleaned.csv"
out_path = "all_responses_round0-3_verified.csv"

os.makedirs('../proc_data_phase0/verified/', exist_ok=True)

manynames = load_results.load_cleaned_results(many_names_path, index_col='vg_img_id')
manynames['spellchecked_min2'] = manynames['spellchecked'].apply(lambda x: Counter({k: x[k] for k in x if x[k] > 1}))

verifications = pd.read_csv(verification_data_path, converters={'name_cluster-nofillers': eval, 'same_object-nofillers': eval})
del verifications['batch']
del verifications['source']
del verifications['hitid']
del verifications['assignmentid']
del verifications['workerid']
del verifications['name_cluster']
del verifications['same_object']

# Remove all manipulated control items
# ad hoc fix for weird bug: elizabethville is not in fact a control item
verifications = verifications.loc[(verifications['name'] == 'elizabethville') |
                                  verifications['control_type'].isna() |
                                  (verifications['control_type'] == 'vg_majority') |
                                  (verifications['control_type'].apply(lambda x: isinstance(x, str) and x.startswith('syn')))]
verifications.rename(columns={'name_cluster-nofillers': 'name_cluster', 'same_object-nofillers': 'same_object'}, inplace=True)


print("ManyNames:", len(manynames))
print(manynames[:10].to_string())


print("\nVerifications:", len(verifications))
print(verifications[:10].to_string())

# aggregate per image+name
scores_per_name = verifications.groupby(['image', 'name']).agg({'adequacy': lambda x: x.tolist(), 'inadequacy_type': lambda x: x.tolist(), 'name_cluster': lambda x: x.tolist(), 'same_object': lambda x: x.tolist()})
# Remove points with <=1 annotator to avoid division by zero
scores_per_name = scores_per_name.loc[scores_per_name['adequacy'].apply(lambda x: len(x) > 1)]
# Are these stats biased towards images with more names? They are counted more times... Then again, there are more names there to agree or not agree on.
scores_per_name['adequacy_bin'] = scores_per_name['adequacy'].apply(lambda x: [0 if a == 0 else 1 for a in x])
scores_per_name['inadequacy_type'] = scores_per_name['inadequacy_type'].apply(lambda x: [str(t) for t in x])
scores_per_name['same_object'] = scores_per_name['same_object'].apply(lambda x: {key: [d[key] for d in x] for key in x[0]})

scores_per_name['adequacy_mean'] = scores_per_name['adequacy'].apply(lambda x: (2-(sum(x)/len(x)))/2)
scores_per_name['inadequacy_type_majority'] = scores_per_name['inadequacy_type'].apply(lambda x: Counter(x).most_common(1)[0][0]) # TODO What if no option has majority?
scores_per_name['same_object_majority'] = scores_per_name['same_object'].apply(lambda x: {k: Counter(x[k]).most_common(1)[0][0] for k in x})
scores_per_name['name_cluster_majority'] = scores_per_name['same_object_majority'].apply(lambda x: tuple(sorted(k for k in x if x[k] == 1)))

print("\nScores_per_name:", len(scores_per_name))
print(scores_per_name[:10].to_string())

# Loop through manynames updating each row with a new 'verified' column (dictionary)
manynames['verified'] = [{} for _ in range(len(manynames))]
for img, row in tqdm(manynames.iterrows(), total=len(manynames)):
    names_for_img = set(list(row['spellchecked_min2'].keys()) + [row['vg_obj_name']])
    if len(names_for_img) == 1:  # imgs with only one name weren't verified; just assume they're correct
        for name in names_for_img: # (there's only one though)
            manynames.at[img, 'verified'][name] = {'cluster': [name],
                                               'adequacy': 1.0,
                                               'inadequacy_type': None,
                                                'cluster_id': 0,
                                                'cluster_weight': 1.0,
                                               }
    else:
        cluster_weights = {}
        for name in names_for_img:
            cluster = scores_per_name.at[(img, name),'name_cluster_majority']
            inadequacy_type = scores_per_name.at[(img, name), 'inadequacy_type_majority']
            if inadequacy_type == 'nan':
                inadequacy_type = None
            manynames.at[img, 'verified'][name] = {'cluster': cluster,
                                                   'adequacy': scores_per_name.at[(img, name),'adequacy_mean'],
                                                   'inadequacy_type': inadequacy_type,
                                                   }
            if cluster not in cluster_weights:
                cluster_weights[cluster] = sum([row['spellchecked_min2'][name] for name in cluster])
        # normalize cluster weights
        clusters_total_weight = sum(cluster_weights.values())
        cluster_weights = {cluster: (cluster_weights[cluster] / clusters_total_weight) for cluster in cluster_weights}
        # sort to find most-picked cluster
        clusters_sorted = [y[0] for y in sorted(list(cluster_weights.items()), key=lambda x: x[1], reverse=True)]
        # add cluster id and weight
        for name in manynames.at[img, 'verified']:
            manynames.at[img, 'verified'][name]['cluster_id'] = clusters_sorted.index(manynames.at[img, 'verified'][name]['cluster'])
            manynames.at[img, 'verified'][name]['cluster_weight'] = cluster_weights[manynames.at[img, 'verified'][name]['cluster']]

manynames.to_csv(out_path, sep="\t")
print("New ManyNames csv with verification column saved to",out_path)