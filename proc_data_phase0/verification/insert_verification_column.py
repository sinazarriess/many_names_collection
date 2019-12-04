from analysis import load_results
import os
import pandas as pd
from collections import Counter
from tqdm import tqdm
import numpy as np
import random
from sklearn import cluster

from itertools import chain, combinations

random.seed(1234)

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

# verifications['matrix'] = verifications['same_object'].copy()
# for i, row in verifications.iterrows():
#     verifications.at[i, 'matrix'] = {row['name']: row['matrix']}
# scores_per_img = verifications.groupby('image').agg({'matrix': lambda x: x.tolist()})
# scores_per_img['matrix'] = scores_per_img['matrix'].apply(lambda x: {name: sum([d[name] for d in x]) for name in x[0]})

# print(scores_per_img.to_string())


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

print(scores_per_name[:10].to_string())

def cluster_from_similarity_matrix(similarities):
    names = sorted(list(similarities.keys())) # first sort to avoid hash randomness
    random.shuffle(names)   # now shuffle to avoid some kind of bias
    distances = np.zeros((len(names), len(names)))
    for i in range(len(names)):
        for j in range(len(names)):
            distances[i,j] = 1 - similarities[names[i]][names[j]]   # invert from closeness to distance
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering
    clustering = cluster.AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage="complete", distance_threshold=.5).fit_predict(distances)
    return {n: cl for n,cl in zip(names, clustering)}

def soft_cluster_from_similarity_matrix(similarities):
    names = sorted(list(similarities.keys())) # first sort to avoid hash randomness
    random.shuffle(names)   # now shuffle to avoid some kind of bias
    name_sets = chain.from_iterable(combinations(names, r) for r in range(len(names), 0, -1))   # ordered from big to small
    soft_clusters = []
    for name_set in name_sets:
        if not any([set(name_set) <= set(cluster) for cluster in soft_clusters]):
            if not any([similarities[n1][n2] < .5 for n1,n2 in combinations(name_set, 2)]):
                soft_clusters.append(name_set)
    return soft_clusters




# Old way:
scores_per_name['same_object_majority'] = scores_per_name['same_object'].apply(lambda x: {k: Counter(x[k]).most_common(1)[0][0] for k in x})
scores_per_name['name_cluster_majority'] = scores_per_name['same_object_majority'].apply(lambda x: tuple(sorted(k for k in x if x[k] == 1)))

# New way:
scores_per_name['same_object'] = scores_per_name['same_object'].apply(lambda x: {key: sum(x[key])/len(x[key]) for key in x})
for (img, name), row in scores_per_name.iterrows():
    scores_per_name.at[(img, name), 'same_object'] = {name: row['same_object']}
scores_per_image = scores_per_name.reset_index().groupby('image').agg({'same_object': lambda x: {key: d[key] for d in x for key in d}})

scores_per_image['cluster'] = [{} for _ in range(len(scores_per_image))]
scores_per_image['soft_cluster'] = [[] for _ in range(len(scores_per_image))]

for img, row in scores_per_image.iterrows():
    scores_per_image.at[img,'cluster'] = cluster_from_similarity_matrix(row['same_object'])
    scores_per_image.at[img, 'soft_cluster'] = soft_cluster_from_similarity_matrix(row['same_object'])

print("\nScores_per_image:", len(scores_per_image))
print(scores_per_image[:10].to_string())

print("\nScores_per_name:", len(scores_per_name))
print(scores_per_name[:10].to_string())

# Loop through manynames updating each row with a new 'verified' column (dictionary)
manynames['verified'] = [{} for _ in range(len(manynames))]
manynames['verified_soft_clusters'] = [{} for _ in range(len(manynames))]
for img, row in tqdm(manynames.iterrows(), total=len(manynames)):
    names_for_img = set(list(row['spellchecked_min2'].keys()) + [row['vg_obj_name']])
    if len(names_for_img) == 1:  # imgs with only one name weren't verified; just assume they're correct
        for name in names_for_img: # (there's only one though)
            manynames.at[img, 'verified'][name] = {'cluster': [name],
                                               'adequacy': 1.0,
                                               'can_be_same_object': [name],
                                               'inadequacy_type': None,
                                                'cluster_id': 0,
                                                'cluster_weight': 1.0,
                                               }
        manynames.at[img, 'verified_soft_clusters'][(name)] = {'index': 0,
                                                                'count': 1.0,
                                                                'adequacy': row['verified'][name]['adequacy'],
                                                                'inadequacy_type': row['verified'][name]['inadequacy_type']}
    else:
        cluster_weights = {}
        clustering = scores_per_image.at[img, 'cluster']
        soft_clustering = scores_per_image.at[img, 'soft_cluster']
        for name in names_for_img:
            cluster = tuple(sorted([n for n in clustering if clustering[n] == clustering[name]]))
            soft_clusters = [sc for sc in soft_clustering if name in sc]
            can_be_same_object = scores_per_name.at[(img, name), 'name_cluster_majority']   # old, majority-based way
            inadequacy_type = scores_per_name.at[(img, name), 'inadequacy_type_majority']
            if inadequacy_type == 'nan':
                inadequacy_type = None
            manynames.at[img, 'verified'][name] = {'cluster': cluster,
                                                   'soft_clusters': soft_clusters,
                                                   'can_be_same_object': can_be_same_object,
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

        soft_cluster_counts = {cluster: sum([row['spellchecked_min2'][name] for name in cluster]) for cluster in soft_clustering}
        # soft_clusters_total_weight = sum(cluster_weights.values())
        # soft_cluster_weights = {cluster: (soft_cluster_weights[cluster] / soft_clusters_total_weight) for cluster in soft_cluster_weights}
        soft_clusters_sorted = [y[0] for y in sorted(list(soft_cluster_counts.items()), key=lambda x: x[1], reverse=True)]
        for sc in soft_clusters_sorted:
            manynames.at[img, 'verified_soft_clusters'][sc] = {'index': soft_clusters_sorted.index(sc),
                                                               'count': soft_cluster_counts[sc],
                                                               'adequacy': sum([row['verified'][name]['adequacy'] for name in sc]) / len(sc),
                                                               'inadequacy_type': Counter([row['verified'][name]['inadequacy_type'] for name in sc]).most_common(1)[0][0]}

manynames.to_csv(out_path, sep="\t")
print("New ManyNames csv with verification column saved to",out_path)
