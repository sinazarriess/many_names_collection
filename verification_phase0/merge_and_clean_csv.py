import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from collections import Counter


BATCHES = [1,2,3,4,5,6,7,8]
out_path = '1_crowdsourced/results/merged_{}'.format('-'.join([str(x) for x in BATCHES]))
os.makedirs(out_path, exist_ok=True)

if os.path.exists(out_path) and input("Recompute & overwrite merged csv files? y/N").lower().startswith('y'):

    name_annotations_paths = [(i, '1_crowdsourced/results/batch{}/name_annotations.csv'.format(i)) for i in BATCHES]
    assignments_paths = [(i, '1_crowdsourced/results/batch{}/per_assignment.csv'.format(i)) for i in BATCHES]

    print("Merging dataframes...")

    name_annotations_dfs = []
    for i, p in name_annotations_paths:
        df = pd.read_csv(p, converters={'colors': eval, 'same_color': eval}).assign(batch=i)
        columns = [col for col in df.columns if not col.startswith('Unnamed')]
        name_annotations_dfs.append(df[columns])
    name_annotations = pd.concat(name_annotations_dfs, sort=True).reset_index(drop=True)
    name_annotations.rename(columns={'same_color': 'name_cluster', 'colors': 'same_object', 'rating': 'adequacy', 'type': 'inadequacy_type', 'requesterannotation': 'source'}, inplace=True)
    columns = ['batch', 'source', 'hitid', 'assignmentid', 'image', 'object', 'name', 'adequacy',
               'inadequacy_type', 'same_object', 'name_cluster', 'control_type', 'reliable1', 'reliable2', 'image_url', 'workerid']
    name_annotations = name_annotations[columns]

    print('Merged into name_annotations of {} rows.'.format(len(name_annotations)))

    # Create "no-filler" variants of same_object and name_cluster:
    print("Creating columns without fillers")
    fillers = {}
    for i, row in tqdm(name_annotations.iterrows()):
        if isinstance(row['control_type'], str) and row['control_type'] != 'vg_majority' and not row['control_type'].startswith('syn'):
            key = (row['image'], row['object'])
            if key not in fillers:
                fillers[key] = []
            fillers[key].append(row['name'])
    name_annotations['name_cluster-nofillers'] = name_annotations['name_cluster'].copy()
    for i, row in tqdm(name_annotations.iterrows()):
        name_annotations.at[i,'name_cluster-nofillers'] = [n for n in row['name_cluster-nofillers'] if n not in fillers[(row['image'], row['object'])]]


    assignments_dfs = []
    for i, p in assignments_paths:
        df = pd.read_csv(p, converters={'mistakes': eval}).assign(batch=i)
        columns = [col for col in df.columns if not col.startswith('Unnamed')]
        assignments_dfs.append(df[columns])

    assignments = pd.concat(assignments_dfs, sort=True).reset_index(drop=True)
    assignments.rename(columns={'requesterannotation': 'source'}, inplace=True)
    columns = ['batch', 'source', 'hitid', 'assignmentid', 'control_score', 'control_score-filtered',
               'decision1', 'decision2', 'explanation', 'mistakes', 'workerid']
    assignments = assignments[columns]

    print("\n-------------")
    print('assignments:', len(assignments))
    print('hits:', len(assignments['hitid'].unique()))
    print('workers:', len(assignments['workerid'].unique()))
    print('rejections:', len(assignments.loc[assignments['decision1'] == 'reject']))
    print('bonuses:', len(assignments.loc[assignments['decision2'] == 'bonus']))
    print(assignments[:5].to_string())
    print("-------------")

    with open(os.path.join(out_path, 'name_annotations.csv'), 'w+') as outfile:
        name_annotations.to_csv(outfile, index=False)
        print("Name annotations written to", outfile.name)

    with open(os.path.join(out_path, 'assignments.csv'), 'w+') as outfile:
        assignments.to_csv(outfile, index=False)
        print("Assignments written to", outfile.name)

    ## Writing anonymous files
    name_annotations_anon = name_annotations.copy()
    assignments_anon = assignments.copy()
    for i, workerid in enumerate(name_annotations_anon['workerid'].unique()):
        name_annotations_anon.replace(to_replace={'workerid': {workerid: 'worker{}'.format(i)}}, inplace=True)
        assignments_anon.replace(to_replace={'workerid': {workerid: 'worker{}'.format(i)}}, inplace=True)

    with open(os.path.join(out_path, 'name_annotations_ANON.csv'), 'w+') as outfile:
        name_annotations_anon.to_csv(outfile, index=False)
        print("Anonymized name annotations written to", outfile.name)

    with open(os.path.join(out_path, 'per_assignment_ANON.csv'), 'w+') as outfile:
        assignments_anon.to_csv(outfile, index=False)
        print("\nAnonymized assignments written to", outfile.name)


annotations_all = pd.read_csv(os.path.join(out_path, 'name_annotations_ANON.csv'), converters={'name_cluster-nofillers': eval, 'name_cluster': eval, 'same_object': eval})
annotations_targets = annotations_all.loc[annotations_all['control_type'].isna() | (annotations_all['control_type'] == 'vg_majority') | (annotations_all['control_type'].apply(lambda x: isinstance(x, str) and x.startswith('syn')))]

print("\n-------------")
print(annotations_all[:5].to_string())
print('name_annotations:', len(annotations_all))
print('targets:', len(annotations_targets))
print('unique images:', len(annotations_all['image'].unique()), "/", len(annotations_targets['image'].unique()))
print('unique names:', len(annotations_all['name'].unique()), "/", len(annotations_targets['name'].unique()))
print("-------------")

# annotations_targets_adequate['cluster_size-nofillers'] = annotations_targets_adequate['name_cluster-nofillers'].apply(len)
# grouped = annotations_targets_adequate.groupby(['image', 'object', 'assignmentid']).agg({'name_cluster-nofillers': lambda x: x.tolist()})
# grouped['n_clusters-nofillers'] = grouped['name_cluster-nofillers'].apply(len)
#
# print('Average cluster size: {}'.format(annotations_targets_adequate['cluster_size-nofillers'].mean()))
# print("Average n_clusters: {}".format(grouped['n_clusters-nofillers'].mean()))

# Compute average cluster size of these adequate names

# Compute how many names are adequate that are NOT in the same cluster



print("\n~~~~~~~~~~~~~~")
print("Inter-annotator agreement:")
scores_per_name = annotations_targets.groupby(['image', 'object', 'name']).agg({'adequacy': lambda x: x.tolist(), 'inadequacy_type': lambda x: x.tolist(), 'name_cluster': lambda x: x.tolist(), 'same_object': lambda x: x.tolist()}).reset_index()
# Are these stats biased towards images with more names? They are counted more times... Then again, there are more names there to agree or not agree on.
scores_per_name['adequacy_bin'] = scores_per_name['adequacy'].apply(lambda x: [0 if a == 0 else 1 for a in x])
scores_per_name['inadequacy_type'] = scores_per_name['inadequacy_type'].apply(lambda x: [str(t) for t in x])
scores_per_name['adequacy_agreement_cat'] = scores_per_name['adequacy'].apply(lambda x: sum([(x.count(score) * (x.count(score) - 1)) / (len(x) * (len(x) - 1)) for score in [0, 1, 2]]))
scores_per_name['adequacy_agreement_bin'] = scores_per_name['adequacy_bin'].apply(lambda x: sum([(x.count(score) * (x.count(score) - 1)) / (len(x) * (len(x) - 1)) for score in [0, 1]]))
scores_per_name['adequacy_agreement_std'] = scores_per_name['adequacy'].apply(np.std)
scores_per_name['inadequacy_type_agreement'] = scores_per_name['inadequacy_type'].apply(lambda x: sum([(x.count(score) * (x.count(score) - 1)) / (len(x) * (len(x) - 1)) for score in['nan', 'linguistic', 'bounding box', 'visual', 'other']]))
scores_per_name['same_object'] = scores_per_name['same_object'].apply(lambda x: {key: [d[key] for d in x] for key in x[0]})
scores_per_name['same_object_agreement'] = scores_per_name['same_object'].apply(lambda x: [sum([(x[k].count(score) * (x[k].count(score) - 1)) / (len(x[k]) * (len(x[k]) - 1)) for score in [0, 1]]) for k in x])
scores_per_name['same_object_agreement'] = scores_per_name['same_object_agreement'].apply(lambda x: sum(x) / len(x))
print(scores_per_name[['adequacy_agreement_cat', 'adequacy_agreement_bin', 'adequacy_agreement_std', 'inadequacy_type_agreement', 'same_object_agreement']].mean())
print("~~~~~~~~~~~~~~")
print()

## Aggregate multiple annotations for a image+name:
# adequacy: mean
# inadequacy_type: majority, otherwise 'unknown'
# same_object: majority, otherwise 'unknown'
scores_per_name['adequacy_mean'] = scores_per_name['adequacy'].apply(lambda x: sum(x)/len(x))
scores_per_name['inadequacy_type_majority'] = scores_per_name['inadequacy_type'].apply(lambda x: Counter(x).most_common(1)[0][0]) # TODO What if no option has majority?
scores_per_name['same_object_majority'] = scores_per_name['same_object'].apply(lambda x: {k: Counter(x[k]).most_common(1)[0][0] for k in x})
scores_per_name['name_cluster_majority'] = scores_per_name['same_object_majority'].apply(lambda x: tuple(sorted(k for k in x if x[k] == 1)))
print(scores_per_name[:20].to_string())

print("Number of names total:", len(scores_per_name))
print("Number of adequate names total:", len(scores_per_name.loc[scores_per_name['adequacy_mean'] < 1]))
print("Adequate names per image (<1):", scores_per_name.loc[scores_per_name['adequacy_mean'] < 1].groupby('image').agg({'name': 'count'}).mean().values[0])
print("Clusters (majority) per image:", scores_per_name.groupby('image').agg({'name_cluster_majority': 'nunique'}).mean().values[0])
print("Adequate clusters (majority) per image:", scores_per_name.loc[scores_per_name['adequacy_mean'] < 1].groupby('image').agg({'name_cluster_majority': 'nunique'}).mean().values[0])
cluster_sizes = scores_per_name.groupby('image').agg({'name_cluster_majority': lambda x: [len(t) for t in x.unique()]})
cluster_sizes_flat = [x for y in cluster_sizes['name_cluster_majority'].tolist() for x in y]
print("Cluster size (majority):", sum(cluster_sizes_flat)/len(cluster_sizes_flat))
cluster_sizes = scores_per_name.loc[scores_per_name['adequacy_mean'] < 1].groupby('image').agg({'name_cluster_majority': lambda x: [len(t) for t in x.unique()]})
cluster_sizes_flat = [x for y in cluster_sizes['name_cluster_majority'].tolist() for x in y]
print("Adequate cluster size (majority):", sum(cluster_sizes_flat)/len(cluster_sizes_flat))