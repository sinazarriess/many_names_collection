import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from collections import Counter


BATCHES = [1,2,3,4,5,6,7,8]
REDONE = ['redone']
out_path = '1_crowdsourced/results/merged_{}'.format('-'.join([str(x) for x in BATCHES]))
if len(REDONE) > 0:
    out_path += '_{}'.format('-'.join(REDONE))

write_csv = not os.path.exists(out_path) or input("Recompute & overwrite merged csv files? y/N").lower().startswith('y')

os.makedirs(out_path, exist_ok=True)
BATCHES = ['batch{}'.format(i) for i in BATCHES] + REDONE

# For computing which assignments to redo:
if len(REDONE) == 0:
    UNRELIABLE_WORKER_THRESHOLD = 0.9
    UNRELIABLE_ASSIGNMENT_TRESHOLD = 0.7
# For actually deleting rows (less strict):
else:
    UNRELIABLE_WORKER_THRESHOLD = 0.5
    UNRELIABLE_ASSIGNMENT_TRESHOLD = 0.6

if write_csv:

    name_annotations_paths = [(i, '1_crowdsourced/results/{}/name_annotations.csv'.format(i)) for i in BATCHES]
    assignments_paths = [(i, '1_crowdsourced/results/{}/per_assignment.csv'.format(i)) for i in BATCHES]

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
assignments = pd.read_csv(os.path.join(out_path, 'per_assignment_ANON.csv'), converters={'mistakes': eval})

per_worker = assignments.groupby('workerid').agg({'control_score-filtered': 'mean'}).reset_index()
unreliable_workers = per_worker.loc[per_worker['control_score-filtered'] < UNRELIABLE_WORKER_THRESHOLD]['workerid']

assignments['reliable'] = (assignments['control_score-filtered'] >= UNRELIABLE_ASSIGNMENT_TRESHOLD) & ~(assignments['workerid'].isin(unreliable_workers))

print("Workers: {}; unreliable: {}".format(len(per_worker), len(unreliable_workers)))
print("Assignments: {}; unreliable: {}".format(len(assignments), len(assignments.loc[~assignments['reliable']])))

if len(REDONE) > 0:
    annotations_all = annotations_all.loc[annotations_all['assignmentid'].isin(assignments.loc[assignments['reliable']]['assignmentid'])]
    del annotations_all['reliable1']
    del annotations_all['reliable2']
    with open(os.path.join(out_path, 'name_annotations_filtered_ANON.csv'), 'w+') as outfile:
        annotations_all.to_csv(outfile, index=False)
        print("Anonymized, filtered name annotations written to", outfile.name)

coverage = assignments.loc[assignments['reliable']].groupby('source').agg({'assignmentid': 'count'})
for i in range(1047):
    src = '../1_pre-pilot/verification_round0_amt.csv_{}'.format(i)
    if src not in coverage.index:
        coverage[src]['assignmentid'] = 0
for i in range(292):
    src = '../1_pre-pilot/verification_round1_amt.csv_{}'.format(i)
    if src not in coverage.index:
        coverage[src]['assignmentid'] = 0
for i in range(1713):
    src = '../1_pre-pilot/verification_round2_amt.csv_{}'.format(i)
    if src not in coverage.index:
        coverage[src]['assignmentid'] = 0
coverage['missing'] = coverage['assignmentid'].apply(lambda x: 0 if x >= 3 else 3-x)
print("MISSING:", coverage['missing'].sum())

round0_amt = pd.read_csv('1_crowdsourced/verification_round0_amt.csv', sep=",", keep_default_na=False)
round1_amt = pd.read_csv('1_crowdsourced/verification_round1_amt.csv', sep=",", keep_default_na=False)
round2_amt = pd.read_csv('1_crowdsourced/verification_round2_amt.csv', sep=",", keep_default_na=False)

rounds_amt = {'round0': round0_amt, 'round1': round1_amt, 'round2': round2_amt}

print(assignments.loc[~assignments['reliable']][:30].to_string())

to_be_redone = []

columns = ['items', 'quality_control', 'source', 'assignments']
for i, row in coverage.loc[coverage['missing'] > 0].iterrows():
    round_amt = rounds_amt[i.split('_')[-3]]
    line = int(i.split('_')[-1])
    to_be_redone.append([round_amt.at[line,'items'], round_amt.at[line,'quality_control'], i, row['missing']])

to_be_redone = pd.DataFrame(to_be_redone, columns=columns)
to_be_redone.to_csv('1_crowdsourced/to_be_redone_amt.csv', index=False)
print("HITs to be redone written to 1_crowdsourced/to_be_redone_amt.csv.")

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


if False:
    # Inspect workers who requester feedback:
    print()
    print("Workers to pay attention to:")
    PAY_ATTENTION_TO_WORKERS = ['AO33H4GL9KZX9']
    by_selected_workers = assignments.loc[assignments['workerid'].isin(PAY_ATTENTION_TO_WORKERS)].sort_values(by='workerid')
    prev_worker = ""
    for i, row in by_selected_workers.iterrows():
        if row['workerid'] != prev_worker:
            print("== {} ==".format(row['workerid']))
            prev_worker = row['workerid']
        print("{}: {}, {}, {}".format(row['assignmentid'], row['decision1'], row['decision2'], row['explanation']))
        print('  ' + '\n  '.join(row['mistakes']))
    print()




print("\n~~~~~~~~~~~~~~")
print("Inter-annotator agreement:")
scores_per_name = annotations_targets.groupby(['image', 'object', 'name']).agg({'adequacy': lambda x: x.tolist(), 'inadequacy_type': lambda x: x.tolist(), 'name_cluster': lambda x: x.tolist(), 'same_object': lambda x: x.tolist()}).reset_index()
# Remove points with <=1 annotator to avoid division by zero
scores_per_name = scores_per_name.loc[scores_per_name['adequacy'].apply(lambda x: len(x) > 1)]
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