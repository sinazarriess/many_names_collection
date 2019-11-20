import pandas as pd
import numpy as np
import os

BATCHES = [3]
out_path = '1_pre-pilot/results/merged_{}'.format('-'.join([str(x) for x in BATCHES]))
os.makedirs(out_path, exist_ok=True)

name_annotations_paths = [(i, '1_pre-pilot/results/batch{}/name_annotations.csv'.format(i)) for i in BATCHES]
assignments_paths = [(i, '1_pre-pilot/results/batch{}/per_assignment.csv'.format(i)) for i in BATCHES]

name_annotations_dfs = []
for i, p in name_annotations_paths:
    df = pd.read_csv(p, converters={'colors': eval}).assign(batch=i)
    columns = [col for col in df.columns if not col.startswith('Unnamed')]
    name_annotations_dfs.append(df[columns])

name_annotations = pd.concat(name_annotations_dfs, sort=True).reset_index(drop=True)
name_annotations.rename(columns={'colors': 'same_object', 'rating': 'adequacy', 'type': 'inadequacy_type', 'requesterannotation': 'source'}, inplace=True)
columns = ['batch', 'source', 'hitid', 'assignmentid', 'image', 'object', 'name', 'adequacy',
           'inadequacy_type', 'same_object', 'control_type', 'reliable1', 'reliable2', 'image_url', 'workerid']
name_annotations = name_annotations[columns]

print("\n-------------")
print('name_annotations:', len(name_annotations))
print('unique images:', len(name_annotations['image'].unique()))
print(name_annotations[:5].to_string())
print("-------------")


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


print("\n~~~~~~~~~~~~~~")
print("Inter-annotator agreement:")
scores_per_name = name_annotations.groupby(['image', 'object', 'name']).agg({'adequacy': lambda x: x.tolist(), 'inadequacy_type': lambda x: x.tolist(), 'same_object': lambda x: x.tolist()}).reset_index(drop=True)
# Are these stats biased towards images with more names? They are counted more times... Then again, there are more names there to agree or not agree on.
scores_per_name['adequacy_bin'] = scores_per_name['adequacy'].apply(lambda x: [0 if a == 0 else 1 for a in x])
scores_per_name['inadequacy_type'] = scores_per_name['inadequacy_type'].apply(lambda x: [str(t) for t in x])
scores_per_name['adequacy_agreement_cat'] = scores_per_name['adequacy'].apply(lambda x: sum([(x.count(score)*(x.count(score)-1)) / (len(x)*(len(x)-1)) for score in [0,1,2]]))
scores_per_name['adequacy_agreement_bin'] = scores_per_name['adequacy_bin'].apply(lambda x: sum([(x.count(score)*(x.count(score)-1)) / (len(x)*(len(x)-1)) for score in [0,1]]))
scores_per_name['adequacy_agreement_std'] = scores_per_name['adequacy'].apply(np.std)
scores_per_name['inadequacy_type_agreement'] = scores_per_name['inadequacy_type'].apply(lambda x: sum([(x.count(score)*(x.count(score)-1)) / (len(x)*(len(x)-1)) for score in ['nan', 'linguistic', 'bounding box', 'visual', 'other']]))
scores_per_name['same_object'] = scores_per_name['same_object'].apply(lambda x: {key: [d[key] for d in x] for key in x[0]})
scores_per_name['same_object_agreement'] = scores_per_name['same_object'].apply(lambda x: [sum([(x[k].count(score)*(x[k].count(score)-1)) / (len(x[k])*(len(x[k])-1)) for score in [0, 1]]) for k in x])
scores_per_name['same_object_agreement'] = scores_per_name['same_object_agreement'].apply(lambda x: sum(x) / len(x))
print(scores_per_name[['adequacy_agreement_cat', 'adequacy_agreement_bin', 'adequacy_agreement_std', 'inadequacy_type_agreement', 'same_object_agreement']].mean())
print("~~~~~~~~~~~~~~")
print()


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