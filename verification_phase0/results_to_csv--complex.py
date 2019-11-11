import pandas as pd
pd.options.display.max_colwidth = 100
import json
import glob

import random

import numpy as np
from collections import Counter
import os
from sklearn.metrics import cohen_kappa_score

from scipy.stats import pearsonr


# Remove unreliable controls
CONTROL_RELIABILITY_THRESHOLD = .5  # Delete control if fewer than this did it correctly

# Approve/bonus assignments/workers
ASSIGNMENT_APPROVAL_TRESHOLD = .7   # will approve any assignment with score higher than this
WORKER_APPROVAL_TRESHOLD = .85   # will approve all assignments even if a few assignments are crap
BONUS_THRESHOLD = 1.0

# Block based on assignments/workers
ASSIGNMENT_BLOCK_THRESHOLD = .6   # will block worker based on single assignment below this
WORKER_BLOCK_THRESHOLD = .85     # will block worker based on mean below this

# Some more absolute params
NO_REJECTION = False
COULANCE = 1

INSPECT_FAILED_CONTROLS = False
INSPECT_REJECTED_ASSIGNMENTS = False


if NO_REJECTION:
    print("Warning: NO_REJECTION is set to true; all assignments will be accepted.")

# TODO Generalize; get paths from a config argument?
resultsdir = '1_pre-pilot/results'

# TODO check overwrite

# Read all assignments from the .json file from MTurk
assignments_from_mturk = []
for filename in glob.glob(os.path.join(resultsdir, '*.json')):
    with open(filename) as file:
        data = json.load(file)
        for hit in data:

            for assignment in hit['Assignments']:
                assignment['HITId'] = hit['HITId']
                assignment['RequesterAnnotation'] = hit['RequesterAnnotation']
                assignment.update(hit['Params'])
                assignment['items'] = eval(assignment['items'])
                answers = {}
                for key in assignment['Answers']:
                    answers.update(assignment['Answers'][key])
                assignment.update(answers)
                del assignment['Answers']
                assignments_from_mturk.append(assignment)


assignments_from_mturk = pd.DataFrame(assignments_from_mturk)
assignments_from_mturk.columns = [x.lower() for x in assignments_from_mturk.columns]
assignments_from_mturk.rename(columns={"stored_font_size": 'control_score_mturk'}, inplace=True)

MAX_N_IMAGES = max([int(col.split('-')[0][3:]) for col in assignments_from_mturk.columns if col.startswith('img')])
MAX_N_NAMES = max([int(col.split('-')[1][4:]) for col in assignments_from_mturk.columns if col.startswith('img') and col.split('-')[1].startswith('name')])

# Merge boolean radiobutton columns into integer-valued rating columns
for i in range(MAX_N_IMAGES):
    for n in range(MAX_N_NAMES):
        assignments_from_mturk['img{}-name{}-rating'.format(i, n)] = np.nan
        assignments_from_mturk['img{}-name{}-type'.format(i, n)] = np.nan
        assignments_from_mturk['img{}-name{}-color'.format(i, n)] = np.nan
for id, assignment in assignments_from_mturk.iterrows():
    for i in range(MAX_N_IMAGES):
        for n in range(MAX_N_NAMES):
            for r in range(3):
                if 'img{}-name{}-rating.{}'.format(i,n,r) in assignments_from_mturk and assignment['img{}-name{}-rating.{}'.format(i, n, r)] == 'true':
                    assignments_from_mturk.at[id, 'img{}-name{}-rating'.format(i, n)] = r
            for r in range(4):
                if 'img{}-name{}-type.{}'.format(i,n,r) in assignments_from_mturk and assignment['img{}-name{}-type.{}'.format(i, n, r)] == 'true':
                    assignments_from_mturk.at[id, 'img{}-name{}-type'.format(i, n)] = r
            for r in list(range(MAX_N_NAMES)) + ['x']:
                if 'img{}-name{}-color.{}'.format(i,n,r) in assignments_from_mturk and assignment['img{}-name{}-color.{}'.format(i, n, r)] == 'true':
                    assignments_from_mturk.at[id, 'img{}-name{}-color'.format(i, n)] = -1 if r == 'x' else r
for i in range(MAX_N_IMAGES):
    for n in range(MAX_N_NAMES):
        for r in range(3):
            if 'img{}-name{}-rating.{}'.format(i, n, r) in assignments_from_mturk:
                del assignments_from_mturk['img{}-name{}-rating.{}'.format(i, n, r)]
        for r in range(4):
            if 'img{}-name{}-type.{}'.format(i, n, r) in assignments_from_mturk:
                del assignments_from_mturk['img{}-name{}-type.{}'.format(i, n, r)]
        for r in list(range(MAX_N_NAMES)) + ['x']:
            if 'img{}-name{}-color.{}'.format(i, n, r) in assignments_from_mturk:
                del assignments_from_mturk['img{}-name{}-color.{}'.format(i, n, r)]


# Split by image
meta = ['hitid', 'requesterannotation', 'assignmentid', 'workerid']
image_annotations = []
for i in reversed(range(MAX_N_IMAGES)): # reversed in order for names with 10 and 1 to not interfere.
    # columns = meta + ['image_url_{}'.format(i), 'quality_control_{}'.format(i)]
    columns = meta + ['items'] + [col for col in assignments_from_mturk.columns if col.startswith('img{}'.format(i))]
    df = assignments_from_mturk[columns].copy()
    df.columns = [col.replace('img{}-'.format(i), '').lower() for col in df.columns]
    df['names'] = df['items'].apply(lambda x: x[i][1] if len(x) > i else np.nan)
    df['n_names'] = df['names'].apply(len)
    df['image_url'] = df['items'].apply(lambda x: x[i][0] if len(x) > i else np.nan)
    # unobfuscate quality control
    df['quality_control'] = df['items'].apply(lambda x: json.loads(bytes.fromhex(x[i][2]).decode('utf-8')) if len(x) > i else {})
    del df['items']
    image_annotations.append(df)

image_annotations = pd.concat(image_annotations, sort=True)
image_annotations.reset_index(inplace=True, drop=True)

# unobfuscate quality control
for i, row in image_annotations.iterrows():
    for name, control_type in row['quality_control'].items():
        control_type = control_type.replace("sans-", "typo-").replace("bold-", "syn-")
        if control_type == "arial":
            control_type = "vg_majority"
        elif control_type == "serif":
            control_type = "alternative"
        elif control_type == "courier":
            control_type = "random"
        image_annotations.at[i, 'quality_control'][name] = control_type

# Clean img_urls
image_annotations['image'] = image_annotations['image_url'].apply(lambda x: x.split('//')[-1].split('_')[0] if x != '' else x)
image_annotations['object'] = image_annotations['image_url'].apply(lambda x: x.split('//')[-1].split('_')[1] if x != '' else x)
# del per_image['image_url']

print('---------------------')
print("image_annotations:", len(image_annotations))
print(image_annotations[:5].to_string())
print('---------------------')


# Split dataframe into one row per name annotation
name_annotations = []
meta = ['hitid', 'requesterannotation', 'assignmentid', 'workerid', 'image', 'object', 'image_url', 'quality_control', 'names']
for n in reversed(range(MAX_N_NAMES)):
    columns = meta + ['name{}-rating'.format(n), 'name{}-type'.format(n), 'name{}-color'.format(n)]
    df = image_annotations[columns].copy()
    df.columns = [col.replace('name{}-'.format(n), '') for col in df.columns]
    df['name'] = df['names'].apply(lambda x: x[n] if len(x) > n else np.nan)
    df = df.loc[~df['name'].isna()]
    name_annotations.append(df)

name_annotations = pd.concat(name_annotations)

# Cleanup, typing, sorting
name_annotations.reset_index(inplace=True)
name_annotations = name_annotations[['hitid', 'requesterannotation', 'assignmentid', 'workerid', 'image', 'object', 'name', 'rating', 'type', 'color', 'image_url', 'quality_control', 'names']]
name_annotations['rating'] = name_annotations['rating'].astype(int)
name_annotations['type'] = name_annotations['type'].replace({0: 'linguistic', 1: 'bounding box', 2: 'visual', 3: 'other'})
name_annotations['color'] = name_annotations['color'].astype(int)
name_annotations = name_annotations.sort_values(by=['workerid', 'hitid']).reset_index(drop=True)

# From 'color' column, compute list of same-colored names, and dict of same-objectness
name_annotations['same_color'] = [[] for _ in range(len(name_annotations))]
name_annotations['colors'] = [{} for _ in range(len(name_annotations))]
for i, row in name_annotations.iterrows():
    if row['color'] != -1:
        same_color = name_annotations.loc[(name_annotations['assignmentid'] == row['assignmentid']) & (name_annotations['image'] == row['image']) & (name_annotations['object'] == row['object']) & (name_annotations['color'] == row['color'])]
        same_colored_names = same_color['name'].unique().tolist()
    else:
        same_colored_names = [row['name']]
    name_annotations.at[i, 'same_color'] = same_colored_names
    name_annotations.at[i, 'colors'].update({name: 1 if name in same_colored_names else 0 for name in name_annotations.at[i, 'names']})

# Create 'control_type' column storing what type of quality control it is
name_annotations['control_type'] = ""
for i, row in name_annotations.iterrows():
    name_annotations.at[i, 'control_type'] = "" if row['name'] not in row['quality_control'] else row['quality_control'][row['name']]

del name_annotations['quality_control']


# Check if quality control items are correct
name_annotations['correct1'] = np.nan
name_annotations['correct2'] = np.nan
name_annotations['correct3'] = np.nan
name_annotations['correct3_explanation'] = ""
for i, row in name_annotations.iterrows():
    if row['control_type'] == 'vg_majority':
        name_annotations.at[i, 'correct1'] = float(row['rating'] == 0)

    elif row['control_type'].startswith('typo'):
        original = '-'.join(row['control_type'].split('-')[1:])
        original_row = name_annotations.loc[name_annotations['assignmentid'] == row['assignmentid']].loc[(name_annotations['image'] == row['image']) & (name_annotations['object'] == row['object'])].loc[name_annotations['name'] == original].squeeze()
        if row['rating'] == 0:
            name_annotations.at[i, 'correct1'] = float(False)
        elif row['type'] != 'linguistic':
            if original_row['rating'] == 0:
                name_annotations.at[i, 'correct1'] = float(False)
            elif row['type'] != original_row['type']:
                name_annotations.at[i, 'correct1'] = float(False)
        else:
            name_annotations.at[i, 'correct1'] = float(True)
        name_annotations.at[i, 'correct2'] = float(original in row['same_color'])

    elif row['control_type'] == 'alternative':
        name_annotations.at[i, 'correct1'] = float(not (row['rating'] == 0 or row['type'] not in ['bounding box', 'other']))
        positive = name_annotations.loc[(name_annotations['assignmentid'] == row['assignmentid']) & (name_annotations['image'] == row['image']) & (name_annotations['object'] == row['object']) & (name_annotations['control_type'] == 'vg_majority')]
        if len(positive) > 0:
            name_annotations.at[i, 'correct2'] = float(positive['name'].squeeze() not in row['same_color'])

    elif row['control_type'] == 'random':
        name_annotations.at[i, 'correct1'] = float(not (row['rating'] == 0 or row['type'] != 'other'))
        name_annotations.at[i, 'correct2'] = float(len(row['same_color']) == 1)

    elif row['control_type'].startswith('syn'):
        synonyms = row['control_type'][4:].split(',')
        same_rating = name_annotations.loc[(name_annotations['assignmentid'] == row['assignmentid']) & (name_annotations['image'] == row['image']) & (name_annotations['object'] == row['object']) & (name_annotations['rating'] == row['rating'])]['name'].unique().tolist()
        name_annotations.at[i, 'correct1'] = float(not any([syn not in same_rating for syn in synonyms]))
        name_annotations.at[i, 'correct2'] = float(not any([syn not in row['same_color'] for syn in synonyms]))

    else: # not a control item, but still a consistency check possible:
        if row['rating'] == 2 and row['type'] == 'bounding box':
            names_deemed_good = name_annotations.loc[(name_annotations['assignmentid'] == row['assignmentid']) & (name_annotations['image'] == row['image']) & (name_annotations['object'] == row['object']) & (name_annotations['rating'] == 0)]['name'].unique().tolist()
            good_names_same_color = [n for n in row['same_color'] if n in names_deemed_good]
            name_annotations.at[i, 'correct3'] = float(len(good_names_same_color) == 0)
            if not name_annotations.at[i, 'correct3']:
                name_annotations.at[i, 'correct3_explanation'] = row['name'] + '_' + ','.join(good_names_same_color)

# Change how rating is represented; this is a bit risky, but it works, so let's not touch it.
# name_annotations['rating'] = name_annotations['rating'].apply(lambda x: (2 - x) / 2)    # mapping 2 to 0, 1 to 1/2, 0 to 1.

# Compute which controls are reliable (>X% correct)
scores_per_name = name_annotations.groupby(['image', 'object', 'name']).agg({'correct1': 'mean', 'correct2': 'mean', 'correct3': 'mean', 'image_url': lambda x: x.tolist()[0], 'control_type': lambda x: x.tolist()[0], 'rating': lambda x: x.tolist(), 'type': lambda x: x.tolist(), 'same_color': lambda x: x.tolist(), 'control_type': lambda x: x.to_list()[0]}).reset_index()
scores_per_name['reliable1'] = scores_per_name['correct1'] >= CONTROL_RELIABILITY_THRESHOLD
scores_per_name['reliable2'] = scores_per_name['correct2'] >= CONTROL_RELIABILITY_THRESHOLD
reliable1_controls = scores_per_name.loc[scores_per_name['reliable1']][['image', 'object', 'name']].values.tolist()
reliable2_controls = scores_per_name.loc[scores_per_name['reliable2']][['image', 'object', 'name']].values.tolist()
name_annotations['reliable1'] = True    # TODO Change to floats including nan
name_annotations['reliable2'] = True
for i, row in name_annotations.iterrows():
    if [row['image'], row['object'], row['name']] not in reliable1_controls:
        name_annotations.at[i, 'reliable1'] = False
    if [row['image'], row['object'], row['name']] not in reliable2_controls:
        name_annotations.at[i, 'reliable2'] = False

print('\n---------------------')
print("name_annotations:", len(name_annotations))
print(name_annotations[:5].to_string())
print('---------------------')

# Some summary stats
scores_per_name.reset_index(inplace=True)
scores_per_name['control_type_trimmed'] = scores_per_name['control_type'].apply(lambda x: x.split('-')[0])
print("Control reliability:")
print(scores_per_name.groupby(['reliable1', 'reliable2', 'control_type_trimmed']).count())
print()

# Inspect unreliable controls
if INSPECT_FAILED_CONTROLS:
    print('\n---------------------')
    print("Failed controls:")
    for c, control in scores_per_name.loc[((~scores_per_name['reliable1'] & ~scores_per_name['correct1'].isna()) |
                                           (~scores_per_name['reliable2'] & ~scores_per_name['correct2'].isna()))].iterrows():
        print("{} \"{}\" ({})\n  {}: {}, {}\n  {}: {}\n".format(
              control['image_url'],
              control['name'],
              control['control_type'],
              control['correct1'],
              control['rating'],
              control['type'],
              control['correct2'],
              control['same_color']))
    print('---------------------')

name_annotations['correct1-filtered'] = name_annotations['correct1'].copy()
name_annotations['correct2-filtered'] = name_annotations['correct2'].copy()
name_annotations['correct3-filtered'] = name_annotations['correct3'].copy()
name_annotations.at[~name_annotations['reliable1'], 'correct1-filtered'] = np.nan
name_annotations.at[~name_annotations['reliable2'], 'correct2-filtered'] = np.nan


# Create assignments dataframe with some basic stats: # TODO Oh my this is ugly code...
assignments = name_annotations.groupby(['hitid', 'requesterannotation', 'assignmentid', 'workerid']).agg({'rating': 'mean', 'color': 'mean',
                                                                                                     'correct1': ['count', 'sum'], 'correct2': ['count', 'sum'], 'correct3': ['count', 'sum'],
                                                                                                     'correct1-filtered': ['count', 'sum'], 'correct2-filtered': ['count', 'sum'], 'correct3-filtered': ['count', 'sum'], })
assignments.columns = ['_'.join(tup).rstrip('_') for tup in assignments.columns.values]
assignments['control_score'] = (assignments['correct1_sum'] + assignments['correct2_sum'] + assignments['correct3_sum']) / (assignments['correct1_count'] + assignments['correct2_count'] + assignments['correct3_count'])
assignments['control_score-filtered'] = (assignments['correct1-filtered_sum'] + assignments['correct2-filtered_sum'] + assignments['correct3-filtered_sum']) / (assignments['correct1-filtered_count'] + assignments['correct2-filtered_count'] + assignments['correct3-filtered_count'])
assignments.drop(['correct1_count', 'correct1_sum', 'correct2_count', 'correct2_sum', 'correct3_count', 'correct3_sum'], axis=1, inplace=True)
assignments.drop(['correct1-filtered_count', 'correct1-filtered_sum', 'correct2-filtered_count', 'correct2-filtered_sum', 'correct3-filtered_count', 'correct3-filtered_sum'], axis=1, inplace=True)


# Add the control score according to M-turk:
assignments = pd.merge(assignments, assignments_from_mturk[['hitid', 'requesterannotation', 'assignmentid', 'control_score_mturk', 'workerid']], on='assignmentid')
assignments['control_score_mturk'] = assignments['control_score_mturk'].apply(lambda x: 1 - float(x.split(',')[2]))


# Make decision and add bookkeeping columns to csv
assignments['decision1'] = ""
assignments['decision2'] = ""
assignments['executed1'] = False
assignments['executed2'] = False
assignments['explanation'] = ""

mean_score_per_worker = assignments.groupby(['workerid']).agg({'control_score-filtered': 'mean', 'control_score': 'mean'})
num_coulance = {worker: 0 for worker in assignments['workerid'].unique()}
# which assignments/workers to ignore in subsequent analysis
assignments_to_ignore = []
workers_to_ignore = []
# sort makes sure that lenience is applied to best of the worst
for i, row in assignments.sort_values(by="control_score-filtered", ascending=False).iterrows():
    if row['control_score-filtered'] >= ASSIGNMENT_APPROVAL_TRESHOLD:
        assignments.at[i, 'decision1'] = 'approve'
        assignments.at[i, 'explanation'] = 'good enough assignment'
    else:
        if mean_score_per_worker.at[row['workerid'], 'control_score-filtered'] >= WORKER_APPROVAL_TRESHOLD:
            assignments.at[i, 'decision1'] = 'approve'
            assignments.at[i, 'explanation'] = 'failed assignment, but good enough average worker score'
        elif num_coulance[row['workerid']] < COULANCE:
            assignments.at[i, 'decision1'] = 'approve'
            assignments.at[i, 'explanation'] = 'failed assignment, but apply lenience'
            num_coulance[row['workerid']] += 1
        elif NO_REJECTION:
            assignments.at[i, 'decision1'] = 'approve'
            assignments.at[i, 'explanation'] = 'failed assignment, but approve everything'
        else:
            assignments.at[i, 'decision1'] = 'reject'
            assignments.at[i, 'explanation'] = 'failed assignment, worker not good enough on average, lenience already applied'
        assignments_to_ignore.append(row['assignmentid'])

    if row['control_score-filtered'] >= BONUS_THRESHOLD:
        assignments.at[i, 'decision2'] = 'bonus'
        assignments.at[i, 'explanation'] += ', bonus because very good assignment'
    if row['control_score-filtered'] < ASSIGNMENT_BLOCK_THRESHOLD:
        assignments.at[i, 'decision2'] = 'block'
        assignments.at[i, 'explanation'] += ', blocked because very bad assignment'
    if mean_score_per_worker.at[row['workerid'], 'control_score-filtered'] < WORKER_BLOCK_THRESHOLD:
        assignments.at[i, 'decision2'] = 'block'
        assignments.at[i, 'explanation'] += ', blocked because low average quality'
        workers_to_ignore.append(row['workerid'])

assignments_to_ignore = set(assignments_to_ignore)
workers_to_ignore = set(workers_to_ignore)

# More bookkeeping: column with mistakes (URL + control_type + rating/type/samecolor)
assignments['mistakes'] = [[] for _ in range(len(assignments))]
for i, row in assignments.iterrows():
    mistakes1 = name_annotations.loc[(name_annotations['assignmentid'] == row['assignmentid']) & (name_annotations['correct1'] == False)][['image_url', 'name', 'rating', 'type']].values.tolist()
    mistakes2 = name_annotations.loc[(name_annotations['assignmentid'] == row['assignmentid']) & (name_annotations['correct2'] == False)][['image_url', 'name', 'same_color']].values.tolist()
    mistakes3 = name_annotations.loc[(name_annotations['assignmentid'] == row['assignmentid']) & (name_annotations['correct3'] == False)][['image_url', 'name', 'correct3_explanation']].values.tolist()
    mistakes1 = ["{} - {}: {}".format(m[0], m[1], m[2] if m[2] == 0 else m[3]) for m in mistakes1]
    mistakes2 = ["{} - {}: {}".format(m[0], m[1], m[2]) for m in mistakes2]
    mistakes3 = ["{} - {} ~ {}".format(m[0], m[1], m[2]) for m in mistakes3]
    assignments.at[i, 'mistakes'] = mistakes1 + mistakes2 + mistakes3
assignments['n_mistakes'] = assignments['mistakes'].apply(len)


print('\n---------------')
print(assignments[:5].to_string())
print('---------------')

print('\n---------------')
print(assignments.groupby(['decision1', 'decision2', 'explanation']).agg({'workerid': 'nunique', 'assignmentid': 'nunique', 'hitid': 'nunique'}).to_string())
print('---------------')

# Are there any workers who got a bonus and a rejection?
decisions_by_worker = assignments.groupby('workerid').agg({'decision1': lambda x: x.tolist(), 'decision2': lambda x: x.tolist()})
bonused_and_rejected = decisions_by_worker.loc[decisions_by_worker['decision2'].apply(lambda x: 'bonus' in x) & (decisions_by_worker['decision2'].apply(lambda x: 'block' in x) | decisions_by_worker['decision1'].apply(lambda x: 'reject' in x))]
if len(bonused_and_rejected) > 0:
    print("WARNING: Some workers were both bonused and blocked/rejected:\n", bonused_and_rejected.to_string())


if INSPECT_REJECTED_ASSIGNMENTS:
    for i, row in assignments.loc[assignments['decision1'] == 'reject'].iterrows():
        print("{}, {}: assignment: {} ({}); worker mean: {} ({})".format(row['assignmentid'], row['workerid'], row['control_score-filtered'], row['control_score'], mean_score_per_worker.at[row['workerid'], 'control_score-filtered'], mean_score_per_worker.at[row['workerid'], 'control_score']))
        print('  ' + '\n  '.join(row['mistakes']))
        print()



with open(os.path.join(resultsdir, 'per_name.csv'), 'w+') as outfile:
    scores_per_name.to_csv(outfile)
    print("Stats per name written to", outfile.name)

## Writing non-anonymous files
with open(os.path.join(resultsdir, 'per_assignment.csv'), 'w+') as outfile:
    assignments.to_csv(outfile, index=False)
    print("\nAssignments written to", outfile.name)

with open(os.path.join(resultsdir, 'name_annotations.csv'), 'w+') as outfile:
    name_annotations.to_csv(outfile)
    print("Annotations per name written to", outfile.name)

## Writing anonymous files
name_annotations_anon = name_annotations.copy()
assignments_anon = assignments.copy()
for i, workerid in enumerate(name_annotations_anon['workerid'].unique()):
    name_annotations_anon.replace(to_replace={'workerid': {workerid: 'worker{}'.format(i)}}, inplace=True)
    assignments_anon.replace(to_replace={'workerid': {workerid: 'worker{}'.format(i)}}, inplace=True)

with open(os.path.join(resultsdir, 'name_annotations_ANON.csv'), 'w+') as outfile:
    name_annotations_anon.to_csv(outfile)
    print("Anonymized name annotations written to", outfile.name)

with open(os.path.join(resultsdir, 'per_assignment_ANON.csv'), 'w+') as outfile:
    assignments_anon.to_csv(outfile, index=False)
    print("\nAnonymized assignments written to", outfile.name)



# randomly sample 3 out of 4 assignments per HIT
print('Name annotations:', len(name_annotations))
if False:
    assignmentids = assignments.groupby('hitid').agg({'assignmentid': lambda x: x.tolist()})
    assignmentids['assignmentid'] = assignmentids['assignmentid'].apply(lambda x: random.sample(x, len(x)-1))
    assignmentids = [a for x in assignmentids['assignmentid'] for a in x]
    name_annotations = name_annotations.loc[name_annotations['assignmentid'].isin(assignmentids)]
    print('Sampled down to:', len(name_annotations))

# name_annotations = name_annotations.loc[~name_annotations['workerid'].isin(workers_to_ignore)]
# print('After removing bad workers:', len(name_annotations))
if False:
    name_annotations = name_annotations.loc[~name_annotations['assignmentid'].isin(assignments_to_ignore)]
    print('After removing further bad assignments:', len(name_annotations))


# names = name_annotations.groupby(['image', 'object', 'image_url', 'name']).agg({
#         'rating': lambda x: (2-(x.mean()))/2,
#         # 'rating': lambda x: {rating: x.tolist().count(rating) for rating in [0, 1, 2]},
#         # 'type': lambda x: {str(t): [str(y) for y in x.tolist()].count(t) for t in ['nan', 'linguistic', 'bounding box', 'visual', 'other']},
#         'colors': lambda x: x.tolist(),
#         # 'assignmentid': 'nunique'
#     }) # .rename(columns={'assignmentid': 'n_assignments'})
#
# # for i, row in names.iterrows():
# #     names.at[i, 'colors'] = {name: sum([c[name] for c in row['colors']]) for name in row['colors'][0]}
#
#
# print('---------------------')
# print('names:', len(names))
# print(names[:10].to_string())
# print('---------------------')


# Turn name_annotations into name_pairs (one pair of names per row)
name_pairs = []
columns = ['image', 'object', 'url', 'workerid', 'name1', 'name2', 'correct_name', 'same_object']
for i, row in name_annotations.iterrows():
    for name in row['colors']:
        name_pairs.append([row['image'], row['object'], row['image_url'], row['workerid'], row['name'], name, row['rating'], row['colors'][name]])
name_pairs = pd.DataFrame(name_pairs, columns=columns).groupby(['image', 'object', 'url', 'name1', 'name2']).mean()

# Load manual annotations, make representations comparable/compatible
manual_annotations = '../raw_data_phase0/verification_pilot/verif_annos_pilot.csv'
name_pairs_amore = pd.read_csv(manual_annotations, sep="\t")
name_pairs_amore['image'] = name_pairs_amore['url'].apply(lambda x: x.split('//')[2].split('_')[0])
name_pairs_amore['object'] = name_pairs_amore['url'].apply(lambda x: x.split('//')[2].split('_')[1])
name_pairs_amore['same_object'] = name_pairs_amore['same_object'].replace({'Yes': 1, 'Not sure': .5, 'No': 0})
name_pairs_amore['correct_name'] = (3 - name_pairs_amore['correct_name']) / 3   # scale from [3,-3] to [0,2]
name_pairs_amore.rename(columns={'mn_obj_name': 'name1', 'vg_obj_name': 'name2'}, inplace=True)
name_pairs_amore = name_pairs_amore.groupby(['image', 'object', 'url', 'name1', 'name2']).agg({'correct_name': 'mean', 'same_object': 'mean'})

# Join amore annotations and crowdsourced annotations, and compute pearson correlations
amore_vs_crowd = name_pairs_amore.join(name_pairs, lsuffix='_amore', rsuffix='_crowd').dropna()
print('\n---------------')
print('AMORE vs. crowd:', len(amore_vs_crowd))
print(amore_vs_crowd[:5].to_string())
print('- - - - - - - -')
print(pearsonr(amore_vs_crowd['correct_name_amore'], amore_vs_crowd['correct_name_crowd']))
print(pearsonr(amore_vs_crowd['same_object_amore'], amore_vs_crowd['same_object_crowd']))
print('---------------')

# # For quick inspection
# print()
# print(annotations.groupby(['workerid', 'hitid'])['rating', 'correct1', 'correct2'].agg(['count', 'mean']).to_string())

# Explore mismaches
amore_vs_crowd_mismaches = amore_vs_crowd.loc[abs(amore_vs_crowd['correct_name_amore'] - amore_vs_crowd['correct_name_crowd']) > 1]
print("Mismatches correct_name:", len(amore_vs_crowd_mismaches))
for i, row in amore_vs_crowd_mismaches.iterrows():
    print(i, 'amore:', row['correct_name_amore'], ' - crowd:', row['correct_name_crowd'])
    input()

amore_vs_crowd_mismaches = amore_vs_crowd.loc[abs(amore_vs_crowd['same_object_amore'] - amore_vs_crowd['same_object_crowd']) > .5]
print("Mismatches same_object:", len(amore_vs_crowd_mismaches))
for i, row in amore_vs_crowd_mismaches.iterrows():
    print(i, 'amore:', row['same_object_amore'], ' - crowd:', row['same_object_crowd'])


