import pandas as pd
pd.options.display.max_colwidth = 100
import json
import glob

import numpy as np

import os


ANONYMIZE = False
CONTROL_RELIABILITY_THRESHOLD = .49  # Delete control if not more than this did it correctly

ASSIGNMENT_APPROVAL_TRESHOLD = .7   # will approve any assignment with score higher than this
WORKER_APPROVAL_TRESHOLD = .8   # will approve all assignments even if a single assignment is crap

WORKER_BLOCK_THRESHOLD = .8     # will block worker based on mean below this
ASSIGNMENT_BLOCK_THRESHOLD = .7   # will block worker based on single assignment below .7

BONUS_THRESHOLD = 1.0

INSPECT_FAILED_CONTROLS = True
INSPECT_REJECTED_ASSIGNMENTS = True
NO_REJECTION = False

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
dfs_per_image = []
for i in reversed(range(MAX_N_IMAGES)): # reversed in order for names with 10 and 1 to not interfere.
    # columns = meta + ['image_url_{}'.format(i), 'quality_control_{}'.format(i)]
    columns = meta + ['items'] + [col for col in assignments_from_mturk.columns if col.startswith('img{}'.format(i))]
    df = assignments_from_mturk[columns].copy()
    df.columns = [col.replace('img{}-'.format(i), '').lower() for col in df.columns]
    df['names'] = df['items'].apply(lambda x: x[i][1] if len(x) > i else np.nan)
    df['image_url'] = df['items'].apply(lambda x: x[i][0] if len(x) > i else np.nan)
    df['quality_control'] = df['items'].apply(lambda x: json.loads(bytes.fromhex(x[i][2]).decode('utf-8')) if len(x) > i else {})
    del df['items']
    dfs_per_image.append(df)

per_image = pd.concat(dfs_per_image, sort=True)
per_image.reset_index(inplace=True, drop=True)

# Clean img_urls
per_image['image'] = per_image['image_url'].apply(lambda x: x.split('//')[-1].split('_')[0] if x != '' else x)
per_image['object'] = per_image['image_url'].apply(lambda x: x.split('//')[-1].split('_')[1] if x != '' else x)
# del per_image['image_url']

print("Per image:", len(per_image))
print(per_image[:5].to_string())

# Split dataframe into one row per annotation
annotations = []
meta = ['hitid', 'requesterannotation', 'assignmentid', 'workerid', 'image', 'object', 'image_url', 'quality_control', 'names']
for n in reversed(range(MAX_N_NAMES)):
    columns = meta + ['name{}-rating'.format(n), 'name{}-type'.format(n), 'name{}-color'.format(n)]
    df = per_image[columns].copy()
    df.columns = [col.replace('name{}-'.format(n), '') for col in df.columns]
    df['name'] = df['names'].apply(lambda x: x[n] if len(x) > n else np.nan)
    del df['names']
    df = df.loc[~df['name'].isna()]
    annotations.append(df)

annotations = pd.concat(annotations)

# Cleanup, typing, sorting
annotations.reset_index(inplace=True)
annotations = annotations[['hitid', 'requesterannotation', 'assignmentid', 'workerid', 'image', 'object', 'name', 'rating', 'type', 'color', 'image_url', 'quality_control']]
annotations['rating'] = annotations['rating'].astype(int)
annotations['type'] = annotations['type'].replace({0: 'linguistic', 1: 'bounding box', 2: 'visual', 3: 'other'})
annotations['color'] = annotations['color'].astype(int)
annotations = annotations.sort_values(by=['workerid', 'hitid']).reset_index(drop=True)


# From 'color' column, compute list of same-colored names
annotations['same_color'] = [[] for _ in range(len(annotations))]
for i, row in annotations.iterrows():
    if row['color'] != -1:
        same_color = annotations.loc[(annotations['assignmentid'] == row['assignmentid']) & (annotations['image'] == row['image']) & (annotations['object'] == row['object']) & (annotations['color'] == row['color'])]
        same_colored_names = same_color['name'].unique().tolist()
        annotations.at[i, 'same_color'] = same_colored_names
    else:
        annotations.at[i, 'same_color'] = [row['name']]


# Un-obfuscate quality control; create 'control_type' column storing what type of quality control it is
annotations['control_type'] = ""
for i, row in annotations.iterrows():
    if row['name'] in row['quality_control']:
        item = row['quality_control'][row['name']].replace("sans-", "typo-").replace("bold-", "syn-")
        if item == "arial":
            item = "vg_majority"
        elif item == "serif":
            item = "alternative"
        elif item == "courier":
            item = "random"
    else:
        item = ""
    annotations.at[i, 'control_type'] = item

del annotations['quality_control']


# Check if quality control items are correct
annotations['correct1'] = np.nan
annotations['correct2'] = np.nan
annotations['correct3'] = np.nan
annotations['correct3_explanation'] = ""
for i, row in annotations.iterrows():
    if row['control_type'] == 'vg_majority':
        annotations.at[i, 'correct1'] = float(row['rating'] == 0)

    elif row['control_type'].startswith('typo'):
        original = '-'.join(row['control_type'].split('-')[1:])
        original_row = annotations.loc[annotations['assignmentid'] == row['assignmentid']].loc[(annotations['image'] == row['image']) & (annotations['object'] == row['object'])].loc[annotations['name'] == original].squeeze()
        if row['rating'] == 0:
            annotations.at[i, 'correct1'] = float(False)
        elif row['type'] != 'linguistic':
            if original_row['rating'] == 0:
                annotations.at[i, 'correct1'] = float(False)
            elif row['type'] != original_row['type']:
                annotations.at[i, 'correct1'] = float(False)
        else:
            annotations.at[i, 'correct1'] = float(True)
        annotations.at[i, 'correct2'] = float(original in row['same_color'])

    elif row['control_type'] == 'alternative':
        annotations.at[i, 'correct1'] = float(not (row['rating'] == 0 or row['type'] not in ['bounding box', 'other']))
        positive = annotations.loc[(annotations['assignmentid'] == row['assignmentid']) & (annotations['image'] == row['image']) & (annotations['object'] == row['object']) & (annotations['control_type'] == 'vg_majority')]
        if len(positive) > 0:
            annotations.at[i, 'correct2'] = float(positive['name'].squeeze() not in row['same_color'])

    elif row['control_type'] == 'random':
        annotations.at[i, 'correct1'] = float(not (row['rating'] == 0 or row['type'] != 'other'))
        annotations.at[i, 'correct2'] = float(len(row['same_color']) == 1)

    elif row['control_type'].startswith('syn'):
        synonyms = row['control_type'][4:].split(',')
        same_rating = annotations.loc[(annotations['assignmentid'] == row['assignmentid']) & (annotations['image'] == row['image']) & (annotations['object'] == row['object']) & (annotations['rating'] == row['rating'])]['name'].unique().tolist()
        annotations.at[i, 'correct1'] = float(not any([syn not in same_rating for syn in synonyms]))
        annotations.at[i, 'correct2'] = float(not any([syn not in row['same_color'] for syn in synonyms]))

    else: # not a control item, but still a consistency check possible:
        if row['rating'] == 2 and row['type'] == 'bounding box':
            names_deemed_good = annotations.loc[(annotations['assignmentid'] == row['assignmentid']) & (annotations['image'] == row['image']) & (annotations['object'] == row['object']) & (annotations['rating'] == 0)]['name'].unique().tolist()
            good_names_same_color = [n for n in row['same_color'] if n in names_deemed_good]
            annotations.at[i, 'correct3'] = float(len(good_names_same_color) == 0)
            if not annotations.at[i, 'correct3']:
                annotations.at[i, 'correct3_explanation'] = row['name'] + '_' + ','.join(good_names_same_color)

# Change how rating is represented; this is a bit risky, but it works, so let's not touch it.
annotations['rating'] = annotations['rating'].apply(lambda x: (2 - x) / 2)    # mapping 2 to 0, 1 to 1/2, 0 to 1.


# Compute which controls are reliable (>X% correct)
scores_per_name = annotations.groupby(['image', 'object', 'name']).agg({'correct1': 'mean', 'correct2': 'mean', 'correct3': 'mean', 'image_url': lambda x: x.tolist()[0],  'control_type': lambda x: x.tolist()[0], 'rating': lambda x: x.tolist(), 'type': lambda x: x.tolist(), 'same_color': lambda x: x.tolist(), 'control_type': lambda x: x.to_list()[0]}).reset_index()
scores_per_name['reliable1'] = scores_per_name['correct1'] >= CONTROL_RELIABILITY_THRESHOLD
scores_per_name['reliable2'] = scores_per_name['correct2'] >= CONTROL_RELIABILITY_THRESHOLD
reliable1_controls = scores_per_name.loc[scores_per_name['reliable1']][['image', 'object', 'name']].values.tolist()
reliable2_controls = scores_per_name.loc[scores_per_name['reliable2']][['image', 'object', 'name']].values.tolist()
annotations['reliable1'] = True
annotations['reliable2'] = True
for i, row in annotations.iterrows():
    if [row['image'], row['object'], row['name']] not in reliable1_controls:
        annotations.at[i, 'reliable1'] = False
    if [row['image'], row['object'], row['name']] not in reliable2_controls:
        annotations.at[i, 'reliable2'] = False

# Some summary stats
# annotations['control_type_trimmed'] = annotations['control_type'].apply(lambda x: x.split('-')[0])
# print(annotations.groupby(['reliable1', 'reliable2', 'control_type_trimmed']).count())
scores_per_name.reset_index(inplace=True)
scores_per_name['control_type_trimmed'] = scores_per_name['control_type'].apply(lambda x: x.split('-')[0])
print(scores_per_name.groupby(['reliable1', 'reliable2', 'control_type_trimmed']).count())

# Inspect unreliable controls
if INSPECT_FAILED_CONTROLS:
    print("\nFailed controls:")
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

# Inspect low-agreement items (lowest-agreement first)


print("\nPer name:", len(annotations))
print(annotations[:10].to_string())


# Create assignments dataframe with some basic stats: # TODO Oh my this is ugly code...
annotations['correct1-filtered'] = annotations['correct1'].copy()
annotations['correct2-filtered'] = annotations['correct2'].copy()
annotations['correct3-filtered'] = annotations['correct3'].copy()
annotations.at[~annotations['reliable1'], 'correct1-filtered'] = np.nan
annotations.at[~annotations['reliable2'], 'correct2-filtered'] = np.nan
assignments = annotations.groupby(['hitid', 'requesterannotation', 'assignmentid', 'workerid']).agg({'rating': 'mean', 'color': 'mean',
                                                                                                     'correct1': ['count', 'sum'], 'correct2': ['count', 'sum'], 'correct3': ['count', 'sum'],
                                                                                                     'correct1-filtered': ['count', 'sum'], 'correct2-filtered': ['count', 'sum'], 'correct3-filtered': ['count', 'sum'],})
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

mean_score_per_worker = assignments.groupby(['workerid']).agg({'control_score-filtered': 'mean', 'control_score': 'mean'})

for i, row in assignments.iterrows():
    if NO_REJECTION or row['control_score-filtered'] >= ASSIGNMENT_APPROVAL_TRESHOLD:
        assignments.at[i, 'decision1'] = 'approve'
        if row['control_score-filtered'] >= BONUS_THRESHOLD:
            assignments.at[i, 'decision2'] = 'bonus'
    else:
        if mean_score_per_worker.at[row['workerid'], 'control_score-filtered'] >= WORKER_APPROVAL_TRESHOLD:
            assignments.at[i, 'decision1'] = 'approve'
        else:
            assignments.at[i, 'decision1'] = 'reject'
    if row['control_score-filtered'] < ASSIGNMENT_BLOCK_THRESHOLD and mean_score_per_worker.at[row['workerid'], 'control_score-filtered'] < WORKER_BLOCK_THRESHOLD:
        assignments.at[i, 'decision2'] = 'block'

# More bookkeeping: column with mistakes (URL + control_type + rating/type/samecolor)
assignments['mistakes'] = [[] for _ in range(len(assignments))]
for i, row in assignments.iterrows():
    mistakes1 = annotations.loc[(annotations['assignmentid'] == row['assignmentid']) & (annotations['correct1'] == False)][['image_url', 'name', 'rating', 'type']].values.tolist()
    mistakes2 = annotations.loc[(annotations['assignmentid'] == row['assignmentid']) & (annotations['correct2'] == False)][['image_url', 'name', 'same_color']].values.tolist()
    mistakes3 = annotations.loc[(annotations['assignmentid'] == row['assignmentid']) & (annotations['correct3'] == False)][['image_url', 'name', 'correct3_explanation']].values.tolist()
    mistakes1 = ["{} - {}: {}".format(m[0], m[1], m[2] if m[2] == 1 else m[3]) for m in mistakes1]
    mistakes2 = ["{} - {}: {}".format(m[0], m[1], m[2]) for m in mistakes2]
    mistakes3 = ["{} - {}: {}".format(m[0], m[1], m[2]) for m in mistakes3]
    assignments.at[i, 'mistakes'] = mistakes1 + mistakes2 + mistakes3
assignments['n_mistakes'] = assignments['mistakes'].apply(len)


print()
print(assignments.to_string())


print()
print(assignments.groupby(['decision1', 'decision2']).agg({'workerid': 'nunique', 'assignmentid': 'nunique', 'hitid': 'nunique'}))

if INSPECT_REJECTED_ASSIGNMENTS:
    for i, row in assignments.loc[assignments['decision1'] == 'reject'].iterrows():
        print(row['assignmentid'], row['workerid'], row['control_score'], row['control_score-filtered'], mean_score_per_worker.at[row['workerid'], 'control_score-filtered'], mean_score_per_worker.at[row['workerid'], 'control_score'], '\n  ' + '\n  '.join(row['mistakes']))
        print()



with open(os.path.join(resultsdir, 'per_assignment.csv'), 'w+') as outfile:
    assignments.to_csv(outfile, index=False)
    print("\nAssignments written to", outfile.name)


if ANONYMIZE:
    for i, workerid in enumerate(annotations['workerid'].unique()):
        annotations.replace(to_replace={'workerid': {workerid: 'worker{}'.format(i)}}, inplace=True)
    print("Worker IDs anonymized.")


with open(os.path.join(resultsdir, 'per_name.csv'), 'w+') as outfile:
    annotations.to_csv(outfile)
    print("Answers per name written to", outfile.name)


# # For quick inspection
# print()
# print(annotations.groupby(['workerid', 'hitid'])['rating', 'correct1', 'correct2'].agg(['count', 'mean']).to_string())


