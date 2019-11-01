import pandas as pd
pd.options.display.max_colwidth = 100
import json
import glob

import numpy as np

import os


ANONYMIZE = False
CONTROL_RELIABILITY_THRESHOLD = .5
APPROVAL_TRESHOLD = .7  # TODO Can be higher once I have more annotations?
BONUS_THRESHOLD = 1.0
BLOCK_THRESHOLD = .8

# TODO Generalize; get paths from a config argument?
resultsdir = '1_pre-pilot/results/complex'

# TODO check overwrite

# Read all assignments from the .json file from MTurk
assignments_from_mturk = []
for filename in glob.glob(os.path.join(resultsdir, '*.json')):
    with open(filename) as file:
        data = json.load(file)
        for hit in data:

            for assignment in hit['Assignments']:
                assignment['HITId'] = hit['HITId']
                assignment.update(hit['Params'])
                answers = {}
                for key in assignment['Answers']:
                    answers.update(assignment['Answers'][key])
                assignment.update(answers)
                del assignment['Answers']
                assignments_from_mturk.append(assignment)

                # TODO Get RequesterAnnotation

assignments_from_mturk = pd.DataFrame(assignments_from_mturk)
assignments_from_mturk.columns = [x.lower() for x in assignments_from_mturk.columns]
assignments_from_mturk.rename(columns={"stored_font_size": 'control_score_mturk'}, inplace=True)

MAX_N_IMAGES = 11    # TODO read this from the dataframe
MAX_N_NAMES = 16

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


print(assignments_from_mturk[:5].to_string())



# Split by image
meta = ['assignmentid', 'hitid', 'workerid']
dfs_per_image = []
for i in reversed(range(MAX_N_IMAGES)):
    columns = meta + [col for col in assignments_from_mturk.columns if (col == 'image_url_{}'.format(i) or
                                                                        col.startswith('name_{}_'.format(i)) or
                                                                        col == 'quality_control_{}'.format(i) or
                                                                        col.startswith('img{}-name'.format(i))) or
                      col == 'img{}-comment'.format(i)]
    df = assignments_from_mturk[columns]
    df.columns = [col.replace('image_url_{}'.format(i), 'image_url').replace('name_{}_'.format(i), 'name_').replace('img{}-'.format(i), '').replace('quality_control_{}'.format(i), 'quality_control').lower() for col in df.columns]
    dfs_per_image.append(df)


per_image = pd.concat(dfs_per_image, sort=True)
per_image.reset_index(inplace=True, drop=True)

# Clean img_urls
per_image['image'] = per_image['image_url'].apply(lambda x: x.split('//')[-1].split('_')[0] if x != '' else x)
per_image['object'] = per_image['image_url'].apply(lambda x: x.split('//')[-1].split('_')[1] if x != '' else x)
# del per_image['image_url']

print("Per image:", len(per_image))

print(per_image[:200].to_string())
# print(per_image.groupby(['image', 'object']).agg({'workerid': lambda x: len(x.unique())}))



# split by name
dfs_per_name = []
for n in reversed(range(MAX_N_NAMES)):
    if 'name{}-rating'.format(n) in per_image:
        columns = ['assignmentid', 'hitid', 'workerid', 'image', 'object', 'name_{}'.format(n), 'name{}-rating'.format(n), 'name{}-type'.format(n), 'name{}-color'.format(n), 'image_url', 'quality_control']
        df = per_image[columns]
        df.columns = [col.replace('name_{}'.format(n), 'name').replace('name{}-'.format(n), '') for col in df.columns]
        df = df.loc[df['name'] != '']
        dfs_per_name.append(df)

annotations = pd.concat(dfs_per_name)
annotations.reset_index(inplace=True)

annotations = annotations[['hitid', 'assignmentid', 'workerid', 'image', 'object', 'name', 'rating', 'type', 'color', 'image_url', 'quality_control']]

annotations['rating'] = annotations['rating'].astype(int)
annotations['type'] = annotations['type'].replace({0: 'linguistic', 1: 'bounding box', 2: 'visual', 3: 'other'})
annotations['color'] = annotations['color'].astype(int)


# from 'color' column, compute list of same-colored names
annotations['same_color'] = [[] for _ in range(len(annotations))]
for i, row in annotations.iterrows():
    if row['color'] != -1:
        same_color = annotations.loc[(annotations['assignmentid'] == row['assignmentid']) & (annotations['image'] == row['image']) & (annotations['object'] == row['object']) & (annotations['color'] == row['color'])]
        same_colored_names = same_color['name'].unique().tolist()
        annotations.at[i, 'same_color'] = same_colored_names
    else:
        annotations.at[i, 'same_color'] = [row['name']]

# Un-obfuscate quality control; create 'control_type' column storing what type of quality control it is
annotations['quality_control'] = annotations['quality_control'].apply(lambda x: json.loads(bytes.fromhex(x).decode('utf-8')))
annotations['control_type'] = ""
for i, row in annotations.iterrows():
    if row['name'] in row['quality_control']:
        item = row['quality_control'][row['name']].replace("sans-", "typo-")
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

# Check if control items are correct
annotations['correct1'] = np.nan
annotations['correct2'] = np.nan
# per_name['correct1'] = per_name['correct1'].astype(bool)    # to convert the Nones to proper nans?
# per_name['correct2'] = per_name['correct2'].astype(bool)
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
        same_rating = annotations.loc[(annotations['assignmentid'] == row['assignmentid']) & (annotations['image'] == row['image']) & (annotations['object'] == row['object']) & (annotations['rating'] == row['rating'])]['name'].unique().to_list()
        annotations.at[i, 'correct1'] = float(not any([syn not in same_rating for syn in synonyms]))
        annotations.at[i, 'correct2'] = float(not any([syn not in row['same_color'] for syn in synonyms]))


annotations = annotations.sort_values(by=['workerid', 'hitid']).reset_index(drop=True)

scores_per_name = annotations.groupby(['image', 'object', 'name']).agg({'correct1': 'mean', 'correct2': 'mean'}).reset_index()

# Compute which controls are reliable (>X% correct)
reliable_controls1 = scores_per_name.loc[scores_per_name['correct1'] > CONTROL_RELIABILITY_THRESHOLD][['image', 'object', 'name']].values.tolist()
reliable_controls2 = scores_per_name.loc[scores_per_name['correct2'] > CONTROL_RELIABILITY_THRESHOLD][['image', 'object', 'name']].values.tolist()
for i, row in annotations.iterrows():
    if [row['image'], row['object'], row['name']] not in reliable_controls1:
        annotations.at[i, 'correct1'] = np.nan
    if [row['image'], row['object'], row['name']] not in reliable_controls2:
        annotations.at[i, 'correct2'] = np.nan


print()

annotations['rating'] = annotations['rating'].apply(lambda x: (2 - x) / 2)    # mapping 2 to 0, 1 to 1/2, 0 to 1.

print("Per name:", len(annotations))
print(annotations[:10].to_string())


assignments = annotations.groupby(['assignmentid', 'workerid']).agg({'rating': 'mean', 'color': 'mean', 'correct1': ['count', 'sum'], 'correct2': ['count', 'sum']})
assignments.columns = ['_'.join(tup).rstrip('_') for tup in assignments.columns.values]

assignments['n_controls'] = assignments[['correct1_count', 'correct2_count']].sum(axis=1)
assignments['control_score'] = (assignments['correct1_sum'] + assignments['correct2_sum']) / (assignments['correct1_count'] + assignments['correct2_count'])

assignments = pd.merge(assignments, assignments_from_mturk[['assignmentid', 'control_score_mturk', 'workerid']], on='assignmentid')
assignments['control_score_mturk'] = assignments['control_score_mturk'].apply(lambda x: 1 - float(x.split(',')[-1]))

del assignments['correct1_count']
del assignments['correct1_sum']
del assignments['correct2_count']
del assignments['correct2_sum']

assignments['decision1'] = assignments['control_score'].apply(lambda x: "approve" if x >= APPROVAL_TRESHOLD else "reject")
assignments['decision2'] = assignments['control_score'].apply(lambda x: "bonus" if x >= BONUS_THRESHOLD else "block" if x < BLOCK_THRESHOLD else np.nan)
assignments['executed1'] = False
assignments['executed2'] = False

# Add column with reliable mistakes (URL + control_type + rating/type/samecolor)
assignments['mistakes'] = [[] for _ in range(len(assignments))]
for i, row in assignments.iterrows():
    mistakes1 = annotations.loc[(annotations['assignmentid'] == row['assignmentid']) & (annotations['correct1'] == False)][['image_url', 'name', 'rating', 'type']].values.tolist()
    mistakes2 = annotations.loc[(annotations['assignmentid'] == row['assignmentid']) & (annotations['correct2'] == False)][['image_url', 'name', 'same_color']].values.tolist()
    mistakes1 = ["{} - {}: {}".format(m[0], m[1], m[2] if m[2] == 1 else m[3]) for m in mistakes1]
    mistakes2 = ["{} - {}: {}".format(m[0], m[1], m[2]) for m in mistakes2]
    assignments.at[i, 'mistakes'] = mistakes1 + mistakes2
assignments['n_mistakes'] = assignments['mistakes'].apply(len)

print(assignments.to_string())

with open(os.path.join(resultsdir, 'assignments.csv'), 'w+') as outfile:
    assignments.to_csv(outfile, index=False)
    print("Assignments written to", outfile.name)

# TODO compare quality scores to those in the MTurk field (merge from the original per_HIT data).


print("Total: {} assignments. Approve: {} (bonus: {}). Reject: {} (block: {})".format(len(assignments),
                                                                                      len(assignments.loc[assignments['decision1'] == 'approve']),
                                                                                      len(assignments.loc[assignments['decision2'] == 'bonus']),
                                                                                      len(assignments.loc[assignments['decision1'] == 'reject']),
                                                                                      len(assignments.loc[assignments['decision2'] == 'block']),
                                                                                      ))

print()

print(annotations.groupby(['workerid', 'hitid'])['rating', 'correct1', 'correct2'].agg(['count', 'mean']).to_string())

# print(per_name.loc[per_name['control_type'] == 'alternative'].to_string())


# print(per_name.loc[per_name['control_type'].apply(lambda x: x.startswith('typo'))].to_string())

if ANONYMIZE:
    for i, workerid in enumerate(annotations['workerid'].unique()):
        annotations.replace(to_replace={'workerid': {workerid: 'worker{}'.format(i)}}, inplace=True)

# per_name['correct2'] = per_name['correct2'].astype(bool) ## TODO Meh this doesn't work.


with open(os.path.join(resultsdir, 'names.csv'), 'w+') as outfile:
    annotations.to_csv(outfile)
    print("Answers per name written to", outfile.name)

