import pandas as pd
import json
import glob

import numpy as np

import os

ANONYMIZE = False

pd.options.display.max_colwidth = 100
# TODO Generalize; get paths from a config argument?

resultsdir = '1_pre-pilot/results/complex'


# Read all assignments from the .json file from MTurk
assignments = []
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
                assignments.append(assignment)

                # TODO Get RequesterAnnotation

assignments = pd.DataFrame(assignments)
assignments.columns = [x.lower() for x in assignments.columns]
assignments.rename(columns={"stored_font_size": 'control_score_mturk'}, inplace=True)

MAX_N_IMAGES = 11
MAX_N_NAMES = 16

# Merge boolean radiobutton columns into integer-valued rating columns
for i in range(MAX_N_IMAGES):
    for n in range(MAX_N_NAMES):
        assignments['img{}-name{}-rating'.format(i,n)] = np.nan
        assignments['img{}-name{}-type'.format(i, n)] = np.nan
        assignments['img{}-name{}-color'.format(i, n)] = np.nan
for id, assignment in assignments.iterrows():
    for i in range(MAX_N_IMAGES):
        for n in range(MAX_N_NAMES):
            for r in range(3):
                if 'img{}-name{}-rating.{}'.format(i,n,r) in assignments and assignment['img{}-name{}-rating.{}'.format(i,n,r)] == 'true':
                    assignments.at[id, 'img{}-name{}-rating'.format(i,n)] = r
            for r in range(4):
                if 'img{}-name{}-type.{}'.format(i,n,r) in assignments and assignment['img{}-name{}-type.{}'.format(i,n,r)] == 'true':
                    assignments.at[id, 'img{}-name{}-type'.format(i,n)] = r
            for r in list(range(MAX_N_NAMES)) + ['x']:
                if 'img{}-name{}-color.{}'.format(i,n,r) in assignments and assignment['img{}-name{}-color.{}'.format(i,n,r)] == 'true':
                    assignments.at[id, 'img{}-name{}-color'.format(i,n)] = -1 if r == 'x' else r
for i in range(MAX_N_IMAGES):
    for n in range(MAX_N_NAMES):
        for r in range(3):
            if 'img{}-name{}-rating.{}'.format(i, n, r) in assignments:
                del assignments['img{}-name{}-rating.{}'.format(i,n, r)]
        for r in range(4):
            if 'img{}-name{}-type.{}'.format(i, n, r) in assignments:
                del assignments['img{}-name{}-type.{}'.format(i, n, r)]
        for r in list(range(MAX_N_NAMES)) + ['x']:
            if 'img{}-name{}-color.{}'.format(i, n, r) in assignments:
                del assignments['img{}-name{}-color.{}'.format(i, n, r)]


print(assignments[:5].to_string())



# Split by image
meta = ['assignmentid', 'hitid', 'workerid']
dfs_per_image = []
for i in reversed(range(MAX_N_IMAGES)):
    columns = meta + [col for col in assignments.columns if (col == 'image_url_{}'.format(i) or
                                                            col.startswith('name_{}_'.format(i)) or
                                                            col == 'quality_control_{}'.format(i) or
                                                            col.startswith('img{}-name'.format(i))) or
                                                            col == 'img{}-comment'.format(i)]
    df = assignments[columns]
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

per_name = pd.concat(dfs_per_name)
per_name.reset_index(inplace=True)

per_name = per_name[['hitid', 'assignmentid', 'workerid', 'image', 'object', 'name', 'rating', 'type', 'color', 'image_url', 'quality_control']]

per_name['rating'] = per_name['rating'].astype(int)
per_name['type'] = per_name['type'].replace({0: 'linguistic', 1: 'bounding box', 2: 'visual', 3: 'other'})
per_name['color'] = per_name['color'].astype(int)


# from 'color' column, compute list of same-colored names
per_name['same_color'] = [[] for _ in range(len(per_name))]
for i, row in per_name.iterrows():
    if row['color'] != -1:
        same_color = per_name.loc[(per_name['assignmentid'] == row['assignmentid']) & (per_name['image'] == row['image']) & (per_name['object'] == row['object']) & (per_name['color'] == row['color'])]
        same_colored_names = same_color['name'].unique().tolist()
        per_name.at[i, 'same_color'] = same_colored_names
    else:
        per_name.at[i, 'same_color'] = [row['name']]

# Un-obfuscate quality control; create 'control_type' column storing what type of quality control it is
per_name['quality_control'] = per_name['quality_control'].apply(lambda x: json.loads(bytes.fromhex(x).decode('utf-8')))
per_name['control_type'] = ""
for i, row in per_name.iterrows():
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
    per_name.at[i,'control_type'] = item

del per_name['quality_control']

# Check if control items are correct
per_name['correct1'] = np.nan
per_name['correct2'] = np.nan
# per_name['correct1'] = per_name['correct1'].astype(bool)    # to convert the Nones to proper nans?
# per_name['correct2'] = per_name['correct2'].astype(bool)
for i, row in per_name.iterrows():
    if row['control_type'] == 'vg_majority':
        per_name.at[i, 'correct1'] = float(row['rating'] == 0)

    elif row['control_type'].startswith('typo'):
        original = '-'.join(row['control_type'].split('-')[1:])
        original_row = per_name.loc[per_name['assignmentid'] == row['assignmentid']].loc[(per_name['image'] == row['image']) & (per_name['object'] == row['object'])].loc[per_name['name'] == original].squeeze()
        if row['rating'] == 0:
            per_name.at[i, 'correct1'] = float(False)
        elif row['type'] != 'linguistic':
            if original_row['rating'] == 0:
                per_name.at[i, 'correct1'] = float(False)
            elif row['type'] != original_row['type']:
                per_name.at[i, 'correct1'] = float(False)
        else:
            per_name.at[i, 'correct1'] = float(True)
        per_name.at[i, 'correct2'] = float(original in row['same_color'])

    elif row['control_type'] == 'alternative':
        per_name.at[i, 'correct1'] = float(not (row['rating'] == 0 or row['type'] not in ['bounding box', 'other']))
        positive = per_name.loc[(per_name['assignmentid'] == row['assignmentid']) & (per_name['image'] == row['image']) & (per_name['object'] == row['object']) & (per_name['control_type'] == 'vg_majority')]
        if len(positive) > 0:
            per_name.at[i, 'correct2'] = float(positive['name'].squeeze() not in row['same_color'])

    elif row['control_type'] == 'random':
        per_name.at[i, 'correct1'] = float(not (row['rating'] == 0 or row['type'] != 'other'))
        per_name.at[i, 'correct2'] = float(len(row['same_color']) == 1)


per_name = per_name.sort_values(by=['workerid', 'hitid']).reset_index(drop=True)

per_name_mean = per_name.groupby(['image', 'object', 'name']).agg({'correct1': 'mean', 'correct2': 'mean'})
# TODO compute average performance on control items;
# TODO exclude controls with low average performance
# TODO compute new quality scores


print()

per_name['rating'] = per_name['rating'].apply(lambda x: (2-x)/2)    # mapping 2 to 0, 1 to 1/2, 0 to 1.

print("Per name:", len(per_name))
print(per_name[:10].to_string())



per_assignment = per_name.groupby(['assignmentid', 'workerid']).agg({'rating': 'mean', 'correct1': ['count', 'sum'], 'correct2': ['count', 'sum']})
per_assignment.columns = ['_'.join(tup).rstrip('_') for tup in per_assignment.columns.values]

per_assignment['n_controls'] = per_assignment[['correct1_count', 'correct2_count']].sum(axis=1)
per_assignment['control_score'] = (per_assignment['correct1_sum'] + per_assignment['correct2_sum']) / (per_assignment['correct1_count'] + per_assignment['correct2_count'])

per_assignment = pd.merge(per_assignment, assignments[['assignmentid', 'control_score_mturk', 'workerid']], on='assignmentid')
per_assignment['control_score_mturk'] = per_assignment['control_score_mturk'].apply(lambda x: 1-float(x.split(',')[-1]))

del per_assignment['correct1_count']
del per_assignment['correct1_sum']
del per_assignment['correct2_count']
del per_assignment['correct2_sum']

per_assignment['decision'] = per_assignment['control_score'].apply(lambda x: "approve" if x > .5 else "reject")
per_assignment['executed'] = False

print(per_assignment.to_string())

with open(os.path.join(resultsdir, 'assignments.csv'), 'w+') as outfile:
    per_name.to_csv(outfile, index=False)
    print("Assignments written to", outfile.name)

# TODO compare quality scores to those in the MTurk field (merge from the original per_HIT data).

# TODO manual inspection

quit()

print()

print(per_name.groupby(['workerid', 'hitid'])['rating','correct1', 'correct2'].agg(['count', 'mean']).to_string())

# print(per_name.loc[per_name['control_type'] == 'alternative'].to_string())


# print(per_name.loc[per_name['control_type'].apply(lambda x: x.startswith('typo'))].to_string())

if ANONYMIZE:
    for i, workerid in enumerate(per_name['workerid'].unique()):
        per_name.replace(to_replace={'workerid': {workerid: 'worker{}'.format(i)}}, inplace=True)

# per_name['correct2'] = per_name['correct2'].astype(bool) ## TODO Meh this doesn't work.


with open(os.path.join(resultsdir, 'names.csv'), 'w+') as outfile:
    per_name.to_csv(outfile)
    print("Answers per name written to", outfile.name)

