import pandas as pd
import json
import glob

import numpy as np

import os



pd.options.display.max_colwidth = 100
# TODO Generalize; get paths from a config argument?

resultsdir = '1_pre-pilot/results/complex'

outdir = os.path.join(resultsdir, 'answers.csv')

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

assignments = pd.DataFrame(assignments)

MAX_N_IMAGES = 11
MAX_N_NAMES = 16

print(assignments[:5].to_string())

# Merge boolean radiobutton columns into integer-valued rating columns
for i in range(MAX_N_IMAGES):
    for n in range(MAX_N_NAMES):
        assignments['IMG{}-NAME{}-rating'.format(i,n)] = np.nan
        assignments['IMG{}-NAME{}-type'.format(i, n)] = np.nan
        assignments['IMG{}-NAME{}-color'.format(i, n)] = np.nan
for id, assignment in assignments.iterrows():
    for i in range(MAX_N_IMAGES):
        for n in range(MAX_N_NAMES):
            for r in range(3):
                if 'IMG{}-NAME{}-rating.{}'.format(i,n,r) in assignments and assignment['IMG{}-NAME{}-rating.{}'.format(i,n,r)] == 'true':
                    assignments.at[id, 'IMG{}-NAME{}-rating'.format(i,n)] = r
            for r in range(4):
                if 'IMG{}-NAME{}-type.{}'.format(i,n,r) in assignments and assignment['IMG{}-NAME{}-type.{}'.format(i,n,r)] == 'true':
                    assignments.at[id, 'IMG{}-NAME{}-type'.format(i,n)] = r
            for r in list(range(MAX_N_NAMES)) + ['X']:
                if 'IMG{}-NAME{}-color.{}'.format(i,n,r) in assignments and assignment['IMG{}-NAME{}-color.{}'.format(i,n,r)] == 'true':
                    assignments.at[id, 'IMG{}-NAME{}-color'.format(i,n)] = -1 if r == 'X' else r
for i in range(MAX_N_IMAGES):
    for n in range(MAX_N_NAMES):
        for r in range(3):
            if 'IMG{}-NAME{}-rating.{}'.format(i, n, r) in assignments:
                del assignments['IMG{}-NAME{}-rating.{}'.format(i,n, r)]
        for r in range(4):
            if 'IMG{}-NAME{}-type.{}'.format(i, n, r) in assignments:
                del assignments['IMG{}-NAME{}-type.{}'.format(i, n, r)]
        for r in list(range(MAX_N_NAMES)) + ['X']:
            if 'IMG{}-NAME{}-color.{}'.format(i, n, r) in assignments:
                del assignments['IMG{}-NAME{}-color.{}'.format(i, n, r)]


print(sorted(assignments.columns.tolist()))

# Split by image
meta = ['AssignmentId', 'HITId', 'WorkerId']
dfs_per_image = []
for i in reversed(range(MAX_N_IMAGES)):
    columns = meta + [col for col in assignments.columns if (col == 'image_url_{}'.format(i) or
                                                            col.startswith('name_{}_'.format(i)) or
                                                            col == 'quality_control_{}'.format(i) or
                                                            col.startswith('IMG{}-NAME'.format(i)))]
    df = assignments[columns]
    df.columns = [col.replace('image_url_{}'.format(i), 'image_url').replace('name_{}_'.format(i), 'name_').replace('IMG{}-'.format(i), '').replace('quality_control_{}'.format(i), 'quality_control').lower() for col in df.columns]
    dfs_per_image.append(df)

per_image = pd.concat(dfs_per_image, sort=True)
per_image.reset_index(inplace=True)

# Clean img_urls
per_image['image'] = per_image['image_url'].apply(lambda x: x.split('//')[-1].split('_')[0] if x != '' else x)
per_image['object'] = per_image['image_url'].apply(lambda x: x.split('//')[-1].split('_')[1] if x != '' else x)
# del per_image['image_url']

print("Per image:", len(per_image))
print(per_image[:5].to_string())
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
        per_name.at[i, 'correct1'] = float(row['type'] == 'linguistic' or (original_row['rating'] != 0 and row['type'] == original_row['type']))
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




print()


per_name['rating'] = per_name['rating'].apply(lambda x: (2-x)/2)    # mapping 2 to 0, 1 to 1/2, 0 to 1.

# anonymize
for i, workerid in enumerate(per_name['workerid'].unique()):
    per_name.replace(to_replace={'workerid': {workerid: 'worker{}'.format(i)}}, inplace=True)

# per_name['correct2'] = per_name['correct2'].astype(bool) ## TODO Meh this doesn't work.

print("Per name:", len(per_name))
print(per_name[:10].to_string())


print()

print(per_name.groupby(['workerid', 'hitid'])['rating','correct1', 'correct2'].agg(['count', 'mean']).to_string())

print(per_name.loc[per_name['control_type'] == 'alternative'].to_string())


# print(per_name.loc[per_name['control_type'].apply(lambda x: x.startswith('typo'))].to_string())

with open(outdir, 'w+') as outfile:
    per_name.to_csv(outfile)
    print("Results written to", outfile.name)

