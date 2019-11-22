import pandas as pd
import json
import glob

import numpy as np

import os


# TODO Generalize; get paths from a config argument?

resultsdir = '1_pre-pilot/results/simple'

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

# Merge boolean radiobutton columns into integer-valued rating columns
for i in range(5):
    for n in range(1, 10):
        if 'IMG{}-NAME{}-rating.1'.format(i,n) in assignments:
            assignments['IMG{}-NAME{}'.format(i,n)] = np.nan
for id, assignment in assignments.iterrows():
    for i in range(5):
        for n in range(1, 10):
            for r in range(1,6):
                if 'IMG{}-NAME{}'.format(i,n) in assignments:
                    if assignment['IMG{}-NAME{}-rating.{}'.format(i,n,r)] == 'true':
                        assignments.at[id, 'IMG{}-NAME{}'.format(i,n)] = r
for i in range(5):
    for n in range(1, 10):
        for r in range(1, 6):
            if 'IMG{}-NAME{}'.format(i, n) in assignments:
                del assignments['IMG{}-NAME{}-rating.{}'.format(i,n, r)]

# Split by image
meta = ['AssignmentId', 'HITId', 'WorkerId']
dfs_per_image = []
for i in range(5):
    columns = meta + [col for col in assignments.columns if (col.startswith('image_url_{}'.format(i)) or
                                                            col.startswith('name_{}'.format(i)) or
                                                            col.startswith('IMG{}'.format(i)))]
    df = assignments[columns]
    df.columns = [col.replace('url_{}'.format(i), 'url').replace('name_{}'.format(i), 'name').replace('IMG{}'.format(i), 'rating').lower() for col in df.columns]
    dfs_per_image.append(df)

print(len(assignments), assignments[:5].to_string())

per_image = pd.concat(dfs_per_image, sort=True)

# Clean img_urls
per_image['image'] = per_image['image_url'].apply(lambda x: x.split('//')[-1].split('_')[0])
per_image['object'] = per_image['image_url'].apply(lambda x: x.split('//')[-1].split('_')[1])
# del per_image['image_url']

print(len(per_image), per_image[:5].to_string())

print(per_image.groupby(['image', 'object']).agg({'workerid': lambda x: len(x.unique())}))

# split by name
dfs_per_name = []
for n in range(11):
    if 'rating-name{}'.format(n) in per_image:
        columns = ['assignmentid', 'hitid', 'workerid', 'image', 'object', 'name_{}'.format(n), 'rating-name{}'.format(n), 'image_url']
        if not 'rating-name0' in per_image:  # simple setup
            columns.append('name_0')
        df = per_image[columns]
        df.columns = [col.replace('name_{}'.format(n), 'name').replace('rating-name{}'.format(n), 'rating') for col in df.columns]
        df = df.loc[df['name'] != '']
        dfs_per_name.append(df)

per_name = pd.concat(dfs_per_name)
per_name = per_name[['hitid', 'assignmentid', 'workerid', 'image', 'object', 'name_0', 'name', 'rating', 'image_url']]

per_name['rating'] = per_name['rating'].apply(lambda x: 6-x).astype(int)


per_name = per_name.sort_values(by=['workerid', 'hitid', 'name_0']).reset_index(drop=True)


# anonymize
for i, workerid in enumerate(per_name['workerid'].unique()):
    per_name.replace(to_replace={'workerid': {workerid: 'worker{}'.format(i)}}, inplace=True)



with open(outdir, 'w+') as outfile:
    per_name.to_csv(outfile)


# pd.options.display.max_colwidth = 100

print()
print(len(per_name), per_name.to_string())

print()

grouped = per_name.groupby(['workerid'])['rating'].agg(['count', 'mean'])

print(grouped)
