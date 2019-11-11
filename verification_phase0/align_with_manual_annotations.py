import pandas as pd
from scipy.stats import pearsonr

ROUNDS = {
    'pilot1': '1_pre-pilot/processed_results/pilot1/name_annotations_ANON.csv',
    # 'pilot2': '1_pre-pilot/processed_results/pilot2/name_annotations_ANON.csv',
}

name_annotations = None
for round in ROUNDS:
    print("Reading", ROUNDS[round])
    df = pd.read_csv(ROUNDS[round], index_col=0, converters={'colors': eval, 'names': eval, 'same_color': eval})
    df['round'] = round
    if name_annotations is None:
        name_annotations = df
    else:
        name_annotations = pd.concat([name_annotations, df])
name_annotations.reset_index(inplace=True, drop=True)
name_annotations['object'] = name_annotations['object'].astype(str)
name_annotations['image'] = name_annotations['image'].astype(str)

print('\n-------------------')
print("name_annotations:", len(name_annotations))
print(name_annotations[:5].to_string())

# Turn name_annotations into name_pairs (one pair of names per row)
name_pairs = []
columns = ['round', 'image', 'object', 'url', 'workerid', 'name1', 'name2', 'correct_name', 'same_object']
for i, row in name_annotations.iterrows():
    for name in row['colors']:
        name_pairs.append([row['round'], row['image'], row['object'], row['image_url'], row['workerid'], row['name'], name, row['rating'], row['colors'][name]])
name_pairs = pd.DataFrame(name_pairs, columns=columns).groupby(['round', 'image', 'object', 'url', 'name1', 'name2']).mean()
name_pairs.reset_index(level=0, inplace=True)

print('\n-------------------')
print("name_pairs:", len(name_pairs), name_pairs.index)
print(name_pairs[:5].to_string())


# Load manual annotations, make representations comparable/compatible
manual_annotations = '../raw_data_phase0/verification_pilot/verif_annos_pilot.csv'
name_pairs_amore = pd.read_csv(manual_annotations, sep="\t")
name_pairs_amore['image'] = name_pairs_amore['url'].apply(lambda x: x.split('//')[2].split('_')[0])
name_pairs_amore['object'] = name_pairs_amore['url'].apply(lambda x: x.split('//')[2].split('_')[1])
name_pairs_amore['same_object'] = name_pairs_amore['same_object'].replace({'Yes': 1, 'Not sure': .5, 'No': 0})
name_pairs_amore['correct_name'] = (3 - name_pairs_amore['correct_name']) / 3   # scale from [3,-3] to [0,2]
name_pairs_amore.rename(columns={'mn_obj_name': 'name1', 'vg_obj_name': 'name2'}, inplace=True)
name_pairs_amore = name_pairs_amore.groupby(['image', 'object', 'url', 'name1', 'name2']).agg({'correct_name': 'mean', 'same_object': 'mean'})

print('\n-------------------')
print("name_pairs_amore:", len(name_pairs_amore), name_pairs_amore.index)
print(name_pairs_amore[:5].to_string())


# Join amore annotations and crowdsourced annotations, and compute pearson correlations
amore_vs_crowd = name_pairs_amore.join(name_pairs, lsuffix='_amore', rsuffix='_crowd').dropna()
print('\n---------------')
print('AMORE vs. crowd:', len(amore_vs_crowd))
print(amore_vs_crowd[:5].to_string())

print('\n===============')
print("Pearson correlations amore ~ crowd:")
for round in list(ROUNDS.keys()) + (['all'] if len(ROUNDS) > 1 else []):
    print('- Round', round)
    if round == 'all':
        df = amore_vs_crowd
    else:
        df = amore_vs_crowd.loc[amore_vs_crowd['round'] == round]
    print('   ', pearsonr(df['correct_name_amore'], df['correct_name_crowd']))
    print('   ', pearsonr(df['same_object_amore'], df['same_object_crowd']))
print('===============')

# # For quick inspection
# print()
# print(annotations.groupby(['workerid', 'hitid'])['rating', 'correct1', 'correct2'].agg(['count', 'mean']).to_string())

quit()

# Explore mismaches
print()

amore_vs_crowd_mismaches = amore_vs_crowd.loc[abs(amore_vs_crowd['correct_name_amore'] - amore_vs_crowd['correct_name_crowd']) > 1]
print("Mismatches correct_name:", len(amore_vs_crowd_mismaches))
for i, row in amore_vs_crowd_mismaches.iterrows():
    print(i, 'amore:', row['correct_name_amore'], ' - crowd:', row['correct_name_crowd'])

print()

amore_vs_crowd_mismaches = amore_vs_crowd.loc[abs(amore_vs_crowd['same_object_amore'] - amore_vs_crowd['same_object_crowd']) > .5]
print("Mismatches same_object:", len(amore_vs_crowd_mismaches))
for i, row in amore_vs_crowd_mismaches.iterrows():
    print(i, 'amore:', row['same_object_amore'], ' - crowd:', row['same_object_crowd'])


