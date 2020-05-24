from analysis import load_results
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt

img_web_dir = "http://object-naming-amore.upf.edu/"

def write_html_table(df, html_filename):

    row_skeleton = \
"""    <tr>
        <td>{i}<td>
        <td><img src="{url}" title="{url}" style="width:350px;height:350px;" /></td>
        <td style="text-align:center"><i><br><br>{main_name}</i><td>
        <td style="text-align:center"><b>Name:<br><br>{name}</b></td>
        <td style="color:#0000ff; text-align:center"><b>Adequacy:<br><br>{adequacy}</b></td>
        <td style="color:#ff0000; text-align:center"><b>Same object:<br><br>{same_object}</b></td>
        <td style="color:#ff0000; text-align:center"><b>Inadequacies: {inadequacy_type}</b></td>
    </tr>""".format

    table_rows = []
    for i, row in df.iterrows():
        table_rows.append(row_skeleton(i=i, **row))

    table_html = "<table>{}<table>".format('\n'.join(table_rows))

    with open(html_filename, "w") as f:
        f.write(table_html)
    print(f"Html file written to {f.name}")


if __name__ == "__main__":
    data = load_results.load_cleaned_results('../proc_data_phase0/verification/all_responses_round0-3_verified_new.csv')
    os.makedirs('html', exist_ok=True)

    # Turn this into an object-name pair dataframe
    columns = ['url', 'main_name', 'name', 'adequacy', 'inadequacy_type', 'same_object']
    rows = []
    for i, row in data.iterrows():
        adequate_names = [name for name, count in row['spellchecked_min2'].most_common() if row['adequacy_mean'][name] >= 0.75]
        if not adequate_names:
            continue
        main_name = adequate_names[0]
        img_url = row['url']
        for name in row['adequacy_mean']:
            rows.append([img_url,
                         main_name,
                         name,
                         row['adequacy_mean'][name],
                         row['inadequacy_type'][name],
                         row['same_object'][main_name][name],
                         ])

    object_name_pairs = pd.DataFrame(rows, columns=columns)

    # 6 main bins, cuts in between:
    adequacy_thresholds = np.arange(-.5*1/6, 1+.6*1/6, 1/6)
    # 3 main bins, cuts in between:
    same_object_thresholds = np.arange(-.5*1/3, 1+.6*1/3, 1/3)

    print(adequacy_thresholds)
    print(same_object_thresholds)

    rules = [
        ('adequate1', lambda x: (x['adequacy'] >= 0.75) & (x['same_object'] >= 0.5), 1), # minor errors that don't cause non-same-object
        ('adequate2', lambda x: x['adequacy'] == 1, 1), # to rule in mattress/bed cases (adequate but not same object)
        ('inadequate1', lambda x: (x['adequacy'] < 1) & (x['same_object'] < 0.5), 0), # minor errors that cause non-same-object
        ('inadequate2', lambda x: x['adequacy'] < 0.42, 0), # to rule out inadequate names for the same object
    ]
    object_name_pairs['gold'] = np.nan
    object_name_pairs['rule'] = ""
    for rule, mask, y in rules:
        na_before = sum(object_name_pairs['gold'].isna())
        object_name_pairs['rule'].loc[object_name_pairs['gold'].isna() & mask(object_name_pairs)] = rule
        object_name_pairs['gold'].loc[mask(object_name_pairs)] = y
        na_after = sum(object_name_pairs['gold'].isna())
        print(f'Rule {rule} decided {sum(mask(object_name_pairs))} ({na_before-na_after} new).')

    print(f'{sum(object_name_pairs["gold"].isna())} unknowns remaining.')

    np.random.seed(123456)

    for rule, _, y in rules:
        top = object_name_pairs.loc[object_name_pairs['rule'] == rule].sort_values(by="adequacy", ascending=y)[:100].reset_index(drop=True)
        write_html_table(top, f'html/automatic_{rule}.html')
    sample = object_name_pairs.loc[object_name_pairs['gold'].isna()].sample(100).reset_index(drop=True)
    write_html_table(sample, 'html/automatic_unknown.html')

    quit()


    object_name_pairs_sample = object_name_pairs.sample(1000)
    write_html_table(object_name_pairs_sample.loc[object_name_pairs_sample['gold'] == 1].loc[(object_name_pairs_sample['adequacy'] < 1) | (object_name_pairs_sample['same_object'] < 1)].reset_index(drop=True),
                     'html/auto-positive.html')
    write_html_table(object_name_pairs_sample.loc[object_name_pairs_sample['gold'] == 0].loc[(object_name_pairs_sample['adequacy'] >= .42)].reset_index(drop=True),
                     'html/auto-negative.html')
    write_html_table(object_name_pairs_sample.loc[object_name_pairs_sample['gold'].isna()].reset_index(drop=True),
                     'html/auto-unknown.html')

    quit()

    np.random.seed(1234)
    for adequacy_threshold1, adequacy_threshold2 in zip(adequacy_thresholds, adequacy_thresholds[1:]):
        df = object_name_pairs.loc[object_name_pairs['adequacy'] >= adequacy_threshold1].loc[object_name_pairs['adequacy'] < adequacy_threshold2]
        print(f"{adequacy_threshold2:.2f}: {len(df)}")
        df_sample = df.sample(min(len(df), 100)).reset_index()
        write_html_table(df_sample, f'html/adeq-{adequacy_threshold2:.2f}-{len(df)}.html')

        # for same_object_threshold in same_object_thresholds:
        #     df = object_name_pairs.loc[object_name_pairs['adequacy'] < adequacy_threshold].loc[object_name_pairs['same_object'] < same_object_threshold]
        #     print(f"{adequacy_threshold:.2f}-{same_object_threshold:.2f}: {len(df)}")
        #     df_sample = df.sample(min(len(df), 100))
        #     write_html_table(df_sample, f'html/{len(df)}_adeq-{adequacy_threshold:.2f}_sameobj-{same_object_threshold:.2f}.html')

