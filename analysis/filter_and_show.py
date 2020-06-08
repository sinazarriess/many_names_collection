from analysis import load_results
import pandas as pd
import numpy as np
import os


MAIN_DATA_DIR = '../proc_data_phase0/verification/all_responses_round0-3_verified_new.csv'
OUT_NAME = "exploration"    # any string; only determines the names of output html files.
N_IMAGES_PER_HTML = 100     # will randomly sample up to this many images for each html
MODAL_NAME_MIN_ADEQUACY = 0.75  # If 0, modal name is simply the most frequent name.

##
# Rules are tuples of the form (name, function, value), where the value is either 0 or 1,
# meaning whether to reject or accept names based on this rule. The value matters only if
# you want to explore the effect of a set of rules altogether. If you merely want to print
# some examples for each rule independently, the value you pick does not matter.
#
# NOTE: Rules are processed in the order in which they are given in this list. Later rules
# do not overrule earlier rules. An image will be shown only in the html file of the
# first rule that applied to it. To avoid this, just make sure your rules apply to
# mutually exclusive domains.
##
RULES = [
        ('same_object_05_adequacy_04-05', lambda x: ((0.5 <= x['same_object']) & (0.4 <= x['adequacy']) & (x['adequacy'] < 0.5)), 1),
        ('same_object_05_adequacy_05-06', lambda x: ((0.5 <= x['same_object']) & (0.5 <= x['adequacy']) & (x['adequacy'] < 0.6)), 1),
        ('same_object_05_adequacy_06-08', lambda x: ((0.5 <= x['same_object']) & (0.6 <= x['adequacy']) & (x['adequacy'] < 0.8)), 1),
        ('same_object_05_adequacy_08-10', lambda x: ((0.5 <= x['same_object']) & (0.8 <= x['adequacy']) & (x['adequacy'] <= 1.0)), 1),
    ]

# Main thresholds, dividing the spectrum into maximally coherent bins, for convenience:
#  adequacy:    0.000, 0.083, 0.250, 0.417, 0.583, 0.750, 0.917, 1.000
#  same_object: 0.000, 0.167, 0.500, 0.833, 1.000


def main():
    """
    Loads data, applies the rules defined in RULES, generates html output.
    """
    print("Loading data.")
    data = load_results.load_cleaned_results(MAIN_DATA_DIR)
    np.random.seed(123456)

    # First turn the verified data into a dataframe with a single object-name pair per row
    columns = ['url', 'modal_name', 'name', 'adequacy', 'inadequacy_type', 'same_object']
    rows = []
    for i, row in data.iterrows():
        # find the modal name
        candidate_modal_names = [name for name, count in row['spellchecked_min2'].most_common()
                          if row['adequacy_mean'][name] >= MODAL_NAME_MIN_ADEQUACY]
        if not candidate_modal_names:
            continue
        modal_name = candidate_modal_names[0]

        # create rows for all verified names
        for name in row['adequacy_mean']:
            rows.append([row['url'], modal_name, name,
                         row['adequacy_mean'][name],
                         row['inadequacy_type'][name],
                         row['same_object'][modal_name][name],
                         ])
    object_name_pairs = pd.DataFrame(rows, columns=columns)

    # Apply the list of rules, resulting in two new columns:
    object_name_pairs['gold'] = np.nan  # 1 if the name is ruled in (e.g., for inclusion in MN2.0) else 0
    object_name_pairs['rule'] = "" # name of the rule responsible for the 'gold' value of this name
    for rule, mask, y in RULES:
        na_before = sum(object_name_pairs['gold'].isna())
        object_name_pairs['rule'].loc[object_name_pairs['gold'].isna() & mask(object_name_pairs)] = rule
        object_name_pairs['gold'].loc[mask(object_name_pairs)] = y
        na_after = sum(object_name_pairs['gold'].isna())
        print(f'Rule {rule} applies to {sum(mask(object_name_pairs))} items ({na_before-na_after} not covered by previous rules).')

    print(f'{sum(object_name_pairs["gold"].isna())} examples remain unclassified by the current set of rules.')

    # Create html files, one for each rule, with a sample of items to which the rule applied
    # NOTE: Only shows examples that weren't already decided by previous rules.
    for rule, _, y in RULES:
        df = object_name_pairs.loc[object_name_pairs['rule'] == rule]
        df = df.sample(min(N_IMAGES_PER_HTML, len(df))).reset_index(drop=True)
        write_html_table(df, f'html/{OUT_NAME}_{rule}.html')
    df = object_name_pairs.loc[object_name_pairs['gold'].isna()]
    df = df.sample(min(N_IMAGES_PER_HTML, len(df))).reset_index(drop=True)
    write_html_table(df, f'html/{OUT_NAME}_unknown.html')


def write_html_table(df, html_filename):
    """
    Writes html files based on a dataframe
    :param df: Contains at least columns url, modal_name, name, adequacy, same_object, inadeuqacy_type
    :param html_filename: path to output file
    :return:
    """

    row_skeleton = \
"""    <tr>
        <td>{i}<td>
        <td><img src="{url}" title="{url}" style="width:350px;height:350px;" /></td>
        <td style="text-align:center"><i><br><br>{modal_name}</i><td>
        <td style="text-align:center"><b>Name:<br><br>{name}</b></td>
        <td style="color:#0000ff; text-align:center"><b>Adequacy:<br><br>{adequacy}</b></td>
        <td style="color:#ff0000; text-align:center"><b>Same object:<br><br>{same_object}</b></td>
        <td style="color:#ff0000; text-align:center"><b>Inadequacies: {inadequacy_type}</b></td>
    </tr>""".format

    table_rows = []
    for i, row in df.iterrows():
        table_rows.append(row_skeleton(i=i, **row))

    table_html = "<table>{}<table>".format('\n'.join(table_rows))

    os.makedirs('html', exist_ok=True)
    with open(html_filename, "w") as f:
        f.write(table_html)
    print(f"Html file written to {f.name}")


if __name__ == "__main__":
    main()

    ## This is old code for computing some number that Gemma asked for:
    # data = load_results.load_cleaned_results('../proc_data_phase0/verification/all_responses_round0-3_verified.csv')
    # data['most_frequent_name'] = data['spellchecked_min2'].apply(lambda x: x.most_common()[0])
    # data['verified_soft_clusters'] = data['verified_soft_clusters'].apply(eval)
    # data['verified_soft_clusters'] = data['verified_soft_clusters'].apply(lambda x: {(tuple([k]) if isinstance(k, str) else k): v for (k, v) in x.items()})
    # data['main_soft_cluster'] = data['verified_soft_clusters'].apply(lambda x: [v for v in x if x[v]['index'] == 0][0])
    # data['most_frequent_from_main_soft_cluster'] = data.apply(
    #     lambda x: sorted([(n, x['spellchecked_min2'][n]) for n in x['main_soft_cluster']], key=lambda y: -y[1])[0],
    #     axis=1)
    # data['most_frequent_name_adequacy'] = data.apply(lambda x: x['verified'][x['most_frequent_name'][0]]['adequacy'], axis=1)
    # data['most_frequent_from_main_soft_cluster_adequacy'] = data.apply(
    #     lambda x: x['verified'][x['most_frequent_from_main_soft_cluster'][0]]['adequacy'],
    #     axis=1)
    #
    # print(data[['url', 'most_frequent_name', 'most_frequent_from_main_soft_cluster', 'main_soft_cluster']][:50].to_string())
    #
    # unequal_data = data.loc[data['most_frequent_name'] != data['most_frequent_from_main_soft_cluster']]
    # unequal_counts = unequal_data.loc[unequal_data['most_frequent_name'].apply(lambda x: x[1]) != unequal_data['most_frequent_from_main_soft_cluster'].apply(lambda x: x[1])]
    # print(len(data), len(data) - len(unequal_data), len(unequal_data), len(unequal_counts))
    # print(unequal_counts.sample(min(len(unequal_counts), 30))[['url', 'most_frequent_name', 'most_frequent_name_adequacy', 'most_frequent_from_main_soft_cluster', 'most_frequent_from_main_soft_cluster_adequacy', 'main_soft_cluster']].to_string())
    # print(unequal_counts['most_frequent_name_adequacy'].mean(), unequal_counts['most_frequent_from_main_soft_cluster_adequacy'].mean())
    # print(sum(unequal_counts['most_frequent_name_adequacy'] < unequal_counts['most_frequent_from_main_soft_cluster_adequacy']), sum(unequal_counts['most_frequent_name_adequacy'] == unequal_counts['most_frequent_from_main_soft_cluster_adequacy']))
    #
    # count = 0
    # total = 0
    # for i, row in data.iterrows():
    #     if 'food' in row['verified'] and 'sauce' in row['verified'] and row['verified']['sauce']['adequacy'] == 1.0:
    #         if 'sauce' in row['verified']['food']['can_be_same_object']:
    #             count += 1
    #         total += 1
    #
    # print(count, total, 100*count/total)
