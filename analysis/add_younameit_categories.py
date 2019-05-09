import os

import pandas

BASE_DIR = os.path.dirname(os.getcwd())

def _add_younameit_cats(df, category_mapping_fname="dataset_creation/categories/category_mapping.tsv"):
    # load category mapping (WordNet --> YouNameIt)
    cat2samplecats_map = pandas.read_csv(os.path.join(BASE_DIR, category_mapping_fname), sep="\t")
    wncat2younameit = {wn_cat:yni_cat for (idx, (wn_cat, yni_cat)) in \
        cat2samplecats_map[["wordnet_category","younameit_category"]]\
            .drop_duplicates(subset="wordnet_category").iterrows()}
    
    df["domain"] = df["cat"].apply(lambda cat: wncat2younameit.get(cat, "UNK"))
    return df

    

