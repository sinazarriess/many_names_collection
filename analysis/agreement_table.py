import pandas as pd
import glob
import os
import json
import sys
from collections import Counter
import numpy as np

def snodgrass_agreement(rdict,vocab,singletons=False):

    # to decide: do we include singleton answers for calculating agreement?
    if singletons:
        vec = np.array([rdict[key] for key in rdict])
    else:
        vec = np.array([rdict[key] for key in rdict if vocab[key] > 1])
    vec_rel = vec/(np.sum(vec))


    agr = np.sum(vec_rel * np.log2(1/vec_rel)) ## check whether this is correct

    return agr

def percent_agreement(rdict):

    # to decide: do we include singleton answers for calculating agreement?
    topname = rdict.most_common(1)[0][0]
    total = sum(rdict.values())
    return rdict[topname]/total

def make_df(filename):
    resdf = pd.read_csv(filename,sep="\t")
    resdf['spellchecked'] = resdf['spellchecked'].apply(lambda x: Counter(eval(x)))
    resdf['clean'] = resdf['clean'].apply(lambda x: Counter(eval(x)))
    resdf['canon'] = resdf['canon'].apply(lambda x: Counter(eval(x)))

    vocab_counter = Counter()
    vg_is_common = []

    for ix,row in resdf.iterrows():

        vocab_counter += row['spellchecked']
        max_name = row['spellchecked'].most_common(1)[0][0]
        vg_is_common.append(int(max_name == row['vg_obj_name']))

    resdf['vg_is_max'] = vg_is_common

    return resdf

def make_agreement_table(resdf):
    resdf['snodgrass'] = resdf['spellchecked'].apply(lambda x: snodgrass_agreement(x,{},True))
    resdf['percent_agree'] = resdf['spellchecked'].apply(lambda x: percent_agreement(x))
    nobjects = len(resdf)

    tablerows = []
    tablerows.append(('all',\

                     str("%.2f"%np.mean(resdf['percent_agree'])),\
                     str("%.2f"%np.mean(resdf['snodgrass'])),\
                     str("%.2f"%(np.sum(resdf['vg_is_max'])/nobjects))))

    for c in set(list(resdf['vg_domain'])):
        catdf = resdf[resdf['vg_domain'] == c]
        ncat = len(catdf)

        synagree = Counter()
        for s in set(list(catdf['synset'])):
            syndf = catdf[catdf['synset'] == s]
            synagree[s] = np.mean(syndf['percent_agree'])
        print(c)
        print(synagree)
        topsyn = synagree.most_common(1)[0][0]
        botsyn = synagree.most_common()[:-1-1:-1][0][0]

        topdf = catdf[catdf['synset'] == topsyn]
        botdf = catdf[catdf['synset'] == botsyn]

        tablerows.append((c,\

                         str("%.2f"%np.mean(catdf['percent_agree'])),\
                         str("%.2f"%np.mean(catdf['snodgrass'])),\
                         str("%.2f"%(np.sum(catdf['vg_is_max'])/ncat)),\
                         topsyn,
                         str("%.2f"%np.mean(topdf['percent_agree'])),\
                         str("%.2f"%np.mean(topdf['snodgrass'])),\
                         str("%.2f"%(np.sum(topdf['vg_is_max'])/len(topdf))),\
                         botsyn,
                         str("%.2f"%np.mean(botdf['percent_agree'])),\
                         str("%.2f"%np.mean(botdf['snodgrass'])),\
                         str("%.2f"%(np.sum(botdf['vg_is_max'])/len(botdf)))\
                         ))

    outdf = pd.DataFrame(tablerows,columns=['domain','% top','SD','=VG','max synset','%','SD','=VG','min synset','%','SD','=VG' ])
    print(outdf.sort_values(by=['% top']).to_latex(index=False))


fn = 'all_responses_round0-3_cleaned.csv'
resdf = make_df(fn)
make_agreement_table(resdf)
