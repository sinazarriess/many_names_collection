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
    vg_prop = []
    vg_overlap = []
    nanno = []
    ntypes = []

    for ix,row in resdf.iterrows():

        vocab_counter += row['spellchecked']
        max_name = row['spellchecked'].most_common(1)[0][0]
        vg_is_common.append(int(max_name == row['vg_obj_name']))
        vg_weight = row['spellchecked'][row['vg_obj_name']]/sum(row['spellchecked'].values())
        vg_prop.append(vg_weight)
        vg_overlap.append(row['spellchecked'][row['vg_obj_name']])
        nanno.append(sum(row['spellchecked'].values()))
        ntypes.append(len(row['spellchecked'].keys()))

    resdf['vg_is_max'] = vg_is_common
    resdf['vg_mean'] = vg_prop
    resdf['vg_overlap'] = vg_overlap
    resdf['n_anno'] = nanno
    resdf['n_types'] = ntypes

    return resdf

def make_agreement_table(resdf):
    resdf['snodgrass'] = resdf['spellchecked'].apply(lambda x: snodgrass_agreement(x,{},True))
    resdf['percent_agree'] = resdf['spellchecked'].apply(lambda x: percent_agreement(x))
    nobjects = len(resdf)

    tablerows = []
    tablerows2 = []

    tablerows.append(('all',\

                     str("%.2f"%(np.mean(resdf['percent_agree'])*100)),\
                     str("%.2f"%np.mean(resdf['snodgrass'])),\
                     str("%.2f"%np.mean(resdf['n_types'])),\
                     str("%.2f"%((np.sum(resdf['vg_is_max'])/nobjects)*100)),\
                     str("%.2f"%((np.sum(resdf['vg_mean'])/nobjects)*100)),\
                     str("%.2f"%((np.sum(resdf['vg_overlap'])/np.sum(resdf['n_anno']))*100))

                     ))

    for c in set(list(resdf['vg_domain'])):
        catdf = resdf[resdf['vg_domain'] == c]
        ncat = len(catdf)

        synagree = Counter()
        for s in set(list(catdf['synset'])):
            syndf = catdf[catdf['synset'] == s]
            synagree[s] = np.mean(syndf['vg_mean'])
        print(c)
        print(synagree)
        topsyn = synagree.most_common(1)[0][0]
        botsyn = synagree.most_common()[:-1-1:-1][0][0]

        topdf = catdf[catdf['synset'] == topsyn]
        botdf = catdf[catdf['synset'] == botsyn]

        tablerows.append((c,\

                         str("%.2f"%(np.mean(catdf['percent_agree'])*100)),\
                         str("%.2f"%np.mean(catdf['snodgrass'])),\
                         str("%.2f"%np.mean(catdf['n_types'])),\
                         str("%.2f"%((np.sum(catdf['vg_is_max'])/ncat)*100)),\
                         str("%.2f"%((np.sum(catdf['vg_mean'])/ncat)*100)),\
                         str("%.2f"%((np.sum(catdf['vg_overlap'])/np.sum(catdf['n_anno']))*100))

                    ))
        tablerows2.append((c,\

                         topsyn,
                         str("%.2f"%((np.mean(topdf['percent_agree'])*100))),\
                         str("%.2f"%(np.mean(topdf['snodgrass']))),\
                         str("%.2f"%((np.sum(topdf['vg_is_max'])/len(topdf))*100)),\
                         str("%.2f"%((np.sum(topdf['vg_mean'])/len(topdf))*100)),\

                         botsyn,
                         str("%.2f"%((np.mean(botdf['percent_agree'])*100))),\
                         str("%.2f"%(np.mean(botdf['snodgrass']))),\
                         str("%.2f"%((np.sum(botdf['vg_is_max'])/len(botdf))*100)),\
                         str("%.2f"%((np.sum(botdf['vg_mean'])/len(botdf))*100))

                         ))

    outdf = pd.DataFrame(tablerows,columns=['domain','% top','SD','N','Max=VG','% VG (Micro)','% VG (Macro)'])
    print(outdf.sort_values(by=['% top']).to_latex(index=False))
    outdf2 = pd.DataFrame(tablerows2,columns=['domain','max synset','% top1','SD','Max=VG','% VG','min synset','% top','SD','Max=VG','% VG' ])
    print(outdf2.sort_values(by=['% top1']).to_latex(index=False))

if __name__ == '__main__':
    fn = 'all_responses_round0-3_cleaned.csv'
    resdf = make_df(fn)
    make_agreement_table(resdf)
