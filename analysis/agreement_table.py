import pandas as pd
import glob
import os
import json
import sys
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import wordnet as wn

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
    ntypesmin1 = []

    for ix,row in resdf.iterrows():

        vocab_counter += row['spellchecked']
        max_name = row['spellchecked'].most_common(1)[0][0]
        vg_is_common.append(int(max_name == row['vg_obj_name']))
        vg_weight = row['spellchecked'][row['vg_obj_name']]/sum(row['spellchecked'].values())
        vg_prop.append(vg_weight)
        vg_overlap.append(row['spellchecked'][row['vg_obj_name']])
        nanno.append(sum(row['spellchecked'].values()))
        ntypes.append(len(row['spellchecked'].keys()))
        min1types = [k for k in row['spellchecked'].keys() if row['spellchecked'][k] > 1]
        ntypesmin1.append(len(min1types))


    resdf['vg_is_max'] = vg_is_common
    resdf['vg_mean'] = vg_prop
    resdf['vg_overlap'] = vg_overlap
    resdf['n_anno'] = nanno
    resdf['n_types'] = ntypes
    resdf['n_types_min1'] = ntypesmin1
    resdf['snodgrass'] = resdf['spellchecked'].apply(lambda x: snodgrass_agreement(x,{},True))
    resdf['percent_agree'] = resdf['spellchecked'].apply(lambda x: percent_agreement(x))

    return resdf

def make_agreement_table(resdf):

    nobjects = len(resdf)

    tablerows = []
    tablerows2 = []


    tablerows.append(('all',\

                     str("%.1f"%(np.mean(resdf['percent_agree'])*100)),\
                     str("%.1f"%np.mean(resdf['snodgrass'])),\
                     str("%.1f"%np.mean(resdf['n_types'])),\
                     str("%.1f"%np.mean(resdf['n_types_min1'])),\
                     str("%.1f"%((np.sum(resdf['vg_is_max'])/nobjects)*100)),\
                     str("%.1f"%((np.sum(resdf['vg_mean'])/nobjects)*100)),\
                     str(len(resdf))
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

                         str("%.1f"%(np.mean(catdf['percent_agree'])*100)),\
                         str("%.1f"%np.mean(catdf['snodgrass'])),\
                         str("%.1f"%np.mean(catdf['n_types'])),\
                         str("%.1f"%np.mean(catdf['n_types_min1'])),\
                         str("%.1f"%((np.sum(catdf['vg_is_max'])/ncat)*100)),\
                         str("%.1f"%((np.sum(catdf['vg_mean'])/ncat)*100)),\
                         str(len(catdf))
                         #str("%.2f"%((np.sum(catdf['vg_overlap'])/np.sum(catdf['n_anno']))*100))

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

    outdf = pd.DataFrame(tablerows,columns=['domain','% top','H','N','Nmin1','top=VG','% VG','#'])
    print(outdf.sort_values(by=['% top']).to_latex(index=False))
    #outdf2 = pd.DataFrame(tablerows2,columns=['domain','max synset','% top1','SD','Max=VG','% VG','min synset','% top','SD','Max=VG','% VG' ])
    #print(outdf2.sort_values(by=['% top1']).to_latex(index=False))
    return outdf

def make_synset_df(resdf):

    objdf = pd.read_json('~/Downloads/objects.json.zip',compression='zip')
    objdf = pd.DataFrame([obj for listobj in list(objdf['objects']) for obj in listobj])
    objdf = objdf[objdf['synsets'].apply(lambda x: len(x) >0)]
    objdf['synset'] = objdf['synsets'].apply(lambda x: x[0])
    objdf['name'] = objdf['names'].apply(lambda x: x[0])
    name2synset = dict(list(zip(objdf['name'],objdf['synset'])))
    #name2lemmas = {n:[l._name for l in wn.synset(n).lemmas()] for n in name2synset}

    syn2resp = {}
    syn2synset = {}
    syn2domain = {}
    syncount = Counter()
    syn2names = {}

    vg_is_common = []
    vg_prop = []
    vg_overlap = []
    nanno = []
    ntypes = []
    ntypesmin1 = []

    for ix,row in resdf.iterrows():
        sn = name2synset[row['vg_obj_name']]
        if sn not in syn2resp:
            syn2resp[sn] = Counter()
            syn2domain[sn] = row['vg_domain']
            syn2synset[sn] = row['synset']
        if sn not in syn2names:
            syn2names[sn] = [l._name for l in wn.synset(sn).lemmas()]
        if row['vg_obj_name'] not in syn2names[sn]:
            syn2names[sn].append(row['vg_obj_name'])
        syn2resp[sn].update(row['spellchecked'])
        #syn2vg[sn][row['vg_obj_name']] += 1
        syncount[sn] += 1

    new_rows = []
    for sn in syn2resp:
        new_rows.append((sn,syn2names[sn],syn2domain[sn],syn2synset[sn],syn2resp[sn],syncount[sn]))

    syndf = pd.DataFrame(new_rows,columns=['vg_obj_synset','vg_obj_names','vg_domain','synset','responses','n_images'])

    for ix,row in syndf.iterrows():

        max_name = row['responses'].most_common(1)[0][0]
        #is_top = max_name in name2lemmas[row['vg_obj_name']]
        #vg_is_common.append(int(max_name == row['vg_obj_name']))
        vg_is_common.append(int(max_name in row['vg_obj_names']))
        vg_weight = sum([row['responses'][n] for n in row['vg_obj_names']])/sum(row['responses'].values())
        vg_prop.append(vg_weight)
        vg_overlap.append(sum([row['responses'][n] for n in row['vg_obj_names']]))
        nanno.append(sum(row['responses'].values()))
        ntypes.append(len(row['responses'].keys()))
        min1types = [k for k in row['responses'].keys() if row['responses'][k] > 1]
        ntypesmin1.append(len(min1types))


    syndf['vg_is_max'] = vg_is_common
    syndf['vg_mean'] = vg_prop
    syndf['vg_overlap'] = vg_overlap
    syndf['n_anno'] = nanno
    syndf['n_types'] = ntypes
    syndf['n_types_min1'] = ntypesmin1
    syndf['snodgrass'] = syndf['responses'].apply(lambda x: snodgrass_agreement(x,{},True))
    syndf['percent_agree'] = syndf['responses'].apply(lambda x: percent_agreement(x))
    #syndf['vg_name_synset'] = syndf['vg_obj_name'].apply(lambda x: name2synset[x])

    return syndf,name2synset


def agreement_boxplot(resdf):
    plotdata = []
    cats = ['all']
    plotdata.append(list(resdf['percent_agree']))

    for c in set(list(resdf['vg_domain'])):
        cats.append(c)
        catdf = resdf[resdf['vg_domain'] == c]
        plotdata.append(list(catdf['percent_agree']))

    fig, ax = plt.subplots()
    ax.boxplot(plotdata,labels=cats,widths=0.8,notch=True)
    #plt.xticks(range(1,len(cats)), cats)
    plt.savefig("agreebox.png")

    fig, ax = plt.subplots()
    plotdata = []
    cats = []
    rangec = []

    for i,c in enumerate(list(set(list(resdf['synset'])))[:20]):
        cats.append(c)
        rangec.append((i+1)*4)
        catdf = resdf[resdf['synset'] == c]
        plotdata.append(list(catdf['percent_agree']))

    print(rangec)
    fig, ax = plt.subplots()
    ax.boxplot(plotdata)
    plt.xticks(range(1,len(cats)+1), cats,fontsize = 6, rotation = 90)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig("agreebox_synset.png")


if __name__ == '__main__':
    fn = 'all_responses_round0-3_cleaned.csv'
    resdf = make_df(fn)
    syndf,name2synsets = make_synset_df(resdf)
    #print(syndf[syndf['vg_is_max'] == 1].head())
    #print(syndf[syndf['vg_is_max'] == 0].head())
    o1 = make_agreement_table(resdf)
    o2 = make_agreement_table(syndf)
    o3 = pd.concat([o1,o2[list(o2.columns[1:])]],axis=1)
    print(o3.to_latex(index=False))


    #agreement_boxplot(resdf)
