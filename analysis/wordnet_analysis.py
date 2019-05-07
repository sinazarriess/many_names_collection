#!/usr/bin/env python
# coding: utf-8


from nltk.corpus import wordnet as wn
import pandas as pd
from agreement_table import make_df
import os
import json
import sys
from collections import Counter
import numpy as np


pd.set_option('display.max_colwidth', -1)



hyp = lambda s:s.hypernyms()

def name_pairs(rdict):

    np_count = Counter()
    if len(rdict.keys()) > 1:
        topname = rdict.most_common(1)[0][0]
        for oname in rdict:
            if not oname == topname:
                np_count[(topname,oname)] = rdict[oname]

    return np_count

def flat_name_pairs(rdict):

    np_count = Counter()
    if len(rdict.keys()) > 1:
        names = rdict.most_common()
        for i,(n,fr) in enumerate(names):
            if i < len(names):
                alt_names = names[i+1:]
                top_name = names[i][0]

                for oname,fr in alt_names:
                    if not oname == top_name:
                        np_count[(top_name,oname)] = fr

    return np_count

def vg_name_pairs(vg_name,rdict):

    np_count = Counter()

    for oname in rdict:
        if not oname == vg_name:
            np_count[(vg_name,oname)] = rdict[oname]

    return np_count


def is_hyponym(w1,w2,max_depth=5):

    for dpt in range(1,max_depth):

        syns1 = wn.synsets(w1,pos="n")
        syns2 = wn.synsets(w2,pos="n")

        for syn1 in syns1:
            hyp_closure = list(syn1.closure(hyp,depth=dpt))

            for syn2 in syns2:
                if syn2 in hyp_closure:
                    return dpt
    return 0

def is_hypernym(w1,w2,max_depth=5):

    for dpt in range(1,max_depth):

        syns1 = wn.synsets(w1,pos="n")
        syns2 = wn.synsets(w2,pos="n")

        for syn2 in syns2:
            hyp_closure = list(syn2.closure(hyp,depth=dpt))

            for syn1 in syns1:
                if syn1 in hyp_closure:
                    return dpt
    return 0

def is_synonym(w1,w2):

    syns1 = wn.synsets(w1,pos="n")
    syns2 = wn.synsets(w2,pos="n")

    for syn2 in syns2:

        for syn1 in syns1:
                if syn1 == syn2:
                    return 1
    return 0

def is_cohyponym(w1,w2,max_depth=1):

    syns1 = wn.synsets(w1,pos="n")
    syns2 = wn.synsets(w2,pos="n")

    for syn2 in syns2:

        hyp_closure2 = list(syn2.closure(hyp,depth=max_depth))


        for syn1 in syns1:
            hyp_closure1 = list(syn1.closure(hyp,depth=max_depth))

            for h2 in hyp_closure2:
                for h1 in hyp_closure1:
                    if h1 == h2:
                        return 1
    return 0



fn = 'all_responses_round0-3_cleaned.csv'
resdf = make_df(fn)

ordered_paircount = Counter()
paircount = Counter()
wordcount = Counter()

for x,row in resdf.iterrows():
    #print(ndict)
    x = flat_name_pairs(row['spellchecked'])
    #x = vg_name_pairs(row['vg_obj_name'],row['spellchecked'])
    ordered_paircount.update(x)
    for pair in x:
        paircount[tuple(sorted(pair))] += x[pair]
    wordcount.update(row['spellchecked'])

print("N name types:",len(wordcount))
singletons = [w for w in wordcount if wordcount[w] == 1]
print("... singletons:",len(singletons))
wn_wordcount = Counter()
wn_map = {}

for w in wordcount:
    if len(wn.synsets(w)) > 0:
        wn_wordcount[w] = wordcount[w]
        wn_map[w] = w
    elif len(wn.synsets('_'.join(w.split(' ')))) > 0:
        wn_wordcount[w] = wordcount[w]
        wn_map[w] = '_'.join(w.split(' '))
    elif len(wn.synsets(''.join(w.split(' ')))) > 0:
        wn_wordcount[w] = wordcount[w]
        wn_map[w] = ''.join(w.split(' '))
    elif len(wn.synsets('-'.join(w.split(' ')))) > 0:
        wn_wordcount[w] = wordcount[w]
        wn_map[w] = '-'.join(w.split(' '))

wn_singletons = [w for w in wn_wordcount if wn_wordcount[w] == 1]
print("N name types covered by WN:",len(wn_wordcount))
print("... singletons:",len(wn_singletons))


wn_paircount = Counter({(w1,w2):paircount[(w1,w2)] for (w1,w2) in paircount if w1 in wn_wordcount \
                                                                               and w2 in wn_wordcount })
wn_ordered_paircount = Counter({(w1,w2):ordered_paircount[(w1,w2)] for (w1,w2) in ordered_paircount if w1 in wn_wordcount and w2 in wn_wordcount })

print("N ordered name variants:",len(ordered_paircount))
print("N name variants :",len(paircount))
print("N ordered name variants covered by WN:",len(wn_ordered_paircount))
singletons = [w for w in wn_ordered_paircount if wn_ordered_paircount[w] == 1]
print("... singletons:",len(singletons))
print("N name variants covered by WN:",len(wn_paircount))
singletons = [w for w in wn_paircount if wn_paircount[w] == 1]
print("... singletons:",len(singletons))



pairrel = {}

for (top,other) in paircount:


    if (top,other) in wn_paircount:
        if is_synonym(wn_map[top],wn_map[other]):
            pairrel[(top,other)] = 'synonymy'
        elif is_hypernym(wn_map[top],wn_map[other],max_depth=10):
            pairrel[(top,other)] = 'hypernymy.1'
        elif is_hypernym(wn_map[other],wn_map[top],max_depth=10):
            pairrel[(top,other)] = 'hypernymy.2'
        elif is_cohyponym(wn_map[top],wn_map[other],max_depth=1):
            pairrel[(top,other)] = 'co-hyponymy'
        else:
            pairrel[(top,other)] = 'crossclassified'
    else:
        pairrel[(top,other)] = 'not-covered'


print("pairrel",len(pairrel))
print("paircount",len(paircount))


typecounts = Counter()
tokencounts = Counter()
for p in pairrel:
    if paircount[p] > 5:
        rel = pairrel[p]
        if '.' in rel:
            rel = rel[:-2]
        typecounts[rel] += 1
        tokencounts[rel] += paircount[p]


outdf = []
totaltypes = sum(typecounts.values())
totaltokens = sum(tokencounts.values())
for p in typecounts:
    outdf.append((p,"%.3f"%(typecounts[p]/totaltypes),"%.3f"%(tokencounts[p]/totaltokens)))


outdff = pd.DataFrame(outdf,columns=['relation','types','tokens'])

print(outdff.sort_values(by=['types']).to_latex(index=False))


pairdf = []

for (top,other) in paircount:
#    if paircount[(top,other)] > 5:
        prow = (top,other,pairrel[(top,other)],\
           paircount[(top,other)],\
           ordered_paircount[(top,other)],\
           ordered_paircount[(other,top)])
        pairdf.append(prow)

pairdf = pd.DataFrame(pairdf,columns=['word1','word2','relation',\
'totalfreq','freq-w1-w2','freq-w2-w1'])
pairdf = pairdf.sort_values(by=['totalfreq'],ascending=False)
pairdf.to_csv("names_pairs_relations_v2.csv")





cat2paircount = {}
for cat in set(list(resdf['vg_domain'])):
    cat2paircount[cat] = Counter()
    catdf = resdf[resdf['vg_domain'] == cat]

    for ndict in catdf['spellchecked']:
    #print(ndict)
        x = name_pairs(ndict)
        cat2paircount[cat].update(x)


# In[254]:


examples = []
for cat in cat2paircount:
    exstr = []
    #print(cat)
    for ((n1,n2),freq) in cat2paircount[cat].most_common(10):
        exstr.append("%s -- %s (%d)"%(n1,n2,freq))
    examples.append((cat,", ".join(exstr)))


exdf = pd.DataFrame(examples,columns=['category',"most frequent naming variants"])
#print(exdf.to_latex(index=False))


# In[ ]:
