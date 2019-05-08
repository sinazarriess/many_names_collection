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
            # Threshold
            if i < len(names) and fr > 1:
                alt_names = names[i+1:]
                top_name = names[i][0]

                for oname,fr in alt_names:
                    if not oname == top_name:
                        np_count[(top_name,oname)] = fr
            else:
                break

    return np_count

def vg_name_pairs(vg_name,rdict):

    np_count = Counter()

    for oname in rdict:
        if not oname == vg_name:
            np_count[(vg_name,oname)] = rdict[oname]

    return np_count


def is_hyponym(w1,w2,max_depth=5):

    syns1 = wn.synsets(w1,pos="n")
    syns2 = wn.synsets(w2,pos="n")

    for syn1 in syns1:
        hyp_closure = list(syn1.closure(hyp,depth=dpt))

        for syn2 in syns2:
            if syn2 in hyp_closure:
                return (True,syn1,syn2)
    return (False,0,0)

def is_hypernym(w1,w2,max_depth=5):


    syns1 = wn.synsets(w1,pos="n")
    syns2 = wn.synsets(w2,pos="n")

    for syn2 in syns2:
        hyp_closure = list(syn2.closure(hyp,depth=max_depth))

        for syn1 in syns1:
            if syn1 in hyp_closure:
                return (True,syn1,syn2)
    return (False,0,0)

def is_synonym(w1,w2):

    syns1 = wn.synsets(w1,pos="n")
    syns2 = wn.synsets(w2,pos="n")

    for syn2 in syns2:

        for syn1 in syns1:
                if syn1 == syn2:
                    return (True,syn1,syn2)
    return (False,0,0)

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
                        return (True,h1,h2)
    return (False,0,0)

def get_word_rel(w1,w2):

    (has_rel,s1,s2) = is_synonym(wn_map[top],wn_map[other])
    if has_rel:
        return ('synonymy',s1,s2)

    (has_rel,s1,s2) = is_hypernym(wn_map[top],wn_map[other],max_depth=10)
    if has_rel:
        return ('hypernymy.1',s1,s2)

    (has_rel,s1,s2) = is_hypernym(wn_map[other],wn_map[top],max_depth=10)
    if has_rel:
        return ('hypernymy.2',s1,s2)

    (has_rel,s1,s2) = is_cohyponym(wn_map[top],wn_map[other],max_depth=1)
    if has_rel:
        return ('co-hyponymy',s1,s2)

    return ('crossclassified',0,0)



fn = 'all_responses_round0-3_cleaned.csv'
resdf = make_df(fn)

ordered_paircount = Counter()
paircount = Counter()
wordcount = Counter()
word2domain = {}
domain_ordered_paircount = Counter()
domain_paircount = Counter()

for x,row in resdf.iterrows():
    #print(ndict)
    x = flat_name_pairs(row['spellchecked'])
    #x = vg_name_pairs(row['vg_obj_name'],row['spellchecked'])
    ordered_paircount.update(x)
    for (x1,x2) in x:
        domain_ordered_paircount[(row['vg_domain'],x1,x2)] += x[(x1,x2)]


    wordcount.update(row['spellchecked'])
    for word in row['spellchecked']:
        if word not in word2domain:
            word2domain[word] = Counter()
        word2domain[word][row['vg_domain']] += row['spellchecked'][word]

for p in ordered_paircount:
    swapp = (p[1],p[0])
    if not swapp in paircount:
        if ordered_paircount[p] > ordered_paircount[swapp]:
            paircount[p] = ordered_paircount[p] + ordered_paircount[swapp]
        else:
            paircount[swapp] = ordered_paircount[p] + ordered_paircount[swapp]

for p in domain_ordered_paircount:
    swapp = (p[0],p[2],p[1])
    if not swapp in domain_paircount:
        if domain_ordered_paircount[p] > domain_ordered_paircount[swapp]:
            domain_paircount[p] = domain_ordered_paircount[p] + domain_ordered_paircount[swapp]
        else:
            domain_paircount[swapp] = domain_ordered_paircount[p] + domain_ordered_paircount[swapp]



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
        pairrel[(top,other)] = get_word_rel(top,other)

    else:
        pairrel[(top,other)] = ('not-covered',0,0)


print("pairrel",len(pairrel))
print("paircount",len(paircount))


typecounts = Counter()
tokencounts = Counter()
for p in pairrel:
    if paircount[p] > 1:
        rel = pairrel[p][0]
        if '.' in rel:
            rel = rel[:-2]
        typecounts[rel] += 1
        tokencounts[rel] += paircount[p]

domtypecounts = {}
domtokencounts = {}
for (domain,top,other) in domain_paircount:
    if (top,other) in pairrel:
        rel = pairrel[(top,other)][0]
    else:
        rel = pairrel[(other,top)][0]
    if '.' in rel:
        rel = rel[:-2]
    if domain not in domtypecounts:
        domtypecounts[domain] = Counter()
        domtokencounts[domain] = Counter()
    domtypecounts[domain][rel] += 1
    domtokencounts[domain][rel] += domain_paircount[(domain,top,other)]




outdf = []
totaltypes = sum(typecounts.values())
totaltokens = sum(tokencounts.values())
relations = typecounts.keys()

for domain in domtypecounts:
    row = [domain]
    for p in relations:
        row.append("%.3f"%(domtypecounts[domain][p]/sum(domtypecounts[domain].values())))
        row.append("%.3f"%(domtokencounts[domain][p]/sum(domtokencounts[domain].values())))
    outdf.append(row)

row = ['all']
for p in relations:
    row.append("%.3f"%(typecounts[p]/totaltypes))
    row.append("%.3f"%(tokencounts[p]/totaltokens))
outdf.append(row)

print(outdf)

colnames = ['domain']
for p in relations:
    colnames.append(p+"_typ")
    colnames.append(p+"_tok")
outdff = pd.DataFrame(outdf,columns=colnames)

print(outdff.to_latex(index=False))


pairdf = []

for (top,other) in paircount:
#    if paircount[(top,other)] > 5:
        prow = (top,other,pairrel[(top,other)][0],\
           paircount[(top,other)],\
           ordered_paircount[(top,other)],\
           ordered_paircount[(other,top)],\
           word2domain[top].most_common(1)[0][0],\
           word2domain[other].most_common(1)[0][0],
           str(pairrel[(top,other)][1]),
           str(pairrel[(top,other)][2]),
           )
        pairdf.append(prow)

pairdf = pd.DataFrame(pairdf,columns=['word1','word2','relation',\
'totalfreq','freq-w1-w2','freq-w2-w1','domain_w1','domain_w2','syn_w1','syn_w2'])
pairdf = pairdf.sort_values(by=['totalfreq'],ascending=False)
pairdf.to_csv("names_pairs_relations_v2.csv")

domainpairdf = []

for (domain,top,other) in domain_paircount:
#    if paircount[(top,other)] > 5:

        if (top,other) in pairrel:
            rel = pairrel[(top,other)]
        else:
            rel = pairrel[(other,top)]

        prow = (
           domain,
           top,other,rel[0],\
           domain_paircount[(domain,top,other)],\
           domain_ordered_paircount[(domain,top,other)],\
           domain_ordered_paircount[(domain,other,top)],\
           word2domain[top].most_common(1)[0][0],\
           word2domain[other].most_common(1)[0][0],
           str(rel[1]),
           str(rel[2]),
           )
        domainpairdf.append(prow)

domainpairdf = pd.DataFrame(domainpairdf,columns=['domain','word1','word2','relation',\
'totalfreq','freq-w1-w2','freq-w2-w1','domain_w1','domain_w2','syn_w1','syn_w2'])
domainpairdf = domainpairdf.sort_values(by=['domain','totalfreq'],ascending=False)
domainpairdf.to_csv("domains_names_pairs_relations_v2.csv")





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
