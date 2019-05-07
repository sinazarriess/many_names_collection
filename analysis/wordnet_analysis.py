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



fn = 'all_responses_round0-3_cleaned.csv'
resdf = make_df(fn)

hyp = lambda s:s.hypernyms()




def name_pairs(rdict):

    np_count = Counter()
    if len(rdict.keys()) > 1:
        topname = rdict.most_common(1)[0][0]
        for oname in rdict:
            if not oname == topname:
                np_count[(topname,oname)] = rdict[oname]

    return np_count


# In[101]:


paircount = Counter()
for ndict in resdf['spellchecked']:
    #print(ndict)
    x = name_pairs(ndict)
    paircount.update(x)


# In[102]:


paircount.most_common(20)


# In[103]:


for ((top,other),freq) in paircount.most_common(20):
    topsyn = wn.synsets(top)
    othersyn = wn.synsets(other)
    print(top,topsyn)
    print(other,othersyn)
    print("***")


# In[104]:


hyp = lambda s:s.hypernyms()
list(sp.closure(hyp,depth=4))






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

def is_cohyponym(w1,w2,max_depth=5):

    for dpt in range(1,6):

        syns1 = wn.synsets(w1,pos="n")
        syns2 = wn.synsets(w2,pos="n")




        for syn2 in syns2:

            hyp_closure2 = list(syn2.closure(hyp,depth=dpt))


            for syn1 in syns1:
                hyp_closure1 = list(syn1.closure(hyp,depth=dpt))

                for h2 in hyp_closure2:
                    for h1 in hyp_closure1:
                        if h1 == h2:
                            return dpt
    return 0


# In[106]:


is_hyponym("person","man")


# In[107]:


is_hypernym("person","man")


# In[129]:


pair_depth = {}

for (top,other) in paircount:
    if ' ' in top:
        top = '_'.join(top.split(' '))
    if ' ' in other:
        other = '_'.join(other.split(' '))
    if len(wn.synsets(top)) > 0 and         len(wn.synsets(other)) > 0:
        pair_depth[(top,other)] = (is_synonym(top,other),                               is_hyponym(top,other,max_depth=10),                               is_hypernym(top,other,max_depth=10),                               is_cohyponym(top,other,max_depth=10)
                              )
    else:
        print("WORD NOUT FOUND in WN")
        print(top,other,len(wn.synsets(top)),len(wn.synsets(other)))


# In[130]:


pair_depth


# In[226]:


pair_rel = {}
no_rel = Counter()
have_rel = Counter()
for p in paircount:
    if p in pair_depth:
        if pair_depth[p] == (0,0,0,0):
            print(p)
            no_rel[p] = paircount[p]
        else:
            pair_rel[p] = pair_depth[p]
            have_rel[p] = paircount[p]
    else:
        no_rel[p] = paircount[p]


# In[168]:


len(no_rel),len(have_rel)


# In[169]:


sum(no_rel.values())/sum(paircount.values())


# In[170]:


sum(have_rel.values())


# In[193]:


unordered_counts = Counter()
unordered_rel = {}
unordered_depth = {}
swapped = []
for (p1,p2) in have_rel:
    swapp= (p2,p1)
    if not (p1,p2) in swapped:
        unordered_counts[(p1,p2)] = have_rel[(p1,p2)]
        unordered_counts[(p1,p2)] += have_rel[swapp]
        if pair_rel[(p1,p2)][0] > 0:
            unordered_rel[(p1,p2)] = 'synonyms'
            unordered_depth[(p1,p2)] = pair_rel[(p1,p2)][0]
        elif pair_rel[(p1,p2)][1] > 0:
            unordered_rel[(p1,p2)] = 'hierarchical'
            unordered_depth[(p1,p2)] = pair_rel[(p1,p2)][1]
        elif pair_rel[(p1,p2)][2] > 0:
            unordered_rel[(p1,p2)] = 'hierarchical'
            unordered_depth[(p1,p2)] = pair_rel[(p1,p2)][2]
        elif pair_rel[(p1,p2)][3] > 0:
            unordered_rel[(p1,p2)] = 'crossclassified'
            unordered_depth[(p1,p2)] = pair_rel[(p1,p2)][3]

        swapped.append(swapp)


# In[179]:


sum(unordered_counts.values())


# In[180]:


sum(have_rel.values())


# In[181]:


len(unordered_rel)


# In[183]:


len(have_rel)


# In[196]:


typecounts = Counter()
tokencounts = Counter()
depth_av = {}
for p in unordered_rel:
    if unordered_rel[p] not in depth_av:
        depth_av[unordered_rel[p]] = []
    typecounts[unordered_rel[p]] += 1
    tokencounts[unordered_rel[p]] += unordered_counts[p]
    depth_av[unordered_rel[p]].append(unordered_depth[p])


# In[197]:


typecounts


# In[198]:


tokencounts


# In[220]:


outdf = []
totaltypes = sum(typecounts.values())
totaltokens = sum(tokencounts.values())
for p in typecounts:
    outdf.append((p,"%.3f"%(typecounts[p]/totaltypes),"%.3f"%(tokencounts[p]/totaltokens),"%.3f"%(np.mean(depth_av[p]))))


# In[221]:


outdff = pd.DataFrame(outdf,columns=['relation','types','tokens','av WN depth'])


# In[222]:


outdff


# In[225]:


print(outdff.to_latex(index=False))


# In[ ]:





# In[253]:


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
    print(cat)
    for ((n1,n2),freq) in cat2paircount[cat].most_common(10):
        exstr.append("%s -- %s (%d)"%(n1,n2,freq))
    examples.append((cat,", ".join(exstr)))



# In[255]:


exdf = pd.DataFrame(examples,columns=['category',"most frequent naming variants"])


# In[256]:


exdf


# In[257]:


pd.set_option('display.max_colwidth', -1)
print(exdf.to_latex(index=False))


# In[ ]:





# In[252]:


for (n1,n2) in unordered_rel:
    if unordered_rel[(n1,n2)] == "hierarchical":
        print(n1,n2, paircount[(n1,n2)],paircount[(n2,n1)])


# In[ ]:





# In[249]:


unordered_rel


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[262]:


lettsyn = wn.synsets('food')
print(lettsyn)
hyp_closure2 = list(lettsyn[2].closure(hyp,depth=5))
print(hyp_closure2)


# In[263]:


lettsyn = wn.synsets('fruit')
print(lettsyn)
hyp_closure2 = list(lettsyn[0].closure(hyp,depth=5))
print(hyp_closure2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[119]:


len(paircount)


# In[120]:


len(pair_depth)


# In[ ]:





# In[110]:


paircount[('hotdog','hot dog')]


# In[111]:


wn.synsets('hotdog')


# In[ ]:





# In[112]:


is_hyponym('giraffe','animal',max_depth=10)


# In[113]:


wn.synsets('giraffe')


# In[114]:


is_hyponym('flamingo','animal',max_depth=10)


# In[126]:


wn.synsets('hat_stand')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[116]:


paircount[('skier','skiier')]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:
