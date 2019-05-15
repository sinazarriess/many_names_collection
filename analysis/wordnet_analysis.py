#!/usr/bin/env python
# coding: utf-8


from nltk.corpus import wordnet as wn
import pandas as pd
from agreement_table import make_df
from agreement_table import make_synset_df
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


def is_hyponym(syns1,syns2,max_depth=5):

    #syns1 = wn.synsets(w1,pos="n")
    #syns2 = wn.synsets(w2,pos="n")

    for syn1 in syns1:
        hyp_closure = list(syn1.closure(hyp,depth=dpt))

        for syn2 in syns2:
            if syn2 in hyp_closure:
                return (True,syn1,syn2)
    return (False,0,0)

def is_hypernym(syns1,syns2,max_depth=5):


    #syns1 = wn.synsets(w1,pos="n")
    #syns2 = wn.synsets(w2,pos="n")

    for syn2 in syns2:
        hyp_closure = list(syn2.closure(hyp,depth=max_depth))

        for syn1 in syns1:
            if syn1 in hyp_closure:
                return (True,syn1,syn2)
    return (False,0,0)

def is_synonym(syns1,syns2):

    #syns1 = wn.synsets(w1,pos="n")
    #syns2 = wn.synsets(w2,pos="n")

    for syn2 in syns2:

        for syn1 in syns1:
            #print("synonym",syn1,syn2)
            if syn1 == syn2:
                return (True,syn1,syn2)
    return (False,0,0)

def is_cohyponym(syns1,syns2,max_depth=1):

    #syns1 = wn.synsets(w1,pos="n")
    #syns2 = wn.synsets(w2,pos="n")

    for syn2 in syns2:

        hyp_closure2 = list(syn2.closure(hyp,depth=max_depth))


        for syn1 in syns1:
            hyp_closure1 = list(syn1.closure(hyp,depth=max_depth))

            for h2 in hyp_closure2:
                for h1 in hyp_closure1:
                    if h1 == h2:
                        return (True,h1,h2)
    return (False,0,0)

def is_meronym(syns1,syns2,max_depth=1):

    #syns1 = wn.synsets(w1,pos="n")
    #syns2 = wn.synsets(w2,pos="n")

    for syn1 in syns1:

        for osyn in syn1.part_meronyms():
            if osyn in syns2:
                return (True,syn1,osyn)
        for osyn in syn1.substance_meronyms():
            if osyn in syns2:
                return (True,syn1,osyn)
    return (False,0,0)

def get_word_rel(w1,w2,wn_map):

    (has_rel,s1,s2) = is_synonym(wn_map[w1],wn_map[w2])
    if has_rel:
        return ('synonymy',s1,s2)

    (has_rel,s1,s2) = is_hypernym(wn_map[w1],wn_map[w2],max_depth=10)
    if has_rel:
        return ('hypernymy',s1,s2)

    (has_rel,s1,s2) = is_hypernym(wn_map[w1],wn_map[w2],max_depth=10)
    if has_rel:
        return ('hyponymy',s1,s2)

    (has_rel,s1,s2) = is_cohyponym(wn_map[w1],wn_map[w2],max_depth=1)
    if has_rel:
        return ('co-hyponymy',s1,s2)

    (has_rel,s1,s2) = is_meronym(wn_map[w1],wn_map[w2],max_depth=10)
    if has_rel:
        return ('meronymy',s1,s2)
    (has_rel,s1,s2) = is_meronym(wn_map[w1],wn_map[w2],max_depth=10)
    if has_rel:
        return ('holonymy',s1,s2)

    return ('rel-not-covered',0,0)

def get_syn_word_rel(syn1,w2,wn_map):

    if w2 not in wn_map:
        return ('word-not-covered',0,0)
    #else:
#        print(w2,wn_map[w2])

    (has_rel,s1,s2) = is_synonym([wn.synset(syn1)],wn_map[w2])
    if has_rel:
        return ('synonym',s1,s2)

    (has_rel,s1,s2) = is_hypernym([wn.synset(syn1)],wn_map[w2],max_depth=10)
    if has_rel:
        return ('hyponym',s1,s2)

    (has_rel,s1,s2) = is_hypernym(wn_map[w2],[wn.synset(syn1)],max_depth=10)
    if has_rel:
        return ('hypernym',s1,s2)

    (has_rel,s1,s2) = is_cohyponym([wn.synset(syn1)],wn_map[w2],max_depth=1)
    if has_rel:
        return ('co-hyponym',s1,s2)

    (has_rel,s1,s2) = is_meronym([wn.synset(syn1)],wn_map[w2],max_depth=10)
    if has_rel:
        return ('meronym',s1,s2)
    (has_rel,s1,s2) = is_meronym(wn_map[w2],[wn.synset(syn1)],max_depth=10)
    if has_rel:
        return ('holonym',s1,s2)

    return ('rel-not-covered',0,0)


def example_table(resdf):

    cat2paircount = {}
    for cat in set(list(resdf['vg_domain'])):
        cat2paircount[cat] = Counter()
        catdf = resdf[resdf['vg_domain'] == cat]

        for ndict in catdf['spellchecked']:
        #print(ndict)
            x = name_pairs(ndict)
            cat2paircount[cat].update(x)

    examples = []
    for cat in cat2paircount:
        exstr = []
        #print(cat)
        for ((n1,n2),freq) in cat2paircount[cat].most_common(10):
            exstr.append("%s -- %s (%d)"%(n1,n2,freq))
        examples.append((cat,", ".join(exstr)))


    exdf = pd.DataFrame(examples,columns=['category',"most frequent naming variants"])
#print(exdf.to_latex(index=False))

def get_vocab(syndf,name2synset):

    wordcount = Counter()
    word2domain = {}
    wn_map = {}


    for ix,row in syndf.iterrows():
        wordcount.update(row['responses'])
        for word in row['responses']:
            if word not in word2domain:
                word2domain[word] = Counter()
                word2domain[word][row['vg_domain']] += row['responses'][word]

                if word in name2synset:
                    #print("found word",word)
                    wn_map[word] = [wn.synset(name2synset[word])]
                    #print(wn_map[word])
                else:
                    if len(wn.synsets(word,pos="n")) > 0:
                        wn_map[word] = wn.synsets(word,pos="n")
                    elif len(wn.synsets('_'.join(word.split(' ')),pos="n")) > 0:
                        wn_map[word] = wn.synsets('_'.join(word.split(' ')),pos="n")
                    elif len(wn.synsets(''.join(word.split(' ')),pos="n")) > 0:
                        wn_map[word] = wn.synsets(''.join(word.split(' ')),pos="n")
                    elif len(wn.synsets('-'.join(word.split(' ')),pos="n")) > 0:
                        wn_map[word] = wn.synsets('-'.join(word.split(' ')),pos="n")



    return wordcount,word2domain,wn_map

def add_rel_to_df(syndf,wn_vocab):

    rel_token = []
    rel_types = []
    rel_names = []

    for ix,row in syndf.iterrows():
        vg_syn = row['vg_obj_synset']

        #print(vg_syn)
        rel_tok_c = Counter()
        rel_typ_c = Counter()
        rel_synsets = {}
        for (mname) in row['responses']:
            if (not mname in row['vg_obj_names']):
            #mname in wn_vocab and \
                rel,s1,s2 = get_syn_word_rel(vg_syn,mname,wn_vocab)
                #if rel == 'hypernym':
                #    print(rel,mname,s1,s2)
                rel_typ_c[rel] += 1
                rel_tok_c[rel] += row['responses'][mname]

                if rel not in rel_synsets:
                    rel_synsets[rel] = []
                rel_synsets[rel].append(mname)

        #print(rel_tok_c)

        rel_token.append(rel_tok_c)
        rel_types.append(rel_typ_c)
        rel_names.append(rel_synsets)

    syndf['rel_tokens'] = rel_token
    syndf['rel_types'] = rel_types
    syndf['rel_names'] = rel_names

    return syndf


def make_rel_table(syndf):

    outdf = []

    relations = ['synonym','meronym','holonym','hypernym','hyponym','rel-not-covered','word-not-covered','co-hyponym']
    rel2freq_types = {r:[] for r in relations}
    rel2freq_tokens = {r:[] for r in relations}
    for ix,row in syndf.iterrows():
        total = sum(row['rel_tokens'].values())
        totalt = sum(row['rel_types'].values())
        for rel in rel2freq_types:
            rel2freq_tokens[rel].append(row['rel_tokens'][rel]/total)
            rel2freq_types[rel].append(row['rel_types'][rel]/totalt)

    for p in rel2freq_types:
        row = [p]
        row.append("%.1f"%(np.mean(rel2freq_types[p])*100))
        row.append("%.1f"%(np.mean(rel2freq_tokens[p])*100))
        #row.append("%.1f"%((typecounts_min1[p]/totaltypes_min1)*100))
        #row.append("%.1f"%((tokencounts_min1[p]/totaltokens_min1)*100))
        outdf.append(row)

    print(outdf)

    #outdff = pd.DataFrame(outdf,columns=['relation','Ftok', 'N','Ftok-min1','Nmin1'])
    outdff = pd.DataFrame(outdf,columns=['relation','% Typ', '% Tok'])


    print(outdff.sort_values(by=['% Typ']).to_latex(index=False))


if __name__ == '__main__':
    fn = 'all_responses_round0-3_cleaned.csv'
    resdf = make_df(fn)
    syndf,name2synset = make_synset_df(resdf)


    print(syndf.head())
    vocab,domain2vocab,wn_vocab = get_vocab(syndf,name2synset)
    syndf = add_rel_to_df(syndf,wn_vocab)
    make_rel_table(syndf)





    #colnames = ['domain']
    #for p in relations:#
    #    colnames.append(p+"_typ")


    # domtypecounts = {}
    # domtokencounts = {}
    # for (domain,top,other) in domain_paircount:
    #     if (top,other) in pairrel:
    #         rel = pairrel[(top,other)][0]
    #     else:
    #         rel = pairrel[(other,top)][0]
    #     if '.' in rel:
    #         rel = rel[:-2]
    #     if domain not in domtypecounts:
    #         domtypecounts[domain] = Counter()
    #         domtokencounts[domain] = Counter()
    #     domtypecounts[domain][rel] += 1
    #     domtokencounts[domain][rel] += domain_paircount[(domain,top,other)]

    ##for domain in domtypecounts:#
    #    row = [domain]
    #    for p in relations:
    #        row.append("%.3f"%(domtypecounts[domain][p]/sum(domtypecounts[domain].values())))
    #        row.append("%.3f"%(domtokencounts[domain][p]/sum(domtokencounts[domain].values())))
    #    outdf.append(row)



    # pairdf = []
    #
    # for (top,other) in paircount:
    # #    if paircount[(top,other)] > 5:
    #         prow = (top,other,pairrel[(top,other)][0],\
    #            paircount[(top,other)],\
    #            ordered_paircount[(top,other)],\
    #            ordered_paircount[(other,top)],\
    #            word2domain[top].most_common(1)[0][0],\
    #            word2domain[other].most_common(1)[0][0],
    #            str(pairrel[(top,other)][1]),
    #            str(pairrel[(top,other)][2]),
    #            )
    #         pairdf.append(prow)
    #
    # pairdf = pd.DataFrame(pairdf,columns=['word1','word2','relation',\
    # 'totalfreq','freq-w1-w2','freq-w2-w1','domain_w1','domain_w2','syn_w1','syn_w2'])
    # pairdf = pairdf.sort_values(by=['totalfreq'],ascending=False)
    # pairdf.to_csv("names_pairs_relations_v2.csv")
    #
    # domainpairdf = []
    #
    # for (domain,top,other) in domain_paircount:
    # #    if paircount[(top,other)] > 5:
    #
    #         if (top,other) in pairrel:
    #             rel = pairrel[(top,other)]
    #         else:
    #             rel = pairrel[(other,top)]
    #
    #         prow = (
    #            domain,
    #            top,other,rel[0],\
    #            domain_paircount[(domain,top,other)],\
    #            domain_ordered_paircount[(domain,top,other)],\
    #            domain_ordered_paircount[(domain,other,top)],\
    #            word2domain[top].most_common(1)[0][0],\
    #            word2domain[other].most_common(1)[0][0],
    #            str(rel[1]),
    #            str(rel[2]),
    #            )
    #         domainpairdf.append(prow)
    #
    # domainpairdf = pd.DataFrame(domainpairdf,columns=['domain','word1','word2','relation',\
    # 'totalfreq','freq-w1-w2','freq-w2-w1','domain_w1','domain_w2','syn_w1','syn_w2'])
    # domainpairdf = domainpairdf.sort_values(by=['domain','totalfreq'],ascending=False)
    # domainpairdf.to_csv("domains_names_pairs_relations_v2.csv")







# In[ ]:
