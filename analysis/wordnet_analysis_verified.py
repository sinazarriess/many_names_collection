import pandas as pd
from nltk.corpus import wordnet as wn
from collections import Counter
from itertools import chain
import numpy as np


VERIFIED = True

hyp = lambda s: s.hypernyms()   # TODO not good python

def is_hyponym(syns1,syns2,max_depth=5):

    #syns1 = wn.synsets(w1,pos="n")
    #syns2 = wn.synsets(w2,pos="n")

    for syn1 in syns1:
        hyp_closure = list(syn1.closure(hyp,depth=max_depth))

        for syn2 in syns2:
            if syn2 in hyp_closure:
                return (True,syn1,syn2)
    return (False,syns1,syns2)

def is_hypernym(syns1,syns2,max_depth=5):


    #syns1 = wn.synsets(w1,pos="n")
    #syns2 = wn.synsets(w2,pos="n")

    for syn2 in syns2:
        hyp_closure = list(syn2.closure(hyp,depth=max_depth))

        for syn1 in syns1:
            if syn1 in hyp_closure:
                return (True,syn1,syn2)
    return (False,syns1,syns2)

def is_synonym(syns1,syns2):

    #syns1 = wn.synsets(w1,pos="n")
    #syns2 = wn.synsets(w2,pos="n")

    for syn2 in syns2:

        for syn1 in syns1:
            #print("synonym",syn1,syn2)
            if syn1 == syn2:
                return (True,syn1,syn2)
    return (False,syns1,syns2)

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
    return (False, syns1, syns2)

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
    return (False, syns1, syns2)


def synset_word_relation(synset, word, word_to_synset):

    if word not in word_to_synset:
        return ('word-not-covered',0,0)
    #else:
#        print(w2,wn_map[w2])

    (has_rel,s1,s2) = is_synonym([wn.synset(synset)], word_to_synset[word])
    if has_rel:
        return ('synonym',s1,s2)

    (has_rel,s1,s2) = is_hypernym([wn.synset(synset)], word_to_synset[word], max_depth=10)
    if has_rel:
        return ('hyponym',s1,s2)

    (has_rel,s1,s2) = is_hypernym(word_to_synset[word], [wn.synset(synset)], max_depth=10)
    if has_rel:
        return ('hypernym',s1,s2)

    (has_rel,s1,s2) = is_cohyponym([wn.synset(synset)], word_to_synset[word], max_depth=1)
    if has_rel:
        return ('co-hyponym',s1,s2)

    (has_rel,s1,s2) = is_meronym([wn.synset(synset)], word_to_synset[word], max_depth=10)
    if has_rel:
        return ('meronym',s1,s2)
    (has_rel,s1,s2) = is_meronym(word_to_synset[word], [wn.synset(synset)], max_depth=10)
    if has_rel:
        return ('holonym',s1,s2)

    return ('rel-not-covered', word, synset)



if __name__ == '__main__':

    path_to_verified_data = '../proc_data_phase0/verification/all_responses_round0-3_verified.csv'

    verified_data = pd.read_csv(path_to_verified_data, sep='\t', converters={'spellchecked': lambda x: Counter(eval(x)), 'verified': eval})

    indices_where_vgname_adequate = []

    # TODO Redo the following based on entry-level name, not based on vg_obj_name's synset.

    # Look only at rows where the vg_obj_name is adequate and in cluster 0.
    for i, row in verified_data.iterrows():
        if row['verified'][row['vg_obj_name']]['adequacy'] == 1 and row['verified'][row['vg_obj_name']]['cluster_id'] == 0:
            indices_where_vgname_adequate.append(i)

    print('before:', len(verified_data))
    verified_data = verified_data.iloc[indices_where_vgname_adequate]
    print('after:', len(verified_data))

    # Create separate column containing only the names that are 1. adequate and 2. in cluster 0
    verified_data['names_to_be_counted'] = [[] for _ in range(len(verified_data))]
    if VERIFIED:
        for i, row in verified_data.iterrows():
            verified_data.at[i,'names_to_be_counted'] = [name for name in row['spellchecked'] if name in row['verified'] and row['verified'][name]['adequacy'] == 1 and row['verified'][name]['cluster_id'] == 0]
    else:
        for i, row in verified_data.iterrows():
            verified_data.at[i, 'names_to_be_counted'] = [name for name in row['spellchecked'] if name in row['verified']]

    # Get vocabulary of remaining names
    vocabulary = set(chain(*verified_data['names_to_be_counted']))
    print('vocabulary:', len(vocabulary))

    # Create word-to-synset mapping once:
    name_to_synset = {}
    for name in vocabulary:
        if len(wn.synsets(name, pos="n")) > 0:
            name_to_synset[name] = wn.synsets(name, pos="n")
        elif len(wn.synsets('_'.join(name.split(' ')), pos="n")) > 0:
            name_to_synset[name] = wn.synsets('_'.join(name.split(' ')), pos="n")
        elif len(wn.synsets(''.join(name.split(' ')), pos="n")) > 0:
            name_to_synset[name] = wn.synsets(''.join(name.split(' ')), pos="n")
        elif len(wn.synsets('-'.join(name.split(' ')), pos="n")) > 0:
            name_to_synset[name] = wn.synsets('-'.join(name.split(' ')), pos="n")

    # Find which types of relations hold
    # verified_data['name_relations'] = [Counter() for _ in range(len(verified_data))]

    rel_not_covereds = []

    total_relation_tokens = Counter()
    total_relation_types = Counter()
    for i, row in verified_data.iterrows():
        relations = [(name, synset_word_relation(row['synset'], name, name_to_synset)) for name in row['names_to_be_counted'] if name != row['vg_obj_name']]
        relations = [(t[0], (t[1][0], t[1][1], row['vg_obj_name'])) for t in relations]
        relations_types = [relation[1][0] for relation in relations]
        relations_tokens = list(chain(*[[relation[1][0] for _ in range(row['spellchecked'][relation[0]])] for relation in relations]))

        for relation in relations:
            if relation[1][0] == 'rel-not-covered':
                rel_not_covereds.append((relation[1][1], relation[1][2]))

        # verified_data.at[i, 'name_relations'].update(relations)
        total_relation_tokens.update(relations_tokens)
        total_relation_types.update(relations_types)

    # Print examples of no-relation:
    rel_not_covereds = Counter(rel_not_covereds).most_common(9999)
    print('\n'.join([str(x[0]) + ' ' + str(x[1]) for x in rel_not_covereds]))


    print()

    print(total_relation_tokens)
    print(total_relation_types)

    # TODO Surely I can create a dataframe more directly from the dictionaries?
    relation_counts_df = []
    for relation in ['synonym','meronym','holonym','hypernym','hyponym','rel-not-covered','word-not-covered','co-hyponym']:
        relation_counts_df.append([relation, total_relation_types[relation], total_relation_tokens[relation]])
    relation_counts_df = pd.DataFrame(relation_counts_df, columns=['relation','% Typ', '% Tok'])
    relation_counts_df['% Typ'] /= relation_counts_df['% Typ'].sum()
    relation_counts_df['% Tok'] /= relation_counts_df['% Tok'].sum()

    relation_counts_df['% Typ'] = relation_counts_df['% Typ'].apply(lambda x: "%.1f" % (x * 100))
    relation_counts_df['% Tok'] = relation_counts_df['% Tok'].apply(lambda x: "%.1f" % (x * 100))

    print(relation_counts_df.to_latex(index=False))
