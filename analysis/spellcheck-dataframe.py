import pandas as pd
import json
from collections import Counter
import numpy as np
import re

import enchant

# https://pypi.org/project/pyspellchecker/
from spellchecker import SpellChecker

whitelist = ['airplane', 'aeroplane', 'countertop', 'donut', 'hoodie',
             'frisbee', 'focaccia', 'railcar', 'armoire', 'tv stand',
             'hompton bench', 'selfie']
common_corrections = {'tshirt':'t-shirt','t shirt':'t-shirt','t shart':'t-shirt',
                      't shrit':'t-shirt', 't shait':'t-shirt', 't sheart':'t-shirt', 't- shirt':'t-shirt', 'teeshirt':'t-shirt',
                      'shrite':'shirt', 't shaer':'t-shirt',
                      'giratte':'giraffe', 'doughnut':'donut', 'helmate':'helmet','aeroplane':'airplane',
                      'airplain':'airplane','aeroplain':'airplane','alroplane':'airplane', 'areplane':'airplane',
                      'baseballer':'baseball player', 'billow':'pillow', 'buillow':'pillow',
                      'bedset':'bed set', 'burker':'burger', 'bress':'dress',
                      'camal':'camel', 'chir':'chair', 'curton':'curtain', 'caboard':'cupboard',
                      'corte':'coat', 'gawn':'gown', 'harce':'hare', 'mirrer':'mirror',
                      'jokket':'jacket', 'joket':'jacket', 'jooket':'jacket', 'juck':'jug', 'labtob':'laptop', 'laptor':'laptop',
                      'lopdop':'laptop', 'pajamas':'pyjamas', 'railcars':'railcar', 'showba':'shower',
                      'scating':'skating', 'skatting':'skating', 'sunglass':'sunglasses', 'shoba':'sofa',
                      'sopa':'sofa', 'shopa': 'sofa', 'scooty':'scooter', 'surfboarder':'surfer', 'sketing board': 'skateboard',
                      'tye':'tie', 'tils':'tile', 'trine':'train', 'trane':'train', 'umberla':'umbrella',
                      'ven':'van', 'washfashion':'washbasin', 'hot dog':'hotdog', 'skiier':'skier'
                     }

def clean_counter(count):
    cleaned_counter = Counter()
    for name in count:
        new_name = "".join(i for i in name if ord(i)<128)
        new_name = re.sub('[\\."_#]+', '', new_name)
        cleaned_counter[new_name] += count[name]
    return cleaned_counter

def canonize(count):
    """merges infrequent names in the counter
    with the biggest category if the edit distance is 1"""
    canon = dict()
    corrected = []
    for name, freq in count.items():
        if name in corrected:
            pass
        if freq <= 3 and name not in common_corrections:
            similar_names = Counter(dict((other_word, v) for (other_word, v) in count.items()
                                 if other_word in spell.edit_distance_1(name) and
                                 other_word not in common_corrections))
            canon[similar_names.most_common(1)[0][0]] = sum([count[n] for n, v in similar_names.most_common()])
            corrected.extend(similar_names.keys())
        elif name not in canon:
            canon[name] = freq
    return canon

def get_corrections(namedf):
    """create a mapping between misspelled responses and their corrections"""
    responses = [word for counter in namedf.canon for word in counter]
    response_count = Counter(responses)

    correction_dict = dict()
    for name in response_count:
        if name in common_corrections:
            correction_dict[name] = common_corrections[name]
        elif not all(dictionary.check(part) or part in whitelist for part in name.split()):
            suggestions = dictionary.suggest(name)
            if len(suggestions)==0:
                correction_dict[name] = name
            elif suggestions[0].lower() != name:
                correction_dict[name] = suggestions[0].lower()
    return correction_dict

def spellcheck(count, correction_dict):
    """apply the typo-correction mappings and update the counts"""
    spellchecked = dict()
    for name, freq in count.items():
        if name in correction_dict:
            this_name = correction_dict[name]
        else:
            this_name = name
        if this_name not in spellchecked:
            spellchecked[this_name] = count[name]
        else:
            spellchecked[this_name] += count[name]
    return spellchecked

if __name__=="__main__":
    namedf = pd.read_csv('../data_phase0/all_results_files_preprocessed_rounds0-3.csv',
                         sep="\t")

    total_resp = []
    vocab_counter = Counter()
    vg_counter = Counter()
    vg_is_mostcommon = 0
    vg_is_common = []
    for ix,row in namedf.iterrows():
        all_resp = Counter(eval(row['responses_r0'])) +\
                   Counter(eval(row['responses_r1'])) +\
                   Counter(eval(row['responses_r2'])) +\
                   Counter(eval(row['responses_r3']))
        total_resp.append(all_resp)
        vocab_counter += all_resp
        vg_counter[row['vg_obj_name']] += 1
        max_name = all_resp.most_common(1)[0][0]
        vg_is_common.append(max_name == row['vg_obj_name'])
        if max_name == row['vg_obj_name']:
            vg_is_mostcommon += 1

    namedf['all_responses'] = total_resp
    whitelist += list(namedf.vg_obj_name)

    language = "en"
    dictionary = enchant.Dict(language)
    spell = SpellChecker()

    namedf['clean'] = namedf.all_responses.apply(clean_counter)
    namedf['canon'] = namedf.clean.apply(canonize)

    correction_dict = get_corrections(namedf)
    namedf['spellchecked'] = namedf.canon.apply(lambda x:
                                                spellcheck(x, correction_dict))

    namedf.to_json('all_responses_round0-3_cleaned.json.gz', orient='split')
    namedf.to_csv('all_responses_round0-3_cleaned.csv', sep='\t')
