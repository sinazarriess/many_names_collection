import pandas as pd
import glob
import os
import json
#import xmltodict
import sys
from collections import Counter
import numpy as np
#from spellchecker import SpellChecker
#from nltk.stem.wordnet import WordNetLemmatizer

def process_answers(resultfile,hit2images):
    
    with open(resultfile, 'r') as handle:
        resultlist = json.load(handle)
    
    obj2answers = {}
    obj2info = {}
    
    for assignm in resultlist:
        hitid = assignm['HITId']
        print(hitid)
        for a in assignm['Assignments']:
            #print(a)
            answers = a['Answers']
            for ix in range(10):
                object_id = hit2images[hitid][str(ix)][1]
                name = answers[str(ix)]['inputbox_objname-'+str(ix)]
                if object_id not in obj2answers:
                    obj2answers[object_id] = Counter()
                    obj2info[object_id] = hit2images[hitid][str(ix)]

                if name:
                    name = name.lower()
                    obj2answers[object_id][name] += 1
                    
                    
    ## optional: spell checking
    ## commented this out, as it didn't really seem to work
    ## obj2clean_answers = {objid:check_spelling(obj2answers[objid]) for objid in obj2answers}


        
    allrows = [] 
    for objid in obj2answers:
        row = obj2info[objid]
        row.append(obj2answers[objid])
        allrows.append(row)
        
    fulldf = pd.DataFrame(allrows,columns=['vg_img_id','vg_object_id','cat','synset','url','responses'])
                    
    return fulldf
                    
def process_hits(hitfile):
    
    with open(hitfile, 'r') as handle:
        hitlist = json.load(handle)
        
    hit2images = {}
    for hit in hitlist:
        hit2images[hit['HIT']['HITId']] = hit['Images']
    return hit2images


def check_spelling(answers):
    d = answers
    misspelled = spell.unknown(d.keys())
    ok = {w:d[w] for w in d.keys() if not w in misspelled}
    #print(misspelled)
    #print(ok)
    
    for mis in misspelled:
        #print("Checking:",mis)
        cands = spell.candidates(mis)
        #print("Cand:",cands)
        corrected = False
        for c in ok:
            if c in cands:
                #print("YAY")
                ok[c] += d[mis]
                corrected = True
                break
        if not corrected:
            ok[mis] = d[mis]
            
    clean_answers = Counter()
    for w in ok:
        lem_w = Lem.lemmatize(w)
        clean_answers[lem_w] += ok[w]
        
    return clean_answers
