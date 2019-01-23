import pandas as pd
from collections import Counter
from random import shuffle
import csv

def check_taboos(rdf):

    im2taboo = {}
    im2names = {}
    problem_workers = []
    problem_hits = 0

    appr = []
    reject = []
    feedback = []

    for rx,row in rdf.iterrows():
        checks = []
        all_taboos = []
        all_names = []
        check = []
        for ix in ['0','1','2','3','4']:
            im = row['Input.img_%s_url'%ix]
            taboo = str(row['Input.taboolist_%s'%ix])

            
            #print("Taboo",str(taboo))
            if taboo != "nan":
                taboo = taboo.split(", ")
                all_taboos.append(taboo)
                name = row['Answer.inputbox_objname-%s'%ix]
                name = name.lower()
                all_names.append(name)
                check.append(name in taboo)

        if check.count(True) > 1:
            print("Problem with Worker:",row['WorkerId'])
            print("Taboo",all_taboos)
            print("Names",all_names)
            problem_hits +=1
            problem_workers.append(row['WorkerId'])
            reject.append('x')
            appr.append('{}')
            feedback.append('"Please DO NOT use one of the taboo words for naming the object! We are sorry if this was not clearly stated in the instructions."')
        else:
            reject.append('{}')
            appr.append('x')
            feedback.append('{}')


    print("Problematic hits:", problem_hits)
    print("Problematic workers:",len(set(problem_workers)))
    print(set(problem_workers))

    return(reject,appr,feedback)
            
            
resultdf = pd.read_csv("Batch_3453438_batch_results.csv")
reject,appr,feedback = check_taboos(resultdf)
resultdf['Approve'] = appr
resultdf['Reject'] = reject
resultdf['RequesterFeedback'] = feedback

resultdf.to_csv("Batch_3453438_batch_results_approved.csv",index=False,quoting=csv.QUOTE_ALL)