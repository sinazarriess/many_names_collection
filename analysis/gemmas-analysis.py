# -*- coding: utf-8 -*-

# created gbt Sun Feb 24 2019

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ast import literal_eval
from collections import Counter

# spellchecked_min2: why this restriction? Did we really not check names with freq=1?

df=pd.read_csv('../proc_data_phase0/verification/all_responses_round0-3_verified.csv', sep="\t")
#df=pd.read_csv('kk.csv', sep="\t")
df=df.drop(['responses_r0','opt-outs','top_response_domain_r0','responses_domains_r0',
            'sample_type','all_responses','clean','canon',
            'responses_domains_r1', 'responses_r1', 'top_response_domain_r1',
       'responses_domains_r2', 'responses_r2', 'top_response_domain_r2',
       'responses_domains_r3', 'responses_r3', 'top_response_domain_r3'], axis=1)
#df=df.head(1)
#print(df)
print(df.columns)

adequacies=[] # number of adequate names (total)
tot_names=0; non_ad=0 # total n of names; n of non-adequate names (see threshold below)
list_n_ad_names=[] # number of adequate names per canonical object
list_entry_names=[] # entry names for each canonical object (cluster 0)
list_freqs_entry_name=[] # entry names for each canonical object (cluster 0)
no_entry_lev=0
for row in df.itertuples():
    name_freqs=eval(row.spellchecked_min2)
    print(name_freqs)
    # find entry-level name; this should be done for the canonical object,
    # but I can't recover the info from the verified column (name data in a dictionary),
    # so I use the one from all data and check if the entry level name form those data is in cluster 0
    # there are only 700 imgs where the most frequent name is not in cluster 0; TODO: check them, and mend this
    entry_name,entry_freq=name_freqs.most_common(1)[0] # returns tuple, e.g. ('man',11)
    d = literal_eval(row.verified) # dictionary with verification data
    n_ad_names_img=0; freqs_canon_names=1; entry_name_in_cluster0=False
    for name in d.keys(): # each name for this object
        tot_names+=1
        #print("\t",name)
        values_dict = d[name]
        ad=values_dict['adequacy']
        adequacies.append(ad); 
        if ad < 0.6: non_ad += 1
        else: # name is adequate
            if values_dict['cluster_id']==0: # labels canonical object
                if name==entry_name: entry_name_in_cluster0=True
                n_ad_names_img += 1 # number of adequate names
                freqs_canon_names += name_freqs[name] # total frequency of canonical names
#    print("\t",freqs_canon_names)
    if entry_name_in_cluster0==False: entry_name,entry_freq=(None,0) ### TO BE MENDED IN THE FUTURE (see comment 'find entry-level name' above)
    list_n_ad_names.append(n_ad_names_img)
    list_entry_names.append(entry_name)
    freq_entry_name=(entry_freq+1)/freqs_canon_names
    list_freqs_entry_name.append(freq_entry_name)


df['n_canonical_names']=list_n_ad_names
df['entry_name']=list_entry_names
df['entry_freq']=list_freqs_entry_name

df.entry_freq.plot(kind='hist')
plt.show()
plt.close()

#"{'man': 
# {'cluster': ('baseball player', 'batter', 'man', 'person', 'player'), 
# 'adequacy': 1.0, 
#'inadequacy_type': None, 
# 'cluster_id': 0, 
# 'cluster_weight': 0.9375}, 
# 'helmet': {'cluster': ('helmet',), 'adequacy': 0.5, 'inadequacy_type': 'bounding box', 'cluster_id': 1, 'cluster_weight': 0.0625}, 'player': {'cluster': ('baseball player', 'batter', 'man', 'person', 'player'), 'adequacy': 1.0, 'inadequacy_type': None, 'cluster_id': 0, 'cluster_weight': 0.9375}, 'batter': {'cluster': ('baseball player', 'batter', 'man', 'person', 'player'), 'adequacy': 1.0, 'inadequacy_type': None, 'cluster_id': 0, 'cluster_weight': 0.9375}, 'baseball player': {'cluster': ('baseball player', 'batter', 'man', 'person', 'player'), 'adequacy': 1.0, 'inadequacy_type': None, 'cluster_id': 0, 'cluster_weight': 0.9375}, 'person': {'cluster': ('baseball player', 'batter', 'man', 'person', 'player'), 'adequacy': 1.0, 'inadequacy_type': None, 'cluster_id': 0, 'cluster_weight': 0.9375}}"

#

### Histogram
#adequacies=np.array(list_n_ad_names)
#plt.plot(x=adequacies)
#plt.grid(axis='y', alpha=0.75)
#plt.title('Number of names')
##to_write='names with adequacy < 0.6: {} / {} ({:.1%})'.format(non_ad, tot_names, non_ad/tot_names)
##plt.text(0.,15000,to_write)
#plt.show()
#plt.close()
#
#
#df['n_canonical_names']=list_n_ad_names
#ndist = df.n_canonical_names.value_counts()
#ndist.plot(kind='bar',rot=0,title="Distribution of number of names per canonical object")
#plt.show()
#plt.close()