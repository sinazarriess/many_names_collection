# -*- coding: utf-8 -*-

# created gbt Sun Feb 24 2019

import pandas as pd
import numpy as np
import os
#from collections import Counter
#from utils_analysis import *

#df=pd.read_csv("../data_phase0/results-created_2019-Feb-20_13_36_05_final.csv",sep="\t")
#print(df.head())
#print(df.columns)

#TO DO (from paper files)
#
#analyse data from Phase 0
#
#Items:
#
#\begin{itemize}
#\item  to what extent do people agree when their task is to give the most straightforward name they can think of to a visual object? (see \ref{sec:snowgrad})
#\item is the level of agreement the same for all categories? (see \ref{sec:snowgrad})
#\item how specific are the most familiar names? link names to WordNet, show that WordNet might not be ideal to assess specificity
#\item assess how representative name annotations in Visual Genome are, when compared to our names (see \ref{sec:vg})
#\end{itemize}
#
#
#Note: I (Gemma) will use three levels of analysis: ALL (all data lumped together), DOMAIN (Gemma's reorganization of Carina's ``supercategories''; see doc 0\_object\_naming\_taboo), COLLECTION NODE (Carina's ``synset / collection node'').
# 
#Plans for analysis (then we see what to put in the paper):
#
#\begin{enumerate}
#\item compute snowgrad measure and do a:
#  \begin{itemize}
#  \item histogram ALL
#  \item boxplot by DOMAIN
#  \item dataframe with mean and sd by COLLECTION NODE 
#  \item[\ra] This will tell us how much agreement there is among subjects about how to name objects in general and within each domain/''subcategory''.
#  \end{itemize}
#\item can we find generalizations about tendencies in agreement? (open: how to go about it)
#\end{enumerate}

df = pd.read_csv('domains_names_pairs_relations_v2.csv',index_col=0)

df.columns

for i in ['home','people','clothing','vehicles','buildings','food','animals_plants']:
    filename= i+'-to-annotate.csv'
    d=df[(df.relation=="crossclassified") & (df.domain==i)].sort_values('totalfreq',ascending=False)
    d=d.head(1000)
    d.to_csv(filename)