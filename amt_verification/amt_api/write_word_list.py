#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 15:27:17 2019

@author: u148188
"""
import boto3
import glob
import os
import json
import sys
import configparser
import datetime
import xmltodict

import amt_api

def get_words(mturk, path_published, path_results, statuses=["Submitted"]):
    '''ask AMT for results'''
    for filename in glob.glob(os.path.join(path_published, '*final.json')):
        print(filename)
        with open(filename, 'r') as handle:
            parsed = json.load(handle)

        all_words=set()
        for item in parsed:
            assignments = amt_api.get_assignments(mturk, item['HIT']['HITId'], statuses)
            print("HIT/Assignments",item['HIT']['HITId'],len(assignments))

            if len(assignments) > 0:
                for assignment in assignments:
                    answer_dict = xmltodict.parse(assignment['Answer'])
                    answers = []
                    for a in answer_dict['QuestionFormAnswers']['Answer']:
                        param = a['QuestionIdentifier']
                        if 'objname-ex' in param:
                            continue
                        if 'objname' in param and a['FreeText']:
                            answers.append(a['FreeText'].lower())
                    all_words = all_words.union(answers)
                    
        outname = os.path.join(path_results, "all_words-" + os.path.basename(filename)+".txt")
        with open(outname, "w") as fout:
            fout.write("\n".join(all_words))
            fout.close()
        

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please give a me a config file as argument")
        sys.exit()


    CONFIG = configparser.ConfigParser()
    CONFIG.read(sys.argv[1])

    data_path = os.path.dirname(sys.argv[1])

    MTURK = amt_api.connect_mturk(CONFIG)
    path_published = data_path 
    statuses=["Approved"] # "Submitted"
    all_words = get_words(MTURK, path_published, CONFIG['data']['resultdir'], statuses=statuses)
    
    
    
