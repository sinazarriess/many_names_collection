#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Carina
"""
import boto3
import glob
import os
import json
import sys
import configparser
import datetime

import amt_api

def associate_qualification(mturk, qualTypeId, workerId, intValue=90):
    # This will assign the QualificationType
    if False:
        mturk.associate_qualification_with_worker(QualificationTypeId=qualTypeId, 
                                                WorkerId=workerId, 
                                                IntegerValue=100)

    # This will set the QualificationScore from 100 to 90
    print("Associating quali value %d for quali %s of worker %s." % (intValue,
                                                                     qualTypeId,
                                                                     workerId))
    mturk.associate_qualification_with_worker(QualificationTypeId=qualTypeId, 
                                               WorkerId=workerId, 
                                               IntegerValue=intValue,
                                               SendNotification=True)

def list_workers_with_qualification(mturk, qualTypeId):
    response = mturk.list_workers_with_qualification_type(
        QualificationTypeId=qualTypeId,
        #Status='Granted', #'Revoked',
        MaxResults=100
        )
    
    qnext = response['NextToken']
    qualifications = response['Qualifications']

    while qnext:
        nresponse = mturk.list_workers_with_qualification_type(
            QualificationTypeId=qualTypeId,
            #Status='Granted', #'Revoked',
            NextToken=qnext,
            MaxResults=100
            )
        nextqualis = nresponse['Qualifications']
        qualifications += nextqualis

        if 'NextToken' in nresponse:
            qnext = nresponse['NextToken']
        else:
            qnext = None

    return qualifications

def load_known_workers(workerlist_file):
    return [workerId_anonymousId.strip().split()[0] for workerId_anonymousId in open(workerlist_file)]

def print_workers_with_qualification(quals, workers_list):
    round1_releasedate = datetime.datetime(2019, 4, 17)
    num_workers_r0 = 0
    for workerQual in quals:
        if workerQual["WorkerId"] in workers_list:
            print(workerQual["WorkerId"], workerQual["IntegerValue"], 
                    workerQual["Status"], workerQual["GrantTime"], 
                    workerQual["WorkerId"] in workers_list)
            grant_time = amt_api.parse_time(workerQual["GrantTime"])
            print(grant_time, round1_releasedate, workerQual["GrantTime"].date() < round1_releasedate.date())
            num_workers_r0 += 1
        if False:
            #else:
                print(workerQual["WorkerId"], workerQual["IntegerValue"], 
                    workerQual["Status"], workerQual["GrantTime"], 
                    workerQual["WorkerId"] in workers_list)
    print(num_workers_r0, len(quals))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please give a me a config file as argument")
        sys.exit()

    CONFIG = configparser.ConfigParser()
    CONFIG.read(sys.argv[1])

    MTURK = amt_api.connect_mturk(CONFIG)
    
    
    # some statements which assign qualifications to workers of previous rounds, such as to manipulate qualification values / exclude them from participating in current round
    workerIDs_round0 = []
    #workerIDs_round0 = load_known_workers("../confidential_data/mapping_worker2fakeId-v21.csv")
    
    for excludequali in CONFIG['qualification']['excludeprotectionid'].split(","):
        quals = list_workers_with_qualification(MTURK, excludequali)
        print_workers_with_qualification(quals, workerIDs_round0)
    
        #associate_qualification(MTURK, excludequali, workerQual["WorkerId"], intValue=0)
        # 
        #if workerQual["WorkerId"] not in workerIDs_round0:
        #    associate_qualification(MTURK, excludequali, workerQual["WorkerId"])
        
        
