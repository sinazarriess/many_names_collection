import boto3
import pandas as pd
import glob
import os
import json
import xmltodict
import sys
import time
import configparser



def connect_mturk(config):

    mturk = boto3.client('mturk',
       aws_access_key_id = CONFIG['credentials']['id'],
       aws_secret_access_key = CONFIG['credentials']['key'],
       region_name='us-east-1',
       endpoint_url = CONFIG['endpoint']['url']
    )
    print("Connected")
    return mturk

def get_assignments(mturk,hitid):

    aresponse = mturk.list_assignments_for_hit(
                    HITId=hitid)
    anext = aresponse['NextToken']
    assignments = aresponse['Assignments']

    print("get Hit",hitid)

    while anext:
        nresponse = mturk.list_assignments_for_hit(
                    HITId=hitid,NextToken=anext)
        print(nresponse.keys())
        nextassign = nresponse['Assignments']
        assignments += nextassign

        if 'NextToken' in nresponse:
            anext = nresponse['NextToken']
        else:
            anext = None
            

    return assignments

        




def monitor_submission(mturk,path_published):
    '''ask AMT for results'''
    known_hits = []
    for filename in glob.glob(os.path.join(path_published, '*.json')):
        #print(filename)
        with open(filename, 'r') as handle:
            parsed = json.load(handle)

        hit_results = []

        for item in parsed:
            #print(item)

            if item['HIT']['HITId'] not in known_hits:

                assignments = get_assignments(mturk,item['HIT']['HITId'])

                print("HIT/Assignments",item['HIT']['HITId'],len(assignments))

                known_hits.append(item['HIT']['HITId'])


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Please give a me a config file as argument")
        sys.exit()


    max_steps = 10
    n_steps = 0

    CONFIG = configparser.ConfigParser()
    CONFIG.read(sys.argv[1])


    MTURK = connect_mturk(CONFIG)
    path_published = "."

    while n_steps < max_steps:
        monitor_submission(MTURK,path_published)
        print("*****")
        time.sleep(20)





