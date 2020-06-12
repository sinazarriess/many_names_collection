import boto3
import pandas as pd
import glob
import os
import json
import xmltodict
import sys
import nltk
import configparser


WORDS = set(nltk.corpus.words.words())


def connect_mturk(config):

    mturk = boto3.client('mturk',
       aws_access_key_id = CONFIG['credentials']['id'],
       aws_secret_access_key = CONFIG['credentials']['key'],
       region_name='us-east-1',
       endpoint_url = CONFIG['endpoint']['url']
    )
    print("Connected")
    return mturk

def is_suspicious(answer_words):

    suscount = 0
    for a in answer_words:
        if a not in WORDS:
            suscount += 1
            if suscount >= len(answer_words)/2:
                return True
    return False


def get_assignments(mturk,hitid):

    aresponse = mturk.list_assignments_for_hit(
                    HITId=hitid,
                    AssignmentStatuses=['Submitted'])
    anext = aresponse['NextToken']
    assignments = aresponse['Assignments']

    print("get Hit",hitid)

    while anext:
        nresponse = mturk.list_assignments_for_hit(
                    HITId=hitid,NextToken=anext,AssignmentStatuses=['Submitted'])
        #print(nresponse.keys())
        nextassign = nresponse['Assignments']
        assignments += nextassign

        if 'NextToken' in nresponse:
            anext = nresponse['NextToken']
        else:
            anext = None
            

    return assignments

def review_results(mturk,path_published):
    '''ask AMT for results'''



    for filename in glob.glob(os.path.join(path_published, '*final.json')):
        print(filename)
        with open(filename, 'r') as handle:
            parsed = json.load(handle)

        outapproved_name = filename[:-5]+"_approved.txt"
        outsuspicious_name = filename[:-5]+"_suspicious.txt"

        appr_file = open(outapproved_name,'a')
        susp_file = open(outsuspicious_name,'a')

        hit_results = []

        for item in parsed:
            #print(item)

            assignments = get_assignments(mturk,item['HIT']['HITId'])

            print("HIT/Assignments",item['HIT']['HITId'],len(assignments))

            if len(assignments) > 0:
                
                for assignment in assignments:
                    
                    mturk.approve_assignment(
                    AssignmentId=assignment['AssignmentId'],
                    RequesterFeedback='Thank you for working for us!',
                    OverrideRejection=False
                    )
                    
                    #appr_file.write(";".join(answers)+"\n")

                    #assignment_out_dict['Answers'] = answer_out_dict
                    #hit_out_dict['Assignments'].append(assignment_out_dict)



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


    review_results(MTURK,path_published)
    
