import boto3
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
       aws_access_key_id = config['credentials']['id'],
       aws_secret_access_key = config['credentials']['key'],
       region_name='us-east-1',
       endpoint_url = config['endpoint']['url']
    )
    print("Connected")
    return mturk

def get_assignments(mturk,hitid):

    aresponse = mturk.list_assignments_for_hit(
                    HITId=hitid,
                    AssignmentStatuses=['Submitted'])
    if aresponse["NumResults"] < 1:
        print("\nNo assignments yet for HIT %s." % (str(hitid)))
        return []
    
    anext = aresponse['NextToken']
    assignments = aresponse['Assignments']

    print("\nget Hit",hitid)

    while anext:
        nresponse = mturk.list_assignments_for_hit(
                    HITId=hitid,NextToken=anext,AssignmentStatuses=['Submitted'])
        print(nresponse.keys())
        nextassign = nresponse['Assignments']
        assignments += nextassign

        if 'NextToken' in nresponse:
            anext = nresponse['NextToken']
        else:
            anext = None
            
    return assignments

def is_suspicious(answer_words):

    suscount = 0
    for a in answer_words:
        if a not in WORDS:
            suscount += 1
            if suscount >= len(answer_words)/2:
                return True
    return False

def review_results(mturk,path_published):
    '''ask AMT for results'''
    for filename in glob.glob(os.path.join(path_published, '*final.json')):
        print(filename)
        with open(filename, 'r') as handle:
            parsed = json.load(handle)

        outapproved_name = filename[:-5]+"_approved.txt"
        outsuspicious_name = filename[:-5]+"_suspicious.txt"

        appr_file = open(outapproved_name,'w')
        susp_file = open(outsuspicious_name,'w')

        for item in parsed:
            #print(item)

            assignments = get_assignments(mturk,item['HIT']['HITId'])

            print("HIT/Assignments",item['HIT']['HITId'],len(assignments))

            if len(assignments) > 0:
                
                for assignment in assignments:
                    answer_dict = xmltodict.parse(assignment['Answer'])
                    #print(answer_dict['QuestionFormAnswers']['Answer'])
                    #print("Worker,",worker_id)
                    #print(answer_dict['QuestionFormAnswers'].keys())

                    print("Assignment")
                    answers = []
                    for a in answer_dict['QuestionFormAnswers']['Answer']:
                        param = a['QuestionIdentifier']
                        if not 'objname-ex' in param:
                            if 'objname' in param:
                                if a['FreeText']:
                                    answers.append(a['FreeText'])


                    
                    info = "**HIT %s, Assignment %s, Worker %s**\n" % \
                    (item['HIT']['HITId'],assignment['AssignmentId'], assignment['WorkerId'])

                    if is_suspicious(answers):
                        susp_file.write(info)
                        susp_file.write(";".join(answers)+"\n")

                    else:
                        mturk.approve_assignment(
                        AssignmentId=assignment['AssignmentId'],
                        RequesterFeedback='Thank you for working for us!',
                        OverrideRejection=False
                        )
                        appr_file.write(info)
                        appr_file.write(";".join(answers)+"\n")

                    #assignment_out_dict['Answers'] = answer_out_dict
                    #hit_out_dict['Assignments'].append(assignment_out_dict)



if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Please give a me a config file as argument")
        sys.exit()
    
    data_path = os.path.dirname(sys.argv[1])
    
    CONFIG = configparser.ConfigParser()
    CONFIG.read(sys.argv[1])
    
    MTURK = connect_mturk(CONFIG)
    
    path_published = data_path
    review_results(MTURK, path_published)
    
