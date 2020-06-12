import boto3
import pandas as pd
import glob
import os
import json
import xmltodict
import sys
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
                    HITId=hitid,
                    AssignmentStatuses=['Approved'])
    anext = aresponse['NextToken']
    assignments = aresponse['Assignments']

    print("get Hit",hitid)

    while anext:
        nresponse = mturk.list_assignments_for_hit(
                    HITId=hitid,NextToken=anext,AssignmentStatuses=['Approved'])
        #print(nresponse.keys())
        nextassign = nresponse['Assignments']
        assignments += nextassign

        if 'NextToken' in nresponse:
            anext = nresponse['NextToken']
        else:
            anext = None
            

    return assignments


def get_results(mturk,path_published,path_results):
    '''ask AMT for results'''
    for filename in glob.glob(os.path.join(path_published, '*final.json')):
        print(filename)
        with open(filename, 'r') as handle:
            parsed = json.load(handle)

        hit_results = []
        comments = []

        for item in parsed:
            #print(item)

            assignments = get_assignments(mturk,item['HIT']['HITId'])

            print("HIT/Assignments",item['HIT']['HITId'],len(assignments))

            if len(assignments) > 0:
                #print("Found assignments")
                #print(assignments)
                
                hit_out_dict = {'HITId':item['HIT']['HITId'], \
                                'Assignments':[]
                               }
                if 'Images' in item['HIT']:
                    hit_out_dict['Images']= item['HIT']['Images']

                for assignment in assignments:
                    assignment_out_dict = {}
                    assignment_out_dict['WorkerId'] = assignment['WorkerId']
                    assignment_out_dict['AssignmentId'] = assignment['AssignmentId']



                    answer_out_dict = {}

                    #print("assignment",assignment.keys())
                    worker_id = assignment['WorkerId']
                    answer_dict = xmltodict.parse(assignment['Answer'])
                    #print(answer_dict['QuestionFormAnswers']['Answer'])
                    #print("Worker,",worker_id)
                    #print(answer_dict['QuestionFormAnswers'].keys())

                    #print("Assignment")
                    for a in answer_dict['QuestionFormAnswers']['Answer']:
                        #print(dict(a))
                        param = a['QuestionIdentifier']
                        if not 'objname-ex' in param:
                            imid = param[-1]
                            if not imid in answer_out_dict:
                                answer_out_dict[imid] = {}
                            answer_out_dict[imid][param] = a['FreeText']

                        if 'comments' in param:
                            if a['FreeText']:
                                comments.append(a['FreeText'])

                    #print(answer_out_dict)
                    assignment_out_dict['Answers'] = answer_out_dict
                    hit_out_dict['Assignments'].append(assignment_out_dict)

                hit_results.append(hit_out_dict)


        outname = os.path.join(path_results, filename)
        with open(outname, 'w') as outfile:
            json.dump(hit_results, outfile)
        outfile.close()

        outname = os.path.join(path_results, "comments.txt")
        with open(outname, 'w') as outfile:
            outfile.write("\n".join(comments))
        outfile.close()


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Please give a me a config file as argument")
        sys.exit()


    CONFIG = configparser.ConfigParser()
    CONFIG.read(sys.argv[1])


    MTURK = connect_mturk(CONFIG)
    

    print("trying to create folder "+CONFIG['data']['resultdir'])
    if os.path.isdir(CONFIG['data']['resultdir']):
        print("Result Directory  "+CONFIG['data']['resultdir']+ " exists! I won't do anything now.")
        sys.exit()

    os.makedirs(CONFIG['data']['resultdir'])


    path_published = "."
    get_results(MTURK,path_published,CONFIG['data']['resultdir'])
