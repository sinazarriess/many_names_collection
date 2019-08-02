import boto3
import pandas as pd
import glob
import os
import json
import xmltodict
import sys
import configparser

import amt_api

# deprecated, see amt_api
def _get_assignments(mturk,hitid,statuses=['Approved']):
    aresponse = mturk.list_assignments_for_hit(
                    HITId=hitid,
                    AssignmentStatuses=statuses)
    if aresponse["NumResults"] < 1:
        print("\nNo assignments yet for HIT %s." % (str(hitid)))
        return []
    
    anext = aresponse['NextToken']
    assignments = aresponse['Assignments']

    print("\nget Hit",hitid)

    while anext:
        nresponse = mturk.list_assignments_for_hit(
                    HITId=hitid,NextToken=anext,AssignmentStatuses=statuses)
        #print(nresponse.keys())
        nextassign = nresponse['Assignments']
        assignments += nextassign

        if 'NextToken' in nresponse:
            anext = nresponse['NextToken']
        else:
            anext = None
            

    return assignments


def get_results(mturk, path_published, path_results, ass_statuses):
    '''ask AMT for results'''    
    for filename in glob.glob(os.path.join(path_published, '*final.json')):
        print(filename)
        with open(filename, 'r') as handle:
            parsed = json.load(handle)

        hit_results = []
        #comments = []
        comments_outfile = open(os.path.join(path_results, "comments.txt"), 'w')

        for item in parsed:
            #print(item)

            assignments = amt_api.get_assignments(mturk, 
                                                  item['HIT']['HITId'],
                                                  statuses=ass_statuses)

            print("HIT/Assignments",item['HIT']['HITId'],len(assignments))

            if len(assignments) > 0:
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
                                #comments.append(worker_id + "\t" + hit_out_dict["HITId"] + "\t" + a['FreeText'])
                                try:
                                    comments_outfile.write(worker_id + "\t" + hit_out_dict["HITId"] + "\t" + a['FreeText'] + "\n")
                                except UnicodeEncodeError:
                                    comments_outfile.write(worker_id + "\t" + hit_out_dict["HITId"] + "\t" + "UNKNOWN SYMBOLS" + "\n") # workers use emojis

                    #print(answer_out_dict)
                    assignment_out_dict['Answers'] = answer_out_dict
                    hit_out_dict['Assignments'].append(assignment_out_dict)

                hit_results.append(hit_out_dict)

        comments_outfile.close()
        
        infix = "" if "Submitted" not in ass_statuses else "submitted-"
        outname = os.path.join(path_results, "answers-%s" %(infix) + os.path.basename(filename))    
        with open(outname, 'w') as outfile:
            json.dump(hit_results, outfile)
        outfile.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please give a me a config file as argument")
        sys.exit()
    else:
        configfile = sys.argv[1]
    
    CONFIG = configparser.ConfigParser()
    CONFIG.read(configfile)
    data_path = os.path.dirname(configfile)
    MTURK = amt_api.connect_mturk(CONFIG)

    print("Trying to create folder "+CONFIG['data']['resultdir'])
    if os.path.isdir(CONFIG['data']['resultdir']):
        print("Result Directory  "+CONFIG['data']['resultdir']+ " exists! I won't do anything now.")
        sys.exit()

    os.makedirs(CONFIG['data']['resultdir'])

    path_published = data_path
    ass_status = ["Approved"] #["Submitted"]
    get_results(MTURK, path_published, 
                CONFIG['data']['resultdir'], 
                ass_statuses=ass_status)
