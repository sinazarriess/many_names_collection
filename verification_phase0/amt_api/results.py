import boto3
import pandas as pd
import glob
import os
import json
import xmltodict
import sys
import configparser

import amt_api


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

    config = configparser.ConfigParser()
    config.read(configfile)

    basepath = os.path.dirname(configfile)
    out_path = os.path.join(basepath, config['data']['admindir'])
    result_path = os.path.join(basepath, config['data']['resultdir'])

    mturk = amt_api.connect_mturk(config)

    print("Trying to create folder " + result_path)
    if os.path.isdir(result_path):
        print("Result Directory  " + result_path + " exists! I won't do anything now.")
        sys.exit()

    os.makedirs(result_path)

    ass_status = ["Submitted"]
    # ass_status = ["Approved"]
    get_results(mturk, out_path, result_path, ass_statuses=ass_status)
