import boto3
import glob
import os
import json
import sys
import time
import configparser

import amt_api

def monitor_submission(mturk, path_published, total_hits, statuses=["Submitted"]):
    '''ask AMT for results'''
    known_hits = []
    total_assignments = 0
    
    publication_files = glob.glob(os.path.join(path_published, '*_final.json'))
    if len(publication_files) == 0:
        # TODO: only take latest, since it contains all previous ones, too
        publication_files = glob.glob(os.path.join(path_published, '*_uptobatch*.json'))
        
    for filename in publication_files:
        print(filename)
        with open(filename, 'r') as handle:
            parsed = json.load(handle)
        for item in parsed:
            if item['HIT']['HITId'] not in known_hits:
                assignments = amt_api.get_assignments(mturk,
                                                      item['HIT']['HITId'],
                                                      statuses=statuses)
                print("HIT/Assignments",item['HIT']['HITId'], len(assignments))
                total_assignments += len(assignments)
                known_hits.append(item['HIT']['HITId'])
    print("%.2f%% (%d/%d) assignments." % ((total_assignments/total_hits*100), total_assignments, total_hits))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please give a me a config file as argument")
        sys.exit()

    max_steps = 9
    n_steps = 0

    CONFIG = configparser.ConfigParser()
    CONFIG.read(sys.argv[1])

    total_hits = int(CONFIG["batch"]["total"]) * int(CONFIG["batch"]["size"]) * int(CONFIG["hit"]["maxassignments"])
    data_path = os.path.dirname(sys.argv[1])

    MTURK = amt_api.connect_mturk(CONFIG)
    path_published = data_path
    statuses = ["Submitted", "Approved"]
    while n_steps < max_steps:
        monitor_submission(MTURK, path_published, total_hits, statuses=statuses)
        print("*****")
        time.sleep(200)





