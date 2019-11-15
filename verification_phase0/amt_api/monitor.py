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


def monitor_from_list_of_hitids():
    total_assignments = 0
    known_hits = []
    manually_monitor = [
    ]
    for hitid in manually_monitor:
        assignments = amt_api.get_assignments(mturk,
                                              hitid,
                                              statuses=statuses)
        print("HIT/Assignments", hitid, len(assignments))
        total_assignments += len(assignments)
        known_hits.append(hitid)

    print(len(manually_monitor), 'hits;', total_assignments, 'assigments')


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please give a me a config file as argument")
        sys.exit()

    max_steps = 9
    n_steps = 0

    configfile = sys.argv[1]

    config = configparser.ConfigParser()
    config.read(configfile)

    # TODO This doesn't work if there are multiple json files.
    total_hits = int(config["batch"]["total_rows"]) * int(config["hit"]["maxassignments"])

    basepath = os.path.dirname(configfile)
    out_path = os.path.join(basepath, config['data']['admindir'])

    mturk = amt_api.connect_mturk(config)

    print("I have $" + mturk.get_account_balance()['AvailableBalance'] + " in my account")

    # hits = amt_api.get_all_hits(MTURK)

    statuses = ["Submitted", "Approved"]    # TODO Extend functionality; print statuses of HITs; annotations...
    while n_steps < max_steps:
        monitor_submission(mturk, out_path, total_hits, statuses=statuses)
        print("*****")
        time.sleep(200)





