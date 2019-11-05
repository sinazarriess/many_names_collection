import pandas as pd
import time
import json
import configparser
import logging
import os
import sys

import amt_api


def create_new_hit(mturk, config, hit_quals):
    new_hit = mturk.create_hit(
        Title=config['hit']['title'],
        Reward=config['hit']['reward'],
        Description=config['hit']['description'],
        LifetimeInSeconds=eval(config['hit']['lifetime']),
        Keywords=config['hit']['keywords'],
        AssignmentDurationInSeconds=eval(config['hit']['assignmentduration']),
        MaxAssignments=int(config['hit']['maxassignments']),
        QualificationRequirements=hit_quals,
        RequesterAnnotation=config['hit']['worker'],
        Question=open(config['hit']['question']).read(),
        AutoApprovalDelayInSeconds=eval(config['hit']['autoapprovaldelay']),
    )

    print("A new HIT has been created. You can preview it here:")
    if 'sandbox' in config['endpoint']['url']:
        print("https://workersandbox.mturk.com/mturk/preview?groupId=" + new_hit['HIT']['HITGroupId'])
    else:
        print("https://worker.mturk.com/mturk/preview?groupId=" + new_hit['HIT']['HITGroupId'])
    print("HITID = " + new_hit['HIT']['HITId'] + " (Use to Get Results)")
    del new_hit['HIT']['Question']
    del new_hit['HIT']['CreationTime']
    del new_hit['HIT']['Expiration']


    return new_hit


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Please give a me a config file as argument")
        sys.exit()
    else:
        configfile = sys.argv[1]

    logging.info("config file: " + configfile)

    config = configparser.ConfigParser()
    config.read(configfile)


    input("This will publish HITs based on {}. Press any key to continue.".format(os.path.basename(configfile)))    # TODO Compute how many HITs and money.
    if not 'sandbox' in config['endpoint']['url']:
        input("WARNING: This is not a drill! Will cost actual money!")

    # Path structure
    basepath = os.path.dirname(configfile)
    out_path = os.path.join(basepath,config['data']['admindir'])
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Special-purpose stuff removed from config:
    config['hit']['question'] = os.path.join(basepath, config['hit']['question'])
    config['hit']['title'] = "HIT for worker {}.".format(config['hit']['worker'])
    config['hit']['description'] = "This HIT is only for worker {}.".format(config['hit']['worker'])
    config['hit']['maxassignments'] = "1"

    moment = time.strftime("%Y-%b-%d_%H_%M_%S", time.localtime())
    logging.basicConfig(filename=os.path.join(out_path, moment + '.log'), level=logging.INFO)

    # Procedure: Connect to MTurk, create qualification, assign qualification to worker, publish HIT
    mturk = amt_api.connect_mturk(config)

    # Create qualification
    r = mturk.create_qualification_type(
        Name='unique_worker_{}'.format(config['hit']['worker']),  # TODO Generate more robust name
        Keywords='worker',
        Description='This qualification is exclusively assigned to worker {}.'.format(config['hit']['worker']),
        QualificationTypeStatus='Active',
    )
    qualification_id = r["QualificationType"]["QualificationTypeId"]
    r['QualificationType']['CreationTime'] = r['QualificationType']['CreationTime'].strftime('%d_%m_%y_%I_%M')
    with open(os.path.join(out_path, "qual_unique_worker_{}_{}.json".format(config['hit']['worker'], r['QualificationType'][ 'CreationTime'])), 'w') as outfile:
        json.dump(r, outfile)
    print("Qualification {} created for unique worker; saved to {}; {}".format(
        r["QualificationType"]["QualificationTypeId"], outfile.name, r))

    # Assign qualification to worker
    r = mturk.associate_qualification_with_worker(
        QualificationTypeId=qualification_id,
        WorkerId=config['hit']['worker'],
        SendNotification=False,
    )
    print("Qualification {} assigned to worker {}; {}".format(qualification_id, config['hit']['worker'], r))

    # Setup the HIT
    qualifications = [({'QualificationTypeId': qualification_id,
                        'Comparator': 'Exists',
                        'ActionsGuarded':'DiscoverPreviewAndAccept'})]

    all_resulting_HITs = []

    # TODO loop if multiple workers specified?
    hit_data = create_new_hit(mturk, config, qualifications)

    all_resulting_HITs.append(hit_data)

    logging.info("New hit: " + str(hit_data['HIT']['HITId']))
    logging.info("New hit groupId: " + hit_data['HIT']['HITGroupId'])

    outname = os.path.join(out_path, 'created_%s_final.json' % (moment))
    with open(outname, 'w') as outfile:
        json.dump(all_resulting_HITs, outfile)
    outfile.close()
    logging.info("done")

