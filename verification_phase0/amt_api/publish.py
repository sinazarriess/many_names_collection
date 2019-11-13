import pandas as pd
import time
import json
import configparser
import logging
import os
import sys

import amt_api


def get_qualifications(config):
    qlist = []

    if 'sandbox' in config['endpoint']['url']:
        print("this is sandbox mode, no qualifications needed")
        return qlist

    if 'entry_id' in config['qualification']:
        print("create hit with data protection qualification")

        qlist.append({'QualificationTypeId': config['qualification']['entry_id'],
                      'Comparator': 'GreaterThanOrEqualTo',
                      'IntegerValues': [100],
                      'ActionsGuarded': 'Accept'})

        # do not allow turkers who have a qualification that blocks them from participating
        if 'block_id' in config['qualification']:
            qlist.append({'QualificationTypeId': config['qualification']['block_id'],
                'Comparator': 'DoesNotExist',
                'ActionsGuarded':'DiscoverPreviewAndAccept'})

        qlist.extend([
            {'QualificationTypeId': '00000000000000000040',
            'Comparator': 'GreaterThanOrEqualTo',
            'IntegerValues': [
                int(config['qualification']['approvedhits']),
                ],
            'ActionsGuarded':'PreviewAndAccept'},
            {'QualificationTypeId': '000000000000000000L0',
            'Comparator': 'GreaterThanOrEqualTo',
            'IntegerValues': [
                int(config['qualification']['approvalrate']),
                ],'ActionsGuarded':'PreviewAndAccept'},
            {'QualificationTypeId' : '00000000000000000071',
                    'Comparator' : 'In',
                    'LocaleValues' : [
                        {'Country':'GB'}, {'Country':'US'},
                        {'Country':'AU'}, {'Country':'CA'},
                        {'Country':'IE'}, {'Country':'NZ'}
                        ],
                    'ActionsGuarded': 'PreviewAndAccept'}
            ])

    return qlist


def create_new_hit(mturk, config, hit_params, hit_quals, annotation):
    new_hit = mturk.create_hit(
        HITLayoutId=config['layout']['id'],
        HITLayoutParameters=hit_params,
        Title=config['hit']['title'],
        Reward=config['hit']['reward'],
        Description=config['hit']['description'],
        LifetimeInSeconds=eval(config['hit']['lifetime']),
        Keywords=config['hit']['keywords'],
        AssignmentDurationInSeconds=eval(config['hit']['assignmentduration']),
        MaxAssignments=int(config['hit']['maxassignments']),
        QualificationRequirements=hit_quals,
        RequesterAnnotation=annotation,
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

    new_hit['HIT']['params'] = hit_params
    # new_hit['Images'] = out_img_dict
    # print(out_img_dict)

    return new_hit


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Please give a me a config file as argument")
        sys.exit()
    else:
        configfile = sys.argv[1]

    config = configparser.ConfigParser()
    config.read(configfile)

    input("This will publish HITs based on {}. Press any key to continue.".format(os.path.basename(configfile)))    # TODO Compute how many HITs and money.
    if not 'sandbox' in config['endpoint']['url']:
        input("WARNING: This is not a drill! Will cost actual money!")

    basepath = os.path.dirname(configfile)
    out_path = os.path.join(basepath,config['data']['admindir'])
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    config['qualification']['question'] = os.path.join(basepath, config['qualification']['question'])
    config['qualification']['answer'] = os.path.join(basepath, config['qualification']['answer'])
    config['data']['csvfile'] = os.path.join(basepath, config['data']['csvfile'])

    moment = time.strftime("%Y-%b-%d_%H_%M_%S", time.localtime())
    logging.basicConfig(filename=os.path.join(out_path, moment + '.log'), level=logging.INFO)

    logging.info("config file: " + configfile)

    mturk = amt_api.connect_mturk(config)

    initial_row = int(config['batch']['initial_row'])
    batch_size = int(config['batch']['size'])
    total_rows = int(config['batch']['total_rows'])

    qualifications = get_qualifications(config)

    logging.info("Qualifications:")
    logging.info(qualifications)
    logging.info("max assignments")
    logging.info(config['hit']['maxassignments'])

    all_resulting_HITs = []

    # Load data specified by config
    data = pd.read_csv(config['data']['csvfile'], sep=",", keep_default_na=False)

    if total_rows == -1:
        total_rows = len(data)

    if initial_row >= len(data):
        logging.warning("Check your initial row. I will exit now.")
        sys.exit()

    batch_idx = initial_row / batch_size

    ## Loop through all data rows from starting index, creating HITs, sleep after every batch size
    n_hits_published = 0
    for row_idx, row in data[initial_row:initial_row+total_rows].iterrows():
        logging.info("Batch {}, HIT {}".format(batch_idx, row_idx))

        param_list = [{'Name': key, 'Value': str(value)} for key, value in row.to_dict().items()]

        # Add unique turker id, or else empty string
        unique_turker_id = config["qualification"]["unique_turker_id"] if "unique_turker_id" in config["qualification"] else ""
        unique_turker_limit = config["qualification"]["unique_turker_limit"] if "unique_turker_limit" in config["qualification"] else "99999"
        param_list = [{'Name': 'unique_turker_id', 'Value': unique_turker_id}, {'Name': 'unique_turker_limit', 'Value': unique_turker_limit}] + param_list
        logging.info(param_list)

        hit_data = create_new_hit(mturk, config, param_list, qualifications, config['data']['csvfile'] + '_' + str(row_idx))
        n_hits_published += 1
        all_resulting_HITs.append(hit_data)

        logging.info("New hit: " + str(hit_data['HIT']['HITId']))
        logging.info("New hit groupId: " + hit_data['HIT']['HITGroupId'])

        # Sleepy time, now and then:.
        if n_hits_published != total_rows and (n_hits_published % batch_size) == 0:
            outname = os.path.join(out_path, 'created_%s_uptobatch%d.json' % (moment, batch_idx))
            with open(outname, 'w') as outfile:
                json.dump(all_resulting_HITs, outfile)
            outfile.close()
            logging.info("saved batch as: " + outname)

            logging.info("sleeping between batches..." + time.strftime("%Y-%b-%d_%H_%M_%S", time.localtime()))
            time.sleep(int(config['batch']['secondsbetween']))

            batch_idx += 1
            logging.info("wakeup ..." + time.strftime("%Y-%b-%d_%H_%M_%S", time.localtime()))

    outname = os.path.join(out_path, 'created_%s_final.json' % (moment))
    with open(outname, 'w') as outfile:
        json.dump(all_resulting_HITs, outfile)
    outfile.close()
    logging.info("done")

