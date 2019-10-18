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

    if 'protectionid' in config['qualification']:
        print("create hit with data protection qualification")

        qlist.append({'QualificationTypeId': config['qualification']['protectionid'], \
                      'Comparator': 'GreaterThanOrEqualTo', \
                      'IntegerValues': [100], \
                      'ActionsGuarded': 'Accept'})

        # # do not allow turkers who have a qualification of excludeprotectionid
        # for excludequali in config['qualification']['excludeprotectionid'].split(","):
        #     qlist.append({'QualificationTypeId': excludequali.strip(),
        #         'Comparator': 'NotIn',
        #         'IntegerValues': [ -100, 100 ],
        #         'ActionsGuarded':'DiscoverPreviewAndAccept'})

    if 'sandbox' in config['endpoint']['url']:
        print("this is sandbox mode, no qualifications needed")
        return qlist

    # qlist.extend([
    #     {'QualificationTypeId': '00000000000000000040',
    #     'Comparator': 'GreaterThanOrEqualTo',
    #     'IntegerValues': [
    #         int(config['qualification']['approvedhits']),
    #         ],
    #     'ActionsGuarded':'PreviewAndAccept'},
    #     {'QualificationTypeId': '000000000000000000L0',
    #     'Comparator': 'GreaterThanOrEqualTo',
    #     'IntegerValues': [
    #         int(config['qualification']['approvalrate']),
    #         ],'ActionsGuarded':'PreviewAndAccept'},
    #     {'QualificationTypeId' : '00000000000000000071',
    #             'Comparator' : 'In',
    #             'LocaleValues' : [
    #                 {'Country':'GB'}, {'Country':'US'},
    #                 {'Country':'AU'}, {'Country':'CA'},
    #                 {'Country':'IE'}, {'Country':'NZ'}
    #                 ],
    #             'ActionsGuarded': 'PreviewAndAccept'}
    #     ])

    return qlist


def create_new_hit(mturk, config, hit_params, hit_quals):
    new_hit = mturk.create_hit(
        HITLayoutId=config['layout']['id'],
        HITLayoutParameters=hit_params,
        Title=config['hit']['title'],
        Reward=config['hit']['reward'],
        Description=config['hit']['description'],
        LifetimeInSeconds=int(config['hit']['lifetime']),
        AssignmentDurationInSeconds=int(config['hit']['assignmentduration']),
        MaxAssignments=int(config['hit']['maxassignments']),
        QualificationRequirements=hit_quals
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

    data_path = os.path.dirname(configfile)

    log_dir = os.path.join(data_path, "logs")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    moment = time.strftime("%Y-%b-%d_%H_%M_%S", time.localtime())
    logging.basicConfig(filename=os.path.join(log_dir, moment + '.log'), level=logging.INFO)

    config = configparser.ConfigParser()
    config.read(configfile)
    config['batch']['initial_row'] = int(config['batch']['initial_row'])
    config['batch']['size'] = int(config['batch']['size'])

    logging.info("config file: " + configfile)

    mturk = amt_api.connect_mturk(config)

    qualifications = get_qualifications(config)

    logging.info("Qualifications:")
    logging.info(qualifications)
    logging.info("max assignments")
    logging.info(config['hit']['maxassignments'])

    all_resulting_HITs = []

    # Load data specified by config
    data = pd.read_csv(config['data']['csvfile'], sep=",")
    print(data.columns.values)

    if config['batch']['initial_row'] >= len(data):
        logging.warning("Check your initial row. I will exit now.")
        sys.exit()

    batch_idx = int(config['batch']['initial_row'] / config['batch']['size'])

    ## Loop through all data rows from starting index, creating HITs, sleep after every batch size
    for row_idx, row in data[config['batch']['initial_row']:].iterrows():
        logging.info("Batch {}, HIT {}".format(batch_idx, row_idx))

        param_list = [{'Name': key, 'Value': value} for key, value in row.to_dict().items()]
        logging.info(param_list)

        hit_data = create_new_hit(mturk, config, param_list, qualifications)
        all_resulting_HITs.append(hit_data)

        logging.info("New hit: " + str(hit_data['HIT']['HITId']))
        logging.info("New hit groupId: " + hit_data['HIT']['HITGroupId'])

        # Sleepy time, now and then:.
        if (row_idx + 1 % config['batch']['size']) == 0:
            outname = os.path.join(data_path, 'created_%s_uptobatch%d.json' % (moment, batch_idx))
            with open(outname, 'w') as outfile:
                json.dump(all_resulting_HITs, outfile)
            outfile.close()
            logging.info("saved batch as: " + outname)

            logging.info("sleeping between batches..." + time.strftime("%Y-%b-%d_%H_%M_%S", time.localtime()))
            time.sleep(int(config['batch']['secondsbetween']))

            batch_idx += 1
            logging.info("wakeup ..." + time.strftime("%Y-%b-%d_%H_%M_%S", time.localtime()))

    outname = os.path.join(data_path, 'created_%s_final.json' % (moment))
    with open(outname, 'w') as outfile:
        json.dump(all_resulting_HITs, outfile)
    outfile.close()
    logging.info("done")

