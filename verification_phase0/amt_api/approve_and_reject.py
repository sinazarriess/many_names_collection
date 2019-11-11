import boto3
import glob
import os
import json
import xmltodict
import sys
import nltk
import configparser
import botocore

import amt_api

import csv
import pandas as pd

###############################

MTURK_URL = 'https://mturk-requester.us-east-1.amazonaws.com'

REJECT_FEEDBACK = 'HIT is rejected because too many control items were answered incorrectly.'
APPROVE_FEEDBACK = 'Thank you for working for us!'
BONUS_AMOUNT = '0.15'
BONUS_FEEDBACK = 'We have awarded a bonus of ${} because you did all control items perfectly.'.format(BONUS_AMOUNT)

###############################


def main():

    if len(sys.argv) < 2:
        print("Please give a me a config file as argument")
        sys.exit()

    configfile = sys.argv[1]
    config = configparser.ConfigParser()
    config.read(configfile)

    basepath = os.path.dirname(configfile)
    adminpath = os.path.join(basepath, config['data']['admindir'])
    resultspath = os.path.join(basepath, config['data']['resultdir'])
    assignmentspath = os.path.join(resultspath, 'per_assignment.csv')

    mturk = amt_api.connect_mturk(config)

    ## One-time correction:
    # for id in ['3JC6VJ2SABJSWDTJA7TOO55XXBO5A6']: # '['30MVJZJNHMDMYTYZ73JITKDI82E9J9', '3IHR8NYAM71HNYVLLLSB98OEV9QP4D', '3R2PKQ87NW85A2XNEU2NM542VMYMIB', '3S06PH7KSR4R62VCTUIEBG0M58W1DL', '37C0GNLMHF3MDOW9Z0UV6CR3DHM6DU']:
    #      mturk.approve_assignment(AssignmentId=id,
    #                          RequesterFeedback='Thank you for your feedback.',
    #                          OverrideRejection = True,)
    #      print(id, "approved")
    # quit()

    assignments = pd.read_csv(assignmentspath)

    if 'block_id' in config['qualification']:
        block_id = config['qualification']['block_id']
    elif 'decision2' in assignments and 'block' in assignments['decision2'].tolist():
        r = mturk.create_qualification_type(
            Name='verification_block',  # TODO Generate more robust name
            Keywords='block participation',
            Description='This qualification is assigned to workers who make too many mistakes on quality control items in our HITs, and can be used to prevent further participation.',
            QualificationTypeStatus='Active',
        )
        block_id = r["QualificationType"]["QualificationTypeId"]
        r['QualificationType']['CreationTime'] = r['QualificationType']['CreationTime'].strftime('%d_%m_%y_%I_%M')
        with open(os.path.join(adminpath, "verification_block_{}.json".format(r['QualificationType']['CreationTime'])), 'w') as outfile:
            json.dump(r, outfile)
        print("Qualification for blocking workers created; saved to {}; {}".format(outfile.name, r))
        print("!!! REMINDER: Copy this into your config: block_id = {}".format(block_id))


    reject = assignments.loc[assignments['decision1'] == 'reject']['assignmentid'].tolist()
    accept = assignments.loc[assignments['decision1'] == 'approve']['assignmentid'].tolist()

    print("About to approve {} assignments, reject {}.".format(len(accept), len(reject)))   # TODO misleading; already take 'executed1,2' into account.
    if input("Enter 'Yes!' to continue") != "Yes!":
        quit()

    if not 'executed1' in assignments:
        assignments['executed1'] = False
    if not 'executed2' in assignments:
        assignments['executed2'] = False

    # simply iterate through assignments, assign status as per 'decision', and write 'executed' to the csv.
    for i, row in assignments.iterrows():

        if not row['executed1']:
            if row['decision1'] == 'approve':
                r = mturk.approve_assignment(
                    AssignmentId=row['assignmentid'],
                    RequesterFeedback=APPROVE_FEEDBACK,
                )
                print("{} approved; {}".format(row['assignmentid'], r))
                assignments.at[i, 'executed1'] = True
            elif row['decision1'] == 'reject':
                r = mturk.reject_assignment(
                    AssignmentId=row['assignmentid'],
                    RequesterFeedback=REJECT_FEEDBACK,
                )
                print("{} rejected; {}".format(row['assignmentid'], r))
                assignments.at[i, 'executed1'] = True
            else:
                print("WARNING: No decision specified:", row['assignmentid'])
        if not row['executed2']:
            if row['decision2'] == 'bonus':
                r = mturk.send_bonus(
                    WorkerId=row['workerid'],
                    BonusAmount=BONUS_AMOUNT,
                    AssignmentId=row['assignmentid'],
                    Reason=BONUS_FEEDBACK,
                    UniqueRequestToken=row['assignmentid'],
                )
                print("{} bonused; {}".format(row['assignmentid'], r))
                assignments.at[i, 'executed2'] = True
            elif row ['decision2'] == 'block':
                try:
                    r = mturk.associate_qualification_with_worker(
                        QualificationTypeId=block_id,
                        WorkerId=row['workerid'],
                        SendNotification=False,
                    )
                    print("{}: {} blocked; {}".format(row['assignmentid'], row['workerid'], r))
                    assignments.at[i, 'executed2'] = True
                except botocore.exceptions.ClientError:
                    print("{}: {} was already blocked".format(row['assignmentid'], row['workerid']))
                    assignments.at[i, 'executed2'] = True
    assignments.to_csv(assignmentspath, index=False)
    print("Updated 'executed' column in", assignmentspath)


if __name__ == "__main__":
    main()
