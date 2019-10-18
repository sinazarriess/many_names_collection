import boto3
import time
import json
import configparser
import logging
import os
import sys
    
import amt_api


def make_qualification(mturk, question, answer):

    response = mturk.create_qualification_type(
        Name='TaskNotDoneBefore_DataProtv1.3',
        Keywords='data protection',
        Description='Please read our data protection policy and accept it',
        QualificationTypeStatus='Active',
        RetryDelayInSeconds=123,
        Test=question,
        AnswerKey=answer,
        TestDurationInSeconds=300,
    )
    print(response)
    del response['QualificationType']['CreationTime']
    
    with open("protectionqual.json", 'w') as outfile:
        json.dump(response, outfile)
        
    return True

def update_qualification(mturk, qualtypeID, question, answer):
        
    response = mturk.update_qualification_type(
        QualificationTypeId=qualtypeID,
        Description='Please read our data protection policy and accept it',
        QualificationTypeStatus='Active',
        RetryDelayInSeconds=123,
        Test=question,
        AnswerKey=answer,
        TestDurationInSeconds=300,
    )
    print(response)
    del response['QualificationType']['CreationTime']
    
    with open("protectionqual.json", 'w') as outfile:
        json.dump(response, outfile)
        
    return True


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Please give a me a config file as argument")
        sys.exit()
    else:
        configfile = sys.argv[1]

    out_path = os.path.dirname(configfile)

    config = configparser.ConfigParser()
    config.read(configfile)
    print(config)

    basepath = os.path.dirname(configfile)
    out_path = os.path.join(basepath, 'out')
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    config['qualification']['question'] = os.path.join(basepath, config['qualification']['question'])
    config['qualification']['answer'] = os.path.join(basepath, config['qualification']['answer'])
    config['data']['csvfile'] = os.path.join(basepath, config['data']['csvfile'])

    moment = time.strftime("%Y-%b-%d_%H_%M_%S", time.localtime())
    logging.basicConfig(filename=os.path.join(out_path, moment+'.log'),level=logging.INFO)

    logging.info("config file: "+configfile)


    questionqual_xml = config['qualification']['question']
    answerqual_xml = config['qualification']['answer']
    with open(questionqual_xml, 'r') as myfile:
        question = myfile.read()
    with open(answerqual_xml, 'r') as myfile:
        answer = myfile.read()

    print(question)
    print(answer)

    MTURK = amt_api.connect_mturk(config)
    
    if "protectionid" in config["qualification"]:
        sys.stdout.write("Config contains qualification protection (id %s). Updating qualification..." % (config["qualification"]["protectionid"]))
        logging.info("Updating qualification:")
        logging.info(config["qualification"]["protectionid"])
        update_qualification(MTURK, config["qualification"]["protectionid"], question, answer)
        print("Qualification {0} updated.".format(config["qualification"]["protectionid"]))
    else:
        make_qualification(MTURK, question, answer)
        print("Qualification created.")
    
