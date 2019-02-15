import boto3
import pandas as pd
import time
import json
import configparser
import logging
import sys
#from boto3.mturk.question import ExternalQuestion, QuestionContent, Question, QuestionForm
#from boto3.mturk.question import Overview,AnswerSpecification,SelectionAnswer,FormattedContent
#from boto3.mturk.qualification import Qualifications, Requirement
    

def connect_mturk(config):

    mturk = boto3.client('mturk',
       aws_access_key_id = CONFIG['credentials']['id'],
       aws_secret_access_key = CONFIG['credentials']['key'],
       region_name='us-east-1',
       endpoint_url = CONFIG['endpoint']['url']
    )
    print("Connected")
    return mturk

def make_qualification(mturk):
    
    with open('questionqual.xml', 'r') as myfile:
        question=myfile.read()
    with open('answerqual.xml', 'r') as myfile:
        answer=myfile.read()
        
    print(question)
    print(answer)
        
    response = mturk.create_qualification_type(
        Name='DataProtectionv0.3',
        Keywords='data protection',
        Description='please just read our data protection policy and accept it',
        QualificationTypeStatus='Active',
        RetryDelayInSeconds=123,
        Test=question,
        AnswerKey=answer,
        TestDurationInSeconds=30,
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

    moment = time.strftime("%Y-%b-%d_%H_%M_%S", time.localtime())
    logging.basicConfig(filename=moment+'.log',level=logging.INFO)


    CONFIG = configparser.ConfigParser()
    CONFIG.read(sys.argv[1])

    logging.info("config file: "+sys.argv[1])

    MTURK = connect_mturk(CONFIG)
    
    make_qualification(MTURK)
    