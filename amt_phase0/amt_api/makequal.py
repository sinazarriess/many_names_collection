import boto3
import pandas as pd
import glob
import os
import json
import xmltodict

MTURK_SANDBOX = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'

mturk = boto3.client('mturk',
   aws_access_key_id = "AKIAIU2JHDHFOAWBHSHQ",
   aws_secret_access_key = "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",
   region_name='us-east-1',
   endpoint_url = MTURK_SANDBOX
)

#mturk.delete_qualification_type(QualificationTypeId='3RFK03H8OLM4TXQY9NPLUSYCXKPM0D')

#mturk.list_qualification_types(MustBeRequestable=True)

QUAL = mturk.create_qualification_type(
    Name='HITCOUNT_NAMEIT',
    Description='monitoring the number of hits of a worker',
    QualificationTypeStatus='Active',
    AutoGranted = True,
    AutoGrantedValue = 0
)

print(QUAL)
qualdict = {'QualificationTypeId':QUAL['QualificationType']['QualificationTypeId'],\
'Name':QUAL['QualificationType']['Name']}
#print(QUAL)
#mturk.list_qualification_types(MustBeRequestable=True)


with open('./pubsand/qualification.json', 'w') as outfile:
    json.dump(qualdict, outfile)