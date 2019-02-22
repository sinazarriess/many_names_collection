import boto3
import os
import sys
import configparser



def connect_mturk(config):

    mturk = boto3.client('mturk',
       aws_access_key_id = config['credentials']['id'],
       aws_secret_access_key = config['credentials']['key'],
       region_name='us-east-1',
       endpoint_url = config['endpoint']['url']
    )
    print("Connected")
    return mturk

def get_all_hits(mturk):
    """
    https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/mturk.html#MTurk.Client.list_hits
    """
    # hit_responses' fields: NextToken NumResults HITs ResponseMetadata
    all_hit_ids = []
    hit_responses = MTURK.list_hits()
    while True:
        for hit in hit_responses["HITs"]:
            print(hit["HITId"])
            all_hit_ids.append(hit["HITId"])
        hit_next = hit_responses.get('NextToken', None)
        if hit_next is None:
            break
        hit_responses = MTURK.list_hits(NextToken=hit_next)
    
    return all_hit_ids


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

    MTURK = connect_mturk(CONFIG)
    path_published = data_path 
    
