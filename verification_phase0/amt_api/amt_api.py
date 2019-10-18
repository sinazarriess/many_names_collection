import boto3
import os
import sys
import configparser
import datetime


def connect_mturk(config):
    mturk = boto3.client('mturk',
       aws_access_key_id = config['credentials']['id'],
       aws_secret_access_key = config['credentials']['key'],
       region_name='us-east-1',
       endpoint_url = config['endpoint']['url']
    )
    sys.stderr.write("Connected")
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

def get_assignments(mturk, hitid, statuses):
    aresponse = mturk.list_assignments_for_hit(
                    HITId=hitid,
                    AssignmentStatuses=statuses)
    if aresponse["NumResults"] < 1:
        print("\nNo assignments yet for HIT %s." % (str(hitid)))
        return []
    
    anext = aresponse['NextToken']
    assignments = aresponse['Assignments']

    print("\nget Hit",hitid)

    while anext:
        nresponse = mturk.list_assignments_for_hit(
                    HITId=hitid,NextToken=anext,AssignmentStatuses=statuses)
        print(nresponse.keys())
        nextassign = nresponse['Assignments']
        assignments += nextassign

        if 'NextToken' in nresponse:
            anext = nresponse['NextToken']
        else:
            anext = None
            
    return assignments

def parse_time(date_string):
    """
    >>> date_string = "2019-04-17 11:41:34+02:00"
    >>> t = parse_time(date_string)
    datetime(2019, 4, 17)
    """
    if isinstance(date_string, datetime.datetime):
        return date_string
    date_string = [int(a) for a in date_string.split()[0].split("-")]
    return datetime.datetime(date_string[0], date_string[1], date_string[2])

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
    
