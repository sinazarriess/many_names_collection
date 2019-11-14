#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 15:27:17 2019

@author: u148188
"""
import boto3
import glob
import os
import json
import sys
import configparser
import datetime

import amt_api

def cancel_hits(mturk, path_published):
    known_hits = []
    hits_canceled = 0
    ## CODE FOR PANIC MODE
    # hits_to_cancel = []
    # for hit in hits_to_cancel:
    #     mturk.update_expiration_for_hit(HITId=hit, ExpireAt=datetime.datetime(2018, 1, 1))
    #     hits_canceled += 1
    #     known_hits.append(hit)
    # quit()
    for filename in glob.glob(os.path.join(path_published, '*_final.json')):
        print(filename)
        with open(filename, 'r') as handle:
            parsed = json.load(handle)
        do_cancel = input("Really canceling %d HITs (Y/N)? " % (len(parsed)))
        if do_cancel.lower() != "y":
            pass
        else:
            for item in parsed:
                if item['HIT']['HITId'] not in known_hits:
                    mturk.update_expiration_for_hit(HITId=item['HIT']['HITId'], ExpireAt=datetime.datetime(2018,1,1))
                    hits_canceled += 1
                    known_hits.append(item['HIT']['HITId'])
            print("%d hits canceled." % hits_canceled)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please give a me a config file as argument")
        sys.exit()

    configfile = sys.argv[1];

    max_steps = 9
    n_steps = 0

    CONFIG = configparser.ConfigParser()
    CONFIG.read(configfile)

    basepath = os.path.dirname(configfile)
    out_path = os.path.join(basepath, CONFIG['data']['admindir'])

    MTURK = amt_api.connect_mturk(CONFIG)
    
    cancel_hits(MTURK, out_path)
