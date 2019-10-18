#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Carina
"""
import boto3
import glob
import os
import json
import sys
import configparser
import datetime

import amt_api

def block_worker(mturk, workerId, message):
    print("Blocking worker %s." % (workerId))
    mturk.create_worker_block(WorkerId=workerId, 
                              Reason=message)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please give a me a config file as argument")
        sys.exit()

    CONFIG = configparser.ConfigParser()
    CONFIG.read(sys.argv[1])

    MTURK = amt_api.connect_mturk(CONFIG)
    
    workers_to_block = []
    message = "You are blocked because XXX"
    for workerid in workers_to_block:
        block_worker(MTURK, workerid, message)
    
        
