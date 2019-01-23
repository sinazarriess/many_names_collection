import pandas as pd
import glob
import os
import json
import xmltodict
import sys

for filename in glob.glob(os.path.join("results1/", '*.json')):
        print(filename)
        with open(filename, 'r') as handle:
            parsed = json.load(handle)

PATH_PUBLISHED = '.'

for filename in glob.glob(os.path.join(PATH_PUBLISHED, '*.json')):
    print(filename)
    with open(filename, 'r') as handle:
        parsed = json.load(handle)
    print(len(parsed))