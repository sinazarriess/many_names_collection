import boto3
import pandas as pd
import time
import json
import configparser
import logging
import sys

    

def connect_mturk(config):

    mturk = boto3.client('mturk',
       aws_access_key_id = CONFIG['credentials']['id'],
       aws_secret_access_key = CONFIG['credentials']['key'],
       region_name='us-east-1',
       endpoint_url = CONFIG['endpoint']['url']
    )
    print("Connected")
    return mturk


def make_new_hit(n_images,imgdf,img_index):

    param_list = []
    out_img_dict = {}
    
    for him in range(n_images):

        param_dict = {}
        param_dict['Name'] = 'img_%d_url'%him

        if img_index >= len(imgdf):
            img_index = 0
            logging.warning("Reset image index to 0!")

        img_url = 'http://www.coli.uni-saarland.de/~carina/object_naming/amt_images/%d_%d_%s.png' % \
        (imgdf.iloc[img_index]['image_id'],\
            imgdf.iloc[img_index]['object_id'],
            imgdf.iloc[img_index]['sample_type'])
        param_dict['Value'] = img_url

        out_img_dict[str(him)] = (str(imgdf.iloc[img_index]['image_id']),\
            str(imgdf.iloc[img_index]['object_id']),
            imgdf.iloc[img_index]['sample_type'],
            imgdf.iloc[img_index]['synset'], img_url)


        img_index += 1
        param_list.append(param_dict)

    return param_list,out_img_dict,img_index

def publish_new_hit(mturk,config,hit_params):

    new_hit = mturk.create_hit(
    HITLayoutId    = config['layout']['id'],
    HITLayoutParameters = hit_params,
    Title = config['hit']['title'],
    Reward = config['hit']['reward'],
    Description = config['hit']['description'],
    LifetimeInSeconds = int(config['hit']['lifetime']),
    AssignmentDurationInSeconds = int(config['hit']['assignmentduration']),
    MaxAssignments = int(config['hit']['maxassignments']),
    QualificationRequirements=[
        {
        'QualificationTypeId': '00000000000000000040',
        'Comparator': 'GreaterThanOrEqualTo',
        'IntegerValues': [
            int(config['qualification']['approvedhits']),
        ],
        'ActionsGuarded':'PreviewAndAccept'
        }
    ]
    )

    print("A new HIT has been created. You can preview it here:")
    print("https://workersandbox.mturk.com/mturk/preview?groupId=" + new_hit['HIT']['HITGroupId'])
    print("HITID = " + new_hit['HIT']['HITId'] + " (Use to Get Results)")
    del new_hit['HIT']['Question']
    del new_hit['HIT']['CreationTime']
    del new_hit['HIT']['Expiration']

    new_hit['HIT']['params'] = hit_params
    #new_hit['Images'] = out_img_dict
    #print(out_img_dict)

    return new_hit


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


    RESULTS = []
    LINKS = []

    
    imgdf = pd.read_csv(CONFIG['data']['csvfile'],sep="\t")
    print(imgdf.columns.values)

    img_index = int(CONFIG['batch']['initial_img'])

    if img_index >= len(imgdf):
        logging.warning("Check your image index. I will exit now.")
        sys.exit()

    for batch_index in range(int(CONFIG['batch']['total'])):
        logging.info("Time: "+time.strftime("%Y-%b-%d_%H_%M_%S", time.localtime()))
        logging.info("Batch: "+str(batch_index))

        for hit_index in range(int(CONFIG['batch']['size'])):
            logging.info("current img index: "+str(img_index))

            hitparam,hitimages,img_index = make_new_hit(int(CONFIG['hit']['nimages']),imgdf,img_index)
            hitdict = publish_new_hit(MTURK,CONFIG,hitparam)
            hitdict['Images'] = hitimages

            logging.info("New hit: "+str(hitdict['HIT']['HITId']))
            logging.info("New hit groupId: "+ hitdict['HIT']['HITGroupId'])
            logging.info("next img index: "+str(img_index))
            

            RESULTS.append(hitdict)
            

        outname = 'created_%s_uptobatch%d.json' % (moment,batch_index)
        with open(outname, 'w') as outfile:
            json.dump(RESULTS, outfile)
        outfile.close()
        logging.info("saved batch as: "+outname)

        logging.info("sleeping between batches..."+time.strftime("%Y-%b-%d_%H_%M_%S", time.localtime()))
        time.sleep(int(CONFIG['batch']['secondsbetween']))
        logging.info("wakeup ..."+time.strftime("%Y-%b-%d_%H_%M_%S", time.localtime()))

    outname = 'created_%s_final.json' % (moment)
    with open(outname, 'w') as outfile:
        json.dump(RESULTS, outfile)
    outfile.close()
    logging.info("done")

