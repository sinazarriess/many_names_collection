
import os
import re
import sys

import pandas

USRNAME ='u148188' # 'carina'


import annotation_utils

data_spec = ["i_corpus", "r_corpus", 
             "image_id", "i_width", "i_height",
             "region_id", "bb", "cat",
             "rex_id", "refexp", 
             "sent_id"]

def flickr2dataframe(fpath, out_fpath=None):
    all_annos = dict()
    if "Sentences" in fpath:
        for fname in os.listdir(fpath):
            img_annos = get_image_annos(fname)
            for sent_id, region_id, reg_cat, refExp in extract_refExpFromSentences(fpath+"/"+fname):
                tagged = []
                #tagged = tag_refExp(refExp, pos_tagger)
                region_anno = {
                    "region_id": region_id,
                    "refexp": refExp,
                    "cat": reg_cat,
                    "tagged": tagged,
                    "sent_id": sent_id}
                region_anno.update(img_annos)
                add_region_annos(region_anno,
                                 all_annos)
            
    refdf = pandas.DataFrame(data=all_annos)
    refdf['rex_id'] = list(range(refdf.shape[0]))
    if out_fpath:
        refdf.to_json(out_fpath, compression='gzip', orient='split')

def flickr2dataframe_bboxes(fpath, out_fpath=None):
    import xml.etree.ElementTree
    all_annos = dict()
    
    if "Annotations" in fpath:
        for fname in os.listdir(fpath):
            img_annos = get_image_annos(fname)
            e = xml.etree.ElementTree.parse(os.path.join(fpath, fname)).getroot()
            width = int(e.find('size').find('width').text)
            height = int(e.find('size').find('height').text)
            img_annos["i_width"] = width
            img_annos["i_height"] = height
            for bbox, reg_id in extract_bboxFromAnnotations(e):
                region_anno = {
                    'region_id': reg_id,
                    'bb': bbox}
                region_anno.update(img_annos)
                add_region_annos(region_anno,
                                 all_annos)
            box_df = pandas.DataFrame(data=all_annos)
            
    if out_fpath:
        box_df.to_json(out_fpath, compression='gzip', orient='split')
    

def extract_refExpFromSentences(inf):
  # (skip the [/EN# beginning each annotation)
  sent_id = 0
  for line in open(inf):
    for entity_id, entity_cat, entityName in re.findall("\[\/EN\#([^\s/]+)/([^\s]+)\s([^\]]+?)\]", line.strip()):
      yield sent_id, int(entity_id), entity_cat, entityName.lower()
    sent_id += 1


# TODO: double-check order of points (format of bbox specs)
def extract_bboxFromAnnotations(e):
    for obj in e.findall('object'):
        for region_id_el in obj.findall('name'):
            region_id = region_id_el.text
            bbox = obj.find('bndbox')
            if bbox is not None:
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                yield [xmin, ymin, xmax, ymax], region_id

      
def get_image_annos(fname):
    image_id = re.sub("([0-9]+)\.(txt|xml)$", r"\1", os.path.basename(fname))
    return {"i_corpus": 1, 
            "r_corpus": "flickr",
            "image_id": image_id}

def add_region_annos(reg_annos, all_annos):
    for d_spec in data_spec:
        if d_spec in reg_annos:
            all_annos.setdefault(d_spec, []).append(reg_annos[d_spec])

if __name__=="__main__":
    fpath = "/media/%s/Carina_2017/UdS/data/Flickr30kEntities/Sentences/" % (USRNAME)
    if len(sys.argv) > 1:
        fpath = sys.argv[1]
        if len(sys.argv) > 2:
            json_foutpath = sys.argv[2]
    
    if "Sentences" in fpath:
        json_foutpath = "../data/flickr30k_refdf.json.gz"
        flickr2dataframe(fpath, out_fpath=json_foutpath)
    elif "Annotations" in fpath:
        json_foutpath = "../data/flickr30k_bbdf.json.gz"
        flickr2dataframe_bboxes(fpath, out_fpath=json_foutpath)
    
