
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
                yield [xmin, ymin, xmax-xmin, ymax-ymin], int(region_id)

      
def get_image_annos(fname):
    image_id = re.sub("([0-9]+)\.(txt|xml)$", r"\1", os.path.basename(fname))
    return {"i_corpus": 1, 
            "r_corpus": "flickr",
            "image_id": int(image_id)}

def add_region_annos(reg_annos, all_annos):
    for d_spec in data_spec:
        if d_spec in reg_annos:
            all_annos.setdefault(d_spec, []).append(reg_annos[d_spec])

def join_parsed_exps(fbasename="/media/%s/Carina_2017/UPF/github/object_naming/names_in_context/data/flickr30k_refexp_p%d.txt.json", 
                     parts=[0], 
                     json_foutpath=None):
    import df_annotation
    #parses = pd.read_json(json_fpath, compression='gzip', orient='columns')
    #indices_fpath = "{0}.idx".format(
    #    re.sub("(.+?)(\.txt)?(\.json)?(\.gz)?", r"\1", json_fpath))
    #indices = pd.read_csv(indices_fpath, sep=",", header=None, index_col=0)
    #indices.rename({0: "rex_id", 1: "image_id", 2: "region_id"}, axis=1, inplace=True)
    parse_parts = []
    for part in parts:
        json_fpath = fbasename % (part) 
        print("Loading ...", json_fpath)
        part_parse_df = df_annotation.add_dep_parses_from_json(json_fpath)
        part_parse_df.set_index("rex_id", inplace=True)
        parse_parts.append(part_parse_df)
    parse_df = pandas.concat(parse_parts)
    if json_foutpath != None:
        parse_df.to_json(json_foutpath, compression='gzip', orient='split')
    return parse_df
    

if __name__=="__main__":
    data_path = "/media/%s/Carina_2017/UPF/github/object_naming/names_in_context/data/" % USRNAME
    parse_df = join_parsed_exps(os.path.join(data_path, "flickr30k_refexp_p%d.txt.json"), 
                               parts=range(1,13),
                               json_foutpath=os.path.join(data_path, "flickr30k_refexp.txt.json.gz"))
    
    if False:
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
    
