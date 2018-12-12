import copy
import json
import os
import pathlib
import random
import re
import sys

from collections import Counter

folders = os.path.abspath(".").split("/")
USRNAME = folders[folders.index("media")+1] #"u148188"

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

from skimage import io
import pandas

from visual_genome import utils as vg_utils
import annotation_utils


BASE_DIR = os.path.dirname(os.getcwd())
PATH_MCRAE = os.path.join(BASE_DIR, "add_data/mcrae_visa")
sys.path.append(PATH_MCRAE)
import load_cat_conc_map_mcrae as mcrae

PATH_VGENOME = os.path.join(BASE_DIR, "add_data/vgenome")
RENDERED_IMGS_DIR = "http://www.coli.uni-saarland.de/~carina/object_naming/amt_images/"


##
# UTILS
def _relativeObjectSize(obj, img):
    return obj["w"] * obj["h"] / (img["width"].values[0] * img["height"].values[0])

def _ss2ssid(ss_list):
    return [ss.name() for ss in ss_list]

def _ss2ssid_map(ss_list):
    return {ss.name().split(".", 1)[0]:ss.name() for ss in ss_list}

def _ss2offset_map(ss_list):
    return {ss.offset():ss.name() for ss in ss_list}

def _ssids2offsets(ssid_set):
    return [annotation_utils.get_synset(ssid).offset() for ssid in ssid_set]

##
# DATASET UTILS
def object_iter(start_index=0, end_index=-1, data_dir=None, skip_compounds=True, objSize=[0.0,1.0], single_word=False):
    """
    param objSize: [minimum size, maximum size] of the object in proportion to the whole image.
    param single_word: Set True if object name should be a single word (i.e., no MWE)
    """
    if data_dir is None:
        data_dir = vg_utils.get_data_dir() 
    object_data = json.load(open(os.path.join(data_dir, 'objects.json')))
    random.shuffle(object_data)
    image_df = load_image_data(data_dir=data_dir)

    # iterate over images
    idx = 0
    for image_objs in object_data[start_index:end_index]:
        image_id = image_objs["image_id"]
        url = image_objs.get("image_url", "UNKN")
        objects = image_objs["objects"]
        img = image_df[image_df["image_id"]==image_id]
        for obj in objects:
            # ignore small objects
            relObjSize = _relativeObjectSize(obj, img)
            if not (relObjSize >= objSize[0] and relObjSize <= objSize[1]):
                continue
            
            synsets, obj_names = obj["synsets"], obj["names"]
            if single_word:
                obj_names = [objname for objname in obj_names if len(objname.split()) == 1]
                if len(obj_names) == 0:
                    continue
            
            if len(synsets) < 1:
                continue
            if len(synsets) > 1 and skip_compounds == True:
                print("len(synsets) > 1: ", synsets, obj_names)
                continue
            if len(obj_names) > 1:
                print("len(obj_names) > 1: ", synsets, obj_names)

            obj_names = re.sub(r'[\\",]', "", " / ".join(obj_names)).strip()
            yield idx, image_id, url, obj_names, synsets, obj
        idx += 1

def load_image_data(start_index=0, end_index=-1, data_dir=None):
    if data_dir is None:
        data_dir = vg_utils.get_data_dir()
    image_data = json.load(open(os.path.join(data_dir, 'image_data.json')))
    image_df = pandas.DataFrame(image_data)
    return image_df

def _image_path(image_df, image_id, imagedir_path="data/"):
    imagedir_path = os.path.join(imagedir_path, "images")
    url = image_df[image_df["image_id"]==image_id]["url"].values[0]
    img_dir, img_file = url.split("/")[-2:]
    return os.path.join(imagedir_path, img_dir, img_file)

def _image_path_bbox(image_id, object_id, sample_type, 
                     imagedir_path="pilot/images/"):
    imgfile_out = "%s_%s_%s.png" % (image_id, object_id, sample_type)
    return os.path.join(imagedir_path, imgfile_out)

def _collect_random_objects(data_dir=None, start_index=0, end_index=-1, objSize=[0.0, 1.0],  single_word=True):
    img_to_cat_coll = dict()
    obj_synset_map = dict()
    super_cat = "OBJECT"
    
    for idx, image_id, url, obj_names, synsets, obj in object_iter(start_index=start_index, 
                end_index=end_index, 
                data_dir=data_dir, objSize=objSize, single_word=single_word):        
        cat_coll = img_to_cat_coll.setdefault(image_id, {})
        # is object member of supercategory (e.g., person)?
        for ssid in synsets:
            target_synset = annotation_utils.get_synset(ssid)
            print(image_id, super_cat, target_synset)
            coll_synsets = cat_coll.setdefault(super_cat, dict())
            # augment {coll_synset --> [obj_id, ...] ...}
            coll_synsets.setdefault(target_synset, []).append(obj["object_id"])
            # augment {obj_id --> (synset, category, object), ...}
            obj_synset_map[obj["object_id"]] = (target_synset, super_cat, obj)
            break
    return img_to_cat_coll, obj_synset_map


def _collect_relevant_objects(only_cats, cat_map, data_dir=None, start_index=0, end_index=-1, min_match=2, objSize=[0.0, 1.0],  single_word=True):
    """
    Create img_to_cat_coll: {img_id --> {category --> {coll_synset --> [obj_id, ...] ...}, ...}, ...}
    
    param objSize: [minimum size, maximum size] of the object in proportion to the whole image.
    param single_word: Set True if object name should be a single word (i.e., no MWE)
    
    Example:
        {416 : {'person': {Synset('woman.n.01'): [1635308, 1635310], Synset('man.n.01'): [1635309]}, 'structure, construction': {Synset('restaurant.n.01'): [3563923]}}, ...}
    """
    if cat_map == None: # collect random objects
        return _collect_random_objects(data_dir, start_index, end_index, objSize,  single_word)
    
    if only_cats:
        min_match = 1

    img_to_cat_coll = dict()
    obj_synset_map = dict()
    prev_img_id = None
    for idx, image_id, url, obj_names, synsets, obj in object_iter(start_index=start_index, 
                end_index=end_index, 
                data_dir=data_dir, objSize=objSize, single_word=single_word):
        #if prev_img_id != None and prev_img_id != image_id:
        #    print(prev_img_id, ":", img_to_cat_coll[prev_img_id])
            
        prev_img_id = image_id
        
        cat_coll = img_to_cat_coll.setdefault(image_id, {})
        # is object member of supercategory (e.g., person)?
        for ssid in synsets:
            target_synset = annotation_utils.get_synset(ssid)
            all_hypers = annotation_utils.get_all_hypernyms(target_synset)
            match = False
            for hyper_path in all_hypers:
                ssoffsets_path = _ss2offset_map(hyper_path)
                if match:
                    break
                # iterate over set of supercategories, compare hypernym paths (supercat, object)
                for (super_cat, cats_offsets) in cat_map.items():
                        offsets = list(zip(*cats_offsets))[1]
                        found_cat = set(offsets).intersection(ssoffsets_path.keys())
                        if len(found_cat) >= min_match:
                            # augment {category --> {coll_synset --> [obj_id, ...] ...}, ...}
                            coll_synsets = cat_coll.setdefault(super_cat, dict())
                            
                            # found_cat: offset of coll_synset (?) (e.g., 10287213 (man.n.01))
                            catmatch_offset = found_cat.pop()
                            synset = annotation_utils.get_ss_from_offset("n%08d"%(catmatch_offset))
                            print(image_id, super_cat, synset)
                            
                            # augment {coll_synset --> [obj_id, ...] ...}
                            coll_synsets.setdefault(synset, []).append(obj["object_id"])
                            
                            # augment {obj_id --> (synset, category, object), ...}
                            obj_synset_map[obj["object_id"]] = (synset, super_cat, obj)
                            
                            match = True
                            break
                        else:
                            continue
    return img_to_cat_coll, obj_synset_map

def sample_objects(relevant_images_objs, collected_objs):
    image_ids = [img_id for (img_id, objs) in relevant_images_objs.items() if len(objs) > 0]
    random.shuffle(image_ids)
    
    # sample objects meeting the criteria
    added_images = ignore_imageids
    added_names = set()
    while len(image_ids) > 0:
        img_id = image_ids.pop()
        if img_id in added_images:
            continue
        # e.g., {'structure, construction': {Synset('restaurant.n.01'): [2406647]}, 'person': {Synset('man.n.01'): [2305827, 3570920], Synset('woman.n.01'): [2438901]}}
        objs = relevant_images_objs.get(img_id)
        #print("\n", objs)
        num_objs = sum([len(val2) for val in objs.values() for val2 in val.values()])
        
        if num_objs < criteria["min_num_objs"]:
            print("sample object singleton (no other objs)", )
            #print("discard (number of objects == %d, but min number: %d)" % (num_objs, criteria["min_num_objs"]))
            rel_cat = list(objs.keys())[0]
            if imgs_to_collect[rel_cat]["singleton_obj"] > 0:
                # object with that name already sampled?
                objID_candidate = list(objs[rel_cat].values())[0][0]
                objName_cand = obj_synset_map[objID_candidate][2]["names"][0]
                if objName_cand in added_names:
                    continue
                
                collected_objs[rel_cat]["singleton_obj"].append((img_id, objID_candidate))
                imgs_to_collect[rel_cat]["singleton_obj"] = imgs_to_collect[rel_cat]["singleton_obj"] - 1
                added_images.add(img_id)
                added_names.add(objName_cand)
            continue
        # e.g., [('structure, construction', [2406647]), ('person', [2305827, 3570920]), ('person', [2438901])]
        cat_objs_per_synset = [(cat, val2) for (cat, val) in objs.items() for val2 in objs[cat].values()]
        # e.g., {'structure, construction': [2406647], 'person': [2305827, 3570920, 2438901]}
        objs_same_cat = dict()
        [objs_same_cat.setdefault(cat, []).extend(val) for (cat, val) in cat_objs_per_synset]
        
        for rel_cat in objs_same_cat:
            # only other categories?
            if len(objs_same_cat[rel_cat]) == criteria["max_num_distractors"] + 1:
                print("sample object of singleton category: ", objs_same_cat[rel_cat])
                if imgs_to_collect[rel_cat]["max_num_distractors"] > 0:
                    objID_candidate = objs_same_cat[rel_cat][0]
                    objName_cand = obj_synset_map[objID_candidate][2]["names"][0]
                    if objName_cand in added_names:
                        continue
                    
                    descr = criteria_descr["max_num_distractors"][criteria["max_num_distractors"]]
                    collected_objs[rel_cat][descr].append((img_id, objID_candidate))
                    imgs_to_collect[rel_cat]["max_num_distractors"] = imgs_to_collect[rel_cat]["max_num_distractors"] - 1
                    added_images.add(img_id)
                    added_names.add(objName_cand)
                    continue
            
            # at least one other object with same synset
            # e.g., [(Synset('man.n.01'), [2305827, 3570920])]
            objs_same_synset = {synset: synsets for (synset, synsets) in objs[rel_cat].items() if len(synsets) > criteria["min_objs_same_synset"]}
            if len(objs_same_synset) > 0:
                print("sample object with distractor(same synset): ", objs_same_synset)
                if imgs_to_collect[rel_cat]["min_objs_same_synset"] > 0:
                    obj_list =  [obj for objects in objs_same_synset.values() for obj in objects]
                    # exclude objects whose name was already added for another image's object
                    obj_list = [objID for objID in obj_list if obj_synset_map[objID][2]["names"][0] not in added_names]
                    if len(obj_list) == 0:
                        continue
                    
                    random.shuffle(obj_list)
                    descr = criteria_descr["min_objs_same_synset"][criteria["min_objs_same_synset"]]
                    collected_objs[rel_cat][descr].append((img_id, obj_list[0]))
                    imgs_to_collect[rel_cat]["min_objs_same_synset"] = imgs_to_collect[rel_cat]["min_objs_same_synset"] - 1
                    added_images.add(img_id)
                    added_names.add(obj_synset_map[obj_list[0]][2]["names"][0])
                    continue
            
            # no other object with same synset, but same category
            # e.g., {Synset('woman.n.01'): [2438901]}
            if len(objs[rel_cat]) > criteria["min_objs_same_cat"]:
                objs_singleton_synset = {synset: synsets for (synset, synsets) in objs[rel_cat].items() if len(synsets) == 1}
                if len(objs_singleton_synset) > 0:
                    print("sample object singleton: ", objs_singleton_synset)
                    if imgs_to_collect[rel_cat]["min_objs_same_cat"] > 0:
                        obj_list =  [obj for objects in objs_singleton_synset.values() for obj in objects]
                        # exclude objects whose name was already added for another image's object
                        obj_list = [objID for objID in obj_list if obj_synset_map[objID][2]["names"][0] not in added_names]
                        if len(obj_list) == 0:
                            continue
                        
                        random.shuffle(obj_list)
                        descr = criteria_descr["min_objs_same_cat"][criteria["min_objs_same_cat"]]
                        collected_objs[rel_cat][descr].append((img_id, obj_list[0]))
                        imgs_to_collect[rel_cat]["min_objs_same_cat"] = imgs_to_collect[rel_cat]["min_objs_same_cat"] - 1
                        added_images.add(img_id)
                        added_names.add(obj_synset_map[obj_list[0]][2]["names"][0])
    return collected_objs

def samples_to_dataframe(collected_objs):
    cols = ["image_id", "object_id", "sample_type", "category", "synset", "obj_names", "bbox_xywh"]
    sampled_data_df = pandas.DataFrame(columns=cols)
    for (category, samples) in collected_objs.items():
        df_row = {col:None for col in cols}
        for sample_type in samples:
            #print(category, sample_type)
            df_row["category"] = category
            df_row["sample_type"] = sample_type
            for (image_id, object_id) in samples[sample_type]:
                obj = obj_synset_map[object_id]
                new_df_item = {"category": category,
                            "sample_type": sample_type,
                            "image_id": image_id,
                            "object_id": object_id,
                            "synset": obj[0].name(),
                            "obj_names": list(set(obj[2]["names"])),
                            "bbox_xywh": [obj[2]["x"], obj[2]["y"], obj[2]["w"], obj[2]["h"]]
                }
                sampled_data_df = sampled_data_df.append(new_df_item, ignore_index=True)
    return sampled_data_df

def _fill_html_table_row(taboo_words, image_url, item_id, amt_exp=False):
    if amt_exp:
        col_img_src = '<img src="${{img_{0}_url}}" alt="{1}" style="width:350px;height:350px;">'.format(item_id, image_url.split("/")[-1])
        col_taboo_words = '<font color="grey"><b>Taboo Words</b><br><br>${{taboolist_{0}}}</font>'.format(item_id)
    else:
        col_img_src = '<img src={0} alt="{1}" style="width:350px;height:350px;">'.format(image_url, image_url.split("/")[-1])
        col_taboo_words = '<font color="grey"><b>Taboo Words</b><br><br>{0}</font>'.format(taboo_words)
        

    col_text_box = '<b>Object Name</b>:  \n\
        \t  <INPUT TYPE="text" ID="inputbox_objname-{0}" NAME="inputbox_objname-{0}" VALUE="" SIZE=35 autofocus placeholder="Enter an object name"><P><br>\n\
        \t  <i>If you cannot name the object, please specify the reason:</i><br><br>\n\
        		<INPUT TYPE="radio" ID="optout_exh-{0}" NAME="optout-{0}" Value="exhausted" onClick=0>All the names I know are on the taboo list<BR>\n\
        		<INPUT TYPE="radio" ID="optout_qual-{0}" NAME="optout-{0}" Value="quality" onClick=0>Cannot recognize the object (e.g., too tiny, occluded)<BR>\n\
        		<INPUT TYPE="radio" ID="optout_bbox-{0}" NAME="optout-{0}" Value="bbox" onClick=0>Bounding box is false or misleading<BR>\n\
        		<INPUT TYPE="radio" ID="optout_other-{0}" NAME="optout-{0}" Value="other" onClick=0>Other: \n\
        		<INPUT TYPE="text" ID="other_reasons-{0}" NAME="other_reasons-{0}" VALUE="" placeholder="Please specify the reasons" SIZE=35>        \
        		<P><br><BR>'.format(item_id)
        		
    html_row = '<tr>\n\t<td width="50">%s</td>\n\t\
    <td align="position" valign="position" width="400"><div style="background-color:#ffff99">%s</div></td>\n\t\
    <td width="170">%s</td>\n\t</tr>\n\t' % (col_img_src, col_text_box, col_taboo_words)
    return html_row

def _fill_html_table_row_analysis(object_names, taboo_lists, image_url, item_id):
    col_img_src = '<img src={0} alt="{1}" style="width:350px;height:350px;">'.format(image_url, image_url.split("/")[-1])
    col_taboo_words = '<font color="black"><b>Taboo Lists/Words</b><br><br>{0}</font>'.format(taboo_lists)
        
    col_text_box = '<b>Object Names</b>:<br>  \n\
        {0}<br><BR>'.format(object_names)
        		
    html_row = '<tr>\n\t<td width="50">%s</td>\n\t\
    <td align="position" valign="position" width="200"><div style="background-color:#ffff99">%s</div></td>\n\t\
    <td width="470">%s</td>\n\t</tr>\n\t' % (col_img_src, col_text_box, col_taboo_words)
    return html_row

def add_hit_to_amt(sampled_data_df, image_df, amt_df, amt_meta_df, img_basedir):
    row_num = 0
    df_row = {c:None for c in amt_df.columns}
    meta_df_row = {c:None for c in amt_meta_df.columns}
    for row in sampled_data_df.iterrows():
        image_id = row[1]["image_id"]
        image_path = _image_path_bbox(image_id, row[1]["object_id"], row[1]["sample_type"], imagedir_path=img_basedir)
        obj_names = row[1]["obj_names"].replace("/", "<br>")
        
        meta_df_row.update({c:row[1][c] for c in amt_meta_df.columns if c != "item_id"})
        meta_df_row["item_id"] = "item-" + str(row_num)
        meta_df_row["obj_names"] = obj_names
        amt_meta_df = amt_meta_df.append(meta_df_row, ignore_index=True)
        
        df_row["img_%d_url" % row_num] = image_path
        df_row["taboolist_%d" % row_num] = obj_names

        row_num += 1
    
    amt_df = amt_df.append(df_row, ignore_index=True)
    
    return amt_df, amt_meta_df

def _pretty_obj_names(obj_names):
    if isinstance(eval(obj_names), Counter):
        obj_names = "/".join(["%s: %d" % (name, count) for (name, count) in eval(obj_names).items()])
    elif isinstance(eval(obj_names), dict):
        obj_names = "/".join(["%s: %s" % (name, str(counts)) for (name, counts) in eval(obj_names).items()])
    elif isinstance(eval(obj_names), list):
        obj_names = "/".join([str(l) for l in eval(obj_names)])
    return obj_names.replace("/", "<br>")

def write_html_table(sampled_data_df, image_df, html_fname="pilot_testgeneration.html", amt_exp=False, img_basedir=None):
    pilot_basedir = os.path.dirname(html_fname)
    if img_basedir == None:
        img_basedir = os.path.join(pilot_basedir, "images")
    table = ''
    table += '<table style="width:100%" cellpadding="10" cellspacing="10" frame="box">\n<tbody>'
    data_info = ["item_id\timage_id\tobject_id\tobj_names\tsynset\tsample_type\n"]
    meta_fname = html_fname.replace(".html", "") + "-meta.txt"
    
    row_num = 0
    for row in sampled_data_df.iterrows():
        image_id = row[1]["image_id"]
        #url = image_df[image_df["image_id"]==image_id]["url"].values[0]
        #image_path = _image_path(image_df, image_id)
        image_path = _image_path_bbox(image_id, row[1]["object_id"], row[1]["sample_type"], imagedir_path=img_basedir)
        data_info.append("{0}\t{1[image_id]}\t{1[object_id]}\t{1[category]}\t{1[synset]}\t{1[sample_type]}".format("item-" + str(row_num), row[1]))

        obj_names = _pretty_obj_names(row[1]["obj_names"])
        if "taboo_lists" in sampled_data_df.columns: # do not create form for annotation, but illustrate collected data
            html_row = _fill_html_table_row_analysis(obj_names, _pretty_obj_names(row[1]["taboo_lists"]), image_path, row_num)
        else:
            html_row = _fill_html_table_row(obj_names, image_path, row_num, amt_exp)
        table += '\t%s\n' % html_row 
        row_num += 1
        
    table += "</tbody></table>\n"
    if not amt_exp:
        table = '<FORM NAME="myform" METHOD="POST" ACTION="save.php" onsubmit="return notEmpty(document, %d)">\n\
        <input id="annotator_name" name="annotator_name"  placeholder="Enter your full name" type="text" autofocus><br><br>\n' % (row_num) + table
        with open(meta_fname, "w") as metafile:
            metafile.write("\n".join(data_info))
    with open(html_fname, "w") as f:
        f.write(table)
    

if __name__=="__main__":
    start_index = 0
    end_index = -1
    shuffle_items = True # shuffle the order of items to be presented in the html

    extend_existing_sample = False
    version = 0


    ## CONSTRAINTS
    min_max_rel_box_size = [0.2, 0.8]
    random_categories = False

    ##
    if len(sys.argv) <= 1 or sys.argv[1] not in ["sample_objects", "check_images", "render_objects", "website", "amt"]:
        sys.stderr.write("Please specify the type of data creation you want to perform.\n  Options: [sample_objects, check_images, render_objects, website, amt]\n")
        sys.exit()
    else:
        creation_type = sys.argv[1]

    if len(sys.argv) > 2: 
        df_sampleddata_outfname = sys.argv[2]
    else:
        df_sampleddata_outfname = "collected_images/sampled_data-v%d.csv" % (version)

    #relevant_categories = "food_categories.csv"
    #relevant_categories = "pilot_categories.csv"
    relevant_categories = "pilot_example_categories.csv"


    ## START ##
    study_basedir = os.path.dirname(df_sampleddata_outfname)
    pathlib.Path(os.path.join(study_basedir, 'images/')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(study_basedir, 'annotations/')).mkdir(parents=True, exist_ok=True)

    amt_exp = True if creation_type.lower() == "amt" else False

    if creation_type.lower() in ["check_images", "render_objects", "website", "amt"]:
        image_df = load_image_data(data_dir=PATH_VGENOME)
        sampled_data_df = pandas.read_csv(df_sampleddata_outfname, sep="\t")

    if creation_type.lower() == "sample_objects":
        ignore_imageids = set()
        if extend_existing_sample:
            ignore_imageids = set(pandas.read_csv("collected_images-amt/pilot_sampled_data-amt.csv", sep="\t")["image_id"].tolist())
        
        fine_match = True # supercategory (e.g., person) does not suffice as match
        only_cats = (True and fine_match)
        
        if not random_categories:
            cat_map, conc_subcat_map = mcrae.load(os.path.join(PATH_MCRAE, relevant_categories), only_subcats=False, only_cats=only_cats)
        else:
            cat_map = None
            end_index = 10000

        min_match = 2
        single_word = True
        relevant_images_objs, obj_synset_map = _collect_relevant_objects(
            only_cats, cat_map, data_dir=PATH_VGENOME, start_index=start_index,
            end_index=end_index, min_match=min_match, objSize=min_max_rel_box_size,
            single_word=single_word)

        criteria = {"max_num_distractors": 0, # only other categories
                    "min_objs_same_synset": 1, # same category, same synset 
                    "min_objs_same_cat": 1, # same category, but other synset
                    "min_num_objs": 2 # min number of relevant objects in image
                    } 

        criteria_descr = {"max_num_distractors": {0: "singleton_category"}, # only other categories
                        "min_objs_same_synset": {1: "ambiguous_synset"}, # same category, same synset 
                        "min_objs_same_cat": {1: "ambiguous_category"} # same category, but other synset
                        } 

        factor = 10
        imgs_to_collect = {cat: {"singleton_obj": 1 * factor,
                                "max_num_distractors": 3 * factor, 
                                "min_objs_same_synset": 3 * factor, 
                                "min_objs_same_cat": 3 * factor} for cat in cat_map}

        collected_objs = {cat: {"singleton_obj": [],
                                "singleton_category": [], 
                                "ambiguous_synset": [], 
                                "ambiguous_category": []} for cat in cat_map}

        # sample objects meeting the criteria
        collected_objs = sample_objects(relevant_images_objs, collected_objs)
        
        # WRITE csv
        sampled_data_df = samples_to_dataframe(collected_objs)
        sampled_data_df.to_csv(df_sampleddata_outfname, columns=sampled_data_df.columns, sep="\t", index=False)
                                  
    if creation_type.lower() == "check_images":
        # Check if image files exist
        f_imgs2extract = open(os.path.join(study_basedir, "imgs2extract.txt"), "w")
        unseen_images = []
        for row in sampled_data_df.iterrows():
            image_id = row[1]["image_id"]
            image_path = _image_path(image_df, image_id)

            if not os.path.exists(image_path):
                url = image_df[image_df["image_id"]==image_id]["url"].values[0]
                sys.stderr.write("Image path not found: %s\n\t(url: %s)\n" % (image_path, url))
                #unseen_images.append(image_path.split("/", 2)[-1])
                f_imgs2extract.write("%s\n" % image_path.split("/", 2)[-1])
                continue
        f_imgs2extract.close()

    if creation_type.lower() == "render_objects":
        save_fig = True    
        # Draw bboxes and save image files
        for row in sampled_data_df.iterrows():
            image_id = row[1]["image_id"]
            image_path = _image_path(image_df, image_id, imagedir_path=PATH_VGENOME)
            
            if not os.path.exists(image_path):
                url = image_df[image_df["image_id"]==image_id]["url"].values[0]
                sys.stderr.write("Image path not found: %s\n\t(url: %s)\n" % (image_path, url))
                continue
            
            try:
                image = plt.imread(image_path)
            except OSError: 
                sys.stderr.write("OSError. Maybe file %s is empty.\n" % (image_path))
                continue
                
            bb = eval(row[1]["bbox_xywh"])
            obj_names = row[1]["obj_names"]
            
            plt.cla()
            plt.imshow(image)
            plt.gca().add_patch(
                plt.Rectangle((bb[0], bb[1]),
                                bb[2], bb[3], fill=False,
                                edgecolor='r', linewidth=4)
                )
            #plt.title('{}'.format(obj_names))
            plt.text(bb[0]+3, bb[1]+bb[3]-3, obj_names, fontsize=20, color="red", backgroundcolor="white")
            if save_fig:
                plt.tight_layout()
                imgfile_out = _image_path_bbox(image_id, 
                                            row[1]["object_id"],
                                            row[1]["sample_type"],
                                            imagedir_path=os.path.join(study_basedir, "images"))
                print(imgfile_out)
                plt.savefig(imgfile_out)
            
            #plt.show()

    if creation_type.lower() == "website":        
        items_per_page = len(sampled_data_df)
        shuffle_items = False
        
        # Write annotation form
        if len(sys.argv) > 3: 
            html_outfname = sys.argv[3]
        else:
            html_outfname = df_sampleddata_outfname.replace(".csv", "") + ".html"
        rendered_imgs_dir = os.path.join(os.path.dirname(html_outfname), "images/")

        if items_per_page == len(sampled_data_df):
            if shuffle_items:
                sampled_data_df = sampled_data_df.sample(frac=1)
            for page_no in range(0, len(sampled_data_df), items_per_page):
                write_html_table(sampled_data_df.iloc[page_no:page_no+items_per_page], image_df, html_fname=html_outfname.replace(".html", "-"+str(page_no)+".html"), amt_exp=amt_exp, img_basedir=rendered_imgs_dir)
            print("html files written to " + html_outfname.replace(".html", "-<hit_no>.html"))
        else:
            categories = {}
            for cat, group in sampled_data_df.groupby("category"):
                categories[cat] = group

            seen_items = 0
            while seen_items < len(sampled_data_df):
                page_no = 0
                page_items = []
                while page_no < items_per_page and seen_items < len(sampled_data_df):
                    for cat in categories.keys():
                        if page_no == items_per_page:
                            break
                        aux = categories[cat]
                        if len(aux) == 0:
                            continue
                        aux, last_row = aux.drop(aux.tail(1).index),aux.tail(1)
                        page_items.append(last_row)
                        if len(aux) > 0:
                            categories[cat] = aux.sample(frac=1)
                        seen_items += 1
                        page_no += 1
                        
                        
                write_html_table(pandas.concat(page_items).sample(frac=1), image_df, html_fname=html_outfname.replace(".html", "-"+str(seen_items)+".html"), amt_exp=amt_exp, img_basedir=RENDERED_IMGS_DIR)
            print("html files written to " + html_outfname.replace(".html", "-<hit_no>.html"))
            
    if creation_type.lower() == "amt":
        items_per_page = 5
        cols = ["img_%d_url" % i for i in range(items_per_page)]
        cols.extend(["taboolist_%d" % i for i in range(items_per_page)])
        amt_df = pandas.DataFrame(columns=cols)
        amt_meta_df = pandas.DataFrame(columns=["item_id", "image_id", "object_id", "obj_names", "category", "synset", "sample_type"])
        img_basedir = "http://www.coli.uni-saarland.de/~carina/object_naming/amt_images/"
        
        # Write annotation form
        if len(sys.argv) > 3:
            html_outfname = sys.argv[3]
        else:
            html_outfname = df_sampleddata_outfname.replace(".csv", "") + "_amt.csv"
        csv_outfname = html_outfname.replace(".html", ".csv")
        meta_fname = html_outfname.replace(".html", "") + "-meta.csv"
        
            
        if items_per_page == len(sampled_data_df):
            # TODO
            if shuffle_items:
                sampled_data_df = sampled_data_df.sample(frac=1)
            for page_no in range(0, len(sampled_data_df), items_per_page):
                write_html_table(sampled_data_df.iloc[page_no:page_no+items_per_page], image_df, html_fname=html_outfname.replace(".html", "-"+str(page_no)+".html"), amt_exp=amt_exp)
            print("html files written to " + html_outfname.replace(".html", "-<hit_no>.html"))
        else:
            categories = {}
            for cat, group in sampled_data_df.groupby("category"):
                categories[cat] = group

            seen_items = 0
            while seen_items < len(sampled_data_df):
                page_no = 0
                page_items = []
                while page_no < items_per_page and seen_items < len(sampled_data_df):
                    for cat in categories.keys():
                        if page_no == items_per_page:
                            break
                        aux = categories[cat]
                        if len(aux) == 0:
                            continue
                        aux, last_row = aux.drop(aux.tail(1).index), aux.tail(1)
                        page_items.append(last_row)
                        if len(aux) > 0:
                            categories[cat] = aux.sample(frac=1)
                        else:
                            categories[cat] = ""
                        seen_items += 1
                        page_no += 1
                amt_df, amt_meta_df = add_hit_to_amt(pandas.concat(page_items).sample(frac=1), image_df, amt_df, amt_meta_df, RENDERED_IMGS_DIR)
                
                if seen_items <= items_per_page:
                    # write html pattern only once
                    write_html_table(pandas.concat(page_items).sample(frac=1), image_df, html_fname=html_outfname, amt_exp=True)
                    print("html pattern for amt written to " + html_outfname)

            amt_df.to_csv(csv_outfname, columns=amt_df.columns, sep=",", index=False)
            amt_meta_df.to_csv(meta_fname, columns=amt_meta_df.columns, sep="\t", index=False)
            print("data csv for amt written to " + csv_outfname)
            print("meta csv for amt written to " + meta_fname)
            
