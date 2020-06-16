import os
import random
import sys

from collections import Counter

import pandas
import numpy as np

#folders = os.path.abspath(".").split("/")
#USRNAME = folders[folders.index("media")+1] #"u148188"

#from skimage import io
#import pandas

#from visual_genome import utils as vg_utils
#import annotation_utils

BASE_DIR = os.path.dirname(os.getcwd())


class Object_sampler():
    """
    Criteria with respect to sampling based on distractor objects:
    
        keyword           |   target                | distractor(s)
    --------------------------------------------------------------------------------------
    supercat_ambiguous    |   supercategory.node    |   same supercategory, other node
                          |                         |   (Graf et al.: "item 22/23: basic sufficient")
    supercat_unique       |   supercategory.node    |   other supercategory  
                          |                         |   (Graf et al.: "item 33: super sufficient")
    singleton             |   supercategory.node    |   --- (no object subsumed by pre-defined node and 
                          |                         |   none of the most frequent objects in VG)
                          |                         |   (Graf et al.: "item 33: super sufficient")
    seed_ambiguous        |   sc.node.object_name   |   same sc.node, other object_name
                          |                         |   (Graf et al.: "item 12: sub necessary")
    objname_ambiguous     |   sc.node.object_name   |   same sc.node, same object_name  <-- difficult to get 
                          |                         |   for some nodes, should be limited to avoid unbalanced
                          |                         |   overall data
                          |                         |   (Graf et al.: --)
    """
    def __init__(self, obj_synset_map, supercategories, unique_names=False, factor=1):
        self.unique_names = unique_names
        self.obj_synset_map = obj_synset_map
        #self.criteria = criteria
        #self.criteria_descr = criteria_descr
        
        self.criteria = {"singleton_obj": self._singleton_object,
                         "supercat_unique": self._supercat_unique,
                         "seed_ambiguous": self._seed_ambiguous,
                         "supercat_ambiguous": self._supercat_ambiguous}
        
        self.imgs_to_collect = {cat: {"singleton_obj": 1 * factor,
                                    "supercat_unique": 3 * factor, 
                                    "seed_ambiguous": 3 * factor, 
                                    "supercat_ambiguous": 3 * factor} for cat in supercategories}

        self.collected_objs = {cat: {"singleton_obj": {},
                                "supercat_unique": {}, 
                                "seed_ambiguous": {}, 
                                "supercat_ambiguous": {}} for cat in supercategories}

    def _object_name(self, objID):
        return "/".join(self.obj_synset_map[objID][2]["names"]) #[0]

    def _other_object(self, obj_IDs):
        """
        No other relevant objects (distractors) in image?
        """
        descr = "other"
        sys.stderr.write("\nsample remaining objects\n")
        for objID_candidate in obj_IDs:
            # object was not filtered out (e.g., because it is not in singular form)
            if objID_candidate in self.obj_synset_map: 
                rel_supercat = self.obj_synset_map[objID_candidate][1]
                yield objID_candidate, rel_supercat, descr

    # TODO: compare also to most frequent objects in VG!!
    def _singleton_object(self, objs):
        """
        No other relevant objects (distractors) in image?
        """
        descr = "singleton_obj"
        num_objs = sum([len(val2) for val in objs.values() for val2 in val.values()])
        if num_objs != 1:
            return None, None, descr
        sys.stderr.write("sample object singleton (no other objs)\n")
        rel_supercat = list(objs.keys())[0]
        objID_candidate = list(objs[rel_supercat].values())[0][0]
        # object was filtered not out (e.g., because it is not in singular form)
        if objID_candidate in self.obj_synset_map:
            return objID_candidate, rel_supercat, descr
        return None, None, descr

    def _supercat_unique(self, objs_same_scat):
        """
        Only objects of other supercategories.
        Example: target: person.man, distractors: Y1.Y2, Y1 != person
        """
        # e.g., [('structure, construction', [2406647]), ('person', [2305827, 3570920]), ('person', [2438901])]
        descr = "supercat_unique"
        if len(objs_same_scat) != 1:
            return None, descr
        sys.stderr.write("supercat_unique: object of unique category: %s\n" % str(objs_same_scat))

        objID_candidate = objs_same_scat[0]
        if objID_candidate in self.obj_synset_map: # object was filtered out (e.g., because it is not in singular form)
            return objID_candidate, descr
        return None, descr
    
    def _seed_ambiguous(self, objs_same_scat):
        """
        Image has at least one other object with the same seed synset.
        """
        descr = "seed_ambiguous"
        objs_same_synset = {synset: synsets for (synset, synsets) in objs_same_scat.items() if len(synsets) > 1}
        if len(objs_same_synset) < 1:
            return None, descr
        obj_list =  [objID for objects in objs_same_synset.values() for objID in objects if objID in self.obj_synset_map]
        
        sys.stderr.write("seed_ambiguous: object with distractor(same synset): " + str(objs_same_synset))
        for objId in obj_list:
            yield objId, descr
    
    def _supercat_ambiguous(self, objs_same_scat):
        """
        Image has at least one other object with the same supercategory, but the seed synset of the candidate object is unique.
        """
        descr = "supercat_ambiguous"
        if len(objs_same_scat) < 2:
            return None, descr
        objs_singleton_synset = {synset: synsets for (synset, synsets) in objs_same_scat.items() if len(synsets) == 1}
        if len(objs_singleton_synset) < 1:
            return None, descr
        
        obj_list =  [objID for objects in objs_singleton_synset.values() for objID in objects if objID in self.obj_synset_map]
        sys.stderr.write("supercat_ambiguous: sample object synset singleton (no other objs of synset): " + str(objs_singleton_synset))
        for objId in obj_list:
            yield objId, descr
        

    def sample_objects_taboo(self, relevant_images_objs, ignore_imageids=set()):
        """
        @param relevant_images_objs: 
            Example:
            {
            2380107: {'instrumentality, instrumentation': {Synset('furnishing.n.02'): [1350521, 1350511]}, 
                      'article of clothing': {Synset('overgarment.n.01'): [1350513]}},  
            2356658: {'vehicle': {Synset('car.n.01'): [818433], Synset('train.n.01'): [818432]}}, 
            2411876: {'animal': {Synset('horse.n.01'): [207891, 207889]}}, 
            2323774: {'vehicle': {Synset('boat.n.01'): [2720310]}}, ...}
        """
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
            all_obj_IDs = [objID for (supercat, val) in objs.items() for objIDs in objs[supercat].values() for objID in objIDs]
            print("\nAll object IDs: ", all_obj_IDs)
            objName_cand = None
            
            # singleton_object: only a single relevant object in image?
            objID_candidate, rel_supercat, descr = self._singleton_object(objs)
            if objID_candidate != None:
                objName_cand = self._object_name(objID_candidate)
                self.collected_objs[rel_supercat][descr].setdefault(objName_cand, []).append((img_id, objID_candidate))
                print("--> sample object singleton: ", rel_supercat, objID_candidate, objName_cand)
                if self.unique_names and objName_cand not in added_names:
                    # only count found object if its name is unseen
                    self.imgs_to_collect[rel_supercat][descr] -= 1
                added_images.add(img_id)
                added_names.add(objName_cand)
                all_obj_IDs.remove(objID_candidate)
                continue
        
            # Process objects grouped by their category (i.e., same category)
            # e.g., [('structure, construction', [2406647]), ('person', [2305827, 3570920]), ('person', [2438901])]
            supercat_objs_per_synset = [(supercat, objIDs) for (supercat, val) in objs.items() for objIDs in objs[supercat].values()]
            # e.g., {'structure, construction': [2406647], 'person': [2305827, 3570920, 2438901]}
            objs_same_scat = dict()
            [objs_same_scat.setdefault(cat, []).extend(objIDs) for (cat, objIDs) in supercat_objs_per_synset]
            
            for rel_supercat in objs_same_scat:
                # supercat_unique: only other supercategories?
                objID_candidate, descr = self._supercat_unique(objs_same_scat[rel_supercat])
                if objID_candidate != None:                    
                    objName_cand = self._process_object_candidate(img_id, objID_candidate, rel_supercat, descr, added_names)
                    
                    print("--> sample object with unique supercat: ", rel_supercat, objID_candidate, objName_cand)
                    added_images.add(img_id)
                    added_names.add(objName_cand)
                    all_obj_IDs.remove(objID_candidate)
                    continue
                
                # seed_ambiguous: at least one other object with same synset
                # e.g., 2359252: {..., 'person': {Synset('woman.n.01'): [3172624, 1891661]}}
                for objID_candidate, descr in self._seed_ambiguous(objs[rel_supercat]):
                    objName_cand = self._process_object_candidate(img_id, objID_candidate, rel_supercat, descr, added_names)
                    if objName_cand != None:
                        added_images.add(img_id)
                        added_names.add(objName_cand)
                        print("--> sample object with distractor (same synset): ", objs[rel_supercat], objName_cand)
                        all_obj_IDs.remove(objID_candidate)
                # TODO
                #if img_added:
                #    continue
                
                # supercat_ambiguous: no other object with same synset, but same supercategory
                # e.g., 2358790: {..., 'person': {Synset('man.n.01'): [3542299], Synset('woman.n.01'): [2053195]}}
                for objID_candidate, descr in self._supercat_ambiguous(objs[rel_supercat]):
                    objName_cand = self._process_object_candidate(img_id, objID_candidate, rel_supercat, descr, added_names)
                    if objName_cand != None:
                        added_images.add(img_id)
                        added_names.add(objName_cand)
                        print("--> sample object synset singleton (no other objs of synset): ", objs[rel_supercat])
                        all_obj_IDs.remove(objID_candidate)
                # TODO
                #if img_added:
                #    continue
            # Add remaining relevant objects not meeting any of the criteria
            if len(all_obj_IDs) > 0:
                for objID_candidate, rel_supercat, descr in self._other_object(all_obj_IDs):
                    objName_cand = self._process_object_candidate(img_id, objID_candidate, rel_supercat, descr, added_names)
                    if objName_cand != None:
                        added_images.add(img_id)
                        added_names.add(objName_cand)
                        print("--> sample other object: ", objs[rel_supercat])
                all_obj_IDs = []
            
        return self.collected_objs
    
    def _process_object_candidate(self, img_id, objID_candidate, rel_supercat, descr, added_names):
        print("AHA", objID_candidate)
        if objID_candidate != None:
            objName_cand = self._object_name(objID_candidate)
            if self.unique_names and objName_cand in added_names:
                print("\tskipping name duplicate: ", objID_candidate, self.unique_names)
                # exclude objects whose name was already added for another image's object
                return None
            self.collected_objs[rel_supercat][descr].setdefault(objName_cand, []).append((img_id, objID_candidate))
            if objName_cand not in added_names:
                # only count found object if its name is unseen
                self.imgs_to_collect[rel_supercat][descr] -= 1
            print("YEAH", objID_candidate, objName_cand)
            return objName_cand
        print("NO", objID_candidate, objName_cand)
        return None

    def samples_to_dataframe(self, debug=False):
        cols = ["image_id", "object_id", "sample_type", "category", "synset", "obj_names", "bbox_xywh"]
        sampled_data_df = pandas.DataFrame(columns=cols)
        for (category, samples) in self.collected_objs.items():
            df_row = {col:None for col in cols}
            for sample_type in samples:
                df_row["category"] = category
                df_row["sample_type"] = sample_type
                for (obj_names, objects) in samples[sample_type].items():
                    for (image_id, object_id) in objects:
                        obj = self.obj_synset_map[object_id]
                        obj_names_list = list(set(obj[2]["names"]))
                        # TODO for dev only
                        if debug:
                            obj_names_list.append(obj[2]["iou_info"])
                        new_df_item = {"category": category,
                                    "sample_type": sample_type,
                                    "image_id": image_id,
                                    "object_id": object_id,
                                    "synset": obj[0].name(),
                                    "obj_names": obj_names_list,
                                    "bbox_xywh": [obj[2]["x"], obj[2]["y"], obj[2]["w"], obj[2]["h"]]
                        }
                        sampled_data_df = sampled_data_df.append(new_df_item, ignore_index=True)
        return sampled_data_df
    
class SampledImages_Picker():
    def __init__(self, data_df_fname):
        self.sampled_data_df = pandas.read_csv(data_df_fname, sep="\t")
        obj_names = self.sampled_data_df["obj_names"].apply(lambda a: eval(a)[0])
        self.sampled_data_df["obj_names"] = obj_names
        
        self.criteria_thresholds = [("gt", 500, 500), 
                        ("btw", (800,1001), "all"),
                        ("gt", 1000, 1000)]
        self.thresholds = [1, 501, 801, 1001, 10000000]
        
        #df_imgs_per_seed = self._determine_images2sample_per_name(
        #    counter_variant="no_objnms_unique_per_img", 
        #    outfname="sampling2_max1000per_objname.csv")
        #self.sample_images(df_imgs_per_seed)
    
    def save_sampled_data(self, collected_objects, outfname):
        collected_objects.sort_values(by=["category", "synset", "image_id", "obj_names"], ascending=True, inplace=True)
        collected_objects.to_csv(outfname, sep="\t", index=False)
        print("Sampled data saved under %s." % outfname)
    
    def sample_images(self, df_imgs_per_seed, allow_img_duplicates=True):
        """
        category  synset sample_type obj_names no_objects  no_objnms_unique_per_img  imgs_to_sample
        66  person  child.n.01  NaN  brat  1  1  1
        347  food, solid food  baked_goods.n.01  NaN   cornbread  1  1  1
        ...
        405  animal   dog.n.01  NaN  dog  1054  980  980
        ...
        """
        df_imgs_per_seed.sort_values(by=["imgs_to_sample"], ascending=True, inplace=True)
        
        added_images = list()
        collected_objects = None
        for idx, row in df_imgs_per_seed.iterrows():
            num_imgs2sample = row.imgs_to_sample
            obj_set = self.sampled_data_df[self.sampled_data_df["obj_names"]==row["obj_names"]]
            #inds_new_imgs = obj_set["image_id"].apply(lambda a: a in new_imgs)
            #objs_new_imgs = obj_set[inds_new_imgs==True]
            if allow_img_duplicates == False:
                img_ids = obj_set[["image_id"]].drop_duplicates(keep="first")
                obj_set = obj_set.loc[img_ids.index]

            all_imgs = set(obj_set["image_id"].tolist())
            # Try to sample only unseen images not collected before
            new_imgs = all_imgs.difference(added_images)
            inds_new_imgs = obj_set["image_id"].apply(lambda a: a in new_imgs)
            objs_new_imgs = obj_set[inds_new_imgs==True]
            sampled_objs = None
            if len(objs_new_imgs) > num_imgs2sample:
                # sample from unseen images
                sampled_objs = objs_new_imgs.sample(n=num_imgs2sample)
                if len(sampled_objs["image_id"]) > len(sampled_objs["image_id"].unique()):
                    print("AHA -- something is wrong", sampled_objs)
                    return sampled_objs, None
            else:
                sampled_objs = objs_new_imgs
                if allow_img_duplicates == True:
                    # Add seen images to objects from unseen images
                    num_seen_imgs2sample = num_imgs2sample - len(objs_new_imgs)
                    objs_seen_imgs = obj_set[inds_new_imgs==False]
                    sampled_objs = sampled_objs.append(objs_seen_imgs.sample(n=min(
                        len(objs_seen_imgs), num_seen_imgs2sample)))
            
            if collected_objects is None:
                collected_objects = sampled_objs
            else:
                collected_objects = collected_objects.append(sampled_objs)
            added_images.extend(sampled_objs["image_id"].tolist())
        
        return collected_objects, Counter(added_images)
    
             
    def _count_imgs_per_seed(self):
        """
        Count images per seed
        """
        seeds = self.sampled_data_df["synset"].unique()
        
        df_imgs_per_seed = pandas.DataFrame(columns=["category", "synset", "sample_type", "obj_names", "no_objects"])
        for seed in seeds:
            objs_seed = self.sampled_data_df[self.sampled_data_df["synset"] == seed]
            if len(np.unique(objs_seed["image_id"])) != len(objs_seed):
                print(seed, len(np.unique(objs_seed["image_id"])), "!=", len(objs_seed))
            names_seed = objs_seed["obj_names"] #.apply(lambda a: eval(a)[0])
            
            imgid_name = names_seed.str.cat(objs_seed["image_id"].astype(str), sep="$")
            unique_names_per_img = imgid_name.unique()
            names_seed_unique = [name_imgid.split("$")[0] for name_imgid in unique_names_per_img.tolist()]
            name_distr_unique = Counter(names_seed_unique)
            name_distr = Counter(names_seed)
            category = objs_seed["category"].tolist()[0]
            for (name, count) in name_distr.items():
                new_df_item = {"category": category, "synset": seed, "obj_names": name, "no_objects": count, "no_objnms_unique_per_img": int(name_distr_unique[name])}
                df_imgs_per_seed = df_imgs_per_seed.append(new_df_item, ignore_index=True)

        df_imgs_per_seed["no_objnms_unique_per_img"] = df_imgs_per_seed["no_objnms_unique_per_img"].astype(int)
        return df_imgs_per_seed
        
    def _determine_images2sample_per_name(self, counter_variant="no_objnms_unique_per_img", outfname=None):
        """
        Apply the following criteria to determine how many objects/images to sample for each object name. Criteria are based on the total number of each object name:
            no_objs <= 500 --> collect all
            no_objs > 500 --> sample 500 objects
            no_objects > 800 --> sample up to 1000 objects
        """
        print("\nSampling procedure:")
        print("no_objs <= 500 --> collect all")
        print("no_objs > 500 --> sample 500 objects")
        print("no_objects > 800 --> sample up to 1000 objects")
        
        df_imgs_per_seed = self._count_imgs_per_seed()
        df_imgs_per_seed.sort_values(by=[counter_variant], ascending=False)
        
        df_imgs_per_seed["imgs_to_sample"] = 0     
        for idx in range(len(self.thresholds)-1):
            indices = df_imgs_per_seed[counter_variant].between(self.thresholds[idx]-1, self.thresholds[idx+1], inclusive=False)
            imgs_gt = sum(df_imgs_per_seed[counter_variant][indices])
            names_gt = len(df_imgs_per_seed[counter_variant][indices])
            print("Intervall ({0:d}, {1:d}) images:\tObj. names: {2:d}\tImages: {3:d}".format(
                self.thresholds[idx]-1, self.thresholds[idx+1], 
                names_gt, imgs_gt))
            df_imgs_per_seed["imgs_to_sample"][indices==True] = df_imgs_per_seed[counter_variant][indices==True]

        
        #df_imgs_per_seed["imgs_to_sample"][df_imgs_per_seed[counter_variant]>500] = 500
        #indices = df_imgs_per_seed[counter_variant].between(800, 1001, inclusive=False)
        #df_imgs_per_seed["imgs_to_sample"][indices] = df_imgs_per_seed[counter_variant][indices]
        #df_imgs_per_seed["imgs_to_sample"][df_imgs_per_seed[counter_variant]>1000] = 1000
        
        for criterium in self.criteria_thresholds:
            operator, threshold, num2sample = criterium
            if operator == "gt":
                df_imgs_per_seed["imgs_to_sample"][df_imgs_per_seed[counter_variant]>threshold] = num2sample
            elif operator == "btw":
                indices = df_imgs_per_seed[counter_variant].between(threshold[0], threshold[1], inclusive=False)
                if num2sample == "all":
                    df_imgs_per_seed["imgs_to_sample"][indices] = df_imgs_per_seed[counter_variant][indices]
                else:
                    sys.stderr.write("No option implemented for " + str(criterium))
                    sys.exit()

        print("# possible images sampled from intervall #images in (%d,%d): %d" % (self.thresholds[0], self.thresholds[-1], sum(df_imgs_per_seed["imgs_to_sample"])))

        df_imgs_per_seed.sort_values(by=["category", "synset", "imgs_to_sample", ], ascending=False, inplace=True)
        
        if outfname != None:
            df_imgs_per_seed[["category", "synset", "obj_names", counter_variant,  "imgs_to_sample"]].to_csv(outfname, index=False)
        
        return df_imgs_per_seed
    
if __name__=="__main__":
    picker = SampledImages_Picker(sys.argv[1])
    df_imgs_per_seed = picker._determine_images2sample_per_name(
            counter_variant="no_objnms_unique_per_img", 
            outfname="sampling2_max1000per_objname.csv")
    collected_objects, image_counts = picker.sample_images(df_imgs_per_seed, allow_img_duplicates=False)
    print("Image duplicates: ", [c for c in image_counts.values() if c > 1])
    
    picker.save_sampled_data(collected_objects, sys.argv[2])
    print(collected_objects.groupby(by=["category"]).describe()["object_id", "count"].sort_values())
    print(collected_objects.groupby(by=["synset"]).describe()["object_id", "count"].sort_values())
    print(collected_objects.groupby(by=["obj_names"]).describe()["object_id", "count"].sort_values())
