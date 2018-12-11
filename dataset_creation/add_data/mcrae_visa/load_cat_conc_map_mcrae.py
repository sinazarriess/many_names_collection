import os
import re

mcrae_fpath = os.path.join(os.path.abspath(".").split("datasets")[0], "datasets/mcrae_visa", "all_concepts_categories-us.csv")

def load(fname=mcrae_fpath, only_subcats=False, only_cats=False):
    cat_map = {}
    conc_subcat_map = {}
    subcats = []
    subcategory = ""
    for line in open(fname):
        if line.strip() == "":
            continue

        cat_match = re.search(r'<concepts category="(.+?)",\s+wn="n([^"]*?)"(,\s*ss=".+?")?>', line)
        if cat_match:
            category = re.sub("s$", "", cat_match.group(1))
            offset = int(cat_match.group(2))
            if only_cats:
                subcats = cat_map.setdefault(category, [])
            else:
                subcats = cat_map.setdefault(category, [(category, offset)])
            subcategory = ""
            continue
        
        subcat_match = re.search('<subcategory name="(.+?)",\s+wn="n([^"]*?)"(,\s*ss=".+?")?>', line)
        if subcat_match:
            subcategory = re.sub("s$", "", subcat_match.group(1).replace("_", " "))
            if only_subcats:
                offset = int(subcat_match.group(2))
                subcats = cat_map.setdefault(category, [(category, offset)])
                subcats.append(subcategory)
            continue
        
        concept, offset = line.strip().split(",n")
        # some temporary fix for, e.g., pepper,n07815956/n07815839
        if "/" in offset:
            offset = offset.split("/")[0]
        offset = int(offset)
        if len(concept) > 0:
            conc_subcat_map[concept] = subcategory
            conc_subcat_map[offset] = subcategory
            if not only_subcats:
                subcats.append((concept, offset)) 
    return cat_map, conc_subcat_map

def load_reverse(fname=mcrae_fpath, only_subcats=False):
    conc_cat_subcat_map = {}
    subcategory = ""
    for line in open(fname):
        offset = 0
        concept = ""
        if line.strip() == "":
            continue

        cat_match = re.search(r'<concepts category="(.+?)",\s+wn="n([^"]*?)"(,\s*ss=".+?")?>', line)
        if cat_match:
            category = re.sub("s$", "", cat_match.group(1))
            subcategory = ""
            continue
        
        subcat_match = re.search('<subcategory name="(.+?)",\s+wn="n([^"]*?)"(,\s*ss=".+?")?>', line)
        if subcat_match:
            if len(subcategory) > 0:
                subcategory += "/"
            subcategory = re.sub("s$", "", subcat_match.group(1).replace("_", " "))
            continue
        
        concept, offsets = line.strip().split(",n")
        # e.g., pepper,n07815956/n07815839
        if len(concept) > 0:
            if "/" in offsets:
                offsets = offsets.split("/")
            else:
                offsets = [offsets]
            for offset in offsets:
                offset = int(offset.replace("n", ""))
                conc_cat_subcat_map[offset] = {"concept": concept,
                                           "category": category,
                                           "subcategory": subcategory}
    return conc_cat_subcat_map
        
        
