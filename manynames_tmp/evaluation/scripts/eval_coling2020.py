import argparse
import os
import sys
from collections import Counter, defaultdict
from types import SimpleNamespace

import numpy as np
import pandas as pd

#sys.path.append("../analysis/")
import utils_vocab
from mn_loader import load_manynames, load_manynames_all
from create_data_splits import VG_MN_DataLoader

#data_dir = "data/"
#vg_data_dir = os.path.join(data_dir, "1600-400-20")

categories = SimpleNamespace( 
    hit='n_{top}',
    same_object='same-obj',
    other_object='other-object',
    # mistakes
    lowcount='singleton',
    lowadequacy='inadequ.',
    #off_related='related',
    off='off'
    )
    
category_list = list(categories.__dict__.values())
   
   
#vg_categories = SimpleNamespace( 
#    hit='hit',
#    same_object='same-obj'
#    )

MNV2 = load_manynames(os.path.join('../', 'proc_data_phase0', 'mn_v2.0', 'manynames-v2.0.tsv'))
with open(os.path.join("vg_data", "vg_name2aliases.tsv")) as f:
    VG_NM2ALIAS = dict([line.strip().split("\t") for line in f])

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Vanilla Object Classifier')
  parser.add_argument('--scores', dest='scores',
                      help='filename with prediction scores (csv or..)',
                      type=str)
  parser.add_argument('--log_file', dest='log_file',
                      help='filename for saving the results (csv/tsv)',
                      type=str, default=None)                    
  parser.add_argument('--imgids', dest='imgids',
                      help='filename with image ids, one per line',
                      type=str)
  parser.add_argument('--targetvocab', dest='vocab_vec',
                      help='filename with the target labels (one per line), matching the order of the prediction scores',
                      type=str)
  parser.add_argument('--nameset', dest='mn_nameset',
                      help='vocabulary: mn442, mnAll, or none',
                      type=str, default="mn442")
  parser.add_argument('--test', dest='test_data',
                      help='evaluate on test data',
                      type=bool)
  parser.add_argument('--outfname', dest='outfname',
                      help='filename into which to write the evaluation information',
                      type=str, default='eval_information.tsv',)
  parser.add_argument('--modelkey', dest='modelkey',
                      help='key identifying model',
                      type=str, default='Model')

  return parser.parse_args()    

#### HELPER FUNCTIONS ####
def _topnames(MNdf):
    return dict(zip(MNdf["vg_image_id"].values, MNdf["topname"].values))

def _alternative_names(MNdf):
    alt_names = MNdf.apply(lambda a: a["same_object"][a["topname"]], axis=1)
    return dict(zip(MNdf["vg_image_id"].values, alt_names.values))

# @deprecated: used for older version of tsv, where singletons were part of the "incorrect" column: {<name>: {count: 1}, ...}
def _low_count_names(MNdf, min_response_cnt=2):
    def _incorr_low_count(incorrect_annos, min_response_cnt):  
        return set([inm for (inm, anno) in incorrect_annos.items() if __has_low_count(anno, min_response_cnt)])
        
    return dict(zip(MNdf["vg_image_id"].values, 
            MNdf.apply(lambda a: _incorr_low_count(a["incorrect"], min_response_cnt), 
                    axis=1).values))
                    
def _singleton_names(MNdf):
    return dict(zip(MNdf["vg_image_id"].values, 
            MNdf.apply(lambda a: set(a["singletons"].keys()), axis=1).values))

def _inadequate_names(MNdf, min_response_cnt=2, adequacy_threshold=0.4):
    def _incorr_inadequate(incorrect_annos, min_response_cnt, adequacy_threshold):
        return set([inm for (inm, anno) in incorrect_annos.items() if not __has_low_count(anno, min_response_cnt) and __is_inadequate(anno, adequacy_threshold)])
        
    return dict(zip(MNdf["vg_image_id"].values, 
                    MNdf.apply(lambda a: _incorr_inadequate(a["incorrect"],
                        min_response_cnt, adequacy_threshold), axis=1).values))

def _other_objects_names(MNdf, min_response_cnt=2, adequacy_threshold=0.4):
    def _incorr_other_object(incorrect_annos, min_response_cnt, adequacy_threshold):
        return set([inm for (inm, anno) in incorrect_annos.items() if not __has_low_count(anno, min_response_cnt) and not __is_inadequate(anno, adequacy_threshold)])
    
    return dict(zip(MNdf["vg_image_id"].values, 
                    MNdf.apply(lambda a: _incorr_other_object(a["incorrect"],
                        min_response_cnt, adequacy_threshold), axis=1).values))

def __has_low_count(anno, min_response_cnt):
    return anno["count"] < min_response_cnt
    
def __is_inadequate(anno, adequacy_threshold):
    return anno["adequacy_mean"] <= adequacy_threshold
   
#### EVALUATION ####

class ErrorCats():
    def __init__(self, MNdf, VG_NM2ALIAS):
        # MN ground truth
        self.top_obj_names = _topnames(MNdf)
        self.alt_names = _alternative_names(MNdf)
        
        #self.low_cnt_names = _low_count_names(MNdf)
        self.low_cnt_names = _singleton_names(MNdf)
        self.oth_objs_names = _other_objects_names(MNdf)
        self.inadequ_names = _inadequate_names(MNdf)
        
        self.vg_nm2alias = VG_NM2ALIAS
        
    def error_category(self, pred_name, imgid, alias_gt_name=True):
        mn_top_orig = self.top_obj_names[imgid]
        if alias_gt_name is True:
            mn_top = self.vg_nm2alias.get(mn_top_orig)
        else:
            mn_top = mn_top_orig
        if pred_name == mn_top:
            return categories.hit
            
        elif pred_name in self.alt_names[imgid]:
            return categories.same_object
        
        elif pred_name in self.low_cnt_names[imgid]:
            return categories.lowcount
        
        elif pred_name in self.oth_objs_names[imgid]:
            return categories.other_object
        
        elif pred_name in self.inadequ_names[imgid]:
            return categories.lowadequacy

        return categories.off
    

def human_upper_bound(MNdf, MNv1, model_imgids, free_naming=False):
    """
    @param free_naming (bool):  Set True to compute upper bound on *all* valid names, False if to only compute upper bound on vocabulary of comparison model. default: False
    """
    if free_naming is True:
        return human_upper_bound_free_naming(MNdf, MNv1, model_imgids)
    with open(os.path.join("vg_data", "vg_name2aliases.tsv")) as f:
        mn2alias = dict([line.strip().split("\t") for line in f])
    evaluator = ErrorCats(MNdf, mn2alias)
    
    MNv1['topname'] = MNv1["spellchecked"].apply(lambda x: mn2alias.get(x.most_common(1)[0][0], x.most_common(1)[0][0]))
    vocab = set(MNv1['topname'].values)
        
    hum_error_cats = Counter()
    num_items = 0
    for img_id in model_imgids:
        rowh = MNv1.query('vg_img_id == %d' % img_id).iloc[0]
        resp = rowh["spellchecked"]
        for name in resp:                
            if name in vocab:
                nm_alias = mn2alias.get(name, None)
                if nm_alias == None: # not in VG (TODO: use prepare_data etc.)
                    print("Skipping image %d -- name %s (original: %s) not in model vocab." % (img_id, nm_alias, name))
                    continue
                err = evaluator.error_category(nm_alias, img_id)
                if err == categories.off:
                    print("OFF: ", img_id, name, nm_alias)
                # weighted by name count
                hum_error_cats[err] += resp[name]
        num_items += 1
            
    return hum_error_cats, num_items
    
def human_upper_bound_free_naming(MNdf, MNv1, model_imgids):
    """
    @param free_naming (bool):  Set True to compute upper bound on *all* valid names, False if to only compute upper bound on vocabulary of comparison model. default: False
    """
    with open(os.path.join("vg_data", "vg_name2aliases.tsv")) as f:
        mn2alias = dict([line.strip().split("\t") for line in f])
    evaluator = ErrorCats(MNdf, mn2alias)
    
    MNv1['topname'] = MNv1["spellchecked"].apply(lambda x: mn2alias.get(x.most_common(1)[0][0], x.most_common(1)[0][0]))
    vocab = set(MNv1['topname'].values)
        
    hum_error_cats = Counter()
    num_items = 0
    for img_id in model_imgids:
        rowh = MNv1.query('vg_img_id == %d' % img_id).iloc[0]
        resp = rowh["spellchecked"]
        for name in resp:                
            if name in vocab:
                err = evaluator.error_category(name, img_id, alias_gt_name=False)
                if err == categories.off:
                    print("OFF: ", img_id, name)
                # weighted by name count
                hum_error_cats[err] += resp[name]
        num_items += 1
            
    return hum_error_cats, num_items

def evaluate(model_scores, model_imgids, vocab_vec, vocab2idx, log_fname=None, verbose=False):
    classes = {cl:0 for cl in category_list}
    write_log = log_fname is not None
    error_cat = None
    if write_log:
        log_file = open(log_fname, "w")
        log_file.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (
            "vg_image_id", "error_category", 
            "topname", "adequacy_mean", 
            "pred_name", "pred_adequacy_mean", "pred_score_converted"))
    
    # ignore __background__
    all_top_preds_inds = np.argmax(model_scores[:,1:], axis=1)+1
    
    # MN ground truth
    top_obj_names = _topnames(MNV2)
    alt_names = _alternative_names(MNV2)
    
    #low_cnt_names = _low_count_names(MNV2)
    low_cnt_names = _singleton_names(MNV2)
    oth_objs_names = _other_objects_names(MNV2)
    inadequ_names = _inadequate_names(MNV2)
    
    total_instances = 0
    for (idx, imgid) in enumerate(model_imgids):
        # TODO idx --> MN name
        top_pred_idx = all_top_preds_inds[idx]
        pred_name = vocab_vec[top_pred_idx]
        mn_top_orig = top_obj_names[imgid]
        mn_top = VG_NM2ALIAS.get(mn_top_orig)            
        if verbose and mn_top != mn_top_orig:
            print("name aliased: %s --> %s" % (mn_top_orig, mn_top))
        
        if mn_top == None: # not in VG (TODO: use prepare_data etc.)
            print("Skipping image %d -- name %s not in model vocab." % (imgid, mn_top))
            continue
        
        mn_top_idx = vocab2idx[mn_top]
        if len(mn_top_idx) > 1:
            sys.stderr.write("More than one class found for MN-Top %s (image %d).\n" % (mn_top, imgid))
        else:
            mn_top_idx = mn_top_idx[0]
        
        if pred_name == mn_top:
            if verbose:
                print("HIT")
            classes[categories.hit] += 1
            error_cat = categories.hit
            
        elif pred_name in alt_names[imgid]:
            if verbose:
                print("alternative name -- pred: %s  ||  gt: %s (image %d)" % (pred_name, mn_top, imgid))
            classes[categories.same_object] += 1
            error_cat = categories.same_object
        
        elif pred_name in low_cnt_names[imgid]:
            if verbose:
                print("low count name -- pred: %s  ||  gt: %s (image %d)" % (pred_name, mn_top, imgid))
            classes[categories.lowcount] += 1
            error_cat = categories.lowcount
        
        elif pred_name in oth_objs_names[imgid]:
            if verbose:
                print("name of other object -- pred: %s  ||  gt: %s (image %d)" % (pred_name, mn_top, imgid))       
            classes[categories.other_object] += 1
            error_cat = categories.other_object
        
        elif pred_name in inadequ_names[imgid]:
            if verbose:
                print("low adequacy name -- pred: %s  ||  gt: %s (image %d)" % (pred_name, mn_top, imgid))
            classes[categories.lowadequacy] += 1
            error_cat = categories.lowadequacy
        
        else:
            if verbose:
                print("OFF? -- pred: %s  ||  gt: %s (image %d)" % (pred_name, mn_top, imgid))
            # TODO
            #classes[categories.off_related] += 1
            classes[categories.off] += 1
            error_cat = categories.off
        
        total_instances += 1
        if write_log:
            log_file.write("%s\n" % "\t".join(["%d"%imgid, error_cat, "?", 
                            mn_top, "-1", pred_name, "-1", "-1"]))
    
    if write_log:
        log_file.close()
        
    return classes


def convert_scores_to_mnvg(vg_model_scores, pred_imgids, inds2zero):
    """
    Sets names not in vg_mn to zero and recalculates distribution (softmax).
    @param inds2zero The indices of the elements to be set to 0.
    """
    sum_diffs = 0.0
    for idx in range(len(pred_imgids)):
        #diffs = sum(vg_model_scores[idx][inds2zero])
        sum_diffs += sum(vg_model_scores[idx][inds2zero])
        vg_model_scores[idx][inds2zero] = 0
        #s2 = sum(vg_model_scores[idx][inds2zero])
        vg_model_scores[idx] = vg_model_scores[idx] / sum(vg_model_scores[idx]) 
        #print(diffs, s2)
    print("Total sum of differences after score conversion to relevant classes: %.3f (avg=%.6f)" % (sum_diffs, sum_diffs / float(len(pred_imgids))))
    return vg_model_scores

if __name__=="__main__":
    args = parse_args()
    EVAL_DOMAINS = True
    
    ########## LOAD AND SETUP DATA ###########
    # load model predictions, images, vector with targets (target_vec)
    model_output_dir = os.path.dirname(args.scores)
    if os.path.splitext(args.scores)[1] == ".csv":
        scores = np.loadtxt(args.scores, delimiter=',')
    else:
        sys.stderr.write("File format not supported: %s.\n" % (os.path.splitext(args.scores)[1]))
    pred_imgids = [int(img.strip().split(".")[0]) for img in open(args.imgids)]
    vocab_vec_fname = args.vocab_vec
    
    # test_vocab: set file name of vocabulary on which to evaluate
    MN_NAMESET = args.mn_nameset
    VGMN_DATALOADER = VG_MN_DataLoader(MN_NAMESET)
    
    if MN_NAMESET == "mn442":
        mnvg_path = "manynames442_vg1600"
        vocab_fname = os.path.join(
                VGMN_DATALOADER.data_dir, 
                "%s_vg1600_vocab.tsv" % MN_NAMESET)
    elif MN_NAMESET == "mnAll":
        mnvg_path = "manynamesAll_vg1600"
        vocab_fname = os.path.join(
                VGMN_DATALOADER.data_dir, 
                "%s_vg1600_vocab.tsv" % MN_NAMESET)
    #else:
    #    mnvg_path = ""
    #    vocab_fname = os.path.join(VGMN_DATALOADER.data_dir, 
    #                               "mn_vg_vocab.tsv")
    
    # create a joint vocabulary vector of predictions' target_vec and names in test_vocab
    mnvg_vocab_vec, mnvg2idx = utils_vocab.create_joint_vocab_vector(
                                            vocab_vec_fname, 
                                            vocab_fname)
    
    # retrieve ids of the images on which to evaluate
    if args.test_data is True:
        eval_imgids = [int(imgid) for imgid in open(os.path.join("data_splits", "mn_vg_imgids.test.txt"))]
    else:
        eval_imgids = VGMN_DATALOADER.load_vg_groundtruth().keys()    
        
    # filter prediction scores to those images on which to evalute
    scores, pred_imgids = VGMN_DATALOADER.filter_model_output(
                                    pred_imgids, 
                                    relevant_imgids=eval_imgids,
                                    scores=scores)


    # convert the prediction scores to the joint vocab vec (relevant for KL --> prob distribution)
    converted_model_scores = convert_scores_to_mnvg(scores, pred_imgids, mnvg2idx["O"]) 
    
    ########## RUN EVALUATION ###########
    domain_fnames = ["All.x"] #"Baseline.x", 
    cols = ["Model"]
    if EVAL_DOMAINS is True:
        cols.append("Domain")
        domain_dir = os.path.join("mn_data", "domains")
        domain_fnames.extend(os.listdir(domain_dir))
    cols.extend(category_list)
    cols.append("total #")     
    
    results = []
    for domain_f in domain_fnames:
        # TODO: human
        domain = os.path.splitext(domain_f)[0]
        if domain.lower() in ["baseline", "all", "human"]:
            model_imgids = pred_imgids
            model_scores = converted_model_scores
        else:
            domain_imgs = np.array([int(l.strip()) for l in open(os.path.join(domain_dir, domain_f))])
            model_scores, model_imgids = VGMN_DATALOADER.filter_model_output(
                            pred_imgids, relevant_imgids=domain_imgs, 
                            scores=converted_model_scores)
        #if domain.lower() == "baseline":
         #   model_scores = baseline_vecs
        #else:
        #    model_scores = converted_model_scores
        
        print("Domain: ", domain)
        
        # Human upper bound:
        MNv1 = load_manynames_all(os.path.join('../proc_data_phase0/spellchecking', 'all_responses_round0-3_cleaned.csv'))
        for (free_key, free_naming) in [("", False), ("-free", True)]:
            human_errors, total_items = human_upper_bound(MNV2, MNv1, model_imgids, free_naming=free_naming)
            total_human = sum(human_errors.values())
            human = {cat:s/total_human*100 for (cat, s) in human_errors.items()}
            res_vec = ["Human%s"%free_key]
            if EVAL_DOMAINS is True:
                res_vec.append(domain)
            res_vec.extend([human.get(err_class, 0) for err_class in category_list])
            res_vec.append(total_items)
            results.append(res_vec)
        
        # All images
        rel_classes = evaluate(model_scores, model_imgids, mnvg_vocab_vec, mnvg2idx, log_fname=args.log_file)
        res_vec = [args.modelkey]
        if EVAL_DOMAINS is True:
            res_vec.append(domain)
        total_items = sum(rel_classes.values())
        res_vec.extend([rel_classes[err_class]/total_items*100 for err_class in category_list])
        res_vec.append(total_items)
        results.append(res_vec)
        
    res_df = pd.DataFrame(results, columns=cols)

    
    #print(res_df.to_latex(float_format="%.1f", index=False))
    print("Vocabulary:", MN_NAMESET)
    sys.stdout.write(res_df.to_latex(float_format="%.1f", index=False))
    
