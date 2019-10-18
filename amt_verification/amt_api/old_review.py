import boto3
import glob
import os
import json
import xmltodict
import sys
import nltk
import configparser

import amt_api

WORDS = set(nltk.corpus.words.words())
WNWORDS = set([n for n in nltk.corpus.wordnet.all_lemma_names()])
STOPWORDS = set(nltk.corpus.stopwords.words("english"))

# deprecated, see amt_api
def _get_assignments(mturk, hitid, statuses=['Submitted']):
    aresponse = mturk.list_assignments_for_hit(
                    HITId=hitid,
                    AssignmentStatuses=statuses)
    if aresponse["NumResults"] < 1:
        print("\nNo assignments yet for HIT %s." % (str(hitid)))
        return []
    
    anext = aresponse['NextToken']
    assignments = aresponse['Assignments']

    print("\nget Hit",hitid)

    while anext:
        nresponse = mturk.list_assignments_for_hit(
                    HITId=hitid,NextToken=anext,AssignmentStatuses=statuses)
        print(nresponse.keys())
        nextassign = nresponse['Assignments']
        assignments += nextassign

        if 'NextToken' in nresponse:
            anext = nresponse['NextToken']
        else:
            anext = None
            
    return assignments

def is_suspicious(answer_words, max_num_answers, allow_commas=False, only_answer_length=False):
    marked_answers = []
    total_answer_length = 0
    suscount = max_num_answers-len(answer_words)
    if allow_commas is True:
        answer_words = [a.split(",")[0] for a in answer_words]
        print(answer_words)
    threshold = 4.5
    
    for (idx, a) in enumerate(answer_words):
        total_answer_length += len(a.strip().replace(" ", ""))
        if only_answer_length is True:
            continue
        
        susfound = False
        if a in ["t shirt", "tshirt"]:
            a = "t-shirt"
        sus_found = False
        if " " in a:
            for (j, w) in enumerate(a.split()):
                if len(w) > 1 and w in STOPWORDS or (w.endswith("ing") and j > 0):
                    if w in ["of"] or w in WNWORDS:
                        suscount += 0.25
                    else:
                        suscount += 4
                    sus_found = True
        if sus_found is True:
            marked_answers.append("**"+a)
            continue
        if a not in WORDS and a not in WNWORDS:
            if a.replace(" ", "") not in WORDS and a.replace(" ", "") not in WNWORDS:
                #suscount += 1
                marked_answers.append("*"+a)
                continue
        marked_answers.append(a) 
        
    if only_answer_length is True:
        return answer_words, total_answer_length < 18, total_answer_length
    else:
        if total_answer_length < 18 or len(set(answer_words)) < 5:
            return marked_answers, True, suscount-threshold
        if suscount >= threshold:
            return marked_answers, True, suscount-threshold
        return marked_answers, False, suscount-threshold


def is_suspicious1(answer_words, max_num_answers, allow_commas=False):
    marked_answers = []
    total_answer_length = 0
    suscount = max_num_answers-len(answer_words)
    if allow_commas is True:
        answer_words = [a.split(",")[0] for a in answer_words]
        print(answer_words)
    threshold = len(answer_words) / 2 + 6
    
    for (idx, a) in enumerate(answer_words):
        total_answer_length += len(a.strip().replace(" ", ""))
        susfound = False
        if a in ["t shirt", "tshirt"]:
            a = "t-shirt"
        if " " in a:
            for w in a.split():
                if len(w) > 1 and w in STOPWORDS or w.endswith("ing"):
                    suscount += 5
                    susfound = True
                    marked_answers.append("**"+a)
                    """
                    if suscount >= threshold:
                        marked_answers.append("#")
                        marked_answers.extend(answer_words[idx+1:-1])
                        return marked_answers, True, suscount-threshold
                    """
                    continue
        if a not in WORDS and a not in WNWORDS:
            if a.replace(" ", "") not in WORDS and a.replace(" ", "") not in WNWORDS:
                suscount += 1
                susfound = True
                marked_answers.append("*"+a)
                """
                if suscount >= threshold:
                    marked_answers.append("#")
                    marked_answers.extend(answer_words[idx+1:-1])
                    return marked_answers, True, suscount-threshold
                """
                continue
        marked_answers.append(a) 
    
    if total_answer_length < 18 or len(set(answer_words)) < 5:
        return marked_answers, True, suscount-threshold
    if suscount >= threshold:
        return marked_answers, True, suscount-threshold
    return marked_answers, False, suscount-threshold

def is_suspicious0(answer_words, max_num_answers, allow_commas=False):
    total_answer_length = 0
    suscount = max_num_answers-len(answer_words)
    if allow_commas is True:
        answer_words = [a.split(",")[0] for a in answer_words]
        print(answer_words)
    num_diff_ans = len(set(answer_words))
    
    for a in answer_words:
        total_answer_length += len(a.strip())
        if " " in a:
            for w in a.split():
                if w in STOPWORDS:
                    suscount += len(a.split())
                    if suscount >= num_diff_ans/2:
                        return True, suscount-num_diff_ans/2
        if a not in WORDS and a not in WNWORDS:
            suscount += 1
            if suscount >= num_diff_ans/2:
                return True, suscount-num_diff_ans/2
    
    if total_answer_length < 18:
        return True, suscount-num_diff_ans/2
    return False, suscount-num_diff_ans/2

def review_results(mturk, path_published, 
                   max_num_answers, 
                   approve_all=False, 
                   statuses=["Submitted"], 
                   only_answer_length=False):
    '''ask AMT for results'''
    for filename in glob.glob(os.path.join(path_published, '*final.json')):
        print(filename)
        with open(filename, 'r') as handle:
            parsed = json.load(handle)

        infix = "-".join(statuses)
        outapproved_name = filename[:-5]+"_approved-%s.txt" % (infix)
        outsuspicious_name = filename[:-5]+"_suspicious-%s.txt" % (infix)

        appr_file = open(outapproved_name,'w')
        susp_file = open(outsuspicious_name,'w')

        for item in parsed:
            #print(item)
            assignments = amt_api.get_assignments(mturk, item['HIT']['HITId'], statuses=statuses)
            print("HIT/Assignments",item['HIT']['HITId'],len(assignments))

            if len(assignments) > 0:
                for assignment in assignments:
                    answer_dict = xmltodict.parse(assignment['Answer'])
                    answers = []
                    for a in answer_dict['QuestionFormAnswers']['Answer']:
                        param = a['QuestionIdentifier']
                        if not 'objname-ex' in param:
                            if 'objname' in param:
                                if a['FreeText']:
                                    answers.append(a['FreeText'].lower())


                    info = "**HIT %s, Assignment %s, Worker %s, Status %s**\t" % (item['HIT']['HITId'], assignment['AssignmentId'], assignment['WorkerId'], assignment['AssignmentStatus'])

                    marked, is_susp, susp_cnt = is_suspicious(answers, max_num_answers, only_answer_length=only_answer_length)
                    if is_susp:
                        susp_file.write(info)
                        susp_file.write(";".join(marked)+" (sus_cnt: %.1f)\n" % susp_cnt)
                        if approve_all is True:
                            mturk.approve_assignment(
                                AssignmentId=assignment['AssignmentId'],
                                RequesterFeedback='Thank you for working for us!',
                                OverrideRejection=False
                                )
                        else:
                            continue # hard-coded, change as needed
                            mturk.reject_assignment(
                                AssignmentId=assignment['AssignmentId'],
                                RequesterFeedback='HIT is rejected because no answers were given or the entered answers were often not names, but phrases (including adjectives, prepositions (e.g., "on", "with") or even verbs).'
                                )
                    else:
                        appr_file.write(info)
                        try:
                            appr_file.write(";".join(marked)+" (sus_cnt: %.1f)\n" % susp_cnt)
                        except UnicodeEncodeError:
                            appr_file.write(str(marked).strip()+" (sus_cnt: %.1f)\n" % susp_cnt)
                        continue
                        mturk.approve_assignment(
                            AssignmentId=assignment['AssignmentId'],
                            RequesterFeedback='Thank you for working for us!',
                            OverrideRejection=False
                        )
                    #assignment_out_dict['Answers'] = answer_out_dict
                    #hit_out_dict['Assignments'].append(assignment_out_dict)



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please give a me a config file as argument")
        sys.exit()
    
    approve_all = False
    only_answer_length = False
    if len(sys.argv) > 2:
        approve_all = sys.argv[2].lower()=="approve_all"
    
    data_path = os.path.dirname(sys.argv[1])
    
    CONFIG = configparser.ConfigParser()
    CONFIG.read(sys.argv[1])
    
    MTURK = amt_api.connect_mturk(CONFIG)
    
    max_answers = 10
    path_published = data_path
    statuses = ["Submitted", "Approved", "Rejected"]
    review_results(MTURK, path_published, max_answers, 
                   approve_all=approve_all, 
                   statuses=statuses,
                   only_answer_length=only_answer_length)
    
