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

def review_results(mturk, path_published, 
                   max_num_answers, 
                   approve_all=False, 
                   do_rejections=True,
                   feedback2worker="",
                   statuses=["Submitted"], 
                   only_answer_length=False):
    '''ask AMT for results'''
    print("Reviewing")
    for filename in glob.glob(os.path.join(path_published, '*final.json')):
        print(filename)
        with open(filename, 'r') as handle:
            parsed = json.load(handle)

        outapproved_name = filename[:-5]+"_approved_done.txt"
        if approve_all is True:
            outsuspicious_name = filename[:-5]+"_suspicious_accepted.txt"
        elif do_rejections is True:
            outsuspicious_name = filename[:-5]+"_suspicious_rejected.txt"
        else:
            outsuspicious_name = filename[:-5]+"_suspicious_notrejected.txt"
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


                    info = "**HIT %s, Assignment %s, Worker %s**\t" % \
                    (item['HIT']['HITId'],assignment['AssignmentId'], assignment['WorkerId'])

                    marked, is_susp, susp_cnt = [''], False, 0
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
                            if do_rejections is True:
                                mturk.reject_assignment(
                                    AssignmentId=assignment['AssignmentId'],
                                    RequesterFeedback=feedback2worker
                                    )
                            else:
                                continue
                    else:
                        appr_file.write(info)
                        try:
                            appr_file.write(";".join(marked)+" (sus_cnt: %.1f)\n" % susp_cnt)
                        except UnicodeEncodeError:
                            appr_file.write(str(marked).strip()+" (sus_cnt: %.1f)\n" % susp_cnt)
                        #continue # hard-coded, change as needed
                        mturk.approve_assignment(
                            AssignmentId=assignment['AssignmentId'],
                            RequesterFeedback='Thank you for working for us!',
                            OverrideRejection=False
                        )
                        print("HIT {} approved.".format(assignment['AssignmentId']))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please give a me a config file as argument")
        sys.exit()
    
    approve_all = True
    only_answer_length = False
    if len(sys.argv) > 2:
        approve_all = sys.argv[2].lower()=="approve_all"
    
    data_path = os.path.dirname(sys.argv[1])
    
    CONFIG = configparser.ConfigParser()
    CONFIG.read(sys.argv[1])
    
    MTURK = amt_api.connect_mturk(CONFIG)
    
    max_answers = 10
    basepath = os.path.dirname(sys.argv[1])
    out_path = os.path.join(basepath, CONFIG['data']['admindir'])

    statuses = ["Submitted"]
    feedback2rejected_worker = 'HIT is rejected because (almost) no answers were given or the entered answers were often not names, but phrases (including adjectives, prepositions and conjunctions (e.g., "on", "with", "X of Y and Z") or even verbs).'
    review_results(MTURK, out_path, max_answers,
                   approve_all=approve_all, 
                   do_rejections=True, 
                   statuses=statuses, 
                   feedback2worker=feedback2rejected_worker,
                   only_answer_length=only_answer_length)
    
