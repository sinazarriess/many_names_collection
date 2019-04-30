# -*- coding: utf-8 -*-
import os
import sys

from collections import Counter, defaultdict

import enchant

def _load_personal_dictionary(word_file):
    """
    Loads a dictionary from a list of words.
    """
    return enchant.PyPWL("all_valid_object_names.txt")

def word_exists(dictionary, word):
    return dictionary.check(word)

def suggestions(dictionary, word):
    return dictionary.suggest(word)

def check_word_list(all_words, dict2load="en"):
    """
    @param dict2load: default is aspell English dictionary
    """
    dictionary = enchant.Dict(dict2load)
    
    checked_words = defaultdict(list)
    for word in all_words:
        if dictionary.check(word):
            checked_words[word] = [True, ""]
        else:
            word2 = word.replace(" ", "")
            if dictionary.check(word2):
                checked_words[word] = [False, word2]
            else:
                checked_words[word] = [False, dictionary.suggest(word2)]
            
    return checked_words
    
def _write_checks(checked_words, foutname):
    with open(outfname, "w") as fout:
        fout.write("{0}\t{1}\t{2}\n".format("word", "correct", "suggestions"))
        for (word, feedback) in checked_words.items():
            fout.write("{0}\t{1[0]}\t{1[1]}\n".format(word, feedback))

if __name__=="__main__":
    if len(sys.argv) < 2:
        print("Please give a me a txt file (1 name per line) as argument")
        sys.exit()
    
    wordfile = sys.argv[1]
    word_list = [a.strip() for a in open(wordfile)]
    checks = check_word_list(word_list)
    outfname = os.path.join(os.path.dirname(wordfile), "checked-"+os.path.basename(wordfile).replace(".txt", ".tsv"))

    if False:
        test_name = sys.argv[1]
        # load some personal word list dictionary
        word_file = "all_valid_object_names.txt"
        my_dict = _load_personal_dictionary(word_file)
        # check if the word exists in the dictionary
        word_exists = word_exists(my_dict, test_name)
        print("word exists: ", word_exists)
        
        if not word_exists:
            # get suggestions for the input word if the word doesn't exist in the dictionary
            suggestions = suggestions(my_dict, test_name)

            print ("input:", test_name)
            print("suggestions:", suggestions)





