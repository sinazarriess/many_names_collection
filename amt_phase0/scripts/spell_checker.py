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
    
    

if __name__=="__main__":
    if len(sys.argv) < 2:
        print("Please give a me a txt file (1 name per line) as argument")
        sys.exit()
        
    word_list = [a.strip() for a in open(sys.argv[1])]
    checks = check_word_list(word_list)

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





