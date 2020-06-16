import os
import re

from nltk.corpus import wordnet as wn
from nltk.tag.stanford import StanfordPOSTagger
from nltk.parse.stanford import StanfordDependencyParser
from nltk.parse.stanford import StanfordNeuralDependencyParser

from itertools import chain

folders = os.path.abspath(".").split(os.sep)
if "\\" in os.sep:
    USRNAME = "C:\\Users\\Carina Silberer\\"
elif "/" in os.sep:
    USRNAME = "/home/u148188/"

stanford_dir = '/home/%s/models/' % (USRNAME)
'/home/carina/models/stanford-postagger-full-2018-02-27/'
modelfile = stanford_dir + '/stanford-postagger-full-2018-02-27/models/english-bidirectional-distsim.tagger'
jarfile = stanford_dir + '/stanford-postagger-full-2018-02-27/stanford-postagger.jar'

stanford_core_dir = stanford_dir + "stanford-corenlp-full-2018-02-27/"
parser_jarpath = stanford_core_dir + "stanford-corenlp-3.9.1.jar"
parser_modeljar = stanford_core_dir + 'stanford-corenlp-3.9.1-models.jar'
parser_modelpath = "edu/stanford/nlp/models/parser/nndep/english_UD.gz"


sizes = ['big','huge','large','little','long','short','small','tall','tiny']
colors = ['blue','red','green','yellow','white','black','gray','grey','pink',\
        'purple','rose','orange','brown','tan','dark']
locations = ['left','right','bottom','top','middle','side','corner','front','background',\
        'very','far','center','anywhere','upper','lower','leftmost','rightmost']

relations = ['below',
            'above',
            'between',
            'not',
            'behind',
            'under',
            'underneath',
            'by',
            'near',
            'with',
            'at',
            'that',
            'who',
            'beside',
            'besides']

## Stanford CoreNLP
def _to_lists_of_words(items):
    if isinstance(items, str): # single sentence
        return [items.split()]
    if not isinstance(items[0], list): 
        return [item.split() for item in items]
    return items

def _to_strings(items):
    if isinstance(items, str): # single sentence
        return [items]
    if not isinstance(items[0], str):
        return [" ".join(item) for item in items]  
    return items


# Stanford PoSTagger
def load_pos_tagger():
    return StanfordPOSTagger(modelfile, path_to_jar=jarfile)

def tag_refExp(refExps, pos_tagger=None):
    if not pos_tagger:
        pos_tagger = load_pos_tagger()
    #return [(word, "UNKN") for word in refExp.split()]
    return pos_tagger.tag_sents(_to_lists_of_words(refExps))


# Stanford Dependency Parser
def load_dep_parser():
    return StanfordNeuralDependencyParser(
        model_path=parser_modelpath,
        path_to_jar=parser_jarpath, path_to_models_jar=parser_modeljar, 
        java_options='-Xmx4g --add-modules java.se.ee',
        corenlp_options='')

def parse_refEpx(refExps, dep_parser=None):
    if not dep_parser:
        dep_parser = load_dep_parser()
    triples = []
    for parses in dep_parser.raw_parse_sents(_to_strings(refExps)):
        for parse in parses:
            tmp_triples = []
            for triple in graph2triples(parse):
                tmp_triples.append(triple)
            triples.append(tmp_triples)
    return triples

def graph2triples(dep_graph, node=None):
    """
    Modified code of nltk.parse.dependencygraph.triples().
    Extract dependency triples of the form:
    ((head word, head tag, head index), rel, (dep word, dep tag, dep index))
    """
    
    if not node:
        node = dep_graph.root

    head = (node['word'], node['ctag'], node['address'])
    for i in sorted(chain.from_iterable(node['deps'].values())):
        dep = dep_graph.get_by_address(i)
        yield (head, dep['rel'], (dep['word'], dep['ctag'], dep['address']))
        for triple in graph2triples(dep_graph, node=dep):
            yield triple
    

## TODO: loop over referring expressions and label attributes and head nouns
def get_refanno(tagged_words):
    head_found = False
    ann_words = []
    for (word,tag) in tagged_words:
        if word in locations:
            ann_words.append((word,"LOC"))
        elif word in sizes:
            ann_words.append((word,"SIZE"))
        elif word in colors:
            ann_words.append((word,"COLOR"))
        elif word in relations:
            # TODO: do something more clever here
            # following words will be attribute/name of a distractor object
            ann_words.append((word,"REL"))
        elif tag.startswith("JJ"): # carina
            ann_words.append((word, "UNKN"))
        elif not head_found and tag in ['NN','NNS']:
            ann_words.append((word,"NAME"))
            head_found = True
    # carina, temporary (all tags=UNKN when no tagger is used):
    if not head_found:
        ann_words.append((word, "UNKN")) # append last word
    return ann_words

def tag2pos(tag):
    tag_map = {"JJ": "a", "JJS": "a", "NN": "n", "NNS": "n", "None": None} # TODO
    return tag_map.get(tag, None)

def _apply(ssid1, ssid2, func):
    ss1 = ssid1
    ss2 = ssid2
    if isinstance(ssid1, str):
        ss1 = get_synset(ssid1)
    if isinstance(ssid2, str):
        ss2 = get_synset(ssid2)
    return func(ss1, ss2)

def get_synset(ssid):
    """
    >>> get_synset('girl.n.01')
    Synset('girl.n.01')
    """
    return wn.synset(ssid)

def get_synset_from_names(names, pos=None):
    names = [name.strip().replace(" ", "_") for name in names.split(",")]
    synsets = get_synset_all(names[0], pos)
    for synset in synsets:
        print("n%08d" % synset.offset(), names, synset.lemma_names())
        if set(names) == set(synset.lemma_names()):
            return synset

def get_ss_from_offset(offset):
    """
    >>> get_synset_from_offset('02084071-n')
    Synset('dog.n.01')
    >>> get_synset_from_offset('n02084071')
    Synset('dog.n.01')
    """
    print(offset[1:], offset[0], offset)
    if not offset[0].isdigit():
        offset = offset[1:] + "-" + offset[0]
    return wn.of2ss(offset)
    
def get_synset_first(word, pos=None):
    all_synsets = wn.synsets(word, pos)
    #if not all_synsets and pos != None:
    #    all_synsets = wn.synsets(word, None)
    if all_synsets:
        return all_synsets[0]
    return None

def get_synset_all(word, pos=None):
    return wn.synsets(word, pos)

def get_ss_first_name_and_lexfile(word, pos=None):
    synset = get_synset_first(word, pos=pos)
    lexfile_info = get_ss_lexfile_info(synset)
    return (get_synset_name(synset), lexfile_info)

def get_synset_name(ss):
    if not ss:
        return None
    return ss.name()

def get_synset_id(ss):
    if not ss:
        return None
    return wn.ss2of(ss)

def get_ss_lexfile_info(ss):
    if not ss:
        return None
    return ss.lexname()

def get_lch(ssid1, ssid2):
    """
    Returns lowest common hypernyms of synsets ssid1 and ssid2.
    >>> get_lch('girl.n.01', 'granddaughter.n.01')
    [Synset('person.n.01')]
    """
    ss1 = ssid1
    ss2 = ssid2
    if isinstance(ssid1, str):
        ss1 = get_synset(ssid1)
    if isinstance(ssid2, str):
        ss2 = get_synset(ssid2)
    return ss1.lowest_common_hypernyms(ss2)        

def get_path_similarity(ssid1, ssid2):
    return _apply(ssid1, ssid2, wn.path_similarity)

## some WN methods which may be useful at some point
def get_subtree(synset):
    """
    Returns a tree of synsets corresponding to all inherited hypernyms from synset to its root hypernym.
    :param synset: WN synset
    """
    return synset.tree(lambda s:s.hypernyms())

def get_all_hypernyms(synset):
    return synset.hypernym_paths()

def get_hyponyms(synset):
    return synset.hyponyms()

def get_hypernyms(synset):
    return synset.hypernyms()

def get_all_hypernyms_ids(synset):
    all_hypers = synset.hypernym_paths()
    all_path_ids = []
    for path in all_hypers:
        all_path_ids.append([ss.name() for ss in path])
    return all_path_ids

def get_max_path_length(synsets):
    return max([len(p) for p in ss.hypernym_paths() for ss in synsets])

def get_base_form(form, pos=None):
    """
    [WN doc:] Find a possible base form for the given form, with the given part of speech, by checking WordNet's list of exceptional forms, and by recursively stripping affixes for this part of speech until a form in WordNet is found.
    """
    return wn.morphy(form, pos)
    
if __name__=="__main__":
    parses = parse_refEpx(["the surfer", "a big wave"])
    print(parses)
    
    
    
