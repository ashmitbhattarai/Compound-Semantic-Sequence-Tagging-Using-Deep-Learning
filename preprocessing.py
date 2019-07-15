#! -*- coding:utf-8 -*-

import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from collections import defaultdict
import json
import numpy as np
INTENTS = {
    "appreciation": 1,
    "complain":2,
    "suggestion":3,
    "information":4,
    "opinion":5,
    "query":6,
    "demand":7

}

num_char_set = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
char2idx = {"PAD":0,"UNK":1}
for c in num_char_set:
    char2idx[c] = len(char2idx)


POS_TAGS = ['RB', 'eos', 'VB', 'VBP', 'JJR', 'TO', 'PDT', 'IN', 'POS', 'WDT', 'VBD', 'WRB', 'VBN', 'NNP', 'WP', 'JJ', 'RP', 'RBR', 'DT', 'NNS', 'CD', 'NN', 'O', 'FW', 'MD', 'JJS', 'RBS', 'UH', 'CC', 'NNPS', 'VBZ', 'EX', 'VBG', 'PRP']

INTENT_TAGS = ['v_q_neu_pr', 'ns__neu_npr', 'ns_s_p_pr', 'ns_ni_p_npr', 'ns_q_p_pr', 'v_d_p_npr', 'ns_n_neu_npr', 's_n_neu_p', 'ns_c_p_pr', 'ns_n_n_npr', 'o_s_n_pr', 'ns_n_neu_pr', 'v_a_neu_npr', 'ns_n_neu_nr', 'o_n_p_pr', 'o_n_n_pr', 's_i_p_pr', 'v_o_neu_pr', 's_a_p_pr', 'ns_e_neu_pr', 's_n_n_pr', 'ns_q_n_pr', 'o_n_neu_pr', 's_n_p_pr', 'o_d_neu_pr', 's_a_neu_npr', 'v_n_n_pr', 'v_s_p_npr', 'ns_a_p_pr', 'n_o_p_npr', 'ns_o_neu_npr', 'nn_i_neu_pr', 'v_o_neu_npr', 'o_e_p_pr', 'v_n_neu)_pr', 'o_i_p_pr', 'nn_n_neu_npr', 'v_i_neu_pr', 's_e_neu_pr', 'v_s_n_pr', 'ns_o_p_npr', 'o_c_neu_pr', 's_n_neu_pr', 'v_n_n_npr', 's_o_p_pr', 's_a_neu_pr', 'v_a_p_npr', 'v_o_p_npr', 'ns_q_neu_pr', 's_q_neu_pr', 'v_o_n_pr', 'v_d_neu_pr', 'ns_i_p_npr', 'ns_a_p_npr', 'v_n_i_pr', 'ns_n_p_pr', 'ns_i_n_pr', 'ns_a_neu_pr', 's_q_n_pr', 's_i_neu_pr', 'v_c_neu_pr', 's_s_neu_pr', 'ns_i_n_npr', 'o_o_n_pr', 'ns_q_n_npr', 'ns_n_eu_pr', 'ns_n_o_pr', 'ns_n_a_npr', 'vns_o_p_npr', 'ns_s_n_pr', 'o_i_n_pr', 'ns_o_p_pr', 'v_o_n_npr', 'o_q_neu_pr', 'v_n_op_pr', 'v_c_neu_npr', 'v_a_n_pr', 'v_s_neu_pr', 'o_a_p_pr', 'ns_p_neu_npr', 'ns_o_neu_n]pr', 'v_c_n_pr', 'o_s_neu_npr', 'v_s_neu_npr', 'v_s_p_pr', 'ns_a_neu_npr', 'v_n_p_n]pr', 'ns_c_neu_npr', 'ns_d_n_pr', 'ns_i_neu_npr', 'o_o_neu_pr', 'ns_n_pn_pr', 'o_i_p_npr', 'ns_c_n_pr', 'v_i_n_pr', 'v_a_p_pr', 's_n_neu_npr', 'v_i_p_npr', 'v_n_pneu_pr', 'v_n_p_pr', 'ns_i_neu_pr', 'ns_e_p_pr', 'ns_s_neu_pr', 'ns_n_c_pr', 'o_c_n_pr', 'ns_s_neu_npr', 'ns_p_neu_pr', 'v_d_neu_npr', 'ns_s_p_npr', 'vb_o_p_pr', 'nnp_a_p_pr', 'o_a_neu_pr', 'ns_o_n_npr', 's_i_n_npr', 'ns_c_n_npr', 'v_q_n_npr', 'v_i_neu_npr', 's_p_n_pr', 'v_d_p_pr', 'v_s_n_npr', 'v_c_p_npr', 'ns_e_p_npr', 'o_i_neu_pr', 'v_c_n_npr', 'ns_n_neu_nprv', 'ns_n_p_npr', 'v_n_p_npr', 'ns_q_neu_npr', 's_c_n_pr', 'v_n_i_npr', 'ns__neu_pr', 'ns_d_neu_npr', 'v_i_p_pr', 'o_n_neu_npr', 'v_d_n_pr', 'ns_n_n_pr', 'ns_n_pneu_npr', 'v_o_p_pr', 's_o_neu_pr', 's_o_neu_npr', 'ns_c_neu_pr', 'o_o_p_pr', 'ns_o_neu_pr', 'ns_o_n_pr', 'o_a_pneu_pr', 's_s_p_pr', 'ns_d_p_pr', 'ns_i_p_pr', 'ns_n_o_npr', 'v_e_neu_npr', 'ns_s_n_npr', 'ns_e_neu_npr', 'v_a_neu_pr', 'v_n_neu_npr', 'v_c_p_pr', 'o_q_p_pr', 'ns_neu_n_pr', 'v_q_neu_npr', 'v_q_n_pr', 'v_i_n_npr', 'ns_n_ne_npr', 'v_on_p_pr', 'v_n_neu_pr', 'p_c_neu_pr', 'ns_d_neu_pr', 'o_a_neu_npr', 'o_s_neu_pr', 's_c_neu_pr', 'o_s_p_pr']

def define_embeddings(word_embed="random",char_embed="random"):
    word_embeddings = np.random.uniform(-0.25,0.25,(27000,300))
    pos_tags_embeddings = np.random.uniform(-0.25,0.25,(200,50))

    if word_embed == "random" and char_embed=="random":
        return word_embeddings,pos_tags_embeddings
    elif word_embed == "glove":
        fEmbeddings = open("embeddings/glove.6B.100d.txt", encoding="utf-8")

        for line in fEmbeddings:
            split = line.strip().split(" ")
            word = split[0]

            if len(word2Idx) == 0:  # Add padding+unknown
                word2Idx["PADDING_TOKEN"] = len(word2Idx)
                vector = np.zeros(len(split) - 1)  # Zero vector vor 'PADDING' word
                wordEmbeddings.append(vector)

                word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
                vector = np.random.uniform(-0.25, 0.25, len(split) - 1)
                wordEmbeddings.append(vector)

            if split[0].lower() in words:
                vector = np.array([float(num) for num in split[1:]])
                wordEmbeddings.append(vector)
                word2Idx[split[0]] = len(word2Idx)
    return word_embeddings,pos_tags_embeddings

def read_data(filename):
    '''

    :param filename:
    :return multiple dataset:
    '''

    # set max length for words

    max_words = 1024
    max_chars = 36

    text_data = open(filename,"r").read().split('\n\n\n\n')
    
    ## load data as sentence

    # Input datasets
    sentence_list = []
    postags_list = []

    # Target/output lists
    intenttags_list = []
    intent_phrase_list = []
    characters_list = []
    max_count = 1

    pos_uniques = []
    label_uniques = []
    word_uniques = []
    for text_chunk in text_data:
        text_chunk = text_chunk.split("\n")

        ## get the tags
        sentence_text = text_chunk[0]
        # print (sentence_text)
        sentence_text = sentence_text.split("||")
        #sentiment_tag,intent_tag = sentence_text[1].split("||")
        intent_tag = sentence_text[-1].strip("|").strip()
        sentiment_tag = sentence_text[-2].strip("|").strip()

        sentence = []
        pos_tags = []
        intent_tags = []
        chars = []
        for word_text in text_chunk[1:]:
            # print (word_text)
            word_text = word_text.split(",")
            word = word_text[0]
            max_count = max(len(word),max_count)
            char = []
            for c in word:
                if c in char2idx:
                    char.append(char2idx[c])
                else:
                    char.append(char2idx["UNK"])
            POStag = word_text[1].replace("$","").strip()
            if word_text[2] != "eos":
                # intent_wtag = "_".join(word_text[2].replace("$","").strip().lower().split("_")[1:])
                intent_wtag = word_text[2].replace("$", "").strip().lower().split("_")

                assert(len(intent_wtag) == 5)

                intent_wtag.pop(-2)
                intent_wtag = "_".join(intent_wtag)

            # if word ends with ! it becomes !eos, . becomes .eos, nothing will be just eos and so on
            if POStag == "eos" and word.strip()=="":
                word = word.strip()
            if intent_wtag.strip() == "":
                print (word,POStag)
                print (sentence_text)
                print (word_text)

            # get the sentence
            sentence.append(word)
            #get the tags
            pos_tags.append(POStag)
            pos_uniques.append(POStag)
            word_uniques.append(word)
            # get intent_tags
            # intent_tags.append(intent_wtag.lower())
            intent_tags.append(INTENT_TAGS.index(intent_wtag))
            label_uniques.append(intent_wtag.lower())
            # get chars for every word
            chars.append(char)
        ### generate sentence list, postag, intent classifcation dataframe or tags
        # ready and pad the sentences etc
        sentence_list.append(sentence)
        postags_list.append(pos_tags)
        characters_list.append(chars)

        # generate numpy level array
        intenttags_list.append(intent_tag)
        intent_phrase_list.append(intent_tags)
    print (len(text_data))
    # print (list(set(pos_uniques)))
    print (len(list(set(label_uniques))))
    # print (list(set(label_uniques)))
    # print (len(list(set(word_uniques))))
    # print (len(intent_wtag))
    return sentence_list,postags_list,intenttags_list,intent_phrase_list,characters_list,max_count


# sentences,postags,intenttags,intent_words =read_data("data/sample.txt")


### convert the sentences to categories,
#print (sentences[0],postags[0],intenttags[0],intent_words[0])
# list of words,
def load_Sentence(wordembed="random",charembed="random"):

    maxchars = 45
    sentences, postags, intenttags, intent_words,char_list,max_count = read_data("data/latest.txt")
    # wordembeddings,POSembeddings,word2idx,char2idx = define_embeddings(wordembed,charembed)
    wordembeddings,POSembeddings = define_embeddings(wordembed,charembed)

    charembeddings = []
    tkns = Tokenizer(filters="",lower=False)
    tkns.fit_on_texts(sentences)

    # adding "UNK", found in medium blog
    tkns.word_index[tkns.oov_token] = len(tkns.word_index) + 1
    # print (len(tkns.word_index)+1)
    word2num = tkns.texts_to_sequences(sentences)

    pos_tokenizer= Tokenizer(filters="",lower=False)
    pos_tokenizer.fit_on_texts(postags)
    pos_tokenizer.word_index[pos_tokenizer.oov_token] = len(pos_tokenizer.word_index)+1

    pos2num = pos_tokenizer.texts_to_sequences(postags)

    ## having to work with wordIndex, charIndex, labelIndex


    #### print out the training statistics on there
    print ("Training and Validation Data Report")
    print ("_______________________________________________________________________________________________________________")
    print ("_______________________________________________________________________________________________________________")
    maxchars = max(max_count,maxchars)
    for idx,char_sent in enumerate(char_list):
        char_list[idx] = pad_sequences(char_sent,maxchars,padding="post")
        # print (word2num[idx],char_list[idx],sentences[idx],intent_words[idx])

    return_dict = {
        "word_embeddings":wordembeddings,
        "char_embeddings":charembeddings,
        "pos_embeddings":POSembeddings,
        "word2num":word2num,
        "pos2num":pos2num,
        "char2num":char_list,
        "int2num":intent_words
    }
    return word2num,pos2num,char_list,intent_words

### while parsing remember to remove STOP words because problem in tagging


# word2num,pos2num,char2num,int2num = load_Sentence()


