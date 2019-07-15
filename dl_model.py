#! -*- coding:utf-8 -*-

from keras import layers
import numpy as np
from keras.models import Model
from preprocessing import load_Sentence,INTENT_TAGS,define_embeddings
from keras.utils import Progbar
import tensorflow as tf
from keras import metrics
from keras import backend as K
from keras.initializers import RandomUniform
from keras.utils.np_utils import to_categorical


# suggestion part 1 for self Embedding level training of the dataset, using 1 level CNN??
# suggestion part 2 check what is happening in character level
# suggestion part 3 find some paper for multi output and study it
# https://towardsdatascience.com/advanced-keras-constructing-complex-custom-losses-and-metrics-c07ca130a618
# https://cwiki.apache.org/confluence/display/MXNET/Multi-hot+Sparse+Categorical+Cross-entropy
# https://en.wikipedia.org/wiki/Bidirectional_recurrent_neural_networks
# https://github.com/minimaxir/char-embeddings
### TENSORBOARD

##
#https://stackoverflow.com/questions/44861149/keras-use-tensorboard-with-train-on-batch

### GRAPHS >>>>>>> MOST IMPORTANT <<<<<<<
## TEST
## TRAIN VAL
## TEST VAL
## TRAIN error

# STANCE DETECTION
tf.logging.set_verbosity(tf.logging.INFO)


word2num,pos2num,char2num,int2num = load_Sentence()

def split_dev_train(word2num):

    length_clust = set()
    length_dict = {}
    for sent in word2num:
        length_clust.add(len(sent))
        length = len(sent)
        if str(length) not in length_dict:
            length_dict[str(length)] = []
        length_dict[str(length)].append(word2num.index(sent))
    length_tup = sorted(length_dict.items(),key= lambda x: len(x[1]),reverse=True)

    return length_tup,length_clust


length_dict,length_clust = split_dev_train(word2num)

def batch_generator(length_clust,word2num,pos2num,char2num,int2num):
    word2num = np.asarray(word2num)
    pos2num = np.asarray(pos2num)
    int2num = np.asarray(int2num)
    char2num = np.asarray(char2num)
    for key,indices in length_dict:
        # print (length_dict[str(length)])
        # indices = length_dict[str(length)]
        # print (len(indices))
        sents = word2num[indices]
        sents = [np.asarray(each) for each in sents]
        sents = np.asarray(sents)

        postags = pos2num[indices]
        postags = [np.asarray(each) for each in postags]
        postags = np.asarray(postags)

        chartags = char2num[indices]
        chartags = [np.asarray(each) for each in chartags]
        chartags = np.asarray(chartags)

        labels = int2num[indices]
        labels= [np.asarray(each) for each in labels]
        labels = np.asarray(labels)

        yield [sents,postags,chartags],labels

def custom_accseq2seq(y_pred,y_true):
    '''
    This was found in Keras GIT issues for sparse categorical loss function
    to compute accuracy in seq2seq
    :param y_pred:
    :param y_true:
    :return accuracy:
    '''
    print (y_pred.shape,y_true.shape)
    print (y_pred,y_true)
    y_pred_ = K.cast(K.argmax(y_pred,axis=-1),'float32')
    correct = K.cast(K.equal(y_true,y_pred_),'float32')
    flattened = K.flatten(correct)
    length = K.shape(flattened)[0]
    acc = tf.reduce_sum(flattened)/length
    return acc

word_embeddings,pos_tags_embeddings = define_embeddings()



words_inp = layers.Input(shape=(None,),dtype='int32',name='words')

word_embed = layers.Embedding(input_dim=word_embeddings.shape[0],output_dim=word_embeddings.shape[1],
                              weights = [word_embeddings],trainable=False)(words_inp)

question =  layers.Embedding(input_dim=word_embeddings.shape[0],output_dim=word_embeddings.shape[1],
                              weights = [word_embeddings],trainable=False)(words_inp)

pos_inp = layers.Input(shape=(None,),dtype='int32',name='pos_tags')

pos_embed = layers.Embedding(input_dim=pos_tags_embeddings.shape[0],output_dim=pos_tags_embeddings.shape[1],
                              weights = [pos_tags_embeddings],trainable=False)(pos_inp)

character_input=layers.Input(shape=(None,45,),name='char_input')
embed_char_out=layers.TimeDistributed(layers.Embedding(93,30,embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding')(character_input)
dropout= layers.Dropout(0.5)(embed_char_out)
conv1d_out= layers.TimeDistributed(layers.Conv1D(kernel_size=3, filters=30, padding='same',activation='tanh', strides=1))(dropout)
maxpool_out= layers.TimeDistributed(layers.MaxPooling1D(15))(conv1d_out)
char = layers.TimeDistributed(layers.Flatten())(maxpool_out)
char = layers.Dropout(0.5)(char)

word2lstm = layers.concatenate([word_embed,pos_embed,char])



# change later when multiple inputs are added onto the systems
blstm =  layers.Bidirectional(layers.LSTM(400,return_sequences=True,dropout=0.2,recurrent_dropout=0.2))(word2lstm)

# 7 classes for 551 categories
output1 = layers.TimeDistributed(layers.Dense(len(INTENT_TAGS),activation='softmax'))(blstm)

model = Model(inputs=[words_inp,character_input,pos_inp],outputs=[output1])

model.compile(loss='categorical_crossentropy',optimizer='nadam',metrics=["accuracy"])

model.summary()

# conditional random fields for seq2seq scikit learn CRFs

#### Now Build the epochss
# model.fit_generator(batch_generator(length_clust, word2num, pos2num, char2num, int2num),steps_per_epoch=1)
for epoch in range(5):
    # print (epoch)
    a = Progbar(len(length_clust)-1)
    loss_dt = []
    acc_dt = []
    count  = 0
    val_x = []
    val_y = []
    for inp,out in batch_generator(length_clust,word2num,pos2num,char2num,int2num):
        sents,postags,chars = inp
        print (chars.shape,sents.shape)

        y = np.expand_dims(out,-1)
        y = to_categorical(y, num_classes=len(INTENT_TAGS))
        # # print (y.shape,sents.shape)
        if count == 0:
            val_x = [sents,chars,postags]
            val_y = y
        else:
            [loss,acc]= model.train_on_batch([sents,chars,postags],y)
            loss_dt.append(loss)
            acc_dt.append(acc)
            a.update(count)
            # print("\n Training Accuracy is {0}, Training Loss is {1}".format(acc, loss))
            if count%50 == 0:
                [val_acc,val_loss] = model.test_on_batch(val_x,val_y)
                print ("\n Validation Accuracy is {0}, Validation Loss is {1}".format(val_loss,val_acc))
        count += 1
    acc = np.mean(np.asarray(acc_dt))
    loss= np.mean(np.asarray(loss_dt))
    print("\n Mean Training Accuracy is {0}, Mean Training Loss is {1}".format(acc, loss))

    a.update(count + 1)
    # [val_los,val_acc]=model.test_on_batch(x=[word,postag],y=y)
    #
    # print (val_los,val_acc)
