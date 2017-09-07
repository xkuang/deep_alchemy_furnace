
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Data-Pre-Processing" data-toc-modified-id="Data-Pre-Processing-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Data Pre-Processing</a></div><div class="lev2 toc-item"><a href="#Load-Data" data-toc-modified-id="Load-Data-11"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Load Data</a></div><div class="lev2 toc-item"><a href="#Word-Segmentation" data-toc-modified-id="Word-Segmentation-12"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Word Segmentation</a></div><div class="lev2 toc-item"><a href="#Explore-the-Data" data-toc-modified-id="Explore-the-Data-13"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Explore the Data</a></div><div class="lev1 toc-item"><a href="#Word-Embedding" data-toc-modified-id="Word-Embedding-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Word Embedding</a></div><div class="lev2 toc-item"><a href="#Tokenize-Text" data-toc-modified-id="Tokenize-Text-21"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Tokenize Text</a></div><div class="lev2 toc-item"><a href="#Create-Word-Embeddings-with-GloVe" data-toc-modified-id="Create-Word-Embeddings-with-GloVe-22"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Create Word Embeddings with GloVe</a></div><div class="lev3 toc-item"><a href="#Read-Glove" data-toc-modified-id="Read-Glove-221"><span class="toc-item-num">2.2.1&nbsp;&nbsp;</span>Read Glove</a></div><div class="lev3 toc-item"><a href="#Use-Glove-to-Initialize-Embedding-Matrix" data-toc-modified-id="Use-Glove-to-Initialize-Embedding-Matrix-222"><span class="toc-item-num">2.2.2&nbsp;&nbsp;</span>Use Glove to Initialize Embedding Matrix</a></div><div class="lev1 toc-item"><a href="#Build-Dateset" data-toc-modified-id="Build-Dateset-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Build Dateset</a></div><div class="lev1 toc-item"><a href="#Save-Dataset" data-toc-modified-id="Save-Dataset-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Save Dataset</a></div><div class="lev1 toc-item"><a href="#Checkpoint" data-toc-modified-id="Checkpoint-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Checkpoint</a></div><div class="lev1 toc-item"><a href="#Build-Model" data-toc-modified-id="Build-Model-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Build Model</a></div><div class="lev2 toc-item"><a href="#Set-Hyperparameters" data-toc-modified-id="Set-Hyperparameters-61"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Set Hyperparameters</a></div><div class="lev2 toc-item"><a href="#Import-Libraries" data-toc-modified-id="Import-Libraries-62"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>Import Libraries</a></div><div class="lev2 toc-item"><a href="#Model-Visualization" data-toc-modified-id="Model-Visualization-63"><span class="toc-item-num">6.3&nbsp;&nbsp;</span>Model Visualization</a></div><div class="lev2 toc-item"><a href="#Train" data-toc-modified-id="Train-64"><span class="toc-item-num">6.4&nbsp;&nbsp;</span>Train</a></div><div class="lev1 toc-item"><a href="#Evaluate" data-toc-modified-id="Evaluate-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Evaluate</a></div>

# #  Data Pre-Processing

# ## Load Data

# In[1]:


import os


# In[2]:


def load_data(path):
    """
    Load date from file
    """
    input_file = os.path.join(path)
    with open(input_file, 'r', encoding='utf-8') as f:
        return f.read()


# In[3]:


train_data_path = 'train_data.txt'
train_data = load_data(train_data_path).strip().split('\t')[1:]
train_data = [line.split('\n')[:2] for line in train_data]
test_data_path = 'test_data.txt'
test_data = load_data(test_data_path).strip().split('\t')[1:]
test_data = [line.split('\n')[:2] for line in test_data]


# In[4]:


train_sent = [line[0] for line in train_data]
test_sent = [line[0] for line in test_data]
train_label = ['Causal' if line[-1] == 'Cause-Effect(e2,e1)' or line[-1] == 'Cause-Effect(e1,e2)' else 'Non-Causal' for line in train_data]
test_label = ['Causal' if line[-1] == 'Cause-Effect(e2,e1)' or line[-1] == 'Cause-Effect(e1,e2)' else 'Non-Causal' for line in test_data]


# ## Word Segmentation

# In[5]:


from nltk import regexp_tokenize


# In[6]:


filename = "stopwords.txt"
stopWords = {w: None for w in open(filename).read().split()}


# In[7]:


def cut(s):
    """
    Word segmentation
    """
    pattern = r'''
              (?x)                   # set flag to allow verbose regexps 
              (?:[A-Z]\.)+           # abbreviations, e.g. U.S.A. 
              |\d+(?:\.\d+)?%?       # numbers, incl. currency and percentages 
              |\w+(?:[-&']\w+)*      # words w/ optional internal hyphens/apostrophe 
           '''  
    return regexp_tokenize(s, pattern=pattern)

def find_pn(ws):
    """
    Find paired nominals
    """
    for i in range(len(ws)):
        if ws[i] == 'e1':
            for j in range(i+1, len(ws)):
                if ws[j] == 'e1':
                    pn1 = ws[i+1:j] 
        if ws[i] == 'e2':
            for j in range(i+1, len(ws)):
                if ws[j] == 'e2':
                    pn2 = ws[i+1:j]
    return pn1, pn2

def del_stop(ws):
    """
    Delete stopwords
    """
    return [i for i in [stopWords.get(i.lower(), i) for i in ws] if i != None]


# In[8]:


trainWords = [cut(s) for s in train_sent] 
testWords = [cut(s) for s in test_sent] 
causalSent = [[' '.join(trainWords[i])] for i in range(len(trainWords)) if train_label[i] == 'Causal']
trainWords = [del_stop(ws) for ws in trainWords]
testWords = [del_stop(ws) for ws in testWords]


# ## Explore the Data

# In[11]:


import numpy as np


# In[12]:


' '.join(trainWords[8])


# In[14]:


#trainPn1[8], trainPn2[8]


# In[15]:


train_label[8]


# In[16]:


len([i for i in train_label if i == 'Causal']) 


# In[17]:


len([i for i in test_label if i == 'Causal'])


# In[18]:


np.max([len(trainWords[i]) for i in range(len(train_label))])


# In[19]:


np.max([len(testWords[i]) for i in range(len(test_label))])


# In[20]:


np.max([len(trainWords[i]) for i in range(len(train_label)) if train_label[i] == 'Causal'])


# In[21]:


np.max([len(testWords[i]) for i in range(len(test_label)) if test_label[i] == 'Causal'])


# In[22]:


MAX_LEN = 45


# # Word Embedding

# ##  Tokenize Text

# In[23]:


from keras.preprocessing.text import Tokenizer


# In[24]:


tokWords = trainWords.copy()
tokWords.extend(testWords)
tokTexts = [' '.join(i) for i in tokWords]
tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(tokTexts)
word2index = tokenizer.word_index
index2word = {i: w for w, i in word2index.items()}
print('Found %s unique tokens.' % len(word2index))


# ## Create Word Embeddings with GloVe

# In[25]:


VOCAB_SIZE = 23595
EMBEDDING_SIZE = 300
SEED = 42


# ### Read Glove

# In[26]:


glove_n_symbols = 1917495
glove_index_dict = {}
glove_embedding_weights = np.empty((glove_n_symbols, EMBEDDING_SIZE))
globale_scale = 0.1
with open('/Users/lizhn7/Downloads/DATA/Glove/glove.42B.300d.txt', 'r') as fp:
    index = 0
    for l in fp:
        l = l.strip().split()
        word = l[0]
        glove_index_dict[word] = index
        glove_embedding_weights[index, :] = [float(n) for n in l[1:]]
        index += 1
glove_embedding_weights *= globale_scale


# ### Use Glove to Initialize Embedding Matrix

# In[27]:


from nltk import WordNetLemmatizer, PorterStemmer, LancasterStemmer


# In[28]:


# Generate random embedding with same scale as glove
np.random.seed(SEED)
shape = (VOCAB_SIZE, EMBEDDING_SIZE)
scale = glove_embedding_weights.std() * np.sqrt(12) / 2 
embedding = np.random.uniform(low=-scale, high=scale, size=shape)


# In[29]:


wnl = WordNetLemmatizer()
porter = PorterStemmer()
lancaster = LancasterStemmer()


# In[30]:


# Copy from glove weights of words that appear in index2word
count = 0 
for i in range(1, VOCAB_SIZE):
    w = index2word[i]
    g = glove_index_dict.get(w)
    if g is None:
        w = wnl.lemmatize(w)
        g = glove_index_dict.get(w)
    if g is None:
        w = porter.stem(w)
        g = glove_index_dict.get(w)
    if g is None:
        w = lancaster.stem(w)
        g = glove_index_dict.get(w)
    if g is not None:
        embedding[i, :] = glove_embedding_weights[g, :]
        count += 1
print('{num_tokens}-{per:.2f}% tokens in vocab found in glove and copied to embedding.'.format(num_tokens=count, per=count/float(VOCAB_SIZE)*100))


# # Build Dateset

# In[32]:


from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


# In[33]:


labelDict = {'Non-Causal': 0, 'Causal': 1}


# In[34]:


def convert_seq(ws, label):
    """
    Pad words sequene to MAX_LEN and encode label to one-hot encoding
    """
    sentText = [' '.join(i) for i in ws]
    sentSeq = tokenizer.texts_to_sequences(sentText)
    sentData = pad_sequences(sentSeq, maxlen=MAX_LEN, padding='post', truncating='post')
    labelData = np.array([[labelDict[i]] for i in label])
    return sentData, labelData 


# In[36]:


xTrain, yTrain = convert_seq(trainWords, train_label)
xTest, yTest = convert_seq(testWords, test_label)


# In[38]:


xTrain, _, yTrain, _ = train_test_split(xTrain, yTrain, test_size=0., random_state=SEED)
xTest, _, yTest, _ = train_test_split(xTest, yTest, test_size=0., random_state=SEED)
causalSent, _, causalSent, _ = train_test_split(causalSent, causalSent, test_size=0., random_state=SEED)


# # Save Dataset

# In[42]:


import h5py
import pickle


# In[43]:


fh = h5py.File('/Users/lizhn7/Downloads/DATA/semeval2010_task8_all_data/causal_detection/data', 'w')
fh['xTrain'] = xTrain
fh['yTrain'] = yTrain
fh['xTest'] = xTest
fh['yTest'] = yTest
fh['embedding'] = embedding
fh.close()


# In[44]:


with open('/Users/lizhn7/Downloads/DATA/semeval2010_task8_all_data/causal_detection/causalSent', 'wb') as fp:
    pickle.dump((causalSent), fp, -1)


# # Checkpoint

# In[45]:


import h5py
import pickle


# In[46]:


with h5py.File('/Users/lizhn7/Downloads/DATA/semeval2010_task8_all_data/causal_detection/data', 'r') as fh:
    xTrain = fh['xTrain'][:]
    yTrain = fh['yTrain'][:]
    xTest = fh['xTest'][:]
    yTest = fh['yTest'][:]
    embedding = fh['embedding'][:]


# In[47]:


with open('/Users/lizhn7/Downloads/DATA/semeval2010_task8_all_data/causal_detection/causalSent', 'rb') as fp:
    causalSent = pickle.load(fp)


# # Build Model

# ## Set Hyperparameters

# In[107]:


MAX_LEN = 45
VOCAB_SIZE = 23595
EMBEDDING_SIZE = 300
RNN_SIZE = 150
DROPOUT_RATE = 0.5
RNN_DROPOUT_RATE = 0.5
CNN_SIZE = 128
WINDOW_SIZE = 3
NUM_EPOCHS = 128
BATCH_SIZE = 32
STEPS_PER_EPOCH = 20
TEST_STEPS = len(xTest)//BATCH_SIZE+1


# In[108]:


print('NUM_EPOCHS: \t\t%d' % NUM_EPOCHS)
print('STEPS_PER_EPOCH: \t%d' % STEPS_PER_EPOCH)
print('TEST_STEPS: \t\t%d' % TEST_STEPS)

print('TRAIN_BATCHES: \t\t%d' % (NUM_EPOCHS * STEPS_PER_EPOCH))
print('NUM_BATCHES \t\t%d' % (len(xTrain)//BATCH_SIZE+1))


# ## Import Libraries

# In[110]:


from keras.layers import Input, Embedding, Dropout, Bidirectional, LSTM, Reshape, concatenate, Conv1D, BatchNormalization, GlobalMaxPooling1D, Dense
from keras.models import Model
import keras.backend as K
from keras.callbacks import*
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")


# In[111]:


K.clear_session()
seq = Input(shape=(MAX_LEN,), name='INPUT') 
emb_seq = Embedding(VOCAB_SIZE, EMBEDDING_SIZE, weights=[embedding], mask_zero=False, input_length=MAX_LEN, trainable=True, name='EMBEDDING')(seq)
emb_seq = Dropout(DROPOUT_RATE, name='DROPOUT_1')(emb_seq)
blstm = Bidirectional(LSTM(RNN_SIZE, return_sequences=True, implementation=0, dropout=RNN_DROPOUT_RATE, recurrent_dropout=RNN_DROPOUT_RATE), merge_mode='concat', name='BiLSTM_1')(emb_seq)
blstm = Dropout(DROPOUT_RATE, name='DROPOUT_2')(blstm)
blstm = Bidirectional(LSTM(RNN_SIZE, return_sequences=True, implementation=0, dropout=RNN_DROPOUT_RATE, recurrent_dropout=RNN_DROPOUT_RATE), merge_mode='concat', name='BiLSTM_2')(blstm)
blstm = Dropout(DROPOUT_RATE, name='DROPOUT_3')(blstm)
conv = Conv1D(CNN_SIZE, WINDOW_SIZE, padding='valid', activation='elu', name='CONV')(blstm)
pool = GlobalMaxPooling1D(name='MAXPOOLING')(conv)
pool = Dropout(DROPOUT_RATE, name='DROPOUT_4')(pool)
output = Dense(1, activation='sigmoid', name='OUTPUT')(pool)
model = Model(inputs=seq, outputs=output)
model.compile(loss='binary_crossentropy', optimizer='rmsprop')


# ## Model Visualization

# In[112]:


model.summary()
SVG(model_to_dot(model).create(prog='dot', format='svg'))


# ## Train

# In[114]:


def data_generator_all(data, label, batch_size):
    """
    Yield batches of all data
    """
    count = 0
    while True:
        if count >= len(data): 
            count = 0
        x = np.zeros((batch_size, MAX_LEN))
        y = np.zeros((batch_size, 1))
        for i in range(batch_size):
            n = i + count
            if n > len(data)-1:
                break
            x[i, :] = data[n]
            y[i, :] = label[n]
        count += batch_size
        yield (x, y)
        
def data_generator(data, label, batch_size): 
    """
    Yield batches 
    """
    index = np.arange(len(data))
    np.random.shuffle(index)    
    batches = [index[range(batch_size*i, min(len(data), batch_size*(i+1)))] for i in range(len(data)//batch_size)]
    while True:
        for i in batches:
            x, y = data[i], label[i]
            yield (x, y)


# In[115]:


gen_train = data_generator(xTrain, yTrain, BATCH_SIZE)
gen_test = data_generator_all(xTest, yTest, BATCH_SIZE)


# In[116]:


filepath = '/Users/lizhn7/Downloads/DATA/semeval2010_task8_all_data/causal_detection/cp_logs/1/weights.{epoch:03d}-{val_loss:.6f}.hdf5'
log_string = '/Users/lizhn7/Downloads/DATA/semeval2010_task8_all_data/causal_detection/tb_logs/1'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
tensorboard = TensorBoard(log_dir=log_string) 


# In[117]:


history = model.fit_generator(gen_train, 
                              steps_per_epoch=STEPS_PER_EPOCH, 
                              epochs=NUM_EPOCHS, 
                              verbose=1,
                              callbacks=[checkpoint, tensorboard],
                              validation_data=gen_test, 
                              validation_steps=TEST_STEPS)


# # Evaluate

# In[118]:


#threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
threshold = [i/10 for i in range(1, 9, 1)]


# In[119]:


def calculate(pred, actu, THRESHOLD):
    """
    Calculate Precision Recall F1-score
    """
    pred = [1 if i >= THRESHOLD else 0 for i in pred]
    actu = sum([list(i) for i in actu], [])
    CTP = sum([1 for i in range(len(pred)) if pred[i] == 1 and actu[i] == 1])
    CFN = sum([1 for i in range(len(pred)) if pred[i] == 0 and actu[i] == 1])
    CFP = sum([1 for i in range(len(pred)) if pred[i] == 1 and actu[i] == 0])
    CTN = sum([1 for i in range(len(pred)) if pred[i] == 0 and actu[i] == 0])
    NCTP = CTN
    NCFN = CFP
    NCFP = CFN
    NCTN = CTP
    CP = CTP/(CTP+CFP)
    CR = CTP/(CTP+CFN)
    CF1 = 2*CP*CR/(CP+CR)
    NCP = NCTP/(NCTP+NCFP)
    NCR = NCTP/(NCTP+NCFN)
    NCF1 = 2*NCP*NCR/(NCP+NCR)
    ACC = (CTP+CTN)/(CTP+CFP+CFN+CTN)
    print('Threshold: \t%.3f' % (THRESHOLD))
    print('Causal: \tPreciion %.3f \tRecall %.3f \tF1-score %.3f' % (CP, CR, CF1))
    print('Non-Causal: \tPreciion %.3f \tRecall %.3f \tF1-score %.3f' % (NCP, NCR, NCF1))
    print('Accuracy: \t%.3f' % (ACC))


# In[120]:


filename = '/Users/lizhn7/Downloads/DATA/semeval2010_task8_all_data/causal_detection/cp_logs/1/weights.033-0.121688.hdf5'
model.load_weights(filename)
result = model.predict(xTest, batch_size=BATCH_SIZE, verbose=1)

model.summary()

for THRESHOLD in threshold:
    calculate(result, yTest, THRESHOLD)
    print('————————————————————————')
    
SVG(model_to_dot(model).create(prog='dot', format='svg'))

