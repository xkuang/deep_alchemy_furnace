
# GPU

import tensorflow as tf 
print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))



# Checkpoint

import h5py
import pickle

with h5py.File('data/allData.h5', 'r') as fh:
    x_train = fh['x_train'][:]
    y_train = fh['y_train'][:]
    x_val = fh['x_val'][:]
    y_val = fh['y_val'][:]
    embedding = fh['embedding'][:]

with open('data/index.pkl', 'rb') as fp:
    word2index, index2word = pickle.load(fp)


    
# Build Model

##  Set Hyperparameters

MAX_SENT_LEN = 80
MAX_ADJL_LEN = 80
VOCAB_SIZE = 69169
EMBEDDING_SIZE = 300
RNN_SIZE = 128
DROPOUT_RATE = 0.2

NUM_EPOCHS = 256
BATCH_SIZE = 32
STEPS_PER_EPOCH = 80
VALIDATION_STEPS = len(x_val)//BATCH_SIZE+1

## Import Libraries

from keras.layers import Input, Embedding, Dropout, Bidirectional, LSTM, TimeDistributed, Dense  
from keras.models import Model
import keras.backend as K
from keras.callbacks import*
from keras.utils import to_categorical

## Build Graph

K.clear_session()
sequence = Input(shape=(MAX_SENT_LEN,), name='INPUT') 
emb_seq = Embedding(VOCAB_SIZE, EMBEDDING_SIZE, weights=[embedding], mask_zero=True, input_length=MAX_SENT_LEN, trainable=True, name='EMBEDDING')(sequence)
emb_seq = Dropout(DROPOUT_RATE)(emb_seq)
blstm = Bidirectional(LSTM(RNN_SIZE, return_sequences=True, implementation=2), merge_mode='concat', name='ENC_BLSTM_1')(emb_seq)
blstm = Dropout(DROPOUT_RATE)(blstm)
blstm = Bidirectional(LSTM(RNN_SIZE, return_sequences=True, implementation=2), merge_mode='concat', name='ENC_BLSTM_2')(blstm)
blstm = Dropout(DROPOUT_RATE)(blstm)
lstm = LSTM(RNN_SIZE, return_sequences=True, implementation=2, name='DEC_LSTM')(blstm)
lstm = Dropout(DROPOUT_RATE)(lstm)
output = TimeDistributed(Dense(VOCAB_SIZE, activation='softmax'), name='OUTPUT')(lstm)
model = Model(inputs=sequence, outputs=output)

## Train

def gen_label(s):
    """
    One-hot encoding
    """
    gen = to_categorical(s, num_classes=VOCAB_SIZE)
    return gen

def data_generator_all(data, label, batch_size):
    """
    Yield batches of all data
    """
    count = 0
    while True:
        if count >= len(data): 
            count = 0
        x = np.zeros((batch_size, MAX_SENT_LEN))
        y = np.zeros((batch_size, MAX_ADJL_LEN, VOCAB_SIZE))
        for i in range(batch_size):
            n = i + count
            if n > len(data)-1:
                break
            x[i, :] = data[n]
            y[i, :, :] = gen_label(label[n])
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
            x, y = data[i], np.array(list(map(gen_label, label[i])))
            yield (x, y)

gen_train = data_generator(x_train, y_train, BATCH_SIZE)
gen_val = data_generator_all(x_val, y_val, BATCH_SIZE)

filename = 'cp_logs/weights.024-2.211675.hdf5'
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

filepath = 'cp_logs/weights.{epoch:03d}-{val_loss:.6f}.hdf5'
log_string = 'tb_logs/continue_2'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True)
tensorboard = TensorBoard(log_dir=log_string, histogram_freq=1) 
#earlystop = EarlyStopping(monitor='val_loss', verbose=1, patience=2)

history = model.fit_generator(gen_train, 
                              steps_per_epoch=STEPS_PER_EPOCH, 
                              epochs=NUM_EPOCHS, 
                              validation_data=gen_val, 
                              validation_steps=VALIDATION_STEPS,
                              verbose=1,
                              callbacks=[checkpoint, tensorboard])
