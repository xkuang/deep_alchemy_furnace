# # GPU

import tensorflow as tf 
print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


# # Checkpoint

import h5py
import pickle

with h5py.File('allData.h5', 'r') as fh:
    x_train = fh['x_train'][:]
    y_train = fh['y_train'][:]
    x_val = fh['x_val'][:]
    y_val = fh['y_val'][:]
    x_train_all = fh['x_train_all'][:]
    y_train_all = fh['y_train_all'][:]
    x_test = fh['x_test'][:]
    y_test = fh['y_test'][:]
    embedding = fh['embedding'][:]

with open('index.pkl', 'rb') as fp:
    word2index, index2word, token2index, index2token = pickle.load(fp)


# # Build Model

# ##  Set Hyperparameters

MAX_SENT_LEN = 120
MAX_ADJL_LEN = 28
VOCAB_SIZE = 63491
NUM_CLASSES = 148
EMBEDDING_SIZE = 300

# 300-64-0.5-512-79-5-1
ENC_RNN_SIZE = 300
DEC_RNN_SIZE = 64
DROPOUT_RATE = 0.5
NUM_EPOCHS = 512
BATCH_SIZE = 79
STEPS_PER_EPOCH = 20
TEST_STEPS = len(x_test)//BATCH_SIZE

VALIDATION_STEPS = 3

# ## Import Libraries

from keras.layers import Input, Embedding, Dropout, Bidirectional, LSTM, TimeDistributed, Dense  
from keras.models import Model
import keras.backend as K
from keras.callbacks import*
from keras.utils import to_categorical

# ## Build Graph

K.clear_session()
sequence = Input(shape=(MAX_SENT_LEN,), name='INPUT') 
emb_seq = Embedding(VOCAB_SIZE, EMBEDDING_SIZE, weights=[embedding], mask_zero=True, input_length=MAX_SENT_LEN, trainable=True, name='EMBEDDING')(sequence)
emb_seq = Dropout(DROPOUT_RATE)(emb_seq)
blstm = Bidirectional(LSTM(ENC_RNN_SIZE, return_sequences=True, implementation=2), merge_mode='concat', name='ENC_BLSTM_1')(emb_seq)
blstm = Dropout(DROPOUT_RATE)(blstm)
blstm = Bidirectional(LSTM(ENC_RNN_SIZE, return_sequences=False, implementation=2), merge_mode='concat', name='ENC_BLSTM_2')(blstm)
blstm = Dropout(DROPOUT_RATE)(blstm)
context = RepeatVector(MAX_ADJL_LEN, name='CONTEXT')(blstm)
lstm = LSTM(DEC_RNN_SIZE, return_sequences=True, implementation=2, name='DEC_LSTM')(context)
lstm = Dropout(DROPOUT_RATE)(lstm)
output = TimeDistributed(Dense(NUM_CLASSES, activation='softmax'), name='OUTPUT')(lstm)
model = Model(inputs=sequence, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
# ## Train

def gen_label(s):
    """
    One-hot encoding
    """
    gen = to_categorical(s, num_classes=NUM_CLASSES)
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
        y = np.zeros((batch_size, MAX_ADJL_LEN, NUM_CLASSES))
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

gen_train_all = data_generator(x_train_all, y_train_all, BATCH_SIZE)
gen_test = data_generator_all(x_test, y_test, BATCH_SIZE)
gen_train = data_generator(x_train, y_train, BATCH_SIZE)
gen_val = data_generator(x_val, y_val, BATCH_SIZE)

# ## Continue Trian

# filename = 'cp_logs/.hdf5'
# model.load_weights(filename)
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

filepath = 'cp_logs/weights.{epoch:03d}-{val_loss:.6f}.hdf5'
log_string = 'tb_logs/300-64-0.5-512-79-20-3'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
tensorboard = TensorBoard(log_dir=log_string, 
                          histogram_freq=1, 
                          write_graph=False, 
                          write_grads=False, 
                          batch_size=BATCH_SIZE, 
                          write_images=True, 
                          embeddings_freq=1, 
                          embeddings_layer_names=None,
                          embeddings_metadata=None) 

history = model.fit_generator(gen_train, 
                              steps_per_epoch=STEPS_PER_EPOCH, 
                              epochs=NUM_EPOCHS, 
                              verbose=1,
                              callbacks=[checkpoint, tensorboard],
                              validation_data=gen_val, 
                              validation_steps=VALIDATION_STEPS)
