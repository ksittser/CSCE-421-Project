# model from https://towardsdatascience.com/word-level-english-to-marathi-neural-machine-translation-using-seq2seq-encoder-decoder-lstm-model-1a913f2dc4a7

import pandas as pd
import numpy as np
import string
from string import digits
#import matplotlib.pyplot as plt
#%matplotlib inline
import re
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model
from keras.models import load_model
import random

filename = input('Write input file name, or nothing to build from scratch: ')

with open('pseudocode.anno', 'r', encoding='utf-8') as f:
    lines_pseudo = f.read().split('\n')

with open('python-code.code', 'r', encoding='utf-8') as f:
    lines_python = f.read().split('\n')

num_samples = 10000
max_passage_len = 70

lines = list(zip(lines_pseudo,lines_python))
lines = [line for line in lines if len(line[0])<max_passage_len and len(line[1])<max_passage_len]
random.shuffle(lines)
lines = lines[:min(num_samples,len(lines))]
lines_pseudo = [line[0] for line in lines]
lines_python = [line[1] for line in lines]

lines_pseudo = ['START_ '+ln.strip()+' _END' for ln in lines_pseudo]
lines_python = ['START_ '+ln.strip()+' _END' for ln in lines_python]

print(len(lines_pseudo),len(lines_python),'num of passages')
print(max([len(ln) for ln in lines_pseudo]),max([len(ln) for ln in lines_python]),'max len of passages')


# Vocabulary of Pseudo
all_ps_words=set()
for ps in lines_pseudo:
    for word in ps.split():
        if word not in all_ps_words:
            all_ps_words.add(word)

# Vocabulary of Python
all_py_words=set()
for py in lines_python:
    for word in py.split():
        if word not in all_py_words:
            all_py_words.add(word)

# Max Length of source sequence
length_list=[]
for l in lines_pseudo:
    length_list.append(len(l.split(' ')))
max_length_src = np.max(length_list)

# Max Length of target sequence
length_list=[]
for l in lines_python:
    length_list.append(len(l.split(' ')))
max_length_tar = np.max(length_list)

input_words = sorted(list(all_ps_words))
target_words = sorted(list(all_py_words))

# Calculate Vocab size for both source and target
num_encoder_tokens = len(all_ps_words)
num_decoder_tokens = len(all_py_words)
num_decoder_tokens += 1 # For zero padding

# Create word to token dictionary for both source and target
input_token_index = dict([(word, i+1) for i, word in enumerate(input_words)])
target_token_index = dict([(word, i+1) for i, word in enumerate(target_words)])

# Create token to word dictionary for both source and target
reverse_input_char_index = dict((i, word) for word, i in input_token_index.items())
reverse_target_char_index = dict((i, word) for word, i in target_token_index.items())


lines = shuffle(lines)

X, y = lines_pseudo, lines_python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)


def generate_batch(X = X_train, y = y_train, batch_size = 128):
    ''' Generate a batch of data '''
    while True:
        for j in range(0, len(X), batch_size):
            encoder_input_data = np.zeros((batch_size, max_length_src),dtype='float32')
            decoder_input_data = np.zeros((batch_size, max_length_tar),dtype='float32')
            decoder_target_data = np.zeros((batch_size, max_length_tar, num_decoder_tokens),dtype='float32')
            for i, (input_text, target_text) in enumerate(zip(X[j:j+batch_size], y[j:j+batch_size])):
                for t, word in enumerate(input_text.split()):
                    encoder_input_data[i, t] = input_token_index[word] # encoder input seq
                for t, word in enumerate(target_text.split()):
                    if t<len(target_text.split())-1:
                        decoder_input_data[i, t] = target_token_index[word] # decoder input seq
                    if t>0:
                        # decoder target sequence (one hot encoded)
                        # does not include the START_ token
                        # Offset by one timestep
                        decoder_target_data[i, t - 1, target_token_index[word]] = 1.
            yield([encoder_input_data, decoder_input_data], decoder_target_data)


latent_dim = 50


# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb =  Embedding(num_encoder_tokens+1, latent_dim, mask_zero = True)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)

# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(num_decoder_tokens+1, latent_dim, mask_zero = True)
dec_emb = dec_emb_layer(decoder_inputs)

# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

# Use a softmax to generate a probability distribution over the target vocabulary for each time step
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = None
encoder_model = None
decoder_model = None

if not filename:

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Compile the model
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])


    train_samples = len(X_train)
    val_samples = len(X_test)
    batch_size = 128
    epochs = 25


    model.fit_generator(generator = generate_batch(X_train, y_train, batch_size = batch_size),
                        steps_per_epoch = train_samples//batch_size,
                        epochs=epochs,
                        validation_data = generate_batch(X_test, y_test, batch_size = batch_size),
                        validation_steps = val_samples//batch_size)


    # Encode the input sequence to get the "thought vectors"
    encoder_model = Model(encoder_inputs, encoder_states)

    # Decoder setup
    # Below tensors will hold the states of the previous time step
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    # Get the embeddings of the decoder sequence
    dec_emb2= dec_emb_layer(decoder_inputs)

    # To predict the next word in the sequence, set the initial states to the states from the previous time step
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
    decoder_states2 = [state_h2, state_c2]

    # A dense softmax layer to generate prob dist. over the target vocabulary
    decoder_outputs2 = decoder_dense(decoder_outputs2)

    # Final decoder model
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs2] + decoder_states2)

    model.save('model-t2.h5')
    encoder_model.save('model-t2-enc.h5')
    decoder_model.save('model-t2-dec.h5')
else:
    model = load_model(filename + '-model.h5')
    encoder_model = load_model(filename + '-enc.h5')
    decoder_model = load_model(filename + '-dec.h5')


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))

    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_token_index['START_']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += ' ' + sampled_char

        # Exit condition: either hit max length or find stop token.
        if (sampled_char == '_END' or len(decoded_sentence) > 50):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence


train_gen = generate_batch(X_train, y_train, batch_size = 1)


for k in range(100):
    (input_seq, actual_output), _ = next(train_gen)
    decoded_sentence = decode_sequence(input_seq)
    print('Input Pseudo sentence:', X_train[k][6:-4])
    print('Actual Python Translation:', y_train[k][6:-4])
    print('Predicted Python Translation:', decoded_sentence[:-4])
    print()
print('\n---\n')

val_gen = generate_batch(X_test, y_test, batch_size = 1)

for k in range(100):
    (input_seq, actual_output), _ = next(val_gen)
    decoded_sentence = decode_sequence(input_seq)
    print('Input Pseudo sentence:', X_test[k][6:-4])
    print('Actual Python Translation:', y_test[k][6:-4])
    print('Predicted Python Translation:', decoded_sentence[:-4])
    print()