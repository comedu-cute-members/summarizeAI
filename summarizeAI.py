#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
임포트
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import re
from tokenizers import BertWordPieceTokenizer


# 단어 모음 생성
tokenizer = BertWordPieceTokenizer(clean_text=True, lowercase=True)
tokenizer.train(files="News_text.txt", vocab_size = 8000, special_tokens = [
    "[SOS]", "[EOS]",
])

START_TOKEN, END_TOKEN = [0], [1]  # <sos> 와 <eos>
VOCAB_SIZE = 8000


# In[33]:

MAX_INPUT_LENGTH = 1000
MAX_OUTPUT_LENGTH = 600

# In[35]:

BATCH_SIZE = 32

# ## transformer 아키텍처 구현

# In[36]:


# 인코더&디코더(포지셔널 인코딩)
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model, name="Positional_Encoding"):
        super(PositionalEncoding, self).__init__(name=name)
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        return position * (1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32)))

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position = tf.range(position, dtype = tf.float32)[:, tf.newaxis],
            i = tf.range(d_model, dtype = tf.float32)[tf.newaxis, :],
            d_model = d_model
        )
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        angle_rads = np.zeros(angle_rads.shape)
        angle_rads[:, 0::2] = sines # even index -> sin
        angle_rads[:, 1::2] = cosines # odd index  -> cos
        pos_encoding = tf.constant(angle_rads)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)
    
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


# In[37]:


# 패딩 마스크 생성
def create_padding_mask(name):
    inputs = tf.keras.Input(shape=(None,))
    mask = tf.cast(tf.math.equal(inputs,0), tf.float32) # [[1,2,0,2,1]] => [[0.,0.,1.,0.,0.]]
    return tf.keras.Model(inputs=inputs, outputs=mask[:,tf.newaxis, tf.newaxis, :], name=name) # 차원 추가


# In[38]:


def create_look_ahead_mask():
    inputs = tf.keras.Input(shape=(None,))
    seq_len = tf.shape(inputs)[1] # [[1,2,0]] => 3
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0) # 모든 원소가 1인 하삼각행렬
    padding_mask = create_padding_mask(name="look_ahead_padding")(inputs=inputs) # x에서 0이었던 부분만 1로 바꿔진 행렬
    return tf.keras.Model(inputs=inputs, outputs=tf.maximum(look_ahead_mask, padding_mask), name="look_ahead_mask")


# In[39]:


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, batch_size, d_model, num_heads, name="Multi_Head_Attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0 # d_model 사이즈의 행렬을 num_heads로 나눠야하기 때문
        
        self.depth = d_model // self.num_heads
        
        # WQ, WK, WV 정의 : d_model 길이의 밀집층(가중치 행렬)
        self.query_dense = tf.keras.layers.Dense(units=d_model) # WQ (size: d_model * d_k)
        self.key_dense = tf.keras.layers.Dense(units=d_model) # WK (size: d_model * d_k)
        self.value_dense = tf.keras.layers.Dense(units=d_model) # WV (size: d_model * d_v)
        
        # WO
        self.dense = tf.keras.layers.Dense(units=d_model) # size: transpose hd_v * d_model
    
    def call(self, query, key, value, mask):
        
        batch_size = tf.shape(query)[0]

        # 신경망 지나기
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # 헤드 나누기
        query = tf.reshape(query, shape=(batch_size, -1, self.num_heads, self.depth))
        key = tf.reshape(key, shape=(batch_size, -1, self.num_heads, self.depth))
        value = tf.reshape(value, shape=(batch_size, -1, self.num_heads, self.depth))
        
        query = tf.transpose(query, perm=[0,2,1,3])
        key = tf.transpose(key, perm=[0,2,1,3])
        value = tf.transpose(value, perm=[0,2,1,3])

        #스케일 닷 프로덕트 어텐션
        multiple_QandK = tf.matmul(query, key, transpose_b=True) # 행렬 곱셈
        d_k = tf.cast(tf.shape(key)[-1], tf.float32)
    
        # 어텐션 에너지
        energy = multiple_QandK / (d_k ** 0.5)
        
        # mask multihead attention 일 때
        if mask is not None :
            energy += (mask * -1e10)

        # 어텐션 스코어
        attention_weights = tf.nn.softmax(energy, axis = -1)

        # scaled dot product attention
        scaled_dot_attention = tf.transpose(tf.matmul(attention_weights, value), perm=[0,2,1,3]) # 행렬 곱셈

        # concat 
        concat_attention = tf.reshape(scaled_dot_attention, (batch_size, -1, self.d_model))

        # WO 밀집층 레이어 지나기
        outputs = self.dense(concat_attention)

        return outputs


# In[40]:


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, batch_size, dff, d_model, num_heads, dropout, name, epsilon=1e-6):
        super(EncoderLayer, self).__init__(name=name)
        self.batch_size = batch_size
        self.dff = dff
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.epsilon = epsilon
        
        # layers
        self.multi_head_attention = MultiHeadAttention(self.batch_size, self.d_model, self.num_heads)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=self.epsilon)
        self.FFN1 = tf.keras.layers.Dense(units=self.dff, activation='relu')
        self.FFN2 = tf.keras.layers.Dense(units=self.d_model)
        
    def call(self, inputs, padding_mask):
        
        # input size == output size
        # dropout after each sublayer
        
        # first sub layer - Multi-head Attention
        attention = self.multi_head_attention(query=inputs, key=inputs, value=inputs, mask=padding_mask)
        attention = tf.keras.layers.Dropout(rate=self.dropout)(attention)
        
        # add, normalization
        attention_norm = self.norm(inputs + attention)

        # second sub layer - Feed Forward layer
        outputs = self.FFN1(attention_norm)
        outputs = self.FFN2(outputs)
        outputs = tf.keras.layers.Dropout(rate=self.dropout)(outputs)

        # add, normalization
        outputs = self.norm(attention_norm + outputs)
        
        return outputs


# In[41]:


# real 인코더
def encoder(batch_size, vocab_size, num_layers, dff, d_model, num_heads, dropout, name="encoder"):

    inputs = tf.keras.Input(shape=(None,), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    # input embedding, positional encoding
    input_embedded = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=d_model)(inputs)
    input_embedded *= tf.cast(d_model, tf.float32) ** 0.5
    input_positional_encoded = PositionalEncoding(position=vocab_size, d_model=d_model)(input_embedded)
    enc_outputs = tf.keras.layers.Dropout(rate=dropout)(input_positional_encoded)
    
    # encoder layer * N
    for i in range(num_layers):
        enc_outputs = EncoderLayer(batch_size, dff, d_model, num_heads, dropout, "encoder_layer_"+str(i))(enc_outputs, padding_mask)
            
    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=enc_outputs, name=name)


# ## 디코더 구현

# In[42]:


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, batch_size, dff, d_model, num_heads, dropout,  name, epsilon=1e-6):
        super(DecoderLayer, self).__init__(name=name)
        self.batch_size = batch_size
        self.dff = dff
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.epsilon = epsilon
        
        # layers
        self.masked_attention = MultiHeadAttention(self.batch_size, self.d_model, self.num_heads, name="Multi_Head_Attention_1")
        self.encoder_decoder_attention = MultiHeadAttention(self.batch_size, self.d_model, self.num_heads, name="Multi_Head_Attention_2")
        self.norm = tf.keras.layers.LayerNormalization(epsilon=self.epsilon)
        self.FFN1 = tf.keras.layers.Dense(units=self.dff, activation="relu")
        self.FFN2 = tf.keras.layers.Dense(units=self.d_model)
    
    def call(self, inputs, enc_outputs, look_ahead_mask, padding_mask):
        
        # input size == output size
        # dropout after each sublayer
        
        # first sub layer - masked multi-head attention
        attention1 = self.masked_attention(query=inputs, key=inputs, value=inputs, mask=look_ahead_mask)

        # add, normorlization
        attention1 = self.norm(attention1 + inputs)

        # second sub layer - encoder-decoder attention
        attention2 = self.encoder_decoder_attention(query=attention1, key=enc_outputs, value=enc_outputs, mask=padding_mask)
        attention2 = tf.keras.layers.Dropout(rate=self.dropout)(attention2)

        # add, normorlization
        attention2 = self.norm(attention2 + attention1)

        # third sub layer - Feed Forward layer (dense layer)
        feed_forward_output = self.FFN1(attention2)
        feed_forward_output = self.FFN2(feed_forward_output)
        feed_forward_output = tf.keras.layers.Dropout(rate=self.dropout)(feed_forward_output)

        # add, normorlization
        outputs = self.norm(attention2 + feed_forward_output)
        
        return outputs


# In[43]:


def decoder(batch_size, vocab_size, num_layers, dff, d_model, num_heads, dropout, name="decoder"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")

    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")
    
    # output embedding
    output_embedded = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=d_model)(inputs)
    output_embedded *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    output_embedded = PositionalEncoding(position=vocab_size, d_model=d_model)(output_embedded)
    dec_outputs = tf.keras.layers.Dropout(rate=dropout)(output_embedded)
    
    # decoder layer * N
    for i in range(num_layers):
        dec_outputs = DecoderLayer(batch_size, dff, d_model, num_heads, dropout, "decoder_layer_"+str(i))(dec_outputs, enc_outputs, look_ahead_mask, padding_mask)
    
    return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask], outputs=dec_outputs, name=name)


# ## 트랜스포머

# In[44]:


def transformer(encoder, decoder, vocab_size, name="transformer"):
    
    # encoder input (type: keras tensor)
    enc_inputs = tf.keras.Input(shape=(None,), name="enc_inputs")

    # decoder input (type: keras tensor)
    dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

    # encoder padding mask
    enc_padding_mask = create_padding_mask(name="enc_padding_mask")(enc_inputs)

    # decoder padding mask - for the first sub layer
    look_ahead_mask = create_look_ahead_mask()(dec_inputs)

    # decoder padding mask - for the second sub layer
    dec_padding_mask = create_padding_mask(name="dec_padding_mask")(enc_inputs)

    # encoder (type: keras model)
    enc_outputs = encoder(inputs=[enc_inputs, enc_padding_mask])

    # decoder (type: keras model)
    dec_outputs = decoder(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    # 다음 단어 예측 출력층(단어 개수만큼 출력 존재)
    outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

    return tf.keras.Model(inputs=[enc_inputs, dec_inputs], outputs=outputs, name=name)


# In[45]:


# 손실 함수 (cross entropy)

# 요약은 문장을 생성해내는 것이고, 이것은 단어 모음에 있는 단어 중
# 현재 문장 뒤에 올 단어 하나를 선택하는 다중 클래스 분류 문제이다.
# 따라서 cross entropy 함수를 손실함수로 사용한다.
def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, MAX_OUTPUT_LENGTH -1))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)


# ## 데이터 정보
# - size: 2176
# - sos: 8127
# - eos: 8128
# 
# - BATCH_SIZE: 32
# - MAX_INPUT_LENGTH: 1000
# - MAX_OUTPUT_LENGTH: 600
# - VOCAB_SIZE: 8129
# 
# ## 하이퍼파라미터 (논문)
# - D_MODEL: 256 (512)
# - NUM_LAYERS: 3 (6)
# - NUM_HEADS: 8 (8)
# - DFF = 512 (1024)
# - DROPOUT: 0.1

# In[46]:


tf.keras.backend.clear_session()

# hyper parameter (논문과 다름)
D_MODEL = 256
NUM_LAYERS = 3
NUM_HEADS = 8
DFF = 512
DROPOUT = 0.1


# In[51]:


def preprocessing(user_input):
    user_input = re.sub(r"([?.!,])", r" \1 ", user_input)
    user_input = re.sub("[^ A-Za-z?.!,$%]+", '', user_input)
    user_input = user_input.strip()

    user_input = tf.expand_dims(START_TOKEN + tokenizer.encode(user_input).ids + END_TOKEN, 0)
    print(len(user_input[0]))
    
    return user_input


# In[56]:


def get_prediction(user_input):
    
    enc = encoder(BATCH_SIZE, VOCAB_SIZE, NUM_LAYERS, DFF, D_MODEL, NUM_HEADS, DROPOUT)
    dec = decoder(BATCH_SIZE, VOCAB_SIZE, NUM_LAYERS, DFF, D_MODEL, NUM_HEADS, DROPOUT)
    
    trained_model = transformer(enc, dec, VOCAB_SIZE)
    trained_model.load_weights("training_final/cp.ckpt").expect_partial()
    
    output = tf.expand_dims(END_TOKEN, 0)
    print(user_input.shape)
    
    for i in range(MAX_OUTPUT_LENGTH):
        predictions = trained_model(inputs=[user_input, output], training=False)
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        #print(predicted_id)

        if tf.equal(predicted_id, END_TOKEN[0]):
            break
        
        output = tf.concat([output, predicted_id], axis=-1)
    
    return tf.squeeze(output, axis=0)


# In[53]:


def predict(user_input):
    preprocessed_input = preprocessing(user_input)
    predicted_output = get_prediction(preprocessed_input)

    predicted_output = tokenizer.decode(predicted_output)
  
    print('Input:', user_input)
    print('Output:', predicted_output)

    return predicted_output



