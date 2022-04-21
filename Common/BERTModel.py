import os
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import tensorflow_addons as tfa
from transformers import DistilBertTokenizer, TFDistilBertModel, DistilBertConfig
from transformers import logging as hf_logging
from tqdm import tqdm
import numpy as np
class BERTModel:
    def __init__(self):
        self.MODEL_NAME = 'distilbert-base-uncased'
        self.MAX_LEN = 360
        self.Model = None
        self.History = None
        pass
    
    def Init(self):
        '''
        Initialize the structure of the model
        '''
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.MODEL_NAME,  
                                                add_special_tokens=True,
                                                max_length=self.MAX_LEN, 
                                                pad_to_max_length=True)
        bert_config = DistilBertConfig.from_pretrained(self.MODEL_NAME, output_hidden_states=True, output_attentions=True)
        TFBert = TFDistilBertModel.from_pretrained(self.MODEL_NAME, config=bert_config)

        input_ids_layer = tf.keras.layers.Input(shape=(self.MAX_LEN,), name='input_token', dtype='int32')
        input_masks_layer = tf.keras.layers.Input(shape=(self.MAX_LEN,), name='masked_token', dtype='int32') 

        X = TFBert(input_ids = input_ids_layer, attention_mask = input_masks_layer)[0]
        # X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))(X)
        # X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128))(X)
        X = tf.keras.layers.Dropout(0.2)(X)
        X = tf.keras.layers.Dense(1024, activation=tfa.activations.mish)(X)
        X = tf.keras.layers.Flatten()(X)
        X = tf.keras.layers.Dense(4, activation=tf.nn.softmax)(X)
        
        # combine the model
        model = tf.keras.Model(inputs=[input_ids_layer, input_masks_layer], outputs = X)
        
        # 每层都标记为trainable
        for layer in model.layers[:3]:
            layer.trainable = True
        self.Model = model
        
        self.ckpt_dir = './ckpt'
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.model_checkpoint = ModelCheckpoint(filepath=self.ckpt_dir + '/weights_val_best.hdf5',
                                   monitor='val_accuracy',
                                   save_weights_only=True,
                                   save_best_only=True,
                                   verbose=0)

        self.early_stopping = EarlyStopping(patience=3,
                                       monitor='val_accuracy',
                                       min_delta=0,
                                       mode='max',
                                       restore_best_weights=False,
                                       verbose=1)

        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      min_lr=0.000001,
                                      patience=1,
                                      mode='min',
                                      factor=0.1,
                                      min_delta=0.0001,
                                      verbose=1)
        
        model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tfa.optimizers.RectifiedAdam(0.0001),
              metrics=['accuracy'])

        pass
    
    def Summary(self):
        self.Model.summary()
        
    def Train(self, X_train, y_train, X_val, y_val):
        history = self.Model.fit(X_train,
                    y_train,
                    epochs=50,
                    batch_size=8,
                    validation_data=(X_val, y_val),
                    callbacks=[self.model_checkpoint, self.early_stopping, self.reduce_lr])
        self.History = history
        # load the best model
        self.Model.load_weights(self.ckpt_dir + '/weights_val_best.hdf5')
        return history

    def Predict(self, X_test):
        pred_probs = self.Model.predict(X_test)
        y_pred = np.argmax(pred_probs, axis=1)
        return y_pred

    def tokenize(self, sentences):
        input_ids, input_masks, input_segments = [], [], []
        for sentence in tqdm(sentences):
            inputs = self.tokenizer.encode_plus(sentence, 
                                           add_special_tokens=True, 
                                           max_length=self.MAX_LEN, 
                                           pad_to_max_length=True, 
                                           return_attention_mask=True, 
                                           return_token_type_ids=True, 
                                           truncation=True)
            input_ids.append(inputs['input_ids'])
            input_masks.append(inputs['attention_mask'])
            input_segments.append(inputs['token_type_ids'])       

        return np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32')