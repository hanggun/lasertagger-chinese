import json
import numpy as np
import os
from bert4keras.backend import keras, K, batch_gather
from bert4keras.backend import multilabel_categorical_crossentropy
from bert4keras.layers import GlobalPointer, EfficientGlobalPointer, Dense, Dropout, Loss
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_gradient_accumulation

from bert4keras.snippets import open, to_array
from keras.models import Model
from tqdm import tqdm
from data_loader import train_loader, dev_loader, label_map, dev_data, tokenizer, max_len, _id_2_tag, train_data
import tensorflow as tf
from models import GAU_alpha
import utils
from snippets import compute_metrics, metric_keys


epochs = 10
learning_rate = 2e-5
k_sparse = 10
# bert配置
config_path = r'D:\PekingInfoResearch\pretrain_models\chinese_GAU-alpha-char_L-24_H-768/bert_config.json'
checkpoint_path = r'D:\PekingInfoResearch\pretrain_models\chinese_GAU-alpha-char_L-24_H-768/bert_model.ckpt'
dict_path = r'D:\PekingInfoResearch\pretrain_models\chinese_GAU-alpha-char_L-24_H-768/vocab.txt'


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        seq2seq_loss = self.compute_seq2seq_loss(inputs, mask)
        # self.add_metric(seq2seq_loss, 'seq2seq_loss')
        return seq2seq_loss

    def compute_seq2seq_loss(self, inputs, mask=None):
        y_mask, y_true, y_pred = inputs
        # 正loss
        pos_loss = batch_gather(y_pred, y_true)[..., 0] # 取正确位置上的值
        # 负loss
        y_pred = tf.nn.top_k(y_pred, k=k_sparse)[0]
        neg_loss = K.logsumexp(y_pred, axis=-1)
        # 总loss
        loss = neg_loss - pos_loss # 让目标类得分变为所有类的最大值，loss为负是优化方向
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


def my_metrics(y_t, y_p):
    y_t = K.cast(K.squeeze(y_t, axis=2), tf.int64)
    res = K.cast(K.equal(K.argmax(y_p, axis=2), y_t), K.floatx())
    b,s = K.shape(res)
    return K.sum(res) / K.cast(b*s, tf.float32)

def evaluate(data):
    total_metrics = {k: 0.0 for k in metric_keys}
    for d in tqdm(data):
        source = d[0][0]
        target = d[1]
        tag = d[2]
        labels = [label_map[str(tag)] for tag in d[2][:max_len - 2]]
        labels = [0] + labels + [0]

        input_ids = tokenizer.tokens_to_ids(['[CLS]'] + d[0][0].split()[:max_len - 2] + ['[SEP]'])
        segment_ids = [0] * len(input_ids)
        attention_mask = [1] * len(input_ids)

        input_ids, labels, segment_ids, attention_mask = to_array([input_ids]), to_array([labels]), to_array(segment_ids), to_array(attention_mask)
        labels = np.expand_dims(labels, -1)
        res = model.predict([input_ids, segment_ids])
        res = K.argmax(res)
        for r in res:
            tags = [_id_2_tag[x.numpy()] for x in r[1:-1]]
            pred_text = utils._realize_sequence(source.split(), tags)
            metrics = compute_metrics(pred_text.split(), target.split())
            for k, v in metrics.items():
                total_metrics[k] += v
    return {k: v / len(data) for k, v in total_metrics.items()}


class Evaluator(keras.callbacks.Callback):
    """训练回调
    """
    def __init__(self):
        self.best_metric = 0.0

    def on_epoch_end(self, epoch, logs=None):
        metrics = evaluate(dev_data)
        if metrics['main'] >= self.best_metric:  # 保存最优
            self.best_metric = metrics['main']
            model.save_weights('saved_model')
        metrics['best'] = self.best_metric
        print(metrics)

model = build_transformer_model(config_path, checkpoint_path, model=GAU_alpha)
output = Dropout(0.1)(model.output)
output = Dense(len(label_map))(output)
# y_true = keras.Input(shape=(None, 1))
# y_mask = keras.Input(shape=(None,))
# output = [y_mask, y_true, output]
# output = CrossEntropy()(output)
# model = Model(model.input + [y_true, y_mask], output[2])
model = Model(model.input, output)
# acc = my_metrics(y_true, output)
# model.add_metric(acc, name='acc')
model.summary()


AdamG = extend_with_gradient_accumulation(Adam, name='AdamG')
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(learning_rate=learning_rate),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
model.summary()
evaluator = Evaluator()

if __name__ == '__main__':
    mode = 'train'
    if mode == 'train':
        model.fit(train_loader.forfit(), epochs=epochs, steps_per_epoch=len(train_loader), callbacks=[evaluator])
    else:
        model.load_weights('saved_model')
        evaluator.on_epoch_end(0)