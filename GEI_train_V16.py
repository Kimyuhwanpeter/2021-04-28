# -*- coding:utf-8 -*-
from model_V16 import *
from random import shuffle
from collections import Counter

import numpy as np
import os
import datetime
import easydict

FLAGS = easydict.EasyDict({"img_height": 128,
                           
                           "img_width": 88,
                           
                           "tr_txt_path": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/train.txt",
                           
                           "tr_txt_name": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/GEI_IDList_train.txt",
                           
                           "tr_img_path": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/GEI/",
                           
                           "te_txt_path": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/test.txt",
                           
                           "te_txt_name": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/GEI_IDList_test_fix.txt",
                           
                           "te_img_path": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/GEI/",
                           
                           "batch_size": 137,
                           
                           "epochs": 300,
                           
                           "num_classes": 86,
                           
                           "lr": 0.0001,
                           
                           "save_checkpoint": "",
                           
                           "graphs": "C:/Users/Yuhwan/Downloads/", 
                           
                           "train": True,
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": ""})

optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)

def tr_func(img_list, lab_list):

    img = tf.io.read_file(img_list)
    img = tf.image.decode_png(img, 1)
    img = tf.image.resize(img, [FLAGS.img_height, FLAGS.img_width]) / 255.

    lab = lab_list

    return img, lab

def te_func(img_list, lab_list):

    img = tf.io.read_file(img_list)
    img = tf.image.decode_png(img, 1)
    img = tf.image.resize(img, [FLAGS.img_height, FLAGS.img_width]) / 255.

    lab = lab_list

    return img, lab

def make_label_V2(ori_label):
    l = []
    for i in range(FLAGS.batch_size):
        label = [1] * (ori_label[i].numpy() + 1) + [0] * (88 - (ori_label[i].numpy() + 1))
        label = tf.cast(label, tf.float32)
        l.append(label)
    return tf.convert_to_tensor(l, tf.float32)

# @tf.function
def run_model(model, images, training=True):
    return model(images, training=training)

def cal_CDF(logits, labels, num_feature):

    logit_CDF = []
    label_CDF = []
    for i in range(FLAGS.batch_size):
        log = 0.
        lab = 0.
        logit = logits[i]
        logit = logit.numpy()
        label = labels[i]
        label = label.numpy()
        log_buf = []
        lab_buf = []
        for j in range(num_feature):
            log += logit[j]
            lab += label[j]
            log_buf.append(log)
            lab_buf.append(lab)

        logit_CDF.append(log_buf)
        label_CDF.append(lab_buf)

    return logit_CDF, label_CDF

def cal_loss(model, images, labels):

    with tf.GradientTape() as tape:
        logits = run_model(model, images, True)
        #c_loss = -tf.reduce_sum(labels * tf.math.log(logits + 0.000001) + (1 - labels) * tf.math.log(1 - logits + 0.000001), 1)
        # c_loss = (-tf.reduce_sum( (tf.math.log_sigmoid(logits)*labels + (tf.math.log_sigmoid(logits) - logits)*(1-labels)), 1))
        #c_loss = tf.reduce_mean(c_loss)

        # c_loss = tfa.losses.SigmoidFocalCrossEntropy(from_logits=True,)(labels, logits)
        alpha_factor = 1.0
        modulating_factor = 1.0
        p_t = (labels * tf.nn.sigmoid(logits)) + ((1 - labels) * (1 - tf.nn.sigmoid(logits)))
        alpha_factor = labels * 0.25 + (1 - labels) * (1 - 0.25)
        modulating_factor = tf.pow((1.0 - p_t), 2.0)
        ce = (-( (tf.math.log_sigmoid(logits)*labels + (tf.math.log(1 - tf.math.sigmoid(logits)))*(1-labels))))        

        c_loss = tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1)

        # c_loss = (-tf.reduce_sum( (tf.math.log_sigmoid(logits)*labels + (tf.math.log(1 - tf.math.sigmoid(logits)))*(1-labels)), 1))
        c_loss = tf.reduce_mean(c_loss)

        logit_distribution = tf.nn.softmax(tf.nn.sigmoid(logits), 1)
        label_distribution = tf.nn.softmax(labels, 1)

        logit_CDF, label_CDF = cal_CDF(logit_distribution, label_distribution, 9)
        logit_CDF, label_CDF = tf.convert_to_tensor(logit_CDF, dtype=tf.float32), tf.convert_to_tensor(label_CDF, dtype=tf.float32)

        CDF_loss = tf.keras.losses.MeanSquaredError()(label_CDF, logit_CDF)

        total_loss = CDF_loss*10 + c_loss

    grads = tape.gradient(total_loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    return total_loss

def cal_mae(model, images, labels):

    logits = run_model(model, images, False)

    ae = 0
    for i in range(137):

        age_predict = tf.nn.sigmoid(logits[i])
        age_predict = tf.cast(tf.less_equal(0.5, age_predict), tf.int32)
        age_predict = tf.reduce_sum(age_predict)

        ae += tf.abs(labels[i] - age_predict)

    return ae

def main():
    model = fix_GL_network(input_shape=(FLAGS.img_height, FLAGS.img_width, 1), num_classes=88)
    model.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(model=model, optim=optim)

        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored the latest checkpoint!")

    if FLAGS.train:
        count = 0

        tr_img = np.loadtxt(FLAGS.tr_txt_name, dtype="<U100", skiprows=0, usecols=0)
        tr_img = [FLAGS.tr_img_path + img + ".png"for img in tr_img]
        tr_lab = np.loadtxt(FLAGS.tr_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        te_img = np.loadtxt(FLAGS.te_txt_name, dtype="<U100", skiprows=0, usecols=0)
        te_img = [FLAGS.te_img_path + img + ".png" for img in te_img]
        te_lab = np.loadtxt(FLAGS.te_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        te_gener = tf.data.Dataset.from_tensor_slices((te_img, te_lab))
        te_gener = te_gener.map(te_func)
        te_gener = te_gener.batch(137)
        te_gener = te_gener.prefetch(tf.data.experimental.AUTOTUNE)

        #############################
        # Define the graphs
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = FLAGS.graphs + current_time + "_V16" + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        val_log_dir = FLAGS.graphs + current_time + "_V16" + '/val'
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        #############################

        for epoch in range(FLAGS.epochs):
            TR = list(zip(tr_img, tr_lab))
            shuffle(TR)
            tr_img, tr_lab = zip(*TR)
            tr_img, tr_lab = np.array(tr_img), np.array(tr_lab, dtype=np.int32)

            tr_gener = tf.data.Dataset.from_tensor_slices((tr_img, tr_lab))
            tr_gener = tr_gener.shuffle(len(tr_img))
            tr_gener = tr_gener.map(tr_func)
            tr_gener = tr_gener.batch(FLAGS.batch_size)
            tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)

            tr_iter = iter(tr_gener)
            tr_idx = len(tr_img) // FLAGS.batch_size

            for step in range(tr_idx):

                batch_images, batch_labels = next(tr_iter)
                batch_labels = make_label_V2(batch_labels)

                loss = cal_loss(model, batch_images, batch_labels)
                with train_summary_writer.as_default():
                    tf.summary.scalar('Loss', loss, step=count)

                if count % 10 == 0:
                    print("Epochs = {} [{}/{}] Loss = {}".format(epoch, step + 1, tr_idx, loss))

                if count % 100 == 0:
                    te_iter = iter(te_gener)
                    te_idx = len(te_img) // 137
                    ae = 0
                    for i in range(te_idx):
                        imgs, labs = next(te_iter)

                        ae += cal_mae(model, imgs, labs)

                    MAE = ae / len(te_img)
                    print("================================")
                    print("step = {}, MAE = {}".format(count, MAE))
                    print("================================")
                    with val_summary_writer.as_default():
                        tf.summary.scalar('MAE', MAE, step=count)

                    #num_ = int(count // 100)
                    #model_dir = "%s/%s" % (FLAGS.save_checkpoint, num_)
                    #if not os.path.isdir(model_dir):
                    #   os.makedirs(model_dir)
                    #   print("Make {} files to save checkpoint".format(num_))

                    #ckpt = tf.train.Checkpoint(model=model, optim=optim)
                    #ckpt_dir = model_dir + "/" + "[7]_{}.ckpt".format(count)
                    #ckpt.save(ckpt_dir)
                    
                count += 1

if __name__ == "__main__":
    main()