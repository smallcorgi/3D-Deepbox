import tensorflow as tf
import tensorflow.contrib.slim as slim
import cv2, os
import numpy as np
import time
from random import shuffle
from data_processing import *
import sys
import argparse
from tqdm import tqdm

#####
#Training setting

BIN, OVERLAP = 2, 0.1
W = 1.
ALPHA = 1.
MAX_JIT = 3
NORM_H, NORM_W = 224, 224
VEHICLES = ['Car', 'Truck', 'Van', 'Tram','Pedestrian','Cyclist']
BATCH_SIZE = 8
learning_rate = 0.0001
epochs = 50
save_path = './model/'

dims_avg = {'Cyclist': np.array([ 1.73532436,  0.58028152,  1.77413709]), 'Van': np.array([ 2.18928571,  1.90979592,  5.07087755]), 'Tram': np.array([  3.56092896,   2.39601093,  18.34125683]), 'Car': np.array([ 1.52159147,  1.64443089,  3.85813679]), 'Pedestrian': np.array([ 1.75554637,  0.66860882,  0.87623049]), 'Truck': np.array([  3.07392252,   2.63079903,  11.2190799 ])}


#### Placeholder
inputs = tf.placeholder(tf.float32, shape = [None, 224, 224, 3])
d_label = tf.placeholder(tf.float32, shape = [None, 3])
o_label = tf.placeholder(tf.float32, shape = [None, BIN, 2])
c_label = tf.placeholder(tf.float32, shape = [None, BIN])


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='3D bounding box')
    parser.add_argument('--mode',dest = 'mode',help='train or test',default = 'test')
    parser.add_argument('--image',dest = 'image',help='Image path')
    parser.add_argument('--label',dest = 'label',help='Label path')
    parser.add_argument('--box2d',dest = 'box2d',help='2D detection path')
    parser.add_argument('--output',dest = 'output',help='Output path', default = './validation/result_2/')
    parser.add_argument('--model',dest = 'model')
    parser.add_argument('--gpu',dest = 'gpu',default= '0')
    args = parser.parse_args()

    return args


def build_model():

  #### build some layer 
  def LeakyReLU(x, alpha):
      return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

  def orientation_loss(y_true, y_pred):
      # Find number of anchors
      anchors = tf.reduce_sum(tf.square(y_true), axis=2)
      anchors = tf.greater(anchors, tf.constant(0.5))
      anchors = tf.reduce_sum(tf.cast(anchors, tf.float32), 1)

      # Define the loss
      loss = (y_true[:,:,0]*y_pred[:,:,0] + y_true[:,:,1]*y_pred[:,:,1])
      loss = tf.reduce_sum((2 - 2 * tf.reduce_mean(loss,axis=0))) / anchors

      return tf.reduce_mean(loss)

  #####
  #Build Graph
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
    net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], scope='pool1')
    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
    net = slim.max_pool2d(net, [2, 2], scope='pool3')
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
    net = slim.max_pool2d(net, [2, 2], scope='pool4')
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
    net = slim.max_pool2d(net, [2, 2], scope='pool5')
    conv5 = tf.contrib.layers.flatten(net)

    #dimension = slim.fully_connected(conv5, 512, scope='fc7_d')
    dimension = slim.fully_connected(conv5, 512, activation_fn=None, scope='fc7_d')
    dimension = LeakyReLU(dimension, 0.1)
    dimension = slim.dropout(dimension, 0.5, scope='dropout7_d')
    #dimension = slim.fully_connected(dimension, 3, scope='fc8_d')
    dimension = slim.fully_connected(dimension, 3, activation_fn=None, scope='fc8_d')
    #dimension = LeakyReLU(dimension, 0.1)

    #loss_d = tf.reduce_mean(tf.square(d_label - dimension))
    loss_d = tf.losses.mean_squared_error(d_label, dimension)

    #orientation = slim.fully_connected(conv5, 256, scope='fc7_o')
    orientation = slim.fully_connected(conv5, 256, activation_fn=None, scope='fc7_o')
    orientation = LeakyReLU(orientation, 0.1)
    orientation = slim.dropout(orientation, 0.5, scope='dropout7_o')
    #orientation = slim.fully_connected(orientation, BIN*2, scope='fc8_o')
    orientation = slim.fully_connected(orientation, BIN*2, activation_fn=None, scope='fc8_o')
    #orientation = LeakyReLU(orientation, 0.1)
    orientation = tf.reshape(orientation, [-1, BIN, 2])
    orientation = tf.nn.l2_normalize(orientation, dim=2)
    loss_o = orientation_loss(o_label, orientation)

    #confidence = slim.fully_connected(conv5, 256, scope='fc7_c')
    confidence = slim.fully_connected(conv5, 256, activation_fn=None, scope='fc7_c')
    confidence = LeakyReLU(confidence, 0.1)
    confidence = slim.dropout(confidence, 0.5, scope='dropout7_c')
    confidence = slim.fully_connected(confidence, BIN, activation_fn=None, scope='fc8_c')
    loss_c = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=c_label, logits= confidence))
   
    confidence = tf.nn.softmax(confidence)
    #loss_c = tf.reduce_mean(tf.square(c_label - confidence))
    #loss_c = tf.losses.mean_squared_error(c_label, confidence)
    
    total_loss = 4. * loss_d + 8. * loss_o + loss_c
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)
    
    return dimension, orientation, confidence, total_loss, optimizer, loss_d, loss_o, loss_c


def train(image_dir, box2d_loc, label_dir):

    # load data & gen data
    all_objs = parse_annotation(label_dir, image_dir)
    all_exams  = len(all_objs)
    np.random.shuffle(all_objs)
    train_gen = data_gen(image_dir, all_objs, BATCH_SIZE)
    train_num = int(np.ceil(all_exams/BATCH_SIZE))
    
    ### buile graph
    dimension, orientation, confidence, loss, optimizer, loss_d, loss_o, loss_c = build_model()

    ### GPU config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)

    # create a folder for saving model
    if os.path.isdir(save_path) == False:
        os.mkdir(save_path)
    variables_to_restore = slim.get_variables()[:26] ## vgg16-conv5

    saver = tf.train.Saver(max_to_keep=100)

    #Load pretrain VGG model
    ckpt_list = tf.contrib.framework.list_variables('./vgg_16.ckpt')[1:-7]
    new_ckpt_list = []
    for name in range(1,len(ckpt_list),2):
        tf.contrib.framework.init_from_checkpoint('./vgg_16.ckpt', {ckpt_list[name-1][0]: variables_to_restore[name]})
        tf.contrib.framework.init_from_checkpoint('./vgg_16.ckpt', {ckpt_list[name][0]: variables_to_restore[name-1]})

    # Initializing the variables
    init = tf.global_variables_initializer()
    sess.run(init)


    # Start to train model
    for epoch in range(epochs):
        epoch_loss = np.zeros((train_num,1),dtype = float)
        tStart_epoch = time.time()
        batch_loss = 0.0
        for num_iters in tqdm(range(train_num),ascii=True,desc='Epoch '+str(epoch+1)+' : Loss:'+str(batch_loss)):
            train_img, train_label = train_gen.next()
            _,batch_loss = sess.run([optimizer,loss], feed_dict={inputs: train_img, d_label: train_label[0], o_label: train_label[1], c_label: train_label[2]})

            epoch_loss[num_iters] = batch_loss 

        # save model
        if (epoch+1) % 5 == 0:
            saver.save(sess,save_path+"model", global_step = epoch+1)

        # Print some information
        print "Epoch:", epoch+1, " done. Loss:", np.mean(epoch_loss)
        tStop_epoch = time.time()
        print "Epoch Time Cost:", round(tStop_epoch - tStart_epoch,2), "s"
        sys.stdout.flush()

def test(model, image_dir, box2d_loc, box3d_loc):

    ### buile graph
    dimension, orientation, confidence, loss, optimizer, loss_d, loss_o, loss_c = build_model()

    ### GPU config 
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)

    # Initializing the variables
    init = tf.global_variables_initializer()
    sess.run(init)

    # Restore model
    saver = tf.train.Saver()
    saver.restore(sess, model)

    # create a folder for saving result
    if os.path.isdir(box3d_loc) == False:
        os.mkdir(box3d_loc)

    # Load image & run testing
    all_image = sorted(os.listdir(image_dir))

    for f in all_image:
        image_file = image_dir + f
        box2d_file = box2d_loc + f.replace('png', 'txt')
        box3d_file = box3d_loc + f.replace('png', 'txt')
        print image_file
        with open(box3d_file, 'w') as box3d:
            img = cv2.imread(image_file)
            img = img.astype(np.float32, copy=False)

            for line in open(box2d_file):
                line = line.strip().split(' ')
                truncated = np.abs(float(line[1]))
                occluded  = np.abs(float(line[2]))

                obj = {'xmin':int(float(line[4])),
                       'ymin':int(float(line[5])),
                       'xmax':int(float(line[6])),
                       'ymax':int(float(line[7])),
                       }

                patch = img[obj['ymin']:obj['ymax'],obj['xmin']:obj['xmax']]
                patch = cv2.resize(patch, (NORM_H, NORM_W))
                patch = patch - np.array([[[103.939, 116.779, 123.68]]])
                patch = np.expand_dims(patch, 0)
                prediction = sess.run([dimension, orientation, confidence], feed_dict={inputs: patch})
                # Transform regressed angle
                max_anc = np.argmax(prediction[2][0])
                anchors = prediction[1][0][max_anc]

                if anchors[1] > 0:
                    angle_offset = np.arccos(anchors[0])
                else:
                    angle_offset = -np.arccos(anchors[0])

                wedge = 2.*np.pi/BIN
                angle_offset = angle_offset + max_anc*wedge
                angle_offset = angle_offset % (2.*np.pi)

                angle_offset = angle_offset - np.pi/2
                if angle_offset > np.pi:
                    angle_offset = angle_offset - (2.*np.pi)

                line[3] = str(angle_offset)
                 
                line[-1] = angle_offset +np.arctan(float(line[11]) / float(line[13]))
                
                # Transform regressed dimension
                if line[0] in VEHICLES:
                    dims = dims_avg[line[0]] + prediction[0][0]
                else:
                    dims = dims_avg['Car'] + prediction[0][0]

                line = line[:8] + list(dims) + line[11:]
                
                # Write regressed 3D dim and oritent to file
                line = ' '.join([str(item) for item in line]) +' '+ str(np.max(prediction[2][0]))+ '\n'
                box3d.write(line)

 

if __name__ == "__main__":
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.image is None:
        raise IOError(('Image not found.'.format(args.image)))
    if args.box2d is None :
        raise IOError(('2D bounding box not found.'.format(args.box2d)))

    if args.mode == 'train':
        if args.label is None:
            raise IOError(('Label not found.'.format(args.label)))

        train(args.image, args.box2d, args.label)
    else:
        if args.model is None:
            raise IOError(('Model not found.'.format(args.model)))

        test(args.model, args.image, args.box2d, args.output)

