'''
Tensorflow implementation of the mtcnn face detection algorithm

Credit: DavidSandBerg for implementing this method on tensorflow
'''
from six import string_types, iteritems
import numpy as np
import tensorflow as tf
import cv2
import os
import sys

model_path = "models"


def layer(op):
    '''Decorator for composable network layers.'''

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated

class Network(object):

    def __init__(self, inputs, trainable=True):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable

        self.setup()

    def setup(self):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path, encoding='latin1').item()  # pylint: disable=no-member

        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in iteritems(data_dict[op_name]):
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, string_types):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape):
        '''Creates a new TensorFlow variable.'''
        init = tf.random_normal(shape)
        return tf.get_variable(name, trainable=self.trainable, initializer = init)

    def validate_padding(self, padding):
        '''Verifies that the padding is one of the supported ones.'''
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self,
             inp,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding='SAME',
             group=1,
             biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = int(inp.get_shape()[-1])
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i // group, c_o])
            # This is the common-case. Convolve the input without any further complications.
            output = convolve(inp, kernel)
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def prelu(self, inp, name):
        with tf.variable_scope(name):
            i = int(inp.get_shape()[-1])
            alpha = self.make_var('alpha', shape=(i,))
            neg = np.zeros((i))
            for k in range(i):
                neg[k] = -1.
            neg = neg.astype(np.float32)
            if (len(inp.get_shape()) == 2):
                nodea = tf.nn.relu(inp)
                nodeb = tf.nn.relu(tf.multiply(neg, inp))
                nodec = tf.multiply(alpha, tf.multiply(neg, nodeb))
            else:
                nodea = tf.nn.relu(tf.nn.max_pool(inp, ksize = [1, 1, 1, 1], strides = [1, 1, 1, 1], padding = 'SAME'))
                nodeb = tf.nn.relu(tf.multiply(neg, tf.nn.max_pool(inp, ksize = [1, 1, 1, 1], strides = [1, 1, 1, 1], padding = 'SAME')))
                nodec = tf.multiply(alpha, tf.multiply(neg, tf.nn.max_pool(nodeb, ksize = [1, 1, 1, 1], strides = [1, 1, 1, 1], padding = 'SAME')))
            output = tf.add(nodec, nodea)
        return output

    @layer
    def max_pool(self, inp, k_h, k_w, s_h, s_w, name, padding='SAME'):
        self.validate_padding(padding)
        return tf.nn.max_pool(inp,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def fc(self, inp, num_out, name, relu=True):
        with tf.variable_scope(name):
            input_shape = inp.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= int(d)
                feed_in = tf.reshape(inp, [-1, dim])
            else:
                feed_in, dim = (inp, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=name)
            return fc

class PNet(Network):
    def setup(self):
        (self.feed('data')  # pylint: disable=no-value-for-parameter, no-member
         .conv(3, 3, 10, 1, 1, padding='VALID', relu=False, name='conv1')
         .prelu(name='PReLU1')
         .max_pool(2, 2, 2, 2, name='pool1')
         .conv(3, 3, 16, 1, 1, padding='VALID', relu=False, name='conv2')
         .prelu(name='PReLU2')
         .conv(3, 3, 32, 1, 1, padding='VALID', relu=False, name='conv3')
         .prelu(name='PReLU3')
         .conv(1, 1, 2, 1, 1, relu=False, name='conv4-1'))

        (self.feed('PReLU3')  # pylint: disable=no-value-for-parameter
         .conv(1, 1, 4, 1, 1, relu=False, name='conv4-2'))

class RNet(Network):
    def setup(self):
        (self.feed('data')  # pylint: disable=no-value-for-parameter, no-member
         .conv(3, 3, 28, 1, 1, padding='VALID', relu=False, name='conv1')
         .prelu(name='prelu1')
         .max_pool(3, 3, 2, 2, name='pool1')
         .conv(3, 3, 48, 1, 1, padding='VALID', relu=False, name='conv2')
         .prelu(name='prelu2')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
         .conv(2, 2, 64, 1, 1, padding='VALID', relu=False, name='conv3')
         .prelu(name='prelu3')
         .fc(128, relu=False, name='conv4')
         .prelu(name='prelu4')
         .fc(2, relu=False, name='conv5-1'))

        (self.feed('prelu4')  # pylint: disable=no-value-for-parameter
         .fc(4, relu=False, name='conv5-2'))

class ONet(Network):
    def setup(self):
        (self.feed('data')  # pylint: disable=no-value-for-parameter, no-member
         .conv(3, 3, 32, 1, 1, padding='VALID', relu=False, name='conv1')
         .prelu(name='prelu1')
         .max_pool(3, 3, 2, 2, name='pool1')
         .conv(3, 3, 64, 1, 1, padding='VALID', relu=False, name='conv2')
         .prelu(name='prelu2')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
         .conv(3, 3, 64, 1, 1, padding='VALID', relu=False, name='conv3')
         .prelu(name='prelu3')
         .max_pool(2, 2, 2, 2, name='pool3')
         .conv(2, 2, 128, 1, 1, padding='VALID', relu=False, name='conv4')
         .prelu(name='prelu4')
         .fc(256, relu=False, name='conv5')
         .prelu(name='prelu5')
         .fc(2, relu=False, name='conv6-1'))
        
        (self.feed('prelu5')  # pylint: disable=no-value-for-parameter
         .fc(10, relu=False, name='conv6-3'))

        (self.feed('prelu5')  # pylint: disable=no-value-for-parameter
         .fc(4, relu=False, name='conv6-2'))

def addPnet(sess, h = 28, w = 38):
    with tf.variable_scope('pnet'):
        data = tf.placeholder(tf.float32, (1, h, w, 3), 'input')
        pnet = PNet({'data': data})
        pnet.load(os.path.join(model_path, 'det1.npy'), sess)
    graph = sess.graph
    pnet_output0 = graph.get_tensor_by_name('pnet/conv4-1/BiasAdd:0')
    pnet_output1 = graph.get_tensor_by_name('pnet/conv4-2/BiasAdd:0')
    pnet_output0_reshape = tf.reshape(pnet_output0, [((h - 9) >> 1) * ((w - 9) >> 1), 2])
    pnet_output1_reshape = tf.reshape(pnet_output1, [((h - 9) >> 1) * ((w - 9) >> 1), 4])
    pnet_output2 = tf.concat([pnet_output0_reshape, pnet_output1_reshape], -1, name = 'pnet/output')

def addRnet(sess):
    with tf.variable_scope('rnet'):
        data = tf.placeholder(tf.float32, (None, 24, 24, 3), 'input')
        rnet = RNet({'data': data})
        rnet.load(os.path.join(model_path, 'det2.npy'), sess)
    graph = sess.graph
    rnet_output0 = graph.get_tensor_by_name('rnet/conv5-1/conv5-1:0')
    rnet_output1 = graph.get_tensor_by_name('rnet/conv5-2/conv5-2:0')
    rnet_output2 = tf.concat([rnet_output0, rnet_output1], -1, name = 'rnet/output')


def addOnet(sess):
    with tf.variable_scope('onet'):
        data = tf.placeholder(tf.float32, (None, 48, 48, 3), 'input')
        onet = ONet({'data': data})
        onet.load(os.path.join(model_path, 'det3.npy'), sess)
    graph = sess.graph
    onet_output0 = graph.get_tensor_by_name('onet/conv6-1/conv6-1:0')
    onet_output1 = graph.get_tensor_by_name('onet/conv6-2/conv6-2:0')
    onet_output2 = graph.get_tensor_by_name('onet/conv6-3/conv6-3:0')
    onet_output2 = tf.concat([onet_output0, onet_output1, onet_output2], -1, name = 'onet/output')

def addDnet(sess):
    with tf.variable_scope('pnet'):
        data = tf.placeholder(tf.float32, (1, None, None, 3), 'input')
        pnet = PNet({'data': data})
        pnet.load(os.path.join(model_path, 'det1.npy'), sess)
    with tf.variable_scope('rnet'):
        data = tf.placeholder(tf.float32, (None, 24, 24, 3), 'input')
        rnet = RNet({'data': data})
        rnet.load(os.path.join(model_path, 'det2.npy'), sess)
    with tf.variable_scope('onet'):
        data = tf.placeholder(tf.float32, (None, 48, 48, 3), 'input')
        onet = ONet({'data': data})
        onet.load(os.path.join(model_path, 'det3.npy'), sess)

def print_usage():
    print ("Usage: python main.py <net name> <pnet h> <pnet w>\r\n<Net names>:\r\np\t\tpnet, image size defined in argv(h > 12 and w > 12)\r\nr\t\trnet, image size 24 * 24\r\no\t\tonet, image size 48 * 48\r\nd\t\tall nets, used for tensorflow, image size None * None\r\n")
    sys.exit(0)

if __name__ == '__main__':
    sess = tf.Session()
    if not model_path:
        model_path, _ = os.path.split(os.path.realpath(__file__))
    if (len(sys.argv) > 1 and sys.argv[1] in ['p', 'r', 'o', 'd']):
        if (sys.argv[1] == 'p'):
            if (len(sys.argv) < 4 or int(float(sys.argv[2])) < 12 or int(float(sys.argv[3])) < 12):
                print_usage()
            else:
                addPnet(sess, int(float(sys.argv[2])), int(float(sys.argv[3])))
        if (sys.argv[1] == 'r'):
            addRnet(sess)
        if (sys.argv[1] == 'o'):
            addOnet(sess)
        if (sys.argv[1] == 'd'):
            addDnet(sess)
        saver = tf.train.Saver(tf.global_variables())
        saver.save(sess, ''.join(sys.argv[1:]) + "/model")
    else:
        print_usage()
