import tensorflow as tf


class TextCNN(object):
    def __init__(self, config):
        sequence_length = config.max_sentences_length
        num_classes = config.num_labels
        embedding_size = config.embedding_size
        filter_sizes = config.filter_sizes
        num_filters = config.num_filters
        l2_reg_lambda = config.l2_reg_lambda
        l2_loss = tf.constant(0.0)
        pooled_outputs = []


        self.input_x=tf.placeholder(tf.float32,[None,sequence_length,embedding_size],name="input_x")
        self.input_y=tf.placeholder(tf.float32,[None,num_classes],name="input_y")
        self.dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_rate")
        self.learning_rate=tf.placeholder(tf.float32,name="lr")


        self.input_x_expended=tf.expand_dims(self.input_x,-1)


        for filter_size in filter_sizes:
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                #[filter_height, filter_width, in_channels, out_channels]
                filter_shape=[filter_size,embedding_size,1,num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")


                #添加卷积层
                conv=tf.nn.conv2d(
                    self.input_x_expended,
                    W,
                    strides=[1,1,1,1],
                    padding="VALID",
                    name="conv"
                )


                #添加偏置 & relu激活函数
                h=tf.nn.relu(tf.nn.bias_add(conv,b),name="relu")


                #添加最大池化层
                pooled=tf.nn.max_pool(
                    h,
                    ksize=[1,sequence_length-filter_size+1,1,1], #[对1个句子 卷积值hight 卷积值width 1个channel]
                    strides=[1,1,1,1],
                    padding="VALID",
                    name="pool"
                )
                pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        self.h_pooled=tf.concat(pooled_outputs, 3)
        self.h_pooled_flat=tf.reshape(self.h_pooled,[-1,num_filters_total])


        #添加dropout层
        with tf.name_scope("dropout"):
            self.h_drop=tf.nn.dropout(self.h_pooled_flat, self.dropout_keep_prob)


        #添加分类层
        with tf.name_scope("output"):
            self.Weight = tf.get_variable(
                "Weight",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            self.bias = tf.Variable(tf.constant(0.1, shape=[num_classes], name="bias"))
            l2_loss += tf.nn.l2_loss(self.Weight)
            l2_loss += tf.nn.l2_loss(self.bias)
            self.result=tf.matmul(self.h_drop,self.Weight)+self.bias
            self.predictions=tf.argmax(self.result,1,name="predictions")
            tf.summary.histogram("weight",self.Weight)
            tf.summary.histogram("bias",self.bias)


                #计算损失
        with tf.name_scope("loss"):
            losses=tf.nn.softmax_cross_entropy_with_logits(logits=self.result, labels=self.input_y)
            self.loss=tf.reduce_mean(losses)+l2_reg_lambda*l2_loss
            tf.summary.scalar("loss",self.loss)

        #计算正确率
        with tf.name_scope("accuracy"):
            correct_predictions=tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy=tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            tf.summary.scalar("accuracy",self.accuracy)

        #训练操作
        with tf.name_scope("train_operation"):
            self.train_op=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        with tf.name_scope("summary"):
            self.merged=tf.summary.merge_all()