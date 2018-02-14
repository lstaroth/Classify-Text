import tensorflow as tf


class TextCNN(object):
    def __init__(self, sequence_length, num_classes,embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        self.input_x=tf.placeholder(tf.float32,[None,sequence_length,embedding_size],name="input_x")
        self.input_y=tf.placeholder(tf.float32,[None,num_classes],name="input_y")
        self.dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_rate")


        #l2损失
        l2_loss = tf.constant(0.0)


        pooled_outputs=[]


        #扩展一个Channel维度
        self.input_x_expended=tf.expand_dims(self.input_x,-1)
        for filter_size in filter_sizes:
            #[filter_height, filter_width, in_channels, out_channels]
            filter_shape=[filter_size,embedding_size,1,num_filters]
            w1=tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name="weight")
            b1=tf.Variable(tf.constant(0.1,shape=[num_filters]),name="bias")


            #添加卷积层
            conv=tf.nn.conv2d(
                self.input_x_expended,
                w1,
                strides=[1,1,1,1],
                padding="VALID",
                name="conv"
            )


            #添加偏置 & relu激活函数
            h=tf.nn.relu(tf.nn.bias_add(conv,b1),name="relu")


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
            self.h_drop=tf.nn.dropout(self.h_pooled_flat, self.dropout_keep_prob)


            #添加分类层
            w2=tf.Variable(tf.truncated_normal([num_filters_total,num_classes],stddev=0.1),name="weight2")
            b2=tf.Variable(tf.constant(0.1,shape=[num_classes]), name="bias2")
            l2_loss += tf.nn.l2_loss(w2)
            l2_loss += tf.nn.l2_loss(b2)
            self.result=tf.matmul(self.h_drop,w2)+b2
            self.predictions=tf.argmax(self.result,1,name="predictions")


            #计算损失
            losses=tf.nn.softmax_cross_entropy_with_logits(logits=self.result, labels=self.input_y)
            self.loss=tf.reduce_mean(losses)+l2_reg_lambda*l2_loss

            #计算正确率
            correct_predictions=tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy=tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

            #训练操作
            self.train_op=tf.train.AdamOptimizer(1e-4).minimize(self.loss)