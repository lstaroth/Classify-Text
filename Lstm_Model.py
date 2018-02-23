import tensorflow as tf


class TextLSTM(object):
    def __init__(self,config):
        self.num_steps=config.max_sentences_length
        self.hidden_size=config.embedding_size
        self.num_classes=config.num_labels
        self.num_layers=config.num_layers
        self.batch_size=config.batch_size
        self.l2_rate=config.l2_rate
        self.input_x=tf.placeholder(tf.float32,[None,self.num_steps,self.hidden_size],name="input_x")
        self.input_y=tf.placeholder(tf.float32,[None,self.num_classes],name="input_y")
        self.dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")



        with tf.variable_scope("Net",initializer=tf.orthogonal_initializer()):
            def lstm_cell():
                return tf.contrib.rnn.BasicLSTMCell(self.hidden_size,forget_bias=1.0,state_is_tuple=True)


            attn_cell = lstm_cell
            if self.dropout_keep_prob is not None:
                def attn_cell():
                    return tf.contrib.rnn.DropoutWrapper(lstm_cell(),output_keep_prob=self.dropout_keep_prob)


            self.cell_fw=tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(config.num_layers)],
                                                     state_is_tuple=True)
            self.cell_bw=tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(config.num_layers)],
                                                     state_is_tuple=True)

            if self.dropout_keep_prob is not None:
                inputs=tf.nn.dropout(self.input_x,self.dropout_keep_prob)
            else:
                inputs=self.input_x

            #shape: (batch_size, num_steps,hidden_size) => (num_steps,batch_size,hidden_size)
            inputs= tf.transpose(inputs, [1,0,2])
            outputs,state = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell_fw,
                                                           cell_bw=self.cell_bw,
                                                           dtype="float32",
                                                           inputs=inputs,
                                                           swap_memory=True,
                                                            time_major=True)
            outputs_fw,outputs_bw=outputs
            output_fw=outputs_fw[-1]
            output_bw=outputs_bw[-1]
            finial_output=tf.concat([output_fw,output_bw],1)
            with tf.name_scope("output"):
                softmax_w=tf.get_variable("softmax_w",[self.hidden_size*2,self.num_classes],dtype=tf.float32)
                softmax_b=tf.get_variable("softmax_b",[self.num_classes],dtype=tf.float32,initializer=tf.random_normal_initializer(stddev=0.01))
                self.result=tf.matmul(finial_output,softmax_w)+softmax_b
                self.final_state=state
                self.predictions=tf.argmax(self.result,1,name="predictions")
                tf.summary.histogram("softmax_w",softmax_w)
                tf.summary.histogram("softmax_b",softmax_b)


        #计算损失
            with tf.name_scope("loss"):
                losses=tf.nn.softmax_cross_entropy_with_logits(logits=self.result, labels=self.input_y)
                self.loss = tf.reduce_mean(losses)
                tf.summary.scalar("loss",self.loss)


            #计算正确率
            with tf.name_scope("accuracy"):
                correct_predictions=tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.correct_num = tf.reduce_sum(tf.cast(correct_predictions, tf.float32))
                self.accuracy=tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
                tf.summary.scalar("accuracy",self.accuracy)

            with tf.name_scope("train_op"):
                tvars = tf.trainable_variables()
                self.l2_loss = 0.001 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tvars])  # 0.001是lambda超参数
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss+self.l2_loss, tvars), config.max_grad_norm)
                optimizer = tf.train.AdamOptimizer(0.001)
                optimizer.apply_gradients(zip(grads, tvars))
                self.train_op = optimizer.apply_gradients(zip(grads, tvars))

            with tf.name_scope("summary"):
                self.summary_op=tf.summary.merge_all()
