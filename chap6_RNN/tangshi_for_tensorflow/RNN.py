#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
import collections

start_token = 'G'
end_token = 'E'
batch_size = 64


# #### 数据预处理部分

# In[ ]:


def process_poems(file_name):
    poems = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        for line in f.readlines():
            try:
                title, content = line.strip().split(':')
                content = content.replace(' ', '')
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or                                 start_token in content or end_token in content:
                    continue
                if len(content) < 5 or len(content) > 80:
                    continue
                content = start_token + content + end_token
                poems.append(content)
            except ValueError as e:
                pass
    # 按诗的字数排序
    poems = sorted(poems, key=lambda line: len(line))
    # 统计每个字出现次数
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]  
    counter = collections.Counter(all_words)  # 统计词和词频。
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # 排序
    words, _ = zip(*count_pairs)
    words = words[:len(words)] + (' ',)
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]
    return poems_vector, word_int_map, words


# ### rnn_lstm model

# In[ ]:


def rnn_model(model, input_data, output_data, vocab_size, rnn_size=128, num_layers=2, batch_size=64,
              learning_rate=0.01):
    end_points = {}
    # 构建RNN基本单元RNNcell
    if model == 'rnn':
        cell_fun = tf.contrib.rnn.BasicRNNCell
    elif model == 'gru':
        cell_fun = tf.contrib.rnn.GRUCell
    else:
        cell_fun = tf.contrib.rnn.BasicLSTMCell
    #？？？？？？？？？？？？？？？？？？？？？？
    # 每层128个小单元，一共有两层，输出的Ct 和 Ht 要分开放到两个tuple中
    # 在下面补全代码 
    #################################################
    cell = cell_fun(num_units=128, state_is_tuple=True)
    cell = tf.contrib.rnn.MultiRNNCell([cell]*2, state_is_tuple=True)
    #################################################
    # 如果是训练模式，output_data不为None，则初始状态shape为[batch_size * rnn_size]
    # 如果是生成模式，output_data为None，则初始状态shape为[1 * rnn_size]
    if output_data is not None:
        initial_state = cell.zero_state(batch_size, tf.float32)
    else:
        initial_state = cell.zero_state(1, tf.float32)

    # 构建隐层
    with tf.device("/cpu:0"):
        embedding = tf.Variable(tf.random_uniform([vocab_size + 1, rnn_size], -1.0, 1.0),name = 'embedding')
        inputs = tf.nn.embedding_lookup(embedding, input_data)
    #？？？？？？？？？？？？？？？？？？？？？？？？？？
    ####################################################    
    outputs, last_state = tf.nn.dynamic_rnn(cell=cell, inputs=inputs, initial_state=initial_state)# 填写里面的内容
    ######################################################
    output = tf.reshape(outputs, [-1, rnn_size])
    
    weights = tf.Variable(tf.truncated_normal([rnn_size, vocab_size + 1]))
    bias = tf.Variable(tf.zeros(shape=[vocab_size + 1]))
    logits = tf.nn.bias_add(tf.matmul(output, weights), bias=bias) # 一层全连接


    if output_data is not None: # 训练模式
        labels = tf.one_hot(tf.reshape(output_data, [-1]), depth=vocab_size + 1)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        total_loss = tf.reduce_mean(loss)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)  # 优化器用的 adam
        end_points['initial_state'] = initial_state
        end_points['output'] = output
        end_points['train_op'] = train_op
        end_points['total_loss'] = total_loss
        end_points['loss'] = loss
        end_points['last_state'] = last_state
    else: # 生成模式
        prediction = tf.nn.softmax(logits)
        end_points['initial_state'] = initial_state
        end_points['last_state'] = last_state
        end_points['prediction'] = prediction
    return end_points


# ### 训练模型部分

# In[ ]:


def run_training():
    # 处理数据集
    poems_vector, word_to_int, vocabularies = process_poems('./poems.txt')
    # 生成batch
    batches_inputs, batches_outputs = generate_batch(64, poems_vector, word_to_int)

    input_data = tf.placeholder(tf.int32, [batch_size, None])
    output_targets = tf.placeholder(tf.int32, [batch_size, None])
    # 构建模型
    end_points = rnn_model(model='lstm', input_data=input_data, output_data=output_targets, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=64, learning_rate=0.01)

    saver = tf.train.Saver()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        for epoch in range(50):
            n = 0
            n_chunk = len(poems_vector) // batch_size
            for batch in range(n_chunk):
                loss, _, _ = sess.run([
                    end_points['total_loss'],
                    end_points['last_state'],
                    end_points['train_op']
                ], feed_dict={input_data: batches_inputs[n], output_targets: batches_outputs[n]})
                n += 1
                print('[INFO] Epoch: %d , batch: %d , training loss: %.6f' % (epoch, batch, loss))
        saver.save(sess, './poem_generator')


# ### 生成 诗歌部分

# In[ ]:


def gen_poem(begin_word):
    batch_size = 1
    poems_vector, word_int_map, vocabularies = process_poems('./poems.txt')

    input_data = tf.placeholder(tf.int32, [batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=None, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=64, learning_rate=0.01)
    # 如果指定开始的字
    if begin_word:
        word = begin_word
    else:
        word = to_word(predict, vocabularies)
        
    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, './poem_generator')# 恢复之前训练好的模型 
        poem = ''
        #???????????????????????????????????????
        # 下面部分代码主要功能是根据指定的开始字符来生成诗歌
        #########################################
        
        
        #########################################
        return poem


# ### 其他的一些处理函数

# In[ ]:


def generate_batch(batch_size, poems_vec, word_to_int):
    # 每次取64首诗进行训练
    n_chunk = len(poems_vec) // batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size

        batches = poems_vec[start_index:end_index]
        # 找到这个batch的所有poem中最长的poem的长度
        length = max(map(len, batches))
        # 填充一个这么大小的空batch，空的地方放空格对应的index标号
        x_data = np.full((batch_size, length), word_to_int[' '], np.int32)
        for row in range(batch_size):
            x_data[row, :len(batches[row])] = batches[row]
        y_data = np.copy(x_data)
        y_data[:, :-1] = x_data[:, 1:]
        """
        x_data             y_data
        [6,2,4,6,9]       [2,4,6,9,9]
        [1,4,2,8,5]       [4,2,8,5,5]
        """
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches

def to_word(predict, vocabs):# 预测的结果转化成汉字
    sample = np.argmax(predict)
    if sample > len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]
def pretty_print_poem(poem):#  令打印的结果更工整
    poem_sentences = poem.split('。')
    for s in poem_sentences:
        if s != '' and len(s) > 10:
            print(s + '。')


# ### 主函数

# In[ ]:


print('[INFO] train tang poem...')
run_training() # 训练模型
print('[INFO] write tang poem...')
poem2 = gen_poem('月')# 生成诗歌
print("#" * 25)
pretty_print_poem(poem2)
print('#' * 25)
#训练模型时间比较长，训练模型完成后每次生成诗歌的时，不需要再次训练 ，可以注销上面的 run_training()。生成部分执行速度很快

