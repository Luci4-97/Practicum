import tensorflow as tf
import numpy as np


class NN(object):
    def __init__(self):
        print("""
                    ███╗   ███╗██╗     ██████╗ 
                    ████╗ ████║██║     ██╔══██╗
                    ██╔████╔██║██║     ██████╔╝
                    ██║╚██╔╝██║██║     ██╔═══╝ 
                    ██║ ╚═╝ ██║███████╗██║     
                    ╚═╝     ╚═╝╚══════╝╚═╝   
        """)
        # 网络参数
        self.learning_rate = 0.001  # 学习率
        self.max_iter = 10000  # 最大迭代次数
        self.n_hidden_1 = 4  # 第1层神经元个数
        self.n_hidden_2 = 4  # 第2层神经元个数
        self.n_hidden_3 = 4  # 第3层神经元个数
        self.n_hidden_4 = 4  # 第4层神经元个数
        self.n_input = 4  # 样本特征数
        # 定义权值和偏置
        self.Weights = {
            'h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1]), name='layer1_w'),
            'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2]), name='layer2_w'),
            'h3': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_hidden_3]), name='layer3_w'),
            'h4': tf.Variable(tf.random_normal([self.n_hidden_3, self.n_hidden_4]), name='layer4_w'),
            'out': tf.Variable(tf.random_normal([self.n_hidden_4, 1]), dtype=tf.float32)
        }
        self.biases = {
            'h1': tf.Variable(tf.zeros([1, self.n_hidden_1]), name='layer1_bias'),
            'h2': tf.Variable(tf.zeros([1, self.n_hidden_2]), name='layer2_bias'),
            'h3': tf.Variable(tf.zeros([1, self.n_hidden_3]), name='layer3_bias'),
            'h4': tf.Variable(tf.zeros([1, self.n_hidden_4]), name='layer4_bias'),
            'out': tf.constant(0.)
        }
        self.model_path = "./model/model.ckpt"  # 模型保存路径
        self.names = ['h1', 'h2', 'h3', 'h4', 'out']  # 便与遍历
        return

    def __add_layer__(self, name, inputs, activation_function=None):
        """
        添加一个神经网络层
        :param inputs: 输入数据
        :param activation_function: 激活函数
        :return: 该层输出
        """
        ys = tf.matmul(inputs, self.Weights[name]) + self.biases[name]
        if activation_function is None:
            outputs = ys
        else:
            outputs = activation_function(ys)
        self.outputs = outputs
        return self.outputs

    def fit(self, X_train, y_train):
        """
        训练分类器
        :param X_train:训练样本
        :param y_train:训练标签
        :return:
        """
        X = tf.placeholder(tf.float32, [None, 4], name='X_train')
        y = tf.placeholder(tf.float32, [None, 1], name='y_train')
        # 隐藏层
        layer1 = self.__add_layer__('h1', X, activation_function=tf.nn.relu)
        layer2 = self.__add_layer__('h2', layer1, activation_function=tf.nn.relu)
        layer3 = self.__add_layer__('h3', layer2, activation_function=tf.nn.relu)
        layer4 = self.__add_layer__('h4', layer3, activation_function=tf.nn.relu)
        # 输出层
        predict = self.__add_layer__('out', layer4)
        # 定义损失函数
        loss = tf.reduce_mean(tf.square(predict - y))
        # 定义优化器
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss)
        # optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
        # 定义保存器
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # 可视化Loss
            tf.summary.scalar('Loss', loss)
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter("tensorboard/", sess.graph)
            # sess.run(init_weight)
            # 开始训练
            for epoch in range(self.max_iter):
                X_train = X_train.reshape((-1, 4))
                y_train = y_train.reshape((-1, 1))
                sess.run(optimizer, feed_dict={X: X_train, y: y_train})
                if (epoch + 1) % 500 == 0:
                    l = sess.run(loss, feed_dict={X: X_train, y: y_train})
                    r = sess.run(merged, feed_dict={X: X_train, y: y_train})
                    print("Epoch:", '%05d' % (epoch + 1), "loss=", "{:.3f}".format(l))
                    writer.add_summary(r, epoch)
            writer.close()
            print("Optimization Finished!")
            training_loss = sess.run(loss, feed_dict={X: X_train, y: y_train})
            print("Training loss=", training_loss, '\n')
            res = np.around(sess.run(predict, feed_dict={X: X_train})).reshape((1, -1))
            print("Training result: ", res)
            saver.save(sess, self.model_path)
            print("Model saved at: ", self.model_path)
        return

    def get_params(self):
        """
        输出网络参数
        :return: 权值，偏置
        """
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.model_path)  # 恢复模型
            weight = self.Weights.copy()
            bias = self.biases.copy()
            for name in self.names:
                weight[name] = self.Weights[name].eval()
                bias[name] = self.biases[name].eval()
        return weight, bias

    def predict(self, X_test):
        """
        使用模型预测
        :param X_test: 测试数据
        :return: 预测结果
        """
        # 重建网络
        X = tf.placeholder(tf.float32, [None, 4])
        layer1 = self.__add_layer__('h1', X, activation_function=tf.nn.relu)
        layer2 = self.__add_layer__('h2', layer1, activation_function=tf.nn.relu)
        layer3 = self.__add_layer__('h3', layer2, activation_function=tf.nn.relu)
        layer4 = self.__add_layer__('h4', layer3, activation_function=tf.nn.relu)
        # 输出层
        predict = self.__add_layer__('out', layer4)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.model_path)  # 恢复模型
            result = np.around(np.abs(sess.run(predict, feed_dict={X: X_test}))).reshape((1, -1))  # 预测
            print("[Result]: ", result)
        return result


def import_data(rate):
    iris = np.loadtxt('iris.csv', dtype=np.float, delimiter=',')
    np.random.shuffle(iris)
    index = round(iris.shape[0]*rate)
    X_train = iris[:index, :-1]
    y_train = iris[:index, -1]
    X_test = iris[index:, :-1]
    y_test = iris[index:, -1]
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    # import data
    X_train, y_train, X_test, y_test = import_data(0.05)
    
    # build net
    nn = NN()
    nn.max_iter = 30000
    nn.learning_rate = 0.0001

    # train
    # nn.fit(X_train, y_train)

    # predict
    result = nn.predict(X_test)

    # get accuracy
    print(y_test)
    compare = np.squeeze(result==y_test)
    accuracy = np.squeeze(np.where(compare == True)).shape[0]/y_test.shape[0]
    print(accuracy)
