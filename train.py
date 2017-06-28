import tensorflow as tf
import time
import model

from tensorflow.examples.tutorials.mnist import input_data

if __name__ == "__main__":

    tf.reset_default_graph()

    seed = int(time.time())
    tf.set_random_seed(seed)

    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    image_size = 28

    x = tf.placeholder(tf.float32, [None, image_size * image_size])
    x_image = tf.reshape(x, [-1, image_size, image_size, 1])

    keep_prob = tf.placeholder(tf.float32)

    k = model.inference(x_image, keep_prob)
    p = tf.nn.softmax(k)

    t = tf.placeholder(tf.float32, [None, 10])

    with tf.name_scope('train') as scope: 
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=k,labels=t)) 
        loss_summary = tf.summary.scalar('cost', loss)
        train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

    with tf.name_scope('test') as scope:
        correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_summary = tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()

    log_dir = "log/"
    test_log_dir = log_dir + "test"
    train_log_dir = log_dir + "train"

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        test_writer = tf.summary.FileWriter(test_log_dir, sess.graph)
        train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)

        # start training
        for i in range(100):
            print i
            batch_xs, batch_ts = mnist.train.next_batch(50)
            feed_train = {x: batch_xs, t: batch_ts, keep_prob: 0.5}
            train_result = sess.run([merged, train_step], feed_dict=feed_train)
            train_writer.add_summary(train_result[0], i)
            if i > 0 and i % 10 == 0: 
                feed_test = {x: mnist.test.images, t: mnist.test.labels, keep_prob: 1.0}
                test_result = sess.run([merged], feed_dict=feed_test)
                test_writer.add_summary(test_result[0], i)
