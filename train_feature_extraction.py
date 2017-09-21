import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
from tqdm import tqdm

LEARNING_RATE = .001
EPOCHS = 10
BATCH_SIZE = 128
NB_CLASSES = 43

# TODO: Load traffic signs data.
with open('train.p', 'rb') as f:
    data = pickle.load(f)

X = data['features']
y = data['labels']

# TODO: Split data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(X,y)

# TODO: Define placeholders and resize operation.
X = tf.placeholder(tf.float32, [None, 32,32,3])
y = tf.placeholder(tf.int32, [None])
one_hot_y = tf.one_hot(y,NB_CLASSES)
resized = tf.image.resize_images(X, [227,227])

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.

shape = (fc7.get_shape().as_list()[-1], NB_CLASSES)
fc8_w = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8_b = tf.Variable(tf.zeros(NB_CLASSES))
logits = tf.matmul(fc7, fc8_w) + fc8_b

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = one_hot_y))
optimizer = tf.train.AdamOptimizer(learning_rate = .001).minimize(cost, var_list=[fc8_w, fc8_b])
predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_score = tf.reduce_mean(tf.cast(predictions, tf.float32))

init = tf.global_variables_initializer()

def eval_on_data(X_data, y_data, sess):
    total_acc = 0
    total_loss = 0
    for offset in range(0, X_data.shape[0], BATCH_SIZE):
        end = offset + BATCH_SIZE
        X_batch = X_data[offset:end]
        y_batch = y_data[offset:end]

        loss, acc = sess.run([cost, accuracy_score], feed_dict={X: X_batch, y: y_batch})
        total_loss += (loss * X_batch.shape[0])
        total_acc += (acc * X_batch.shape[0])

    return total_loss/X_data.shape[0], total_acc/X_data.shape[0]

# TODO: Train and evaluate the feature extraction model.
with tf.Session() as sess:
    sess.run(init)
    num_batches = X_train.shape[0] // BATCH_SIZE + 1

    for epoch in range(EPOCHS):
        # print("-----")
        # print('EPOCH: ', epoch+1)

        batches_qbar = tqdm(range(num_batches), desc='Epoch {}/{}'.format(epoch+1, EPOCHS), unit='batches')
        for batch in batches_qbar:
            # for batch in range(num_batches):
            start = batch * BATCH_SIZE
            end = (batch+1) * BATCH_SIZE
            batch_x, batch_y = X_train[start:end], y_train[start:end]

            sess.run(optimizer, feed_dict = {X: batch_x, y: batch_y})
        # train_accuracy_score = eval_on_data(X_train, y_train, sess)
        validation_accuracy_score = eval_on_data(X_valid, y_valid, sess)
        # print("Training accuracy: ", train_accuracy_score)
        print("Validation accuracy: ", validation_accuracy_score)
