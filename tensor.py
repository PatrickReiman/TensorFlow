import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Sequential: one input tensor, one output tensor
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),
    # below 0: output is none, past 0: linearly increasing output
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

# each class will return a vector of logits
predictions = model(x_train[:1]).numpy()
predictions

# converts logits to probabilities for each "class"
tf.nn.softmax(predictions).numpy()
# possible to integrate nn.softmax into activation function but apparently this makes loss difficult or impossible to calculate

# our loss fxn, takes a vector of actual true values and a vector of logits and the output is the scalar loss
# loss is equal to -log(probability) of the "true" class. loss = 0 when model is sure answer is correct
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

# class/label = output/target variable that the model predicts (SUPERVISED)
# positive/negative class = output (class) is one or the other (cat or a dog, is a bus or isn't a bus, is cancer or isn't cancer)
# "True" class = output (class) was able to be identified (by the model)
# "False" class = output (class) was not able to be identified (by the model)
# confusion matrix: think of punnet square positive (p) negative (n) on both axes (top is actual values, left is predicted values) x(actual)/x(predicted)
# true positive: p/p, both predicted and actual values are true
# false negative: p/n, actual value is true but predicted to be false
# false positive: n/p, actual value false but predicted to be true
# true negative: n/n, both predicted and actual values are false
# ideal: no false positives/negatives
# actual:       cat       dog
#predicted: cat  5 (t/p)   2 (f/p)
#predicted: dog  3 (f/n)   3 (t/n)
# true (t) negative (n) false (f) positive (p)
# positive in this case just cat, while negative just means dog (hence predicted cat being actual dog means it falsely predicts it is a cat (positive))
# the same is true for false negative where it is a cat (positive) but it is a dog (negative)

# untrained model gives probabilities close to random (1/10 for each class), so initial loss should be to -tf.math.log(1/10) =~2.3
# this is because as we said loss is equal to -log(probability of a true class)??

loss_fn(y_train[:1], predictions).numpy()

# this compiles the model (optimizer: how the )
model.compile(optimizer = 'adam', loss = loss_fn, metrics = ['accuracy'])

# checks [model] parameters, minimizes loss
model.fit(x_train, y_train, epochs = 5)

# checks [model] performance on test test
model.evaluate(x_test, y_test, verbose = 2)

# returns probability
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
probability_model(x_test[:5])