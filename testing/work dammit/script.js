import {MnistData} from './data.js';

// this is what defines the entire neural network
function createModel() {
    const model = tf.sequential();
    //inputShape: width, height, channels (a 2D layer)
    // kernelSize: int/list of 2 ints, specifying size of window
    // filters: the dimension of the output space (# number of filters)
    // strides: int/list of 2 ints, if input tensor has 4 dimensions (batch, height, width, channels), determines how much the widnow shifts by in each of the dimensions
    model.add(tf.layers.conv2d({
      inputShape: [28, 28, 1],
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));
    
    // down-samples the features from the image to better identify differences (layer 2)
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    
    // layer 3
    model.add(tf.layers.conv2d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));
    //layer 4
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    
    // here you "flatten" the input 2D array into a linear output
    model.add(tf.layers.flatten());
  
    // this is the output layer, and has 10 units for how many numbers it can identify (0-9)
    model.add(tf.layers.dense({
      units: 10,
      kernelInitializer: 'varianceScaling',
      activation: 'softmax'
    }));
  
    
    // here you just pick what optimizer you want to use, what loss fxn you will use, and how to report your accuracy
    const optimizer = tf.train.adam();
    model.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });
  
    return model;
}

// This runs the entire model
async function run() {  
    const data = new MnistData();
    await data.load();
    const model = createModel();
  
    await train(model, data);

    await showAccuracy(model, data);
}

document.addEventListener('DOMContentLoaded', run);



async function train(model, data) {
    // this is the training data set (we train it on 5500 images)
    const [trainXs, trainYs] = tf.tidy(() => {
      const d = data.nextTrainBatch(5500);
      return [
        d.xs.reshape([5500, 28, 28, 1]),
        d.labels
      ];
    });
  
    // this is the testing data set (we will test it on 1000 images)
    const [testXs, testYs] = tf.tidy(() => {
      const d = data.nextTestBatch(1000);
      return [
        d.xs.reshape([1000, 28, 28, 1]),
        d.labels
      ];
    });
    
    return model.fit(trainXs, trainYs, {
      batchSize: 512,
      validationData: [testXs, testYs],
      epochs: 10,
      shuffle: true
    });
  }

const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

// predicts what the image will be 
function doPrediction(model, data, testDataSize = 500) {
    const testData = data.nextTestBatch(testDataSize);
    const testxs = testData.xs.reshape([testDataSize, 28, 28, 1]);
    const labels = testData.labels.argMax(-1);
    const preds = model.predict(testxs).argMax(-1);

    testxs.dispose();
    return [preds, labels];
}

// shows how accurate the predictions made are
async function showAccuracy(model, data) {
    const [preds, labels] = doPrediction(model, data);
    const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
    const container = document.getElementById("what").innerHTML
    tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

    labels.dispose();
}