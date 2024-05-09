function Ex1() {
    const a = tf.tensor([[1,2], [3,4]]);
    console.log("shape: ", a.shape);
    // Result of 2 from shape means the array is 2 dimensional (i.e. a matrix)
}

function Ex2() {
    const shape = [2, 2];
    const b = tf.tensor([1, 2, 3, 4], shape);
    console.log("shape: ", b.shape);
    b.print();
    //Same output as Ex1, but you started with a 1 dimensional array (const shape) and converted it to be 2 dimensional
}

//int32 default
function Ex3() {
    const a = tf.tensor([[1, 2], [3, 4]], [2, 2], 'int32');
    console.log('rank :', a.size);
    console.log('shape:', a.shape);
    console.log('dtype', a.dtype);
    a.print();
    // this is a combination of all things, size, shape, type (in that order)
    // You can't add [5, 6] to size, unless you also increase the shape
}

function Ex4() {
    const a = tf.tensor([[1, 2], [3, 4]]);
    console.log('a shape:', a.shape);
    a.print();

    const b = a.reshape([4, 1]);
    console.log("b shape: ", b.shape);
    b.print();
    // the shape of the matrix is vertical, horizontal (y, x) so when we reshape it, all values get shoved into a column that is 4 high
    // notice how the order is preserved, meaning the matrix is read, starting on the first line, left-to-right (per line), then moving down to the next line
}

function Ex5() {
    // async means that multiple functions can be ran, even before the one before finishes, making it faster than sync which has to wait for the one before to finish before it can go
    // also allows you to wait until data is processed until printing ressponse out
    const a = tf.tensor([[1, 2], [3, 4]]);
    a.array().then(array => console.log(array));

    a.data().then(data => console.log(data));
    // in this example it shows us that the a.data is actually faster than the a.array and since they both start at the same time, it finishes first
}

// operations (ops) allow the manipulation of data, while tensors allow storage of data

function Ex6() {
    const x = tf.tensor([1, 2, 3, 4]);
    const y = x.square();
    y.print();
    // x.square(); is the same as tf.square(x); which means it squares every element of the array (without a for loop too!)
}

function Ex7() {
    const a = tf.tensor([1, 2, 3, 4]);
    const b = tf.tensor([10, 20, 30, 40]);
    const y = a.add(b);
    y.print();
    // a.add(b); is the same as tf.add(a, b); this adds together the two arrays at the matching points in their arrays
}

//tensors are immutable, meaning they will hold their values unless you expressly change it, hence ops don't change values they return new ones

// if you don't purposely destroy non-used memory anymore then memory you are using might accidently get overwritten even if you didn't want it to be
function Ex8() {
    const a = tf.tensor([[1, 2], [3, 4]]);
    a.dispose();
    // same as tf.dispose(a);
}

// this will clean up all tf.tensor that aren't returned by a fxn
function Ex9() {
    const a = tf.tensor([[1, 2], [3, 4]]);
    console.log(tf.memory());
    const y = tf.tidy(() => {
        const result = a.square().log().neg();
        return result;
    })
    // the result that is return from a.square().log() is automatically disposed, the result of neg will not be disposed since it is the return value of tf.tidy()
    // not sure if this was supposed to work but it doesn't for some reason
}

function Ex10() {
    console.log(tf.getBackend());
    //this doesn't work (no idea why)
    tf.setBackend('cpu');
    console.log(tf.getBackend());
    // I guess I don't have a pre-defined backend?
}
// can only have one backend and tensorflow will typically switch to whatever is the best

// WebGL is 100x faster than CPU, though you need to explicitly destroy memory
// first pass requires the compiltion of shaders which is slow, however, it is fast after that (can be observed when pressing the button the first time and it is slow)
// when tf.tensor is made, it is not uploaded to GPU until the first pass (on second pass it is already in GPU)
// this means weights are uploaded during first prediction and subsequent passes are faster

// if first prediction matters (performance), you can use warming upwhich passing an input tensor of the same shape before the real data (strange?)



// model is a function with learnable parameters that map inputs to outputs
// two models: (Layers API) use layers or (Core API) use lower-level ops like tf.matMul(), tf.add(), etc.
// Layers API: higher-level api
// function vs sequential model

// sequential: linear stack of layers (most common)
function Ex11() {
    const model = tf.sequential({
        layers: [
            tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}),
            tf.layers.dense({units: 10, activation: 'softmax'})
        ]
    });

    
} 
// This is equivalent to the below mode
function Ex12() {
    const model = tf.sequential();
    model.add(tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}));
    model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
}
// tf.layers.dense creates a fully connected layer
// inputShape: creates input layer to insert before the dense (hidden) layer, defines number of inputs?
// units: must be positive, the dimensionality of the output space (unit = node, output unit, bias unit, hidden unit, input unit)
// acitvation: activation function that will trigger output, relu is no response below 0 and linear increase past 0

// Functional model (tf.model): allows you to create an arbitrary graph of layers (if they don't have cycles)?
function Ex13() {
    const input = tf.input({shape: [784]});
    const dense1 = tf.layers.dense({units: 32, activation: 'relu'}).apply(input);
    const dense2 = tf.layers.dense({units: 10, activation: 'softmax'}).apply(dense1);
    const model = tf.model({inputs: input, outputs: dense2});
    model.summary();
    // creates an arbitrary graph of layers and connects them via .apply(), this .apply() specifices what layer they are assigned to as you can see
    // the input layer is 784, the hidden layer is 32, and the output layer is 10 (using softmax instead of relu)
}
// the result of apply() is a "SymbolicTensor", which is the same as a tensor but without concrete values
// SymbolicTensor is made through tf.input instead of using inputShape
// can use apply() to get normal tensor too if you pass a non-abstract tensor through it
function Ex14() {
    const t = tf.tensor([-2, 1, 0, 5]);
    const o = tf.layers.activation({activation: 'relu'}).apply(t);
    o.print();
}


// Layers API has validation: forces you to specify input shape and will use it to validate input
// knowing the shape in advance allows model to automatically create parameters and can tell you if layers are compatible

// model/summary() gives: name + type of all layers, output shape for each layer, # of weight parameters for each layer, # of trainable.non-trainable parameters
// and general topology?
// a parameter would be like a weight (?)
// NULL in outputShape: model expects input to have a batch size as the outermost dimension


// Layers API: serialization (ability to save and load a model)
// this means it knows about weights, architecture, training parameters (loss, optimization), state of optimizer

// to save or load a model
function Ex15() {
    //const saveResult = await model.save('localstorage://my-model-1');
    //const model = await tf.loadLayersModel('localstorage://my-model-1');

    // await was being stupid so i had to make them into comments
}
// saves in local browser storage

// Custom Layers: allows for custom computation
// defines custom layer that computes sum of squares:
function Ex16() {
    class SquaredSumLayer extends tf.layers.Layer {
        constructor() {
            super({});
        }
        // output is a scalar (magnitude no direction, hence not a vector)
        computeOutputShape(inputShape) { return []; }

        // call() is where computation is done
        call(input, kwargs) {return input.square().sum();}

        // every layer needs unique name
        getClassName() {return 'squareSum';}
    }

    const t = tf.tensor([-2, 1, 0, 5]);
    const o = new SquaredSumLayer().apply(t);
    o.print();
    // this program outputs 30 which is (2^2 = 4, 1^2 = 1, 0^2 = 0, 5^2 = 25, thus 4 + 1 + 0 + 25 = 30)
}
// adding custom layers removes your ability to use serialization



// two ways to train machine learning: Layers API (LayersModel.fit() or LayersModel.fitDataset()) or Core API (Optimizer.minimize())

// model: fxn with learnable parameters that map input to output, optimal parameters (weights) are obtained by training model on data
// training: get batch of data to model, ask model to make prediction, compare prediction with actual (true) value, determine change to weights

// 2-layer model
function Ex17() {
    const model = tf.sequential({
        layers: [
            tf.layers.dense({inputShape: 784, units: 32, activation: 'relu'}),
            tf.layers.dense({units: 10, activation: 'softmax'})
            // softmax is exponential, but a very soft (slow to get there) exponential, hence soft max
        ]
    });
    model.weights.forEach(w => {
        console.log(w.name, w.shape);
    });
    // prints name and shape of weights (parameters)
    // 4 weights total: 2 per dense layer 
    // dense layers represent a fxn that maps the input tensor x to an output tensor y via y = Ax + b (A: kernel, b: bias are the weights of the dense layer)
    // dense layers have bias by default
    // Kernel: gives you the input and output number of nodes, bias gives solely input (in this case)

    // each weight is backend by variable object (variable: floating-point tensor with one additional method assign() used for value updating)
    // Laers API automatically initializes weights using "best practices" (?) can overwrite weights using:
    model.weights.forEach(w => {
        const newVals = tf.randomNormal(w.shape);
        //w.val is an instance of tf.Variable
        w.val.assign(newVals);
        console.log(w.name, w.shape);
    })
}


// Optimizer: decide how much to alter weights, given current prediction, Layers API allows you to use existing optimizers like sdg or adam (or optimizer class)
// Loss Fxn: what the model tries to minimize, tells model how wrong it was, loss is computed every data so weights can be updated
// can use existing loss function (categoricalCrossentropy) or any fxn that takes a predicted and true value and returns loss
// Metrics List: compute a single number that determines the success (accuracy, etc.) of the model, computed on he whole data at end of
// epoch (single passing through of the model), ideal to see loss going down over time, can either use existing metric (accuracy),
// or any function that takes predicted and true value and returns score

function Ex18() {
    const model = tf.sequential({
        layers: [
            tf.layers.dense({inputShape: 784, units: 32, activation: 'relu'}),
            tf.layers.dense({units: 10, activation: 'softmax'})
        ]
    });

    model.compile({
        optimizer: 'sgd',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });
}


// Two ways to train LayersModel
// model.fit(): provide data as one large tensor
// model.fitDataset(): provide data via Dataset object

// model.fit(): if dataset fits in main memory (RAM, very volatile) and available as single tensor
function Ex19() {
    const model = tf.sequential({
        layers: [
            tf.layers.dense({inputShape: 784, units: 32, activation: 'relu'}),
            tf.layers.dense({units: 10, activation: 'softmax'})
        ]
    });

    model.compile({
        optimizer: 'sgd',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    // making dummy data
    const data = tf.randomNormal([100, 784]);
    const labels = tf.randomUniform([100, 10]);

    function onBatchEnd(batch, logs) {
        console.log('Accuracy: ', logs.acc);
    }

    // train for 5 epochs (iterations) with batch size (number of samples ran per iteration, sample being a set of inputs) of 32
    // if you have 64 data points (which are just samples), to make it faster you group samples together into batches, if your batch size is 2 then you will
    // have 32 batches of 2 samples [1][2], [3][4], etc. in the above case there would be only 2 batches total 
    // all batches must be done before moving onto the next epoch, but the more you clump the samples, the faster it will go (less stored in memory)
    model.fit(data, labels, {
        epochs: 5,
        batchSize: 32,
        callbacks: {onBatchEnd}
    }).then(info => {
        console.log('Final Accuracy: ', info.history.acc);
    });
}
// benefits:
// splits data into a train/validation set and uses validation set to measure progress
// shuffles data but only after split, you should pre-shuffle data before passing through fit()
// splits large data tensor into smaller tensors of size batchSize
// call optimizer.minimize() while computing loss with respect to batch
// can notify on the start/end of each epoch/batch (in above example you are notified every time a batch ends [data set was ~100 samples])
// uses main thread (RAM?) to mean that queued tasks can be handled


// model.fitDataset()
// if it doesn't fit entirely to memory or is being streamed (real-time datasets)
function Ex20() {
    const model = tf.sequential({
        layers: [
            tf.layers.dense({inputShape: 784, units: 32, activation: 'relu'}),
            tf.layers.dense({units: 10, activation: 'softmax'})
        ]
    });

    model.compile({
        optimizer: 'sgd',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    function* data() {
        for (let i = 0; i < 100; i++) {
            // generates one sample at a time
            yield tf.randomNormal([784]);
        }
    }

    function* labels() {
        for (let i = 0; i < 100; i++){
            // generates one sample at a time (again?)
            yield tf.randomUniform([10]);
        }
    }

    const xs = tf.data.generator(data);
    const ys = tf.data.generator(labels);
    // zip the data and labels together, shuffle and batch 32 samples at a time
    const ds = tf.data.zip({xs, ys}).shuffle(100 /* bufferSize (hence 100 bufferSize / 32 batchSize = 3.125 batches) */).batch(32);
    // by zip I assume this means each data and label are bound to each other so they don't got seperated in the shuffling

    // train model for 5 epochs
    model.fitDataset(ds, {epochs: 5}).then(info => {
        console.log('Accuracy: ', info.history.acc);
    });

    // Once model is trained you can use model.predict() to make predictions
    // predict 3 random samples
    const prediction = model.predict(tf.randomNormal([3, 784]));
    prediction.print();
}


// i skipped saving and loading models


// model conversion
// converter: SavedModel (default format in which models are saved), KerasModel (generally saved as HDF5 file), TensorFlow Hub Module (packaged for distribution on TensorFlow Hub)

// Ex: Keras model (model.h5) in tmp/
// convert using this command: $tensorflowjs_converter --input_format=keras /tmp/model.h5 /tmp/tfjs_model
// converts model at /tmp/model.h5 and output model.json along with binary weight files at tmp/tfjs_model/ directory

// API for converted TensorFlow SavedModel: const model = await tf.loadGraphModel('path/to/model.json');
// API for converted Keras Model: const model= await tf.loadLayersModel('path/to/model.json');

// honestly you'll have to optimize it if you do this


// if you want to go through differences between python and javascript keras you can
// https://www.tensorflow.org/js/guide/layers_for_keras_users

