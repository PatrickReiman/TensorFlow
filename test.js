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
    // x.square(); is the same as tf.square(x);
}