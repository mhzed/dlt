dlt
--------

Deep learning in typescript.  

The hello-world of Deep Learning is the mnist digits recognition.  This project implements a feed forward neural network to train on the mnist dataset.  Even though it's implemented in typescript (javascript), the performance is excellent as it uses the [vectorious](https://github.com/mateogianolio/vectorious) library that utlizes [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms).  A full epoch on 50k samples with minibatch size of 20 takes only 2+ seconds on a 2013 Macbook Pro with Interl Core I5.

To start training, run:

    # Using npm
    npm run train
    # Using yarn
    yarn train

