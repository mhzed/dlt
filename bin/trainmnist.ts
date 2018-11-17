
import * as path from 'path' 

import { prepareMnistData } from "../src/mnist/prepareMnistData";
import { InputLayer } from '../src/layers/inputlayer';
import { DenseLayer } from '../src/layers/denselayer';
import { Train } from '../src/train';
// import { Rand } from '../src/core/rand';
import { BaseLayer } from '../src/layers/baselayer';
import { SigmoidActivation } from '../src/activation/sigmoid';
import { NdArray } from "../src/core/ndarray";
// import { DropoutLayer } from '../src/layers/dropoutlayer';
// import { L2Regularizer, L1Regularizer } from '../src/types';

import { ReluActivation } from '../src/activation/relu';
// import { SoftmaxActivation } from '../src/activation/softmax';
import { StopWhenNoBetterThanAverage } from '../src/stoppers';
// import { StopAt } from '../src/stoppers';

const NeuronSize = 30;
const ConvertImg = (img: Uint8Array) => NdArray.fromCol(img).mul(1/255);

async function main () {
    // Rand.seed("1234");   // fix seed for predictable results
    console.log(`Training on MNIST data using a single ${NeuronSize} neuron hidden layer.`)
    console.log("Training will stop automatically if accuracy stops impproving on validation dataset");
    let dataset = await prepareMnistData(
      path.resolve(__dirname, "../data/mnist/train-images-idx3-ubyte"), 
      path.resolve(__dirname, "../data/mnist/train-labels-idx1-ubyte"),
      ConvertImg
    );
    console.log(`${dataset.length} items loaded`);

    const ValidationSize = dataset.length/10;
    const TrainSize = dataset.length - ValidationSize;
    const MiniBachSize = 20;
    const LearnRate = 0.1;

    let trainDataset = dataset.slice(0,TrainSize);
    let validateDataset = ValidationSize>0?dataset.slice(TrainSize, TrainSize + (ValidationSize)): null;
  
    let net = new InputLayer([28*28, 1]);
    let layer: BaseLayer = net;
    //layer = new DropoutLayer(layer, 0.2);
    layer = new DenseLayer(layer, NeuronSize
      // , new SigmoidActivation()
      , new ReluActivation()
      , null, 
      //new L2Regularizer(3, learnRate, train.length)
      //new L1Regularizer(2, learnRate, train.length)
    );
    // layer = new DropoutLayer(layer, 0.2);
    layer = new DenseLayer(layer, 10
      , new SigmoidActivation()
      //, new SoftmaxActivation()
      , "cross-entropy", 
      //new L2Regularizer(3, learnRate, train.length)
      //new L1Regularizer(2, learnRate, train.length)
    );

    Train.sgd(net, trainDataset, MiniBachSize, LearnRate,
      new StopWhenNoBetterThanAverage(5, ()=>Train.evaluate(net, validateDataset), 2, true)
    );

    let gradeDataset = await prepareMnistData(
      path.resolve(__dirname, "../data/mnist/t10k-images-idx3-ubyte"), 
      path.resolve(__dirname, "../data/mnist/t10k-labels-idx1-ubyte"),
      ConvertImg
    );
    let finalgrade = Train.evaluate(net, gradeDataset);
    console.log(`Final: ${finalgrade}`);
  }
  
  main().catch((err)=>console.error(err)).then();