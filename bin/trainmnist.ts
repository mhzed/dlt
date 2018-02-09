
import * as path from 'path' 

import { prepareMnistData } from "../src/mnist/prepareMnistData";
import { InputLayer } from '../src/layers/inputlayer';
import { DenseLayer } from '../src/layers/denselayer';
import { Train } from '../src/train';
import { Rand } from '../src/core/rand';
import { BaseLayer } from '../src/layers/baselayer';
import { SigmoidAcitivation } from '../src/activation/sigmoid';
// import { DropoutLayer } from '../src/layers/dropoutlayer';
// import { L2Regularizer, L1Regularizer } from '../src/types';
import { StopAt } from '../src/stoppers';
// import { StopWhenNoBetterThanAverage } from '../src/stoppers';

async function main () {
    Rand.seed("1234");
    console.log("Training on MNIST data using a single 100 neuron hidden layer.")
    console.log("Training will stop automatically if accuray stops impproving on validation dataset");
    console.log("You should see about 97.7% to 98% accuracy on the 10k test data.")
    let dataset = await prepareMnistData(path.resolve(__dirname, "../data/mnist/train-images-idx3-ubyte"), 
      path.resolve(__dirname, "../data/mnist/train-labels-idx1-ubyte"));
    console.log(`${dataset.length} items loaded`);

    const testSize = dataset.length/10;
    const trainSize = dataset.length - testSize;
    // const testSize = 1000;
    // const trainSize = 1000;
    const miniBachSize = 20;
    const learnRate = 0.1;

    let train = dataset.slice(0,trainSize);
    let validate = testSize>0?dataset.slice(trainSize, trainSize + (testSize)): null;
  
    let net = new InputLayer([28*28, 1]);
    let layer: BaseLayer = net;
    //layer = new DropoutLayer(layer, 0.2);
    //layer = new DenseLayer(layer, 200, new SigmoidAcitivation(), null, );
    layer = new DenseLayer(layer, 30, new SigmoidAcitivation(), null, 
      //new L2Regularizer(3, learnRate, train.length)
      //new L1Regularizer(2, learnRate, train.length)
    );
    //layer = new DropoutLayer(layer, 0.2);
    layer = new DenseLayer(layer, 10, new SigmoidAcitivation(), "cross-entropy", 
      //new L2Regularizer(3, learnRate, train.length)
      //new L1Regularizer(2, learnRate, train.length)
    );

    Train.sgd(net, train, miniBachSize, learnRate,
      //new StopWhenNoBetterThanAverage(20, ()=>Train.evaluate(net, validate), 2, true)
      new StopAt(300, (i, sec)=>{
        console.log(`${i}[${sec}] `, Train.evaluate(net, train).toFixed(6), 
          Train.evaluate(net, validate).toFixed(6))
      })
    );

    let grade = await prepareMnistData(path.resolve(__dirname, "../data/mnist/t10k-images-idx3-ubyte"), 
      path.resolve(__dirname, "../data/mnist/t10k-labels-idx1-ubyte"));
    let finalgrade = Train.evaluate(net, grade);
    console.log(`Final: ${finalgrade}`);
  }
  
  main().catch((err)=>console.error(err)).then();

  /**
   * 
350 1 0.874
Final: 0.8726
   * 
   * 
L1 2:
   350 0.965 0.848
Final: 0.8611
   * 
   * 
   * L2 5:
   * 350 0.964 0.872
Final: 0.8711

Dropotu 0.2
350 1 0.886
Final: 0.8857

Dropotu 0.4
350 0.988 0.881
Final: 0.88
   */