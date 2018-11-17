
import * as path from 'path';
import { prepareMnistData } from "../src/mnist/prepareMnistData";
import { LabelData, L2Regularizer, L1Regularizer } from '../src/types';
import { StopWhenNoBetterThanAverage } from '../src/stoppers';
import { Rand } from '../src/core/rand';
import { InputLayer } from '../src/layers/inputlayer';
import { DenseLayer } from '../src/layers/denselayer';
import { Train } from '../src/train';
import { BaseLayer } from '../src/layers/baselayer';
import { SigmoidActivation } from '../src/activation/sigmoid';
import { NdArray } from '../src/core/ndarray';

Rand.seed();
//Rand.seed('abc');  // fix randomness: result in predicatble output for verification

let dataset: LabelData[];
const DataSize = 1000;
const ImageSize = 28*28;
const LableSize = 10;
const MiniBachSize = 20;
const LearnRate = 0.5;
const MinAcceptableAccurancy = 0.7;
const TestSize = DataSize / 10;
const TrainSize = DataSize - TestSize;

const ConvertImg = (img: Uint8Array) => NdArray.fromCol(img).mul(1/255);
describe('train mnist via dense layers', function() {

  it('load data', async function() {
    this.timeout(10000);
    dataset = await prepareMnistData(path.resolve(__dirname, "../data/mnist/train-images-idx3-ubyte"), 
      path.resolve(__dirname, "../data/mnist/train-labels-idx1-ubyte"), ConvertImg, DataSize);
    dataset.length.should.equal(DataSize);
  })

  let epoch1;
  it('train with sigmoid quadratic', async function () {
    this.timeout(10000);
    const HiddenSize = 30;
    let testData = dataset.slice(TrainSize);
    let trainData = dataset.slice(0, TrainSize);

    let input = new InputLayer([ImageSize, 1]);
    let layer: BaseLayer = new DenseLayer(input, HiddenSize, new SigmoidActivation());
    layer = new DenseLayer(layer, LableSize, new SigmoidActivation(), "quadratic");
    epoch1 = Train.sgd(input, trainData, MiniBachSize, LearnRate,
      new StopWhenNoBetterThanAverage(5, ()=>Train.evaluate(input, testData), 1, false));
    epoch1.should.lessThan(70);  // 
    (Train.evaluate(input, testData)).should.greaterThan(MinAcceptableAccurancy);
  })

  let accuracy;
  it('train with sigmoid cross entroy', async function () {
    this.timeout(10000);
    const HiddenSize = 30;
    let testData = dataset.slice(TrainSize);
    let trainData = dataset.slice(0, TrainSize);

    let input = new InputLayer([ImageSize, 1]);
    let layer: BaseLayer = new DenseLayer(input, HiddenSize, new SigmoidActivation());
    layer = new DenseLayer(layer, LableSize, new SigmoidActivation(), "cross-entropy");
    let epochs = Train.sgd(input, trainData, MiniBachSize, LearnRate,
      new StopWhenNoBetterThanAverage(5, ()=>Train.evaluate(input, testData), 1, false));
      epochs.should.lessThan(epoch1);  // converges faster than quadratic
    accuracy = Train.evaluate(input, testData);
    accuracy.should.greaterThan(MinAcceptableAccurancy);
  })

  it('train with sigmoid cross entroy l2 weight decay', async function () {
    this.timeout(10000);
    const HiddenSize = 30;
    let testData = dataset.slice(TrainSize);
    let trainData = dataset.slice(0, TrainSize);
    
    let input = new InputLayer([ImageSize, 1]);
    let layer: BaseLayer = new DenseLayer(input, HiddenSize, new SigmoidActivation(), null, 
      new L2Regularizer(0.5, LearnRate, trainData.length));
    layer = new DenseLayer(layer, LableSize, new SigmoidActivation(), "cross-entropy", 
      new L2Regularizer(0.5, LearnRate, trainData.length));
    let epochs = Train.sgd(input, trainData, MiniBachSize, LearnRate,
      new StopWhenNoBetterThanAverage(5, ()=>Train.evaluate(input, testData), 1, false));
      epochs.should.lessThan(epoch1);  // converges faster than quadratic
    let thisaccuracy = Train.evaluate(input, testData);
    thisaccuracy.should.greaterThan(MinAcceptableAccurancy);
  })

  it('train with sigmoid cross entroy l1 weight decay', async function () {
    this.timeout(10000);
    const HiddenSize = 30;
    let testData = dataset.slice(TrainSize);
    let trainData = dataset.slice(0, TrainSize);
    
    let input = new InputLayer([ImageSize, 1]);
    let layer: BaseLayer = new DenseLayer(input, HiddenSize, new SigmoidActivation(), null, 
      new L1Regularizer(0.3, LearnRate, trainData.length));
    layer = new DenseLayer(layer, LableSize, new SigmoidActivation(), "cross-entropy", 
      new L1Regularizer(0.3, LearnRate, trainData.length));
    let epochs = Train.sgd(input, trainData, MiniBachSize, LearnRate,
      new StopWhenNoBetterThanAverage(5, ()=>Train.evaluate(input, testData), 1, false));
      epochs.should.lessThan(epoch1);  // converges faster than quadratic
    let thisaccuracy = Train.evaluate(input, testData);
    thisaccuracy.should.greaterThan(MinAcceptableAccurancy);
  })

})

