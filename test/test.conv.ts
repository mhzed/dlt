import * as path from 'path';
import "should";

import { prepareMnistData } from "../src/mnist/prepareMnistData";
import { NdArray } from "../src/core/ndarray";
import { LabelData, WeightInitializer } from "../src/types";
// import { InputLayer } from '../src/layers/inputlayer';
// import { ConvolutionLayer } from '../src/layers/convlayer';
// import { ReluActivation } from '../src/activation/relu';

// import * as assert from'assert';
//import * as should from "should";
let dataset: LabelData[];
const DataSize = 1000;
const ImageShape = [28, 28, 1];

// Initialize with 1's, for testing.
export class OneWeightInitializer implements WeightInitializer {
  weights(shape: number[]): NdArray {
    const ret = NdArray.zeros(shape);
    ret.data.forEach((v, i) => ret.data[i] = 1);
    return ret;
  }
}

describe('train mnist via convolution', function() {

  it('load data', async function() {
    dataset = await prepareMnistData(
      path.resolve(__dirname, "../data/mnist/train-images-idx3-ubyte"), 
      path.resolve(__dirname, "../data/mnist/train-labels-idx1-ubyte"), 
      (img: Uint8Array) => NdArray.from(img, ImageShape).muleq(1/255),
      DataSize);
    dataset.length.should.equal(DataSize);
  })

  it('train', async function() {
    // let input = new InputLayer(ImageShape);
    // let layer = new ConvolutionLayer(input, new ReluActivation(), 64, [5,5], 2, 1);
  })
})

