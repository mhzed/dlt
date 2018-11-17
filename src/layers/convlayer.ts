import { Layer, ActivationFunction, NormWeightInitializer, WeightInitializer } from "../types";
import {  NdArray } from "../core/ndarray";
import { Rand } from "../core/rand";
import { BaseLayer } from "./baselayer";
import { InputLayer } from "./inputlayer";
import { Matrix } from "../core/matrix";
import { Volume } from "../core/volume";

type Feature = {
  weight: NdArray;
  bias: number;
}
export class ConvolutionLayer extends BaseLayer {
  readonly type = Layer.Convolution;

  private features: Feature[] = []
  private z: NdArray;
  private forwardOutput: NdArray;
  /**
   * 
   * @param inputLayer 
   * @param probability the probablity of drop out: 0.2 => 20% 
   */
  constructor(inputLayer:BaseLayer, 
      private activation: ActivationFunction,
      featureSize: number,
      private kernel: [number, number],   // row, col
      private padding: number = 0,
      private stride: number = 1,
      wi: WeightInitializer = new NormWeightInitializer()) {

    super(inputLayer);
    let inputDepth: number;
    switch (inputLayer.type) {
      case Layer.Input:
        if ((inputLayer as InputLayer).shape.length !== 3) throw new Error("Expecing a 3D volume");
        inputDepth = (inputLayer as InputLayer).shape[2];
        break;
      case Layer.Convolution:
        inputDepth = (inputLayer as ConvolutionLayer).features.length;
        break;
      default: 
        throw new Error(`${inputLayer.type} not supported`)
    }
    this.features = [];
    for (let i=0; i<featureSize; i++) {
      this.features.push({
        weight: wi.weights(kernel.concat([inputDepth])),
        bias: Rand.rand()
      });
    }
  }

  forward(input: NdArray): NdArray {
    const inputHeight = input.shape[0],   // row size = height
          inputWidth = input.shape[1],
          kernelWidth = this.kernel[1],
          kernelHeight = this.kernel[0],
          outWidth = Math.ceil((inputWidth + this.padding * 2 - kernelWidth + 1) / this.stride),
          outHeight = Math.ceil((inputHeight + this.padding * 2 - kernelHeight + 1) / this.stride);

    this.z = NdArray.zeros([outHeight, outWidth, this.features.length]);

    let outr = 0, outc = 0;
    for (let r = -this.padding; r < inputHeight + this.padding - this.stride + 1; r += this.stride, outr++) {
      for (let c = -this.padding; c < inputWidth + this.padding - this.stride + 1; c += this.stride, outc++) {
        // convolute input volume by this.kernel 
        let window = Volume.sliceWindow(input, r, c, kernelWidth, kernelHeight);
        if (window.data.length != this.features[0].weight.data.length) throw new Error("unexpected size");
        // compute output feature by feature
        for (let f = 0; f < this.features.length; f++) {
          let weight = this.features[f].weight, 
              bias = this.features[f].bias;
          let v = Matrix.dot(weight.data, window.data) + bias;
          let retoffset = this.z.stride[0] * outr + this.z.stride[1] * outc + f;
          this.z.data[ retoffset ] = v;
        }
      }
    }
    this.forwardOutput = this.activation.activate(this.z);
    return this.forwardOutput;
  }

  backprop(input: NdArray, stop: boolean) {
    
  }
}
