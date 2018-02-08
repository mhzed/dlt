import { NdArray } from "./core/ndarray";

export interface LabelData {
  input: NdArray    // the image pixesl = size of input layer 
  label: NdArray    // the desired output
}
export interface WeightInitializer {
  weights(shape: [number, number]): NdArray;
}
export type CostType = "cross-entropy" | "quadratic";

export interface ActivationFunction {
  activate(input: NdArray) : NdArray;
  derivative(input: NdArray) : NdArray;
  cost(label: NdArray, z: NdArray, activation: NdArray, c: CostType): NdArray;
}

export enum Layer {
  Undefined,
  Input,
  Dense,      // AKA fully connected
  Dropout,
  Convolution,
  Maxpool
}

export interface EpochStopper {
  onEpoch(iEpoch: number, elapsedSec: number): boolean;
}

export interface Regularizer {
  // it's safe to update weights in place
  regularize(weights: NdArray): NdArray;
}

export class L2Regularizer {
  private scale: number;
  constructor(private decayRate: number,  // or λ
    private eta: number, 
    private sizeTrain: number)  {
      this.scale = 1-this.eta*this.decayRate/this.sizeTrain 
    }

  regularize(weights: NdArray): NdArray {
    weights.muleq(this.scale)
    return weights;
  }
}

export class L1Regularizer {
  private delta: number;
  constructor(private decayRate: number,  // or λ
    private eta: number, 
    private sizeTrain: number)  {
      this.delta = this.eta*this.decayRate/this.sizeTrain 
    }

  regularize(weights: NdArray): NdArray {
    // TODO: verify
    (weights as any).data.forEach((n,i)=>(weights as any).data[i] -= this.delta * (n>0?1:-1))
    return weights;
  }
}

export class NormWeightInitializer implements WeightInitializer {
  weights(shape: [number, number]): NdArray {
    return NdArray.randn(shape).muleq(1/Math.sqrt(shape[1]));
  }
}