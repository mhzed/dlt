import { Matrix } from "./core/matrix";

export interface LabelData {
  input: Matrix    // the image pixesl = size of input layer 
  label: Matrix    // the desired output
}
export interface WeightInitializer {
  weights(shape: [number, number]): Matrix;
}
export type CostType = "cross-entropy" | "quadratic";

export interface ActivationFunction {
  activate(input: Matrix) : Matrix;
  derivative(input: Matrix) : Matrix;
  cost(label: Matrix, z: Matrix, activation: Matrix, c: CostType): Matrix;
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
  regularize(weights: Matrix): Matrix;
}

export class L2Regularizer {
  private scale: number;
  constructor(private decayRate: number,  // or λ
    private eta: number, 
    private sizeTrain: number)  {
      this.scale = 1-this.eta*this.decayRate/this.sizeTrain 
    }

  regularize(weights: Matrix): Matrix {
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

  regularize(weights: Matrix): Matrix {
    // TODO: verify
    (weights as any).data.forEach((n,i)=>(weights as any).data[i] -= this.delta * (n>0?1:-1))
    return weights;
  }
}

export class NormWeightInitializer implements WeightInitializer {
  weights(shape: [number, number]): Matrix {
    return Matrix.randn(shape).muleq(1/Math.sqrt(shape[1]));
  }
}