import { ActivationFunction, CostType } from "../types";
import { NdArray } from "../core/ndarray";

export class LinearActivation implements ActivationFunction {
  activate(z: NdArray) : NdArray {
    return z;
  }
  private _gradient: NdArray = null;
  gradient(input: NdArray): NdArray {
    if (this._gradient && this._gradient.length == input.length) {
      return this._gradient;
    } else {
      this._gradient = NdArray.consts(input.shape, 1);
      return this._gradient;
    }
  }
  cost(label: NdArray, z: NdArray, activation: NdArray, c: CostType): NdArray {
    // TODO:
    return activation.sub(label);
    // throw new Error("not implemented");
    // if (c == "cross-entropy") {
    //   return activation.sub(label);
    // } else if (c == "quadratic") {
    //   return activation.sub(label).muleqn(this.derivative(z));
    // } else throw new Error("invalid cost " + c);
  }
}
