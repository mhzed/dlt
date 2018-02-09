import { ActivationFunction, CostType } from "../types";
import { NdArray } from "../core/ndarray";

export class SigmoidAcitivation implements ActivationFunction {
  activate(weightedInputs: NdArray) : NdArray {
    return weightedInputs.sigmoid();
  }
  derivative(input: NdArray): NdArray {
    let sigmoid = input.sigmoid();
    return sigmoid.muleqn(sigmoid.mul(-1).addeq(1));
  }
  cost(label: NdArray, z: NdArray, activation: NdArray, c: CostType): NdArray {
    if (c == "cross-entropy") {
      return activation.sub(label);
    } else if (c == "quadratic") {
      return activation.sub(label).muleqn(this.derivative(z));
    } else throw new Error("invalid cost " + c);
  }
}
