import { ActivationFunction, CostType } from "../types";
import { NdArray } from "../core/ndarray";

export class ReluActivation implements ActivationFunction {
  activate(z: NdArray) : NdArray {
    const ret = z.dup();
    ret.data.forEach((v, i) => { if (v<0) ret.data[i]=0; })
    return ret;
  }
  gradient(z: NdArray, a?: NdArray): NdArray {
    const ret = z.dup();
    ret.data.forEach((v, i) => ret.data[i] = (v>0)?1:0)
    return ret;
  }
  cost(label: NdArray, z: NdArray, activation: NdArray, c: CostType): NdArray {
    // rely could be used in the output layer, but is generally not recommended, and the cost computation
    // is much more expensive than sigmoid or softmax.  Here we simply forbid it.
    throw new Error("relu should only be used on hidden layers");
  }
}
