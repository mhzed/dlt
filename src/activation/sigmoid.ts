import { ActivationFunction, CostType } from "../types";
import { NdArray } from "../core/ndarray";

export class SigmoidActivation implements ActivationFunction {
  // σ(x) = 1 / (1 + e^-x)
  activate(z: NdArray) : NdArray {
    let ret = z.dup();
    ret.data.forEach((n,i) => ret.data[i] = 1/(1+Math.exp(-n)) );
    return ret;
  }
  // σ'(x) = σ(x) * (1-σ(x)) 
  gradient(z: NdArray, a?: NdArray): NdArray {
    let sigmoid = a ? a.dup() : this.activate(z);
    return sigmoid.muleqn(sigmoid.mul(-1).addeq(1));
  }

  cost(label: NdArray, z: NdArray, activation: NdArray, c: CostType): NdArray {
    // a is activation is σ(z)
    // label is y
    if (c == "cross-entropy") {
      // cross-entropy cost for sigmoid is: C = -1 * (y * ln(a) + (1 - y) * ln(1 - a))
      // calculus:  y = ln(a), y' = 1/a
      // C`(a) is: -1 * (y / a - (1 - y) / (1 - a))
      // error is ∂(c)/∂(z), using chain rule:
      // ∂(c)/∂(a) * ∂(a)/∂(z) => 
      // -1 * ( y / a - (1 - y) / (1 -a) ) * σ'(z) =>
      // -1 * ( y / σ(z) - (1 - y) / (1 - σ(z)) ) * σ'(z) =>
      // -1 * ( y / σ(z) - (1 - y) / (1 - σ(z)) ) * (σ(z) * (1-σ(z))) =>
      // -1 * (y - σ(z)) => σ(z) - y
      return activation.sub(label);
    } else if (c == "quadratic") {
      // quadratic cost is: C = 1/2 * (y - a) ^ 2, => C' = (a - y)
      // error is ∂(c)/∂(z), chain rule =>  ∂(c)/∂(a) * ∂(a)/∂(z) => 
      // (a - y) * σ'(z)
      return activation.sub(label).muleqn(this.gradient(z));
    } else throw new Error("invalid cost " + c);
  }
}
