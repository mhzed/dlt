import { ActivationFunction, CostType } from "../types";
import { NdArray } from "../core/ndarray";


export class SoftmaxActivation implements ActivationFunction {
  // softmax =>  e^x / Σ(e^x)
  // returns stable softmax: to avoid NaN, shift by -max, 
  activate(z: NdArray) : NdArray {
    let ret = z.dup();
    let max = ret.max();
    let denom = ret.data.reduce((a,n)=>a+Math.exp(n - max), 0);
    ret.data.forEach((n,i)=>ret.data[i] = Math.exp(n - max)/denom);
    return ret;
  }
  /**
   * First the caclulus:
   * f(x) = g(x)/h(x)
   * f'(x) = (g'(x) * h(x) - h'(x) * g(x)) / h(x)^2
   * in softmax: g(x) = e^x,  h(x) = Σ(e^x)
   * case 1: f'(x) of f(x) at i WRT to input x at j, where i == j
   * f'(x) = (e^x * Σ - e^x * e^x) / Σ^2
   *       = (e^x / Σ) * (1 - e^x / Σ)
   *       = S(x) * (1 - S(x))      S is softmax activation above
   * case 2: f'(x) of f(x) at i WRT to input x at j, where i != j
   * f'(x) = (0 - e^x[j] * e^x[i]) / Σ^2
   *       = -1 * S(x[i]) * S(x[j])
   * 
   * Noe the softmax equation implies every input affects every output, so the 'gradient' of z is 
   * a Jacobian matrix of dimension N,N
   * This will break batch backprop in hidden layers where input is stacked into a single matrix.
   * And also softmax is generally only used in the output layer, not hidden layers.  Thus we throw.
   */
  gradient(z: NdArray, a?: NdArray): NdArray {
    throw new Error("not supported");
  }

  cost(label: NdArray, z: NdArray, activation: NdArray, c: CostType): NdArray {
    if (c == "cross-entropy") {   // aka negative log likelihood
      // - Σ(y * ln(a)),  y is one hot vector
      // => - ln(a[i]),  i is the index of 1 in y
      let ret = NdArray.zeros(activation.shape);
      for (let i = 0; i < activation.length; i++ ) {
        if (label.data[i] == 1) {
          ret.data[i] = -( 1- activation.data[i]);
        } else {
          ret.data[i] = activation.data[i];
        }
      }
      return ret;
    } else throw new Error("invalid cost " + c);
  }
}