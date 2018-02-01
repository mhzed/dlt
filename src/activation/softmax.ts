// import { ActivationFunction, CostType } from "../types";
// import { Matrix } from "../matrix";

// export class SoftmaxAcitivation implements ActivationFunction {
//   activate(weightedInputs: Matrix) : Matrix {
//     return weightedInputs.softmax();
//   }
//   derivative(input: Matrix): Matrix {
//   }
//   cost(label: Matrix, z: Matrix, activation: Matrix, c: CostType): Matrix {
//     if (c == "cross-entropy") {
//       return activation.sub(label);
//     } else if (c == "quadratic") {
//       return activation.sub(label).muleqn(this.derivative(z));
//     } else throw new Error("invalid cost " + c);
//   }
// }
