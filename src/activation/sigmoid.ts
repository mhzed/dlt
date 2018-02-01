import { ActivationFunction, CostType } from "../types";
import { Matrix } from "../core/matrix";

export class SigmoidAcitivation implements ActivationFunction {
  activate(weightedInputs: Matrix) : Matrix {
    return weightedInputs.sigmoid();
  }
  derivative(input: Matrix): Matrix {
    let sigmoid = input.sigmoid();
    return sigmoid.muleqn(sigmoid.mul(-1).addeq(1));
  }
  cost(label: Matrix, z: Matrix, activation: Matrix, c: CostType): Matrix {
    if (c == "cross-entropy") {
      return activation.sub(label);
    } else if (c == "quadratic") {
      return activation.sub(label).muleqn(this.derivative(z));
    } else throw new Error("invalid cost " + c);
  }
}
