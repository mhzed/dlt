import { Layer, ActivationFunction, WeightInitializer, 
    CostType, NormWeightInitializer, Regularizer } from "../types";
import { Matrix } from "../core/matrix";
import * as _ from "lodash";
import { BaseLayer } from "./baselayer";

export type DensesForwardType = Matrix;
export type DensesBackPropType = Matrix;

export class DenseLayer extends BaseLayer {
  readonly type = Layer.Dense;

  private weight: Matrix;
  private bias:   Matrix;

  /**
   * 
   * @param inputLayer     previous layer
   * @param size           size of neurons on this layer
   * @param activation     the activation function
   * @param cost           the cost function, set this only on the output layer
   * @param regularizer    the regularizer to use
   * @param wi             random weight initializer, optional
   */
  constructor(inputLayer:BaseLayer, public size: number,
    private activation: ActivationFunction,
    private cost: CostType = null,
    private regularizer: Regularizer = null,
    wi: WeightInitializer = new NormWeightInitializer()) {
      super(inputLayer);
      if (!_([Layer.Input, 
              Layer.Dense,
              Layer.Dropout
            ]).includes(inputLayer.type)) { 
        throw new Error(`${Layer[inputLayer.type]} not supported`);
      }
      if ((inputLayer as any).cost) {
        throw new Error(`Can not append after output layer`);
      }

      this.bias = wi.weights([size, 1]);
      this.weight = wi.weights([size, (this.inputLayer as any).size]);  
  }

  /**
   * 
   * @param input the return of inputLayer.forward() 
   * @returns output activation of this layer
   * weightedIputs = weight * input + bias
   * activation = σ(weightedIputs)
   * 
   */
  forward(input: any): DensesForwardType {
    //if (this.inputLayer.type == Layer.Dense) input = input[1];
    this.inputActivation = input as Matrix;
    this.weightedOutputs = this.weight.matmul(input).addeqCol(this.bias);  // w * input + b
    this.outputActivation = this.activation.activate(this.weightedOutputs); // σ(w * input + b)
    return this.outputActivation;
  }
  private weightedOutputs: Matrix;   // trasient values saved in forward pass, to be used in backprop
  private inputActivation: Matrix;
  private outputActivation: Matrix;

  private nabla_b: Matrix;         // transient values saved during backprop, to be updated
  private nabla_w: Matrix;

  /** 
   * input: for output layer, this is the label
   *        for hidden layers, the return value of backprop of next layer
   * stop:  whether backprop stops at this layer, aka the hidden layer immediately after input layer
   *        if stop is true, then a matMul can be skipped, which saves LOTS of time.
   */
  backprop(input: DensesBackPropType, stop: boolean): DensesBackPropType {
    let nabla_b: Matrix;
    let nabla_w: Matrix;

    if (this.cost) { // last layer, input is label
      nabla_b = this.activation.cost(input, this.weightedOutputs, this.outputActivation, this.cost);  
      nabla_w = nabla_b.matmul(this.inputActivation.T);
    } else {    // hiddenlayers
      let sp = this.activation.derivative(this.weightedOutputs);
      nabla_b = input.muleqn(sp);
      nabla_w = nabla_b.matmul(this.inputActivation.T);
    }
    if (this.nabla_b == null) {
      this.nabla_b = nabla_b;
      this.nabla_w = nabla_w;
    } else {
      this.nabla_b.addeqn(nabla_b);
      this.nabla_w.addeqn(nabla_w);
    }
    this.weightedOutputs = this.outputActivation = this.inputActivation = null;
    if (!stop) {
      return this.weight.T.matmul(nabla_b);
    } else {
      return null;
    } 
  }

  // scale down nabla_w and nabla_b by learningRate before updating weights/biases
  update(learningRate: number): void {
    if (this.regularizer) {
      this.weight = this.regularizer.regularize(this.weight);
    }
    this.weight.subeqn(this.nabla_w.muleq(learningRate));
    if (this.nabla_b.shape[1] > 1) {
      // bias is normally a column vector, but for batch data, the column size == batch size, thus sum into 1 column
      this.bias.subeqn(this.nabla_b.sumCols().muleq(learningRate));
    } else {
      this.bias.subeqn(this.nabla_b.muleq(learningRate));
    }
    this.nabla_w = this.nabla_b = null;
  }
}