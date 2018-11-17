import {  LabelData, EpochStopper, Layer } from "./types";

import { NdArray, Matrix, Rand } from "./core";
import { BaseLayer } from "./layers/baselayer";
import { InputLayer } from "./layers/inputlayer";

// import { Matrix } from './Matrix';
// import { LayerClass, LabelData, WeightInitializer, Layer,
//   ActivationFunction, InputLayer, FullyConnectedLayer } from './DlTypes';

function batchUpdate(net: InputLayer, batch: LabelData[], eta: number) : void {
  let stackedIputs: NdArray, stackedLabels: NdArray;
  stackedIputs = Matrix.stackCols(batch.map((b)=>b.input));
  stackedLabels = Matrix.stackCols(batch.map((b)=>b.label))

  let activation: any = stackedIputs;
  let layer: BaseLayer;
  for (layer of net.step(0)) {
    activation = layer.forward(activation);
  }
  let backpropstate = stackedLabels;
  for (; layer.type != Layer.Input; layer=layer.inputLayer) {
    let lastlayer = layer.inputLayer.type == Layer.Input
      || (layer.inputLayer.type == Layer.Dropout && layer.inputLayer.inputLayer.type == Layer.Input);
    backpropstate = (layer).backprop(backpropstate, lastlayer);
  }
  for (layer of net.step(1)) {
    if ((layer as any).update) (layer as any).update(eta/batch.length);
  }
}

function singleUpdate(net: InputLayer, batch: LabelData[], eta: number) : void {

  for (let b of batch) {
    let activation: any = b.input;
    let layer: BaseLayer;
    for (layer of net.step(0)) {
      activation = layer.forward(activation);
    }
    let backpropstate = b.label;
    for (; layer.type != Layer.Input; layer=layer.inputLayer) {
      let lastlayer = layer.inputLayer.type == Layer.Input
        || (layer.inputLayer.type == Layer.Dropout && layer.inputLayer.inputLayer.type == Layer.Input);
      backpropstate = (layer).backprop(backpropstate, lastlayer);
    }
  }
  for (let layer of net.step(1)) {
    if ((layer as any).update) (layer as any).update(eta/batch.length);
  }
}

export class Train {

  // perform a forward pass for evaluation purpose
  private static forwardEvaluate(net: InputLayer, input: NdArray): NdArray {
    let activation: any = input;
    for (let layer of net.step(0)) {
      if (layer.type != Layer.Dropout) {
        activation = layer.forward(activation);
      }
    }
    return activation;
  }
  static evaluate(net: InputLayer, inputs: LabelData[]): number {
    let c = 0;
    for (let input of inputs) {
      if (Train.forwardEvaluate(net, input.input).argmax() == input.label.argmax()) c++;
    }
    return c/inputs.length;
  }

  static sgd (net: InputLayer, train: LabelData[], 
    mini_batch_size: number, eta: number, 
    step: EpochStopper): number {

    let updateFunc ;
    if (Train.isAllDense(net)) {
      updateFunc = batchUpdate;
    } else {
      updateFunc = singleUpdate;
    }
    updateFunc = singleUpdate;

    let nepochs = 0;
    while (true) {
      const beg = Date.now();
      train = Rand.shuffle(train);  
      for (let i=0; i<train.length; i+=mini_batch_size) {
        updateFunc(net, train.slice(i, i+mini_batch_size), eta)
      } 
      const elaspsed = ((Date.now()-beg)/1000);
      if (!step.onEpoch(nepochs++, elaspsed)) return nepochs;
    }
  }

  static isAllDense(net: InputLayer): boolean {
    for (const layer of net.step(1)) {
      if (layer.type !== Layer.Dense) return false;
    } 
    return true;
  }
}
