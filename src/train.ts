import {  LabelData, EpochStopper, Layer } from "./types";
import { NdArray } from "./core/ndarray";
import { ndm } from "./core/ndm.matrix";
import { Rand } from "./core/rand";
import { BaseLayer } from "./layers/baselayer";
import { InputLayer } from "./layers/inputlayer";
// import * as _ from 'lodash'
// import { Matrix } from './Matrix';
// import { LayerClass, LabelData, WeightInitializer, Layer,
//   ActivationFunction, InputLayer, FullyConnectedLayer } from './DlTypes';
import { applyblas } from '../src/core/applyblas';
import * as nblas from 'nblas';
applyblas(nblas);

function batchUpdate(net: InputLayer, batch: LabelData[], eta: number) : void {
  let stackedIputs: NdArray, stackedLabels: NdArray;
  stackedIputs = ndm.matrix.stackCols(batch.map((b)=>b.input));
  stackedLabels = ndm.matrix.stackCols(batch.map((b)=>b.label))

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

    let nepochs = 0;
    while (true) {
      const beg = Date.now();
      train = Rand.shuffle(train);  
      for (let i=0; i<train.length; i+=mini_batch_size) {
        batchUpdate(net, train.slice(i, i+mini_batch_size), eta)
        //singleUpdate(net, train.slice(i, i+mini_batch_size), eta)
      } 
      const elaspsed = ((Date.now()-beg)/1000);
      if (!step.onEpoch(nepochs++, elaspsed)) return nepochs;
    }
  }
}
