import { Layer } from "../types";

export abstract class BaseLayer {
  readonly type: Layer;

  // layer is a double linked list;
  inputLayer: BaseLayer = null;
  nextLayer: BaseLayer = null;
  constructor(inputLayer: BaseLayer) {
    this.inputLayer = inputLayer;
    if (!inputLayer) return;
    if (inputLayer.nextLayer != null) {
      throw new Error(`Input layer is already linked`);
    }
    inputLayer.nextLayer = this;
  }

  abstract forward(input: any): any;
  abstract backprop(input: any, stop: boolean): any;

}
