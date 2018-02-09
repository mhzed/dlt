import { Layer } from "../types";
import { NdArray } from "../core/ndarray";
import { BaseLayer } from "./baselayer";

export class InputLayer extends BaseLayer {
  readonly type = Layer.Input;
  public size: number;
  constructor(public shape: number[]) {
    super(null);
    this.size = shape.reduce((a, n) => a * n, 1);
    this.inputLayer = null;
    this.nextLayer = null;
  }
  validate(input: NdArray) {
    if (!input.sameShape(this.shape)) throw new Error("shape mismatch");
  }
  forward(input: NdArray): NdArray {
    return input;
  }
  backprop(input: any, stop: boolean): any {
    throw new Error("Can't backprop on input, check your code");
  }
  
  *step(offset:number): IterableIterator<BaseLayer> {
    let layer: BaseLayer = this;
    let i = 0;
    while (layer != null) {
      if (i++>=offset) yield layer;
      layer = layer.nextLayer;
    }
  }
}
