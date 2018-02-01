import { Layer } from "../types";
import { Shape, Matrix } from "../core/matrix";
import { BaseLayer } from "./baselayer";

export class InputLayer extends BaseLayer {
  readonly type = Layer.Input;
  public size: number;
  constructor(public shape: Shape) {
    super(null);
    this.size = shape[0] * shape[1];
    this.inputLayer = null;
    this.nextLayer = null;
  }
  forward(input: Matrix): Matrix {
    //if (input.shape[0] != this.size) throw new Error(`Incorrect size ${input.length}, expect ${this.size}`);    
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
