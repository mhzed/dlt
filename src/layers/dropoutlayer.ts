import { Layer } from "../types";
import {  NdArray } from "../core/ndarray";
import * as _ from "lodash";
import { Rand } from "../core/rand";
import { BaseLayer } from "./baselayer";


export class DropoutLayer extends BaseLayer {
  readonly type = Layer.Dropout;

  size: number;
  /**
   * 
   * @param inputLayer 
   * @param probability the probablity of drop out: 0.2 => 20% 
   */
  constructor(inputLayer:BaseLayer, public probability: number) {
    super(inputLayer);
    if (!_([Layer.Input, Layer.Dense]).includes(inputLayer.type)) { 
      throw new Error(`Can not append dropout layer after ${Layer[inputLayer.type]}`);
    }
    this.size = (inputLayer as any).size;
    
  }

  private mask: NdArray;  
  forward(input: NdArray): NdArray {
    if (!this.mask) {
      this.mask = NdArray.zeros(input.shape);
      let sizeOnes = this.mask.length*(1-this.probability)
      for (let i=0; i<sizeOnes; i++) (this.mask as any).data[i] = 1;
      Rand.shuffle((this.mask as any).data);
    } else {
      Rand.shuffle((this.mask as any).data, this.mask.length/2);
    }
    //this.mask = Matrix.from(Rand.rbinom(input.length, 1, 1-this.probability), input.shape);
    return input.muleqn(this.mask).muleq(1/(1-this.probability));
  }
  backprop(input: any, stop: boolean) {
    if (!stop) {
      return input.muleqn(this.mask)
    } else {
      return null;
    }
  }
}
