// import { Layer, ActivationFunction, NormWeightInitializer, WeightInitializer } from "../types";
// import {  NdArray } from "../core/ndarray";
// import * as _ from "lodash";
// import { Rand } from "../core/rand";
// import { BaseLayer } from "./baselayer";

// type Feature = {
//   weight: NdArray;
//   bias: number;
// }
// export class ConvolutionLayer extends BaseLayer {
//   readonly type = Layer.Convolution;

//   private features: Feature[] = []
//   /**
//    * 
//    * @param inputLayer 
//    * @param probability the probablity of drop out: 0.2 => 20% 
//    */
//   constructor(inputLayer:BaseLayer, 
//       private activation: ActivationFunction,
//       features: number,
//       private kernel: [number, number],
//       private padding: number = 0,
//       private stride: number = 1,
//       wi: WeightInitializer = new NormWeightInitializer()) {

//     super(inputLayer);
//     this.features = [];
//     for (let i=0; i<features; i++) {
//       this.features.push({
//         weight: wi.weights(kernel),
//         bias: Rand.rand()
//       });
//     }
//   }

//   forward(input) {
//     // let win = new Float32Array(this.kernel[0] * this.kernel[1]);
//     // let cs = input.shape[1];
//     // for (let r = 0; r < input.shape[0]-this.kernel[0]; r++ ) {
//     //   for (let c = 0; c < input.shape[1]-this.kernel[1]; c++) {

//     //     let inputoffset = r*cs+c;
//     //     let wincolsize = this.kernel[1], winoffset = 0;
//     //     for (let wr = 0; wr < this.kernel[0]; wr++ ) {
//     //       win.set((input as any).data.slice(inputoffset, inputoffset + wincolsize), winoffset);
//     //       winoffset += wincolsize;
//     //       inputoffset += cs;
//     //     } 

//     //     Matrix.fromData(win, []).matmul( features);
//     //   }
//     // }    
//   }
//   backprop(input: any, stop: boolean) {
//   }
// }
