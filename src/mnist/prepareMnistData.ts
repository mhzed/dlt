import { MnistStream, MnistStreamData } from "./MnistStream";
import { LabelData } from "../types";
import * as fs from 'fs';
import { asyncIterateStream } from "async-iterate-stream/asyncIterateStream";
import { NdArray } from "../core/ndarray";
import { ndm } from "../core/ndm.matrix"

export async function prepareMnistData(imageFile:string, labelFile:string, n?:number): Promise<LabelData[]> {
  const ds = new MnistStream(fs.createReadStream(imageFile), fs.createReadStream(labelFile));
  let dataset: LabelData[] = [];
  for await (const _item of asyncIterateStream(ds, true)) {
    let item = _item as MnistStreamData;
    dataset.push({
      input: NdArray.fromCol(item.image).mul(1/255),
      label: ndm.matrix.oneHotCol(item.label, 10)
    })
    if (n && dataset.length>=n) break;
  }
  return dataset;
}
