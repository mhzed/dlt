import { NdArray } from "./ndarray";

export class Volume {

  static sliceWindow(vol: NdArray, row: number, col: number, width: number, height: number): NdArray {
    if (vol.dimension < 2 || vol.dimension > 3) throw new Error("supprt 2/3d array only");
    if (vol.isTransposed()) vol = vol.rearrange();

    let depth = vol.dimension == 2 ? 1 : vol.shape[2];
    let ret = NdArray.zeros([height, width, depth]);

    let ret_c = 0, ret_r = 0,                     // keep track of r/c in ret
        copy_width = width, copy_height = height  // adjusted for padding
        ;
    if (col < 0) { 
      copy_width -= (-col);
      ret_c = -col;
      col = 0;
    }
    if (row < 0) {
      copy_height -= (-row);
      ret_r = -row;
      row = 0;
    }
    // copy row by row (since row major storage)
    for (let vol_r = row; vol_r < row + copy_height; vol_r++) {
      let off = (vol_r) * vol.stride[0] + col * vol.stride[1];  // already adjusted for depth
      // ensure size stops at the end of the row
      let w = (col + copy_width) > vol.shape[1] ? ((col + copy_width) - vol.shape[1]) :  copy_width;
      let rowbuff = vol.data.subarray(off, off + w * depth);
      ret.data.set(rowbuff, ((ret_r++) * width + ret_c) * depth);
    }
    return ret;
  }
  
}