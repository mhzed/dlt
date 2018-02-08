import { NdArray, NdArrayDataType } from "./ndarray";

export namespace ndm {
  // using class instead of namespace because of applying blas.
  export class matrix {

    /**
     * create an identity matrix
     * @param size 
     */
    static identiy(size: number): NdArray {
      let ret = NdArray.zeros([size,size]),
          cs = ret.shape[1];
      for (let i =0; i<size; i++) ret.data[i*cs+i] = 1;
      return ret;
    }
    /**
     * Create a one hot column vector-matrix (1 column matrix)
     * @param n  the index of element to set to 1
     * @param size total size of vector
     */
    static oneHotCol(n: number, size: number): NdArray {
      if (n>=size) throw new Error(`out of bound`);
      let ret = NdArray.zeros([size,1])
      ret.data[n] = 1;
      return ret;
    }

    /**
     * Print a matrix
     * @param n 
     * @param colSize 
     */
    static toString(n: NdArray, colSize = 5): string {
      dimensionCheck(n);      
      let res = '';
      for (let r = 0; r<n.shape[0]; r++) {
        for (let c = 0; c<n.shape[1]; c++) {
          res += padL(n.get(r,c), colSize);
        }
        res += "\n";
      }
      return res;
    }

    /**
     * matrix multiplicaiton of l * r
     */
    static matmul(l: NdArray, r: NdArray): NdArray {
      let r1 = l.shape[0],   // rows in this matrix
          c1 = l.shape[1],   // columns in this matrix
          r2 = r.shape[0], // rows in multiplicand
          c2 = r.shape[1], // columns in multiplicand
          d1 = l.data,
          d2 = r.data;

      if (c1 !== r2) throw new Error('sizes do not match');

      let data = new NdArray.Type(r1 * c2),
          i, j, k,
          sum;
      for (i = 0; i < r1; i++) {
        for (j = 0; j < c2; j++) {
          sum = +0;
          for (k = 0; k < c1; k++) sum += d1[i * c1 + k] * d2[j + k * c2];
          data[i * c2 + j] = sum;
        }
      }
      return NdArray.fromData(data, [r1, c2]);
    }

    /**
     * Transpose the parameter so that it can be safely matmul-ed
     * Data may or may not be copied.
     * 
     * @param rhs two dimensional array to be transposed
     */
    static T(rhs: NdArray): NdArray { 
      dimensionCheck(rhs);
      if (rhs.shape[0] == 1 || rhs.shape[1] == 1)  {
        return NdArray.fromData(rhs.data, [rhs.shape[1], rhs.shape[0]]);
      } else  {
        let r = rhs.shape[0],
            c = rhs.shape[1],
            i, j;
        let data = new NdArray.Type(c * r);
        for (i = 0; i < r; i++) {
          for (j = 0; j < c; j++) {
            data[j * r + i] = rhs.data[i * c + j];
          }
        }
        return NdArray.fromData(data, [c, r]);
      }
    }

    // sum this column by column, return a col vector with same number of rows as this
    static sumCols(rhs: NdArray): NdArray {
      dimensionCheck(rhs);
      let ret = NdArray.zeros([rhs.shape[0],1]);
      let colsize = rhs.shape[1];
      for (let r = 0; r<rhs.shape[0]; r++) {
        let sum = 0;
        let li = colsize*r
        for (let c = 0; c<rhs.shape[1]; c++) {
          sum += rhs.data[li++];
        }
        ret.data[r] = sum;
      }
      return ret;
    }

    // add 'col' to lhs column by column, 'col' must be a column vector of same number of rows
    // returns lhs
    static addeqCol(lhs: NdArray, col: NdArray): NdArray {
      if (lhs.shape[0] != col.shape[0] || col.shape[1] !=1) {
        throw new Error(`Expected shape ${lhs.shape[1]},1`)
      }
      let colsize = lhs.shape[1];
      for (let r = 0; r<lhs.shape[0]; r++) {
        let li = r * colsize,
            val = col.data[r];
        for (let c = 0; c<lhs.shape[1]; c++) {
          lhs.data[li++] += val;
        }
      }
      return lhs;
    }  
    
      
    /**
     * stack rows vertically, colSize must match
     * @param rows 
     */
    static stackRows(rows: NdArray[]): NdArray {
      let colsize = rows[0].shape[1],
          rowsize = rows.reduce((a,m)=>a+m.shape[0], 0),
          ret = NdArray.from(new NdArray.Type(rowsize * colsize), [rowsize, colsize]),
          offset = 0;
      for ( let row of rows) {  
        // by default storage is row based, so we copy data directly
        if (row.shape[1] != colsize) throw new Error(`Col size ${row.shape[1]} (expect ${colsize}) can't be stacked`);
        ret.data.set(row.data, offset);
        offset += row.length;
      }
      return ret;
    }
    /**
     * stack cols horizontally, rowSize must match
     * @param cols
     */
    static stackCols(cols: NdArray[]): NdArray {
      // for row major storage, it's more efficient to do transpose, instdead of 'picking' data points to copy from cols
      let rows = cols.map((c)=>
        // if a col vector, simply reuse data
        c.shape[1] == 1 ? NdArray.fromData(c.data, [1,c.data.length]) : matrix.T(c)
      );
      return matrix.T(matrix.stackRows(rows));
    }     
      
    static dotat(m1: NdArrayDataType, off1: number, m2: NdArrayDataType, off2: number, size: number): number {
      const lhs = m1.subarray(off1, off1 + size);
      const rhs = m2.subarray(off2, off2 + size);
      return matrix.dot(lhs, rhs);
    }
    static dot(l: NdArrayDataType, r: NdArrayDataType): number {
      let sum = 0;
      for (let i = 0; i< l.length; i++) sum += l[i]*r[i];
      return sum;
    }
  }
}

function dimensionCheck(rhs: NdArray) {
  if (rhs.dimension != 2) throw new Error('2 dimensional matrix only');
}

function padL(a: number, length: number, pad:string=' '): string {
  let digit = a.toString()||''
  if (digit.length > length-1) {  // no room for pads
    digit = digit.slice(0, length-2) + '?';
  }
  return (new Array((length)+1).join(pad)+digit).slice(-(length))
};
