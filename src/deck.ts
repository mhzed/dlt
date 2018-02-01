import { Matrix as Mtrx } from "vectorious";
import { Rand } from "./core/rand";
import * as nblas from "nblas";
// supported external type for 'from'
type ExternData = number[] | Int8Array | Int16Array | Int32Array |
    Uint8Array | Uint16Array | Uint32Array |
    Float32Array | InternalDataType | Uint8ClampedArray;

// Float32Array is considerably faster than Float64Array, use Float64Array only if you absoluely need the precision
type InternalDataType = Float32Array; // internal data type
let InternalT = Float32Array;

export type Shape = [number, number]

/**
 * 
 * - All algebraic operations have two or three versions
 *   - op(param):   param check, shape compatability check, perform op, reutrn result in a new array
 *   - opeq(num):   perform op with a scalar num in place
 *   - opeqn(array):perform op with another DlMatrix in place
 * 
 * Generally speaking, the in place operations are faster (low single digit percentage), 
 * but are trickier to use (easy to have unintended
 * side effects), so use with care and only during the final stage of performane tuning.
 * 
 * - constructor is protected, use DlMatrix.from to create DlMatrix
 */

export class Matrix {
  protected get data(): InternalDataType { return (this.matrix as any).data as InternalDataType }
  get length(): number  { return this.matrix.shape[0] * this.matrix.shape[1] }
  get shape(): Shape { return this.matrix.shape as Shape }  
  // T returns a brand new matrix with re-arranged data:  this may seem costly but is necessary to utilize BLAS, so
  // the cost is neglagible overall.
  get T(): Matrix      { 
    if (this.shape[0] == 1 || this.shape[1] == 1)  {
      return new Matrix(Mtrx.fromTypedArray(this.data.slice(0), [this.shape[1], this.shape[0]]))
    } else  {
      return new Matrix(this.matrix.transpose())
    }
  }
  get(row: number, col: number): number { return this.matrix.get(row, col); }

  sameShape(p: Matrix): boolean {
    return (this.shape[0] == p.shape[0] && this.shape[1] == p.shape[1]);
  }
  equals(p: Matrix): boolean {
    return this.matrix.equals(p.matrix);
  }
  // find and return max value
  max(): number {
    return this.data[this.argmax()];
  }
  // find and return the index (in data) of max value
  argmax(): number {
    return this._argmax();
  }
  // naive impl without blas
  private _argmax(): number {
    let max = Number.MIN_VALUE;
    let reti = 0;
    for (let i = 0; i<this.data.length; i++) {
      if (this.data[i] > max ) {
        max = this.data[i];
        reti = i;
      }
    }
    return reti;
  }
  // return an exact copy of this
  dup(): Matrix {
    return new Matrix(Mtrx.fromTypedArray(this.data.slice(0), this.shape));
  }

  /**
   * element wise add p
   * return result
   */
  add(p: number | Matrix): Matrix {
    let ret = this.dup();
    if (typeof p == 'number') {
      return ret.addeq(p as number);
    } else {
      this._shapeCheck(p as Matrix);
      return ret.addeqn(p as Matrix);
    }
  }
  addeq(p: number): Matrix {
    this.data.forEach((n,i)=>this.data[i]+=p);
    return this;
  }
  addeqn(p: Matrix): Matrix {
    this.matrix.add(p.matrix);
    return this;
  }
  // add p to this column by column, p must be a column vector of same number of rows
  addeqCol(p: Matrix): Matrix {
    if (this.shape[0] != p.shape[0] || p.shape[1] !=1) {
      throw new Error(`Expected shape ${this.shape[1]},1`)
    }
    let colsize = this.shape[1];
    for (let r = 0; r<this.shape[0]; r++) {
      let li = r * colsize;
      let val = p.data[r]
      for (let c = 0; c<this.shape[1]; c++) {
        this.data[li++] += val;
      }
    }
    return this;
  }  
  /**
   * element wise subtract p
   * return this
   */
  sub(p: number | Matrix): Matrix {
    let ret = this.dup();    
    if (typeof p == 'number') {
      return ret.subeq(p as number);
    } else {
      this._shapeCheck(p as Matrix);
      return ret.subeqn(p as Matrix);
    }
  }
  subeq(p: number): Matrix {
    return this.addeq(-p);
  }
  subeqn(p: Matrix): Matrix {
    this.matrix.subtract(p.matrix);
    return this;
  }
  /**
   * element wise multiply p
   * return this
   */
  mul(p: number | Matrix): Matrix {
    let ret = this.dup();
    if (typeof p == 'number') {
      return ret.muleq(p);
    } else {
      this._shapeCheck(p as Matrix);
      return ret.muleqn(p as Matrix);
    }
  }
  muleq(p: number): Matrix {
    this.matrix.scale(p);
    return this;
  }
  muleqn(p: Matrix): Matrix {
    this.matrix.product(p.matrix);
    return this;
  }
  
  /**
   * ^-1 element wise: x -> 1/x
   */
  inv(): Matrix {
    return this.dup().inveq();
  }
  inveq(): Matrix {
    this.data.forEach((n,i)=>this.data[i] = 1/n);
    return this;
  }
  /**
   * exp element wise: x -> e^x
   */
  exp(): Matrix {
    return this.dup().expeq();
  }
  expeq(): Matrix {
    this.data.forEach((n,i)=>this.data[i] = Math.exp(n));
    return this;    
  }
  /**
   * ln element wise: x -> ln(x)
   */
  ln(): Matrix {
    return this.dup().lneq();
  }
  lneq(): Matrix {
    this.data.forEach((n,i)=>this.data[i] = Math.log(n));
    return this;    
  }
  /**
   * sigmoid element wise: x -> 1/(1+e^-x)
   */
  sigmoid(): Matrix {
    return this.dup().sigmoideq();
  }
  sigmoideq(): Matrix {
    this.data.forEach((n,i) => this.data[i] = 1/(1+Math.exp(-n)) );
    return this;    
  }
  /**
   * tanh element wise: x -> tanh(x), e^z - e^-z / e^z + e^-z
   */
  tanh(): Matrix {
    return this.dup().tanheq();
  }
  tanheq(): Matrix {
    this.data.forEach((n,i)=>this.data[i] = Math.tanh(n));
    return this;    
  }
  /**
   * relu element wise: x -> max(0,x)
   */
  relu(): Matrix {
    return this.dup().relueq();
  }
  relueq(): Matrix {
    this.data.forEach((n,i)=>this.data[i] = n>0?n:0);
    return this;    
  }
  /**
   * apply softmax element wise: x => e^x / sum(e^X) (X=0...n)
   */
  softmax(): Matrix {
    return this.dup().softmaxeq();
  }
  softmaxeq(): Matrix {
    let denom = this.data.reduce((a,n)=>a+Math.exp(n), 0);
    this.data.forEach((n,i)=>this.data[i] = Math.exp(n)/denom);
    return this;    
  }
  

  /**
   * matrix multiplicaiton of this * p
   * ! this or p are not mutated:  a new array is returned
   */
  matmul(p: Matrix): Matrix {
    return new Matrix(this.matrix.multiply(p.matrix));
  }

  sum(): number {
    return this._sum();
  }
  _sum(): number {
    return this.data.reduce((a, n) => a+ n, 0)
  }
  // sum this column by column, return a col vector with same number of rows as this
  sumCols(): Matrix {
    let ret = Matrix.zeros([this.shape[0],1]);
    let colsize = this.shape[1];
    for (let r = 0; r<this.shape[0]; r++) {
      let sum = 0;
      let li = colsize*r
      for (let c = 0; c<this.shape[1]; c++) {
        sum += this.data[li++];
      }
      ret.data[r] = sum;
    }
    return ret;
  }

  /**
   * create an array with data and shape
   */
  static from(p: ExternData, shape: Shape): Matrix {
    let data = new InternalT(p.length);
    for (let i=0; i<data.length; i++) data[i] = p[i]
    return new Matrix(Mtrx.fromTypedArray(data, shape));
  }
  /**
   * short hand for from(p, [p.length, 1]); returns a column
   * @param p a col vector
   */
  static fromCol(p: ExternData): Matrix {
    return Matrix.from(p, [p.length, 1])
  }
  /**
   * short hand for from(p, [1,p.length]); returns a row
   * @param p a row vector
   */
  static fromRow(p: ExternData): Matrix {
    return Matrix.from(p, [1,p.length])
  }
  /**
   * construct Matri directly from data: no copying
   * @param data use this data
   * @param shape 
   */
  static fromData(data: InternalDataType, shape: Shape): Matrix {
    return new Matrix(Mtrx.fromTypedArray(data, shape));
  }
  /**
   * stack rows vertically, colSize must match
   * @param rows 
   */
  static stackRows(rows: Matrix[]): Matrix {
    let colsize = rows[0].shape[1];
    let rowsize = rows.reduce((a,m)=>a+m.shape[0], 0);
    let ret = new Matrix(Mtrx.fromTypedArray(new InternalT(rowsize * colsize), [rowsize, colsize]));
    let offset = 0;
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
  static stackCols(cols: Matrix[]): Matrix {
    // for row major storage, it's more efficient to do transpose, instdead of 'picking' data points to copy from cols
    let rows= cols.map((c)=>
      // if a col vector, simply reuse data
      c.shape[1] == 1? new Matrix(Mtrx.fromTypedArray(c.data, [1,c.data.length])) : c.T
    );
    return Matrix.stackRows(rows).T;
  }  
  /**
   * create an array of zeroes with provide shape
   */
  static zeros(shape: Shape): Matrix {
    const size = shape.reduce((a,n)=>a*n, 1);
    let data = new InternalT(size);
    for (let i=0;i<size; i++) data[i] = 0;
    return new Matrix(Mtrx.fromTypedArray(data, shape));
  }

  /**
   * create an identity matrix of [size, size]
   */
  static identiy(size: number): Matrix {
    let ret = Matrix.zeros([size,size])
    let cs = ret.shape[1];
    for (let i =0; i<size; i++) ret.data[i*cs+i] = 1;
    return ret;
  }
  /**
   * Create a one hot column vector
   * @param n  the index of element to set to 1
   * @param size total size of vector
   */
  static oneHotCol(n: number, size: number): Matrix {
    if (n>=size) throw new Error(`out of bound`);
    let ret = Matrix.zeros([size,1])
    ret.data[n] = 1;
    return ret;
  }

  /**
   * create an array of random nubmers between [-1,1) with provide shape
   * optional params: deviation scales min/max, default = 1, mean moves middle, default 0.
   */
  static randn(shape: Shape, deviation=1, mean=0): Matrix {
    const size = shape.reduce((a,n)=>a*n, 1);
    let data = new InternalT(size);
    data.forEach((n,i)=>data[i] = (Rand.rand()*2-1 + mean)*deviation);
    return new Matrix(Mtrx.fromTypedArray(data, shape));
  }

  /**
   * 
   * @param m1 
   * @param off1 
   * @param m2 
   * @param off2 
   * @param size 
   */
  static dotm(m1: Matrix, off1: number, m2: Matrix, off2: number, size: number): number {
    const lhs = m1.data.subarray(off1, off1 + size);
    const rhs = m2.data.subarray(off2, off2 + size);
    return Matrix.dot(lhs, rhs);
  }
  static dot(l: InternalDataType, r: InternalDataType): number {
    return Matrix._dot(l, r);
  }
  static _dot(l: InternalDataType, r: InternalDataType): number {
    let sum = 0;
    for (let i = 0; i< l.length; i++) sum += l[i]*r[i];
    return sum;
  }
  public toString(colSize = 5): string {
    let res = '';
    for (let r = 0; r<this.shape[0]; r++) {      
      for (let c = 0; c<this.shape[1]; c++) {
        res += padL(this.get(r,c), colSize);
      }
      res += "\n";
    }
    return res;
  }
  hasNaN(): boolean { // for debugging mainly
    return this.data.reduce((a,n)=>isNaN(n)||a, false);
  }

  /********************** private section ******************************* */
  protected _shapeCheck(p: Matrix): void {
    if (!this.sameShape(p)) throw new Error(`shape mismatch: ${this.shape} <= ${p.shape}`);
  }
  protected constructor(protected matrix: Mtrx) {
  }
}

function padL(a: number, length: number, pad:string=' '): string {
  let digit = a.toString()||''
  if (digit.length > length-1) {  // no room for pads
    digit = digit.slice(0, length-2) + '?';
  }
  return (new Array((length)+1).join(pad)+digit).slice(-(length))
}

try {
  const _nblas = require("nblas");
  Matrix.prototype.sum = function() {
    return _nblas.asum(this.data);
  }
  Matrix.prototype.argmax = function() {
    return _nblas.iamax(this.data);
  } 
  Matrix.dot = function(l, r) {
    return _nblas.dot(l, r);
  }
} catch (e) {
}
