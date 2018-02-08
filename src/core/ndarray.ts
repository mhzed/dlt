import { Rand } from "./rand";

// Straigh-forward impl of n-dimensional array, with blas optimization
type ExternData = number[] | Int8Array | Int16Array | Int32Array |
    Uint8Array | Uint16Array | Uint32Array |
    Float32Array | Float64Array | Uint8ClampedArray;

export type NdArrayDataType = Float32Array; // internal data type

export class NdArray {
  public static Type = Float32Array;

  get data(): NdArrayDataType { return this._data }
  get length(): number  { return this._data.length }
  get shape(): number[] { return this._shape }  
  get dimension()       { return this._shape.length }

  // get element base on dimensional axis index, slow!! useful for test only
  get(...args: number[]): number { return this._data[this.offset(...args)]; }

  sameShape(p: NdArray): boolean {
    return (arrayEqual(this._shape, p._shape));
  }
  // test equality WRT transposed shapes
  equals(p: NdArray): boolean {
    if (!this.sameShape(p)) return false;
    else {
      let cl = this.walkIndex();
      let cr = p.walkIndex();
      while (true) {
        let left = cl.next();
        let right = cr.next();
        if (left.done || right.done) break;
        else {
          if (this._data[left.value] !== p._data[right.value]) return false;
        }
      }
      return true;
    }
  }
  // swap axis for N dimensional array
  // for 1D shape, transpose() call is meaningless since there is only 1 axis
  // for >=2D shapes, transpose() return a new array with !!shared!!! data but different shape/sride
  // args should be the re-arranged axis:
  // i.e. for 3d array, transpose (2,0,1):  axis 2 -> 0, axis 0 -> 1, axis 1 -> 2
  transpose(args: number[]): NdArray {
    if (args.length != this.dimension) throw new Error(`shape dictates ${this.dimension} parameters`);
    let newshape = args.map((n)=>this._shape[n])
    let newstride = args.map((n)=>this._stride[n])    
    let ret = new NdArray(this._data, newshape, newstride);
    return ret;
  }
  // returns if this is a transposed array:  the strides are not in descending order, meaning the data is 
  // not ordered the same way as default
  isTransposed(): boolean {
    // should always be in descending order from index 0 if not transposed, see strideOf
    for (let i=1; i<this._stride.length ;i++) {
      if (this._stride[i] > this._stride[i-1]) return true;
    }
    return false;
  }
  // return a new NdArray with shape set to newShape but with !!shared!! data as this
  toshape(newShape: number[]): NdArray {
    return NdArray.fromData(this.data, newShape);
  }
  // create a new array with shape set to new Shape, if shape is omitted, this.shape is used
  // data is always copied into new array
  // returned array is always isTransposed() => false
  // thus: array.reshape()  => return a copy of array that removes transposed property (if any)
  reshape(newShape?: number[]): NdArray {
    if (!newShape) newShape = this.shape;
    const size = newShape.reduce((a,n)=>a*n, 1);
    let ret = new NdArray(new NdArray.Type(size), newShape, strideOf(newShape));
    let i = 0;
    for (let offset of this.walkIndex()) {
      ret._data[i++] = this._data[offset];
      if (i>=size) break;
    }
    return ret;
  }
  // return an exact copy of this
  dup(): NdArray {
    return new NdArray(this._data.slice(0), this._shape.slice(0), this._stride.slice(0));
  }

  /*** Internal algebraic */
  sum(): number {
    return this._data.reduce((a, n) => a + n, 0);
  }
  // find and return max value
  max(): number {
    return this._data[this.argmax()];
  }
  argmax(): number {
    let max = Number.MIN_VALUE;
    let reti = 0;
    for (let i = 0; i<this._data.length; i++) {
      if (this._data[i] > max ) {
        max = this._data[i];
        reti = i;
      }
    }
    return reti;
  }
  magnitude(): number {
    return Math.sqrt(this._data.reduce((a, n) => a + n * n, 0));
  }
  
  /**
   * element wise add p
   * return result
   */
  add(p: number | NdArray): NdArray {
    let ret = this.dup();
    if (typeof p == 'number') {
      return ret.addeq(p as number);
    } else {
      this._shapeCheck(p as NdArray);
      return ret.addeqn(p as NdArray);
    }
  }
  addeq(p: number): NdArray {
    this._data.forEach((n,i)=>this._data[i]+=p);
    return this;
  }
  addeqn(p: NdArray): NdArray {
    this._data.forEach((n,i)=>this._data[i]+=p._data[i]);
    return this;
  }
  /**
   * element wise subtract p
   * return this
   */
  sub(p: number | NdArray): NdArray {
    let ret = this.dup();    
    if (typeof p == 'number') {
      return ret.subeq(p as number);
    } else {
      this._shapeCheck(p as NdArray);
      return ret.subeqn(p as NdArray);
    }
  }
  subeq(p: number): NdArray {
    return this.addeq(-p);
  }
  subeqn(p: NdArray): NdArray {
    this._data.forEach((n,i)=>this._data[i]-=p._data[i]);
    return this;
  }
  /**
   * element wise multiply p
   * return this
   */
  mul(p: number | NdArray): NdArray {
    let ret = this.dup();
    if (typeof p == 'number') {
      return ret.muleq(p);
    } else {
      this._shapeCheck(p as NdArray);
      return ret.muleqn(p as NdArray);
    }
  }
  muleq(p: number): NdArray {
    this._data.forEach((n,i)=>this._data[i] = n * p);
    return this;
  }
  // hadamard
  muleqn(p: NdArray): NdArray {
    this._data.forEach((n,i)=>this._data[i] = n * p._data[i]);
    return this;
  }
  
  /**
   * inverse this: x -> 1/x
   */
  inv(): NdArray {
    return this.dup().inveq();
  }
  inveq(): NdArray {
    this._data.forEach((n,i)=>this._data[i] = 1/n);
    return this;
  }
  /**
   * exp this: x -> e^x
   */
  exp(): NdArray {
    return this.dup().expeq();
  }
  expeq(): NdArray {
    this._data.forEach((n,i)=>this._data[i] = Math.exp(n));
    return this;    
  }
  /**
   * sigmoid element wise: x -> 1/(1+e^-x)
   */
  sigmoid(): NdArray {
    return this.dup().sigmoideq();
  }
  sigmoideq(): NdArray {
    this.data.forEach((n,i) => this.data[i] = 1/(1+Math.exp(-n)) );
    return this;    
  }
  /**
   * relu element wise: x -> max(0,x)
   */
  relu(): NdArray {
    return this.dup().relueq();
  }
  relueq(): NdArray {
    this.data.forEach((n,i)=>this.data[i] = n>0?n:0);
    return this;    
  }
  /**
   * apply softmax element wise: x => e^x / sum(e^X) (X=0...n)
   */
  softmax(): NdArray {
    return this.dup().softmaxeq();
  }
  softmaxeq(): NdArray {
    let denom = this.data.reduce((a,n)=>a+Math.exp(n), 0);
    this.data.forEach((n,i)=>this.data[i] = Math.exp(n)/denom);
    return this;    
  }
  hasNaN(): boolean {
    for (let i=0; i<this._data.length; i++) if (isNaN(this._data[i])) return true;
    return false;
  }
  
  /**
   * create an array with data and shape
   * stride by default right most dimension = least significant
   */
  static from(p: ExternData, shape: number[], stride?: number[] ): NdArray {
    let data = new NdArray.Type(p.length);
    for (let i=0; i<data.length; i++) data[i] = p[i]
    return new NdArray(data, shape, stride || strideOf(shape))
  }
  /**
   * short hand for from(p, [p.length, 1]); returns a single column matrix
   * @param p a col vector
   */
  static fromCol(p: ExternData): NdArray {
    return NdArray.from(p, [p.length, 1])
  }
  /**
   * short hand for from(p, [1,p.length]); returns a single row matrix
   * @param p a row vector
   */
  static fromRow(p: ExternData): NdArray {
    return NdArray.from(p, [1,p.length])
  }
  /**
   * construct Matri directly from data: no copying
   * @param data use this data
   * @param shape 
   */
  static fromData(data: NdArrayDataType, shape: number[]): NdArray {
    return new NdArray(data, shape, strideOf(shape));
  }  
  /**
   * create an array of zeroes with provided shape
   */
  static zeros(shape: number[]): NdArray {
    const size = shape.reduce((a,n)=>a*n, 1);
    let data = new NdArray.Type(size); // by default zero-ed, assuming TypedArray
    return new NdArray(data, shape, strideOf(shape))
  }

  /**
   * create an array of random nubmers between [-1,1) with provide shape
   */
  static randn(shape: number[], deviation=1, mean=0): NdArray {
    const size = shape.reduce((a,n)=>a*n, 1);
    let data = new NdArray.Type(size);
    data.forEach((n,i)=>data[i] = (Rand.rand()*2-1 + mean)*deviation);
    return new NdArray(data, shape, strideOf(shape))
  }

  /********************** private section ******************************* */
  // walkIndex WRT to this._strides, always increment the least significant (right most) dimension first
  protected *walkIndex() {
    // think of dindex as [d0, d1, d2 ...],  each d can be incremened up to size defined in this._shapes
    let dindex = new Array<number>(this._shape.length);
    for (let i=0; i<dindex.length; i++) dindex[i] = 0;
    while (true) {
      let offset = 0;
      for (let i=0; i<dindex.length; i++) offset += dindex[i]*this._stride[i];
      yield offset;

      // incremnent dindex WRT to shapes
      for (let p = dindex.length -1; p>=0; p--) {
        let n = dindex[p] + 1;
        if (n<this._shape[p]) {
          dindex[p] = n;
          break;
        } else {
          if (p==0) return;    // no more sig digit, all iterated
          else dindex[p] = 0;  // continue for for more significant digit
        }
      }
    }      
  }
  protected _shapeCheck(p: NdArray): void {
    if (!this.sameShape(p)) throw new Error(`shape mismatch: ${this._shape} <= ${p._shape}`);
    if (!arrayEqual(this._stride, p._stride)) { 
      throw new Error(`stride mismatch, call reshape on transposed NdArray`);
    }
  }
  protected constructor(protected _data: NdArrayDataType, protected _shape: number[], protected _stride: number[]) {
  }
  // calcualte index offset in data based on dimension axis index, slow!
  protected offset(...args): number {
    return this._stride.reduce((a,v,i)=>a+v*args[i], 0);
  }
}

/*****************************
 * Local private helpers
 * ********************** */
function arrayEqual(a1, a2): boolean {
  if (a1.length !== a2.length) return false;
  else {
    for (let i=0; i<a1.length; i++) {
      if (a1[i] != a2[i]) return false;
    }
    return true;
  }
}
function strideOf(shape: number[]): number[] {
  let stride = []
  for (let i=0; i<shape.length-1; i++) {
    stride.push(shape.slice(i+1).reduce((a,n)=>a*n,1))
  }
  stride.push(1);
  return stride;
}
