import { Rand } from "./rand";

// Straigh-forward impl of n-dimensional array
type ExternData = number[] | Int8Array | Int16Array | Int32Array |
    Uint8Array | Uint16Array | Uint32Array |
    Float32Array | Float64Array | Uint8ClampedArray;

type InternalDataType = Float32Array; // internal data type
let InternalT = Float32Array;

export class NdArray {
  get length(): number  { return this.data.length }
  get shape(): number[] { return this._shape }  
  get dimension()       { return this._shape.length }

  // get element base on dimensional axis index, slow!! useful for test only
  get(...args: number[]): number { return this.data[this.offset(...args)]; }

  sameShape(p: NdArray): boolean {
    return (arrayEqual(this._shape, p._shape));
  }
  sameShapeAs(s: number[]): boolean {
    return (arrayEqual(this._shape, s));
  }
  // test equality
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
          if (this.data[left.value] !== p.data[right.value]) return false;
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
    let ret = new NdArray(this.data, newshape, newstride);
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

  // create a new array with shape set to new Shape, if shape is omitted, this.shape is used
  // data is copied into new array
  // returned array is always isTransposed() => false
  // thus: array.reshape()  => return a copy of array that removes transposed property (if any)
  reshape(newShape?: number[]): NdArray {
    if (!newShape) newShape = this.shape;
    const size = newShape.reduce((a,n)=>a*n, 1);
    let ret = new NdArray(new InternalT(size), newShape, strideOf(newShape));
    let i = 0;
    for (let offset of this.walkIndex()) {
      ret.data[i++] = this.data[offset];
      if (i>=size) break;
    }
    return ret;
  }
  // return an exact copy of this
  dup(): NdArray {
    return new NdArray(this.data.slice(0), this._shape.slice(0), this._stride.slice(0));
  }

  /*** Inernal algebraic, some may be blas optimized */
  // find and return max value
  max(): number {
    return this.data[this.argmax()];
  }
  argmax(): number {
    return this._argmax();
  }
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
    this.data.forEach((n,i)=>this.data[i]+=p);
    return this;
  }
  addeqn(p: NdArray): NdArray {
    this.data.forEach((n,i)=>this.data[i]+=p.data[i]);
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
    for (let i=0; i<this.data.length; i++) this.data[i] -= p.data[i];
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
    this.data.forEach((n,i)=>this.data[i] = n * p);
    return this;
  }
  muleqn(p: NdArray): NdArray {
    this.data.forEach((n,i)=>this.data[i] = n * p.data[i]);
    return this;
  }
    
  /**
   * inverse this: x -> 1/x
   */
  inv(): NdArray {
    return this.dup().inveq();
  }
  inveq(): NdArray {
    this.data.forEach((n,i)=>this.data[i] = 1/n);
    return this;
  }
  /**
   * exp this: x -> e^x
   */
  exp(): NdArray {
    return this.dup().expeq();
  }
  expeq(): NdArray {
    this.data.forEach((n,i)=>this.data[i] = Math.exp(n));
    return this;    
  }

  /**
   * create an array with data and shape
   * shape is by default a 2D column vector:  [data.length, 1]
   * stride by default right most dimension = least significant
   */
  static from(p: ExternData, shape?: number[], stride?: number[] ): NdArray {
    if (!shape ) shape = [p.length, 1];
    let data = new InternalT(p.length);
    for (let i=0; i<data.length; i++) data[i] = p[i]
    return new NdArray(data, shape, stride || strideOf(shape))
  }
  /**
   * create an array of zeroes with provide shape
   */
  static zeros(shape: number[]): NdArray {
    const size = shape.reduce((a,n)=>a*n, 1);
    let data = new InternalT(size); // by default zero-ed, assuming TypedArray
    return new NdArray(data, shape, strideOf(shape))
  }

  /**
   * create an array of random nubmers between [-1,1) with provide shape
   */
  static randn(shape: number[], deviation=1, mean=0): NdArray {
    const size = shape.reduce((a,n)=>a*n, 1);
    let data = new InternalT(size);
    data.forEach((n,i)=>data[i] = (Rand.rand()*2-1 + mean)*deviation);
    return new NdArray(data, shape, strideOf(shape))
  }

  public toString(colSize = 5): string {
    if (this.dimension <= 2) {
      let res = '';
      for (let r = 0; r<this._shape[0]; r++) {
        if (this.dimension == 2) {
          for (let c = 0; c<this._shape[1]; c++) {
            res += padL(this.get(r,c), colSize);
          }
        } else {
          res += padL(this.get(r),colSize);
        }
        res += "\n";
      }
      return res;
    } else {
      return '[Dimension too large]';
    }
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
  protected constructor(protected data: InternalDataType, protected _shape: number[], protected _stride: number[]) {
  }
  // calcualte index offset in data based on dimension axis index, slow!
  protected offset(...args): number {
    return this._stride.reduce((a,v,i)=>a+v*args[i], 0);
  }
  hasNaN(): boolean {
    for (let i=0; i<this.data.length; i++) if (isNaN(this.data[i])) return true;
    return false;
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
function padL(a: number, length: number, pad:string=' '): string {
  let digit = a.toString()||''
  if (digit.length > length-1) {  // no room for pads
    digit = digit.slice(0, length-2) + '?';
  }
  return (new Array((length)+1).join(pad)+digit).slice(-(length))
}
