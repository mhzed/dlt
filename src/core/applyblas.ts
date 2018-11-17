import { NdArray } from "./ndarray";
import { Matrix } from "./matrix";
import *  as nblas from 'nblas';

Matrix.mul = (l, r) => {
  let r1 = l.shape[0],
      c1 = l.shape[1],
      r2 = r.shape[0],
      c2 = r.shape[1],
      data = new NdArray.Type(r1 * c2);

  if (c1 !== r2) throw new Error('size mis-match');

  nblas.gemm(l.data, r.data, data, r1, c2, c1);
  return NdArray.fromData(data, [r1, c2]);
}
Matrix.dot = (l, r) => {
  return nblas.dot(l, r);
}

NdArray.prototype.sum = function() {
  return nblas.asum(this._data);
}
NdArray.prototype.argmax = function() {
  return nblas.iamax(this._data);
} 
NdArray.prototype.addeqn = function(p: NdArray) {
  nblas.axpy((p as any).data, this._data, 1);
  return this;
}
NdArray.prototype.subeqn = function(p: NdArray) {
  nblas.axpy((p as any).data, this._data, -1);
  return this;
}

NdArray.prototype.muleq = function(p: number) {
  nblas.scal(this._data, p);
  return this;
}
NdArray.prototype.magnitude = function() {
  if (!this.length) return 0;
  else return nblas.nrm2(this._data);
}

