
import { NdArray } from "../src/core/ndarray";
import { Matrix } from "../src/core/matrix";
import * as assert from'assert';
import * as should from "should";
import { Rand } from "../src/core/rand";

describe('ndarray', function() {

  it('from', function() {
    NdArray.zeros([2,1]).data.reduce((a, v) => a && (v==0), true).should.equal(true);
    assert.ok(NdArray.zeros([2,1]).equals(NdArray.fromCol([0,0])));
    assert.ok(NdArray.zeros([1,2]).equals(NdArray.fromRow([0,0])));
    assert.ok(NdArray.zeros([1,2]).equals(NdArray.fromData(new NdArray.Type([0,0]), [1,2])));
    assert.ok(NdArray.randn([2,1]).length == 2);
    
  })

  it('element wise scalar algebra', function() {
    let a = NdArray.fromCol([1,2,3]);
    let b = a.add(1).sub(2).mul(2)
    a.sum().should.equal(6);
    NdArray.fromCol([3,4]).magnitude().should.equal(5);
    assert.ok(b.equals(NdArray.fromCol([0,2,4])));
    assert.equal(b.max(), 4);
    assert.equal(b.argmax(), 2);
    assert.ok(a.inv().equals(NdArray.fromCol([1,1/2,1/3])));
    assert.ok(a.mul(-1).equals(NdArray.fromCol([-1,-2,-3])));
    a = NdArray.fromCol([0,0,0]);
    assert.ok(a.exp().equals(NdArray.fromCol([1,1,1])));
    a = NdArray.fromCol([0,1]);
  })

  it('element wise algebra', function() {
    let a = NdArray.fromCol([1,2,3]);
    let b = NdArray.fromCol([1,-1,1]);
    assert.ok(a.add(b).equals(NdArray.fromCol([2,1,4])));
    assert.ok(a.sub(b).equals(NdArray.fromCol([0,3,2])));
    assert.ok(a.mul(b).equals(NdArray.fromCol([1,-2,3])));
    should.throws(()=>{
      a.add(NdArray.from([1,2,3], [1, 3]))
    })
  })

  it('shape', function() {
    let a = NdArray.from([1,1,2,2, 3, 3], [3,2])
    assert.equal(a.get(0,0), 1);
    assert.equal(a.get(0,1), 1);
    assert.equal(a.get(1,0), 2);
    assert.equal(a.get(1,1), 2);
    assert.equal(a.get(2,0), 3);
    assert.equal(a.get(2,1), 3);
    let b = a.transpose([1,0]);
    assert.equal(b.get(0,0), 1);
    assert.equal(b.get(0,1), 2);
    assert.equal(b.get(0,2), 3);
    assert.equal(b.get(1,0), 1);
    assert.equal(b.get(1,1), 2);
    assert.equal(b.get(1,2), 3);
    
    let r = NdArray.from([1,-1,2,-2,3,-3],[3,2]); 
    let rt = NdArray.from([1,2,3,-1,-2,-3],[2,3]);  // rt is transpose of r 
    // ensure transpose does not change equality
    assert.ok(rt.transpose([1,0]).equals(r));
    assert.ok(r.transpose([1,0]).equals(rt));
    r.isTransposed().should.equals(false);
    r.transpose([1,0]).isTransposed().should.equals(true);
    should.throws(()=>{
      r.transpose([1]);
    })
    NdArray.zeros([1,2]).equals(NdArray.zeros([2,1])).should.equals(false);
    // r.T.add(rt) will fail because of stride mismatch
    should.throws(()=>{
      assert.ok(r.transpose([1,0]).add(rt).equals(NdArray.from([2,4,6,-2,-4,-6], [2,3])));
    })

    r.reshape([2,3]).equals(NdArray.fromData(r.data, [2,3])).should.equals(true);
    should.throws(()=>r.reshape([1,3]));
    
    let rr = r.transpose([1,0]).rearrange();
    rr.equals(rt).should.equal(true);
    rr.isTransposed().should.equal(false);
  })

  it('rand', function() {
    should.ok(!NdArray.randn([2,2]).equals(NdArray.randn([2,2])));
    Rand.seed('1');
    let l1 = NdArray.randn([2,2]);
    Rand.seed('1');
    let l2 = NdArray.randn([2,2]);
    should.ok(l1.equals(l2));
    should.ok(Rand.rand()!=NaN);
    let ar = [1,2,3]
    Rand.seed();  // reset to random seed
    // console.log(Rand.rbinom(10, 3, 0.8));
    Rand.rbinom(10, 3, 0.1).reduce((a,n)=>a && n<=3, true).should.equal(true);
    should.deepEqual(Rand.shuffle(ar).sort(), [1,2,3]);
    should.equal(NdArray.fromCol([1,NaN, 2]).hasNaN(), true);
    should.equal(NdArray.fromCol([1,1, 2]).hasNaN(), false);
  });

  it('print', function() {
    Matrix.toString(NdArray.from([112355,12345,1234], [1,3]), 5);
  })
  
  it('matrix', function() {
    Matrix.identiy(2).equals(NdArray.from([1,0,0,1], [2,2])).should.equal(true);
    Matrix.oneHotCol(1,3).equals(NdArray.from([0,1,0], [3,1])).should.equal(true);
    should.throws(()=>{
      Matrix.oneHotCol(3,3)
    });
    Matrix.T(NdArray.from([1,0,1], [3,1])).equals(NdArray.from([1,0,1], [1,3])).should.equal(true);
    Matrix.T(NdArray.from([1,0,1,0], [2,2])).equals(NdArray.from([1,1,0,0], [2,2])).should.equal(true);
  })

  it('matrix matmul', function() {
    let a = Matrix.T(NdArray.fromCol([1,2,3]));
    let b = NdArray.from([1,2,3], [3,1]);
    let c = Matrix.mul(a, b)
    assert.ok(c.equals(NdArray.from([14], [1,1])))    
    let d = Matrix.mul(b, a);
    assert.ok(d.equals(NdArray.from([1,2,3,2,4,6,3,6,9], [3,3])))

    let x = NdArray.from([1,2,3, -1,-2,-3], [2,3]);
    let y = NdArray.from([1,2,3, -1,-2,-3], [3,2]);
    assert.ok(Matrix.mul(x, y).equals(NdArray.from([1,-9,-1,9], [2,2])))
    
    should.throws(()=>{
      Matrix.mul(NdArray.zeros([2,3]), NdArray.zeros([2,3]));
    })

    Matrix.dot(a.data, b.data).should.equals(14);
    Matrix.dotat(a.data, 1, b.data, 0, 2).should.equals(8);
  })

  it('matrix stack', function() {
    let x1 = NdArray.fromRow([1,2,3]);
    let x2 =  NdArray.from([1,2,3, -1,-2,-3], [2,3]);
    let x3 = NdArray.fromRow([0,0,0]);
    should.ok(Matrix.stackRows([x1, x2, x3]).equals(
      NdArray.from([1,2,3,1,2,3,-1,-2,-3,0,0,0], [4,3])
    ));
    should.throws(()=>{
      Matrix.stackRows([
        NdArray.fromRow([1,2]), 
        NdArray.fromRow([1,2,3])
      ]);
    })

    let y1 =  NdArray.fromCol([1,2,3]);
    let y2 =  NdArray.from([1,2,3, -1,-2,-3], [3,2]);
    let y3 = NdArray.fromCol([0,0,0]);
    should.ok(Matrix.stackCols([y1, y2, y3]).equals(
      NdArray.from([1,1,2,0,2,3,-1,0,3,-2,-3,0], [3,4])
    ));
    should.throws(()=>{
      Matrix.stackCols([
        NdArray.fromCol([1,2]), 
        NdArray.fromCol([1,2,3])
      ]);
    })
    
    let x =  NdArray.from([1,2,3, -1,-2,-3], [2,3]);
    should.ok(Matrix.addeqCol(x, NdArray.fromCol([1,-1])).equals(
      NdArray.from([2,3,4,-2,-3,-4], [2,3])
    ))
    should.throws(()=>{
      Matrix.addeqCol(x, NdArray.fromCol([1,-1, 0]))
    })
    should.ok(Matrix.sumCols(x).equals(NdArray.fromCol([9,-9])));
  })  

  it('blas', function() {
    require("../src/core/applyblas");
    let a = Matrix.T(NdArray.fromCol([1,2,3]));
    let b = NdArray.from([1,2,3], [3,1]);
    let c = Matrix.mul(a, b)
    assert.ok(c.equals(NdArray.from([14], [1,1])))    
    should.throws(()=>{
      Matrix.mul(NdArray.zeros([2,3]), NdArray.zeros([2,3]));
    })
    
    b.sum().should.equal(6);
    b.argmax().should.equals(2);
    b.addeqn(NdArray.fromCol([1,1,1])).equals(NdArray.fromCol([2,3,4])).should.equals(true);
    b.subeqn(NdArray.fromCol([1,1,1])).equals(NdArray.fromCol([1,2,3])).should.equals(true);
    b.muleq(-1).equals(NdArray.fromCol([-1,-2,-3])).should.equals(true);
    NdArray.fromCol([3,4]).magnitude().should.equal(5);
    NdArray.fromCol([]).magnitude().should.equal(0);
    Matrix.dot(a.data, b.data).should.equals(-14);
  })
})

