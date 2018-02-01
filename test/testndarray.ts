
import { NdArray } from "../src/core/ndarray";
import * as assert from'assert';
import * as should from "should";
import { Rand } from "../src/core/rand";

describe('ndarray', function() {

  it('from', function() {
    assert.ok(NdArray.zeros([2,1]).equals(NdArray.from([0,0])));
    assert.ok(NdArray.randn([2,1]).length == 2);
  })

  it('scalar algebra', function() {
    let a = NdArray.from([1,2,3]);
    let b = a.add(1).sub(2).mul(2)
    assert.ok(b.equals(NdArray.from([0,2,4])));
    assert.equal(b.max(), 4);
    assert.equal(b.argmax(), 2);
    assert.equal((b as any)._argmax(), 2);
    assert.ok(a.inv().equals(NdArray.from([1,1/2,1/3])));
    assert.ok(a.mul(-1).equals(NdArray.from([-1,-2,-3])));
    a = NdArray.from([0,0,0]);
    assert.ok(a.exp().equals(NdArray.from([1,1,1])));
    a = NdArray.from([0,1]);
  })

  it('matrix algebra', function() {
    let a = NdArray.from([1,2,3]);
    let b = NdArray.from([1,-1,1]);
    assert.ok(a.add(b).equals(NdArray.from([2,1,4])));
    assert.ok(a.sub(b).equals(NdArray.from([0,3,2])));
    assert.ok(a.mul(b).equals(NdArray.from([1,-2,3])));
    should.throws(()=>{
      a.add(NdArray.from([1,2,3], [1, 3]))
    })
    let l = new Float32Array([1,2,3]);
    let r = new Float32Array([2,2,1]);
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

    // r.T.add(rt) will fail because of stirde mismatch, so reshape() first
    should.throws(()=>{
      assert.ok(r.transpose([1,0]).add(rt).equals(NdArray.from([2,4,6,-2,-4,-6], [2,3])));
    })
  })

  it('print', function() {
    NdArray.from([112355,12345,1234], [1,3]).toString(5);
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
    should.equal(NdArray.from([1,NaN, 2]).hasNaN(), true);
    should.equal(NdArray.from([1,1, 2]).hasNaN(), false);
  });

})

