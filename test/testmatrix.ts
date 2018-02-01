
import { Matrix } from "../src/core/matrix";
import * as assert from'assert';
import * as should from "should";
import { Rand } from "../src/core/rand";

describe('matrix', function() {

  it('from', function() {
    assert.ok(Matrix.zeros([2,1]).equals(Matrix.fromCol([0,0])));
    assert.ok(Matrix.randn([2,1]).length == 2);
  })

  it('scalar', function() {
    let a = Matrix.fromCol([1,2,3]);
    let b = a.add(1).sub(2).mul(2)
    assert.ok(b.equals(Matrix.fromCol([0,2,4])));
    assert.equal(b.max(), 4);
    assert.equal(b.argmax(), 2);
    assert.equal((b as any)._argmax(), 2);
    assert.ok(a.inv().equals(Matrix.fromCol([1,1/2,1/3])));
    assert.ok(a.mul(-1).equals(Matrix.fromCol([-1,-2,-3])));
    a = Matrix.fromCol([0,0,0]);
    assert.ok(a.exp().equals(Matrix.fromCol([1,1,1])));
    assert.ok(a.exp().ln().equals(Matrix.fromCol([0,0,0])));
    a = Matrix.fromCol([0,1]);
    assert.ok(a.sigmoid().equals(Matrix.fromCol([0.5, 1/(1+Math.exp(-1))])));
    assert.ok(a.tanh().equals(Matrix.fromCol([Math.tanh(0), Math.tanh(1)])));
    a = Matrix.fromCol([-1,0,1]);
    assert.ok(a.relu().equals(Matrix.fromCol([0,0,1])));
    // verify softmax by summing up all element and user it's about 1
    assert.ok(Math.abs((a.softmax() as any).data.reduce((a,n)=>a+n,0) -1)< 0.00001);

    Matrix.fromCol([1,2,3])._sum().should.equal(6);
    Matrix.fromCol([1,2,3]).sum().should.equal(6);
  })

  it('matrix', function() {
    let a = Matrix.fromCol([1,2,3]);
    let b = Matrix.fromCol([1,-1,1]);
    assert.ok(a.add(b).equals(Matrix.fromCol([2,1,4])));
    assert.ok(a.sub(b).equals(Matrix.fromCol([0,3,2])));
    assert.ok(a.mul(b).equals(Matrix.fromCol([1,-2,3])));
    should.throws(()=>{
      a.add(Matrix.fromRow([1,2,3]))
    })
    let l = new Float32Array([1,2,3]);
    let r = new Float32Array([2,2,1]);
    Matrix._dot(l, r).should.equal(9);
    Matrix.dot(l, r).should.equal(9);
    // [2, 3] dot [2, 2]
    Matrix.dotm(Matrix.fromData(l, [3,1]), 1, Matrix.fromData(r, [3,1]), 0, 2).should.equal(10);
  })

  it('shape', function() {
    let a = Matrix.from([1,1,2,2, 3, 3], [3,2])
    assert.equal(a.get(0,0), 1);
    assert.equal(a.get(0,1), 1);
    assert.equal(a.get(1,0), 2);
    assert.equal(a.get(1,1), 2);
    assert.equal(a.get(2,0), 3);
    assert.equal(a.get(2,1), 3);
    let b = a.T
    assert.equal(b.get(0,0), 1);
    assert.equal(b.get(0,1), 2);
    assert.equal(b.get(0,2), 3);
    assert.equal(b.get(1,0), 1);
    assert.equal(b.get(1,1), 2);
    assert.equal(b.get(1,2), 3);

    
    let r = Matrix.from([1,-1,2,-2,3,-3],[3,2]); 
    let rt = Matrix.from([1,2,3,-1,-2,-3],[2,3]);  // rt is transpose of r 
    // ensure transpose does not change equality
    assert.ok(rt.T.equals(r));
    assert.ok(r.T.equals(rt));

    // r.T.add(rt) will fail because of stirde mismatch, so reshape() first
    assert.ok(r.T.add(rt).equals(Matrix.from([2,4,6,-2,-4,-6], [2,3])));

  })

  it('matmul', function() {
    let a = Matrix.fromCol([1,2,3]).T;
    let b = Matrix.from([1,2,3], [3,1]);
    let c = a.matmul(b)
    assert.ok(c.equals(Matrix.from([14], [1,1])))    
    let d = b.matmul(a);
    assert.ok(d.equals(Matrix.from([1,2,3,2,4,6,3,6,9], [3,3])))

    let x = Matrix.from([1,2,3, -1,-2,-3], [2,3]);
    let y = Matrix.from([1,2,3, -1,-2,-3], [3,2]);
    assert.ok(x.matmul(y).equals(Matrix.from([1,-9,-1,9], [2,2])))

    assert.ok(Matrix.identiy(3).equals(Matrix.from([1,0,0,0,1,0,0,0,1], [3,3])));
  })

  it('print', function() {
    Matrix.from([112355,12345,1234], [1,3]).toString(5);
  })

  it('stack', function() {
    let x1 = Matrix.fromRow([1,2,3]);
    let x2 =  Matrix.from([1,2,3, -1,-2,-3], [2,3]);
    let x3 = Matrix.fromRow([0,0,0]);
    should.ok(Matrix.stackRows([x1, x2, x3]).equals(
      Matrix.from([1,2,3,1,2,3,-1,-2,-3,0,0,0], [4,3])
    ));
    should.throws(()=>{
      Matrix.stackRows([
        Matrix.fromRow([1,2]), 
        Matrix.fromRow([1,2,3])
      ]);
    })

    let y1 =  Matrix.fromCol([1,2,3]);
    let y2 =  Matrix.from([1,2,3, -1,-2,-3], [3,2]);
    let y3 = Matrix.fromCol([0,0,0]);
    should.ok(Matrix.stackCols([y1, y2, y3]).equals(
      Matrix.from([1,1,2,0,2,3,-1,0,3,-2,-3,0], [3,4])
    ));
    should.throws(()=>{
      Matrix.stackCols([
        Matrix.fromCol([1,2]), 
        Matrix.fromCol([1,2,3])
      ]);
    })
    
    let x =  Matrix.from([1,2,3, -1,-2,-3], [2,3]);
    should.ok(x.addeqCol(Matrix.fromCol([1,-1])).equals(
      Matrix.from([2,3,4,-2,-3,-4], [2,3])
    ))
    should.ok(x.sumCols().equals(Matrix.fromCol([9,-9])));
  })
  it('rand', function() {
    should.ok(!Matrix.randn([2,2]).equals(Matrix.randn([2,2])));
    Rand.seed('1');
    let l1 = Matrix.randn([2,2]);
    Rand.seed('1');
    let l2 = Matrix.randn([2,2]);
    should.ok(l1.equals(l2));
    should.ok(Rand.rand()!=NaN);
    let ar = [1,2,3]
    Rand.seed();  // reset to random seed
    // console.log(Rand.rbinom(10, 3, 0.8));
    Rand.rbinom(10, 3, 0.1).reduce((a,n)=>a && n<=3, true).should.equal(true);
    should.deepEqual(Rand.shuffle(ar).sort(), [1,2,3]);
    should.equal(Matrix.fromCol([1,NaN, 2]).hasNaN(), true);
    should.equal(Matrix.fromCol([1,1, 2]).hasNaN(), false);
  });

})

