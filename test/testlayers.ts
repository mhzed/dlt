
import { Matrix } from "../src/core/matrix";
import * as assert from'assert';
//import * as should from "should";

describe('layers', function() {

  it('input', function() {
    assert.ok(Matrix.zeros([2,1]).equals(Matrix.fromCol([0,0])));
    assert.ok(Matrix.randn([2,1]).length == 2);
  })


})

