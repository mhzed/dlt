import * as RND from "seedrandom";

// hidden gloabal rand number generator
let g_rnd: RND.prng = RND(Math.random().toString());

export class Rand {
  /**
   * 
   * @param seed set the seed for randn
   */
  static seed(seed?: string) {
    if (seed) g_rnd = RND(seed);
    else g_rnd = RND(Math.random().toString());
  }
  /**
   * return a number between [0,1)
   */
  static rand(): number {
    return g_rnd();
  }
  static int32(): number {
    return g_rnd.int32();
  }

  static shuffle<T>(array: T[], n?: number): T[] {
    if (!n) n = array.length;
    for (let i=0; i<n; i++) {
      let x = Math.abs(Rand.int32())%array.length
      let y = Math.abs(Rand.int32())%array.length
      let temp = array[x]
      array[x] = array[y];
      array[y] = temp;
    }
    return array;
  }
  
  /**
   *
   * @param n Number of variates to return.
   * @param size Number of Bernoulli trials to be summed up. Defaults to 1
   * @param p Probability of a "success". Defaults to 0.5
   * @returns {Array} Random variates array
   */
  static rbinom (n: number, size:number, p: number): number[] {
    var toReturn = [];
    for(var i=0; i<n; i++) {
        var result = 0;
        for(var j=0; j<size; j++) {
          let r= Rand.rand();
            if(r < p) {
                result++
            }
        }
        toReturn[i] = result;
    }
    return toReturn
  };

}