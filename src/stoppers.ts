import { EpochStopper } from "./types";

export class StopWhenNoBetterThanAverage implements EpochStopper {

  /**
   * 
   * @param percEvaluator
   * @param lastN average over this many evaluations
   * @param verbose wheter to print progress on epoch
   */
  constructor(
    private lastN: number, 
    private percEvaluator: ()=>number,
    private nMatch = 2,
    private verbose = false) {
  }

  private history = [];
  private _match = 0;
  // EpochStopper override
  onEpoch(iEpoch: number, elapsedSec: number): boolean {
    let perc = this.percEvaluator();
    if (this.verbose) console.log(`${iEpoch}[${elapsedSec.toFixed(2)}s] ${(perc*100).toFixed(2)}%`);
    let history = this.history;
    let average = history.reduce((a,p)=>a+p, 0) / history.length;
    let shouldStop = (perc <= average && history.length>=this.lastN);
    if (shouldStop) this._match++;
    else this._match = 0;
    history.push(perc);
    if (history.length > this.lastN) history.shift();
    return this._match >= this.nMatch ? false : true;
  }

}

export class StopAt implements EpochStopper {

  /**
   * 
   * @param N stop at Nth loop
   */
  constructor(
    private N: number,
    private cb: (i: number, elapsedsec: number)=>void) {
  }

  private n=0;
  // EpochStopper override
  onEpoch(iEpoch: number, elapsedSec: number): boolean {
    if (this.cb) this.cb(iEpoch, elapsedSec);
    if (this.n++ >= this.N) return false;
    else return true;
  }

}