import { assert } from './assert';

export function sampleWeighted (p : number[]) {
  const r = Math.random();
  let c   = 0.0;

  for (let i = 0, n = p.length; i < n; i++) {
    c += p[i];
    if (c >= r) {
      return i;
    }
  }

  assert(false, 'wtf');
}
