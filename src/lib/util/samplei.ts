import { randf } from './random';

export function samplei (w : number[]) {
  // sample argmax from w, assuming w are
  // probabilities that sum to one

  const r = randf(0, 1);
  let x = 0.0;
  let i = 0;

  while (true) {
    x += w[i];
    if (x > r) { return i; }
    i++;
  }
}
