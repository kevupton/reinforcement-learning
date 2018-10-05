export function maxi (w : number[] | Float64Array) {
  // argmax of array w
  let maxv  = w[0];
  let maxix = 0;

  for (let i = 1, n = w.length; i < n; i++) {
    const v = w[i];
    if (v > maxv) {
      maxix = i;
      maxv  = v;
    }
  }

  return maxix;
}
