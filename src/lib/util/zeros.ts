// helper function returns array of zeros of length n
// and uses typed arrays if available
export function zeros (n : number) : number[] | Float64Array {
  if (typeof(n) === 'undefined' || isNaN(n)) {
    return [];
  }
  if (typeof ArrayBuffer === 'undefined') {
    // lacking browser support
    const arr = new Array(n);
    for (let i = 0; i < n; i++) {
      arr[ i ] = 0;
    }
    return arr;
  }
  else {
    return new Float64Array(n);
  }
}
