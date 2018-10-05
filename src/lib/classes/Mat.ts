import { assert } from '../../util/assert';
// Mat holds a matrix
import { zeros } from '../../util/zeros';

export interface MatrixData {
  n : number;
  d : number;
  w : any[] | Float64Array;
  dw? : any[] | Float64Array;
}

export class Mat implements MatrixData {

  constructor (
    readonly n : number,
    readonly d : number,
    readonly w  = zeros(n * d),
    readonly dw = zeros(n * d),
  ) {
  }

  get (row : number, col : number) {
    // slow but careful accessor function
    // we want row-major order
    const ix = (this.d * row) + col;
    assert(ix >= 0 && ix < this.w.length);
    return this.w[ix];
  }

  set (row : number, col : number, v : number) {
    // slow but careful accessor function
    const ix = (this.d * row) + col;
    assert(ix >= 0 && ix < this.w.length);
    this.w[ix] = v;
  }

  setFrom (arr : number[] | Float64Array) {
    for (let i = 0, n = arr.length; i < n; i++) {
      this.w[i] = arr[i];
    }
  }

  setColumn (matrix : Mat, i : number) {
    for (let q = 0, n = matrix.w.length; q < n; q++) {
      this.w[(this.d * q) + i] = matrix.w[q];
    }
  }

  toJSON () : MatrixData {
    return {
      n: this.n,
      d: this.d,
      w: this.w,
    };
  }

  static fromJSON (json : MatrixData) {
    json = JSON.parse(JSON.stringify(json));

    return new Mat(json.n, json.d, json.w, json.dw);

    // for (let i = 0, n = this.rows * this.columns; i < n; i++) {
    //   this.w[ i ] = json.w[ i ]; // copy over weights
    // }
  }
}

// var Mat = function (n, d) {
//   // n is number of rows d is number of columns
//   this.n  = n;
//   this.d  = d;
//   this.w  = zeros(n * d);
//   this.dw = zeros(n * d);
//
// }
