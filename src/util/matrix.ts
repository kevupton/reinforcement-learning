// return Mat but filled with random numbers from gaussian
import { Mat } from '../lib/Mat';
import { randf, randn } from './random';

export function RandMat (n, d, mu, std) {
  const matrix = new Mat(n, d);
  fillRandn(matrix, mu, std);
  //fillRand(m,-std,std); // kind of :P
  return matrix;
}

// Mat utils
// fill matrix with random gaussian numbers
export function fillRandn (m, mu, std) {
  for (let i = 0, n = m.w.length; i < n; i++) {
    m.w[ i ] = randn(mu, std);
  }
}

export function fillRand (m, lo, hi) {
  for (let i = 0, n = m.w.length; i < n; i++) {
    m.w[ i ] = randf(lo, hi);
  }
}

export function gradFillConst (m, c) {
  for (let i = 0, n = m.dw.length; i < n; i++) {
    m.dw[ i ] = c
  }
}

export function copyMat (b : Mat) {
  const a = new Mat(b.n, b.d);
  a.setFrom(b.w);
  return a;
}

export function copyNet (net) {
  // nets are (k,v) pairs with k = string key, v = Mat()
  const new_net = {};
  for (const p in net) {
    if (net.hasOwnProperty(p)) {
      new_net[ p ] = copyMat(net[ p ]);
    }
  }
  return new_net;
}

export function updateMat (m, alpha) {
  // updates in place
  for (let i = 0, n = m.n * m.d; i < n; i++) {
    if (m.dw[ i ] !== 0) {
      m.w[ i ] += -alpha * m.dw[ i ];
      m.dw[ i ] = 0;
    }
  }
}

export function updateNet (net, alpha) {
  for (const p in net) {
    if (net.hasOwnProperty(p)) {
      updateMat(net[ p ], alpha);
    }
  }
}

export function netToJSON (net : Mat[]) {
  const j = {};
  for (const p in net) {
    if (net.hasOwnProperty(p)) {
      j[ p ] = net[ p ].toJSON();
    }
  }
  return j;
}

export function netFromJSON (j) {
  const net = {};
  for (const p in j) {
    if (j.hasOwnProperty(p)) {
      net[ p ] = Mat.fromJSON(j[ p ]);
    }
  }
  return net;
}

export function netZeroGrads (net) {
  for (var p in net) {
    if (net.hasOwnProperty(p)) {
      var mat = net[ p ];
      gradFillConst(mat, 0);
    }
  }
}

export function netFlattenGrads (net : Mat[]) {
  let rows = 0;
  for (const p in net) {
    if (net.hasOwnProperty(p)) {
      const matrix = net[ p ];
      rows += matrix.dw.length;
    }
  }
  const g  = new Mat(rows, 1);
  let ix = 0;
  for (const p in net) {
    if (net.hasOwnProperty(p)) {
      const mat = net[ p ];
      for (let i = 0, m = mat.dw.length; i < m; i++) {
        g.w[ ix ] = mat.dw[ i ];
        ix++;
      }
    }
  }
  return g;
}
