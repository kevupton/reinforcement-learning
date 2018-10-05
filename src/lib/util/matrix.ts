import { Mat } from '../classes/Mat';
import { randf, randn } from './random';
import { Net } from '../interfaces/index';

// return Mat but filled with random numbers from gaussian
export function RandMat (n : number, d : number, mu : number, std : number) {
  const matrix = new Mat(n, d);
  fillRandn(matrix, mu, std);
  // fillRand(m,-std,std); // kind of :P
  return matrix;
}

// Mat utils
// fill matrix with random gaussian numbers
export function fillRandn (m : Mat, mu : number, std : number) {
  for (let i = 0, n = m.w.length; i < n; i++) {
    m.w[i] = randn(mu, std);
  }
}

export function fillRand (m : Mat, lo : number, hi : number) {
  for (let i = 0, n = m.w.length; i < n; i++) {
    m.w[i] = randf(lo, hi);
  }
}

export function gradFillConst (m : Mat, c : number) {
  for (let i = 0, n = m.dw.length; i < n; i++) {
    m.dw[i] = c;
  }
}

export function copyMat (b : Mat) {
  const a = new Mat(b.n, b.d);
  a.setFrom(b.w);
  return a;
}

export function copyNet (net : Net) {
  // nets are (k,v) pairs with k = string key, v = Mat()
  const new_net : Net = {};
  for (const p in net) {
    if (net.hasOwnProperty(p)) {
      new_net[p] = copyMat(net[p]);
    }
  }
  return new_net;
}

export function updateMat (m : Mat, alpha : number) {
  // updates in place
  for (let i = 0, n = m.n * m.d; i < n; i++) {
    if (m.dw[i] !== 0) {
      m.w[i] += -alpha * m.dw[i];
      m.dw[i] = 0;
    }
  }
}

export function updateNet (net : Net, alpha : number) {
  for (const p in net) {
    if (net.hasOwnProperty(p)) {
      updateMat(net[p], alpha);
    }
  }
}

export function netToJSON (net : Net) {
  const j : any = {};
  for (const p in net) {
    if (net.hasOwnProperty(p)) {
      j[p] = net[p].toJSON();
    }
  }
  return j;
}

export function netFromJSON (j : any) {
  const net : Net = {};
  for (const p in j) {
    if (j.hasOwnProperty(p)) {
      net[p] = Mat.fromJSON(j[p]);
    }
  }
  return net;
}

export function netZeroGrads (net : Net) {
  for (const p in net) {
    if (net.hasOwnProperty(p)) {
      const mat = net[p];
      gradFillConst(mat, 0);
    }
  }
}

export function netFlattenGrads (net : Mat[]) {
  let rows = 0;
  for (const p in net) {
    if (net.hasOwnProperty(p)) {
      const matrix = net[p];
      rows += matrix.dw.length;
    }
  }
  const g = new Mat(rows, 1);
  let ix  = 0;
  for (const p in net) {
    if (net.hasOwnProperty(p)) {
      const mat = net[p];
      for (let i = 0, m = mat.dw.length; i < m; i++) {
        g.w[ix] = mat.dw[i];
        ix++;
      }
    }
  }
  return g;
}
