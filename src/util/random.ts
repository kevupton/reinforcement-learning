// Random numbers utils

let return_v = false;
let v_val    = 0.0;

export function gaussRandom () {
  if (return_v) {
    return_v = false;
    return v_val;
  }
  const u = 2 * Math.random() - 1;
  const v = 2 * Math.random() - 1;
  const r = u * u + v * v;
  if (r == 0 || r > 1) return gaussRandom();
  const c    = Math.sqrt(-2 * Math.log(r) / r);
  v_val    = v * c; // cache this
  return_v = true;
  return u * c;
}

export function randf (a : number, b : number) {
  return Math.random() * (b - a) + a;
}

export function randi (a : number, b : number) {
  return Math.floor(Math.random() * (b - a) + a);
}

export function randn (mu : number, std : number) {
  return mu + gaussRandom() * std;
}
