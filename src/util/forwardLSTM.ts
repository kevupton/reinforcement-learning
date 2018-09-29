export function forwardLSTM (G, model, hidden_sizes, x, prev) {
  // forward prop for a single tick of LSTM
  // G is graph to append ops to
  // model contains LSTM parameters
  // x is 1D column vector with observation
  // prev is a struct containing hidden and cell
  // from previous iteration

  let hidden_prevs;
  let cell_prevs;

  if(prev == null || typeof prev.h === 'undefined') {
    let hidden_prevs = [];
    let cell_prevs = [];
    for(let d=0;d<hidden_sizes.length;d++) {
      hidden_prevs.push(new R.Mat(hidden_sizes[d],1));
      cell_prevs.push(new R.Mat(hidden_sizes[d],1));
    }
  } else {
    hidden_prevs = prev.h;
    cell_prevs = prev.c;
  }

  const hidden = [];
  const cell = [];

  for(let d=0;d<hidden_sizes.length;d++) {

    const input_vector = d === 0 ? x : hidden[d-1];
    hidden_prev = hidden_prevs[d];
    cell_prev = cell_prevs[d];

    // input gate
    const h0 = G.mul(model['Wix'+d], input_vector);
    const h1 = G.mul(model['Wih'+d], hidden_prev);
    const input_gate = G.sigmoid(G.add(G.add(h0,h1),model['bi'+d]));

    // forget gate
    const h2 = G.mul(model['Wfx'+d], input_vector);
    const h3 = G.mul(model['Wfh'+d], hidden_prev);
    const forget_gate = G.sigmoid(G.add(G.add(h2, h3),model['bf'+d]));

    // output gate
    const h4 = G.mul(model['Wox'+d], input_vector);
    const h5 = G.mul(model['Woh'+d], hidden_prev);
    const output_gate = G.sigmoid(G.add(G.add(h4, h5),model['bo'+d]));

    // write operation on cells
    const h6 = G.mul(model['Wcx'+d], input_vector);
    const h7 = G.mul(model['Wch'+d], hidden_prev);
    const cell_write = G.tanh(G.add(G.add(h6, h7),model['bc'+d]));

    // compute new cell activation
    const retain_cell = G.eltmul(forget_gate, cell_prev); // what do we keep from cell
    const write_cell = G.eltmul(input_gate, cell_write); // what do we write to cell
    const cell_d = G.add(retain_cell, write_cell); // new cell contents

    // compute hidden state as gated, saturated cell activations
    const hidden_d = G.eltmul(output_gate, G.tanh(cell_d));

    hidden.push(hidden_d);
    cell.push(cell_d);
  }

  // one decoder to outputs at end
  const output = G.add(G.mul(model['Whd'], hidden[hidden.length - 1]),model['bd']);

  // return cell memory, hidden representation and output
  return {'h':hidden, 'c':cell, 'o' : output};
}
