import { Mat } from '../lib/classes/Mat';

export function initLSTM (input_size, hidden_sizes, output_size) {
  // hidden size should be a list

  const model = {};
  for (let d = 0; d < hidden_sizes.length; d++) { // loop over depths
    const prev_size   = d === 0 ? input_size : hidden_sizes[d - 1];
    const hidden_size = hidden_sizes[d];

    // gates parameters
    model['Wix' + d] = new RandMat(hidden_size, prev_size, 0, 0.08);
    model['Wih' + d] = new RandMat(hidden_size, hidden_size, 0, 0.08);
    model['bi' + d]  = new Mat(hidden_size, 1);
    model['Wfx' + d] = new RandMat(hidden_size, prev_size, 0, 0.08);
    model['Wfh' + d] = new RandMat(hidden_size, hidden_size, 0, 0.08);
    model['bf' + d]  = new Mat(hidden_size, 1);
    model['Wox' + d] = new RandMat(hidden_size, prev_size, 0, 0.08);
    model['Woh' + d] = new RandMat(hidden_size, hidden_size, 0, 0.08);
    model['bo' + d]  = new Mat(hidden_size, 1);
    // cell write params
    model['Wcx' + d] = new RandMat(hidden_size, prev_size, 0, 0.08);
    model['Wch' + d] = new RandMat(hidden_size, hidden_size, 0, 0.08);
    model['bc' + d]  = new Mat(hidden_size, 1);
  }

  // decoder params
  model['Whd'] = new RandMat(output_size, hidden_size, 0, 0.08);
  model['bd']  = new Mat(output_size, 1);
  return model;
}
