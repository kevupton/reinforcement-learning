import { Mat } from '../classes/Mat';

export interface AgentOptions {
  update : 'qlearn' | 'slearn';
  gamma : number; // discount factor, [0, 1)
  epsilon : number; // initial epsilon for epsilon-greedy policy, [0, 1)
  alpha : number; // value function learning rate
  experience_add_every : number; // number of time steps before we add another experience to replay memory
  experience_size : number; // size of experience replay memory
  learning_steps_per_iteration : number;
  tderror_clamp : number; // for robustness
  num_hidden_units : number; // number of neurons in hidden layer
}

export interface AgentEnvironment {
  getMaxNumActions() : number;
  getNumStates() : number;
}

export interface Net {
  [key : string] : Mat;
}
