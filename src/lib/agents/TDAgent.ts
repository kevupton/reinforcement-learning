// QAgent uses TD (Q-Learning, SARSA)
// - does not require environment model :)
// - learns from experience :)
import { getopt } from '../../util/getopt';
import { randi } from '../../util/random';
import { sampleWeighted } from '../../util/sampleWeighted';
import { setConst } from '../../util/setconst';
import { zeros } from '../../util/zeros';

export class TDAgent {
  ns : number;
  na : number;
  explored : boolean;

  update : any;
  gamma : any;
  epsilon : any;
  alpha : any;

  smooth_policy_update : any;
  beta : any;

  lambda : any;
  replacing_traces : any;

  q_init_val : any;
  planN : any;

  Q : any;
  P : any;
  e : any;
  env_model_s : any;

  env_model_r : any;


  constructor (
    readonly env : any,
    opt : any
  ) {
    this.update  = getopt(opt, 'update', 'qlearn'); // qlearn | sarsa
    this.gamma   = getopt(opt, 'gamma', 0.75); // future reward discount factor
    this.epsilon = getopt(opt, 'epsilon', 0.1); // for epsilon-greedy policy
    this.alpha   = getopt(opt, 'alpha', 0.01); // value function learning rate

    // class allows non-deterministic policy, and smoothly regressing towards the optimal policy based on Q
    this.smooth_policy_update = getopt(opt, 'smooth_policy_update', false);
    this.beta                 = getopt(opt, 'beta', 0.01); // learning rate for policy, if smooth updates are on

    // eligibility traces
    this.lambda           = getopt(opt, 'lambda', 0); // eligibility trace decay. 0 = no eligibility traces used
    this.replacing_traces = getopt(opt, 'replacing_traces', true);

    // optional optimistic initial values
    this.q_init_val = getopt(opt, 'q_init_val', 0);

    this.planN = getopt(opt, 'planN', 0); // number of planning steps per learning iteration (0 = no planning)

    this.Q           = null; // state action value function
    this.P           = null; // policy distribution \pi(s,a)
    this.e           = null; // eligibility trace
    this.env_model_s = null;
     // environment model (s,a) -> (s',r)
    this.env_model_r = null;
     // environment model (s,a) -> (s',r)
    this.env = env; // store pointer to environment
    this.reset();
  }
  reset(){
    // reset the agent's policy and value function
    this.ns = this.env.getNumStates();
    this.na = this.env.getMaxNumActions();
    this.Q = zeros(this.ns * this.na);
    if (this.q_init_val !== 0) { setConst(this.Q, this.q_init_val); }
    this.P = zeros(this.ns * this.na);
    this.e = zeros(this.ns * this.na);

    // model/planning vars
    this.env_model_s = zeros(this.ns * this.na);
    setConst(this.env_model_s, -1); // init to -1 so we can test if we saw the state before
    this.env_model_r = zeros(this.ns * this.na);
    this.sa_seen = [];
    this.pq = zeros(this.ns * this.na);

    // initialize uniform random policy
    for (let s = 0; s < this.ns; s++) {
      const poss = this.env.allowedActions(s);
      for (let i = 0, n = poss.length; i < n; i++) {
        this.P[poss[i] * this.ns + s] = 1.0 / poss.length;
      }
    }
    // agent memory, needed for streaming updates
    // (s0,a0,r0,s1,a1,r1,...)
    this.r0 = null;
    this.s0 = null;
    this.s1 = null;
    this.a0 = null;
    this.a1 = null;
  }
  resetEpisode() {
    // an episode finished
  }
  act(s){
    // act according to epsilon greedy policy
    const poss = this.env.allowedActions(s);
    const probs = [];
    for (let i = 0, n = poss.length; i < n; i++) {
      probs.push(this.P[poss[i] * this.ns + s]);
    }
    // epsilon greedy policy
    if (Math.random() < this.epsilon) {
      const a = poss[randi(0, poss.length)]; // random available action
      this.explored = true;
    } else {
      const a = poss[sampleWeighted(probs)];
      this.explored = false;
    }
    // shift state memory
    this.s0 = this.s1;
    this.a0 = this.a1;
    this.s1 = s;
    this.a1 = a;
    return a;
  }
  learn(r1){
    // takes reward for previous action, which came from a call to act()
    if (!(this.r0 == null)) {
      this.learnFromTuple(this.s0, this.a0, this.r0, this.s1, this.a1, this.lambda);
      if (this.planN > 0) {
        this.updateModel(this.s0, this.a0, this.r0, this.s1);
        this.plan();
      }
    }
    this.r0 = r1; // store this for next update
  }
  updateModel(s0, a0, r0, s1) {
    // transition (s0,a0) -> (r0,s1) was observed. Update environment model
    const sa = a0 * this.ns + s0;
    if (this.env_model_s[sa] === -1) {
      // first time we see this state action
      this.sa_seen.push(a0 * this.ns + s0); // add as seen state
    }
    this.env_model_s[sa] = s1;
    this.env_model_r[sa] = r0;
  }
  plan() {

    // order the states based on current priority queue information
    const spq = [];
    for (let i = 0, n = this.sa_seen.length; i < n; i++) {
      const sa = this.sa_seen[i];
      const sap = this.pq[sa];
      if (sap > 1e-5) { // gain a bit of efficiency
        spq.push({sa: sa, p: sap});
      }
    }
    spq.sort(function(a, b){ return a.p < b.p ? 1 : -1; });

    // perform the updates
    const nsteps = Math.min(this.planN, spq.length);
    for (let k = 0; k < nsteps; k++) {
      // random exploration
      // var i = randi(0, this.sa_seen.length); // pick random prev seen state action
      // var s0a0 = this.sa_seen[i];
      const s0a0 = spq[k].sa;
      this.pq[s0a0] = 0; // erase priority, since we're backing up this state
      const s0 = s0a0 % this.ns;
      const a0 = Math.floor(s0a0 / this.ns);
      const r0 = this.env_model_r[s0a0];
      const s1 = this.env_model_s[s0a0];
      const a1 = -1; // not used for Q learning
      if (this.update === 'sarsa') {
        // generate random action?...
        const poss = this.env.allowedActions(s1);
        const a1 = poss[randi(0, poss.length)];
      }
      this.learnFromTuple(s0, a0, r0, s1, a1, 0); // note lambda = 0 - shouldnt use eligibility trace here
    }
  }
  learnFromTuple(s0, a0, r0, s1, a1, lambda) {
    const sa = a0 * this.ns + s0;

    // calculate the target for Q(s,a)
    if (this.update === 'qlearn') {
      // Q learning target is Q(s0,a0) = r0 + gamma * max_a Q[s1,a]
      const poss = this.env.allowedActions(s1);
      let qmax = 0;
      for (let i = 0, n = poss.length; i < n; i++) {
        const s1a = poss[i] * this.ns + s1;
        const qval = this.Q[s1a];
        if (i === 0 || qval > qmax) { qmax = qval; }
      }
      const target = r0 + this.gamma * qmax;
    } else if (this.update === 'sarsa') {
      // SARSA target is Q(s0,a0) = r0 + gamma * Q[s1,a1]
      const s1a1 = a1 * this.ns + s1;
      const target = r0 + this.gamma * this.Q[s1a1];
    }

    if (lambda > 0) {
      // perform an eligibility trace update
      if (this.replacing_traces) {
        this.e[sa] = 1;
      } else {
        this.e[sa] += 1;
      }
      const edecay = lambda * this.gamma;
      const state_update = zeros(this.ns);
      for (let s = 0; s < this.ns; s++) {
        const poss = this.env.allowedActions(s);
        for (let i = 0; i < poss.length; i++) {
          const a = poss[i];
          const saloop = a * this.ns + s;
          const esa = this.e[saloop];
          const update = this.alpha * esa * (target - this.Q[saloop]);
          this.Q[saloop] += update;
          this.updatePriority(s, a, update);
          this.e[saloop] *= edecay;
          const u = Math.abs(update);
          if (u > state_update[s]) { state_update[s] = u; }
        }
      }
      for (let s = 0; s < this.ns; s++) {
        if (state_update[s] > 1e-5) { // save efficiency here
          this.updatePolicy(s);
        }
      }
      if (this.explored && this.update === 'qlearn') {
        // have to wipe the trace since q learning is off-policy :(
        this.e = zeros(this.ns * this.na);
      }
    } else {
      // simpler and faster update without eligibility trace
      // update Q[sa] towards it with some step size
      const update = this.alpha * (target - this.Q[sa]);
      this.Q[sa] += update;
      this.updatePriority(s0, a0, update);
      // update the policy to reflect the change (if appropriate)
      this.updatePolicy(s0);
    }
  }
  updatePriority(s, a, u) {
    // used in planning. Invoked when Q[sa] += update
    // we should find all states that lead to (s,a) and upgrade their priority
    // of being update in the next planning step
    u = Math.abs(u);
    if (u < 1e-5) { return; } // for efficiency skip small updates
    if (this.planN === 0) { return; } // there is no planning to be done, skip.
    for (let si = 0; si < this.ns; si++) {
      // note we are also iterating over impossible actions at all states,
      // but this should be okay because their env_model_s should simply be -1
      // as initialized, so they will never be predicted to point to any state
      // because they will never be observed, and hence never be added to the model
      for (let ai = 0; ai < this.na; ai++) {
        const siai = ai * this.ns + si;
        if (this.env_model_s[siai] === s) {
          // this state leads to s, add it to priority queue
          this.pq[siai] += u;
        }
      }
    }
  }
  updatePolicy(s) {
    const poss = this.env.allowedActions(s);
    // set policy at s to be the action that achieves max_a Q(s,a)
    // first find the maxy Q values
    let qmax, nmax;
    const qs = [];
    for (let i = 0, n = poss.length; i < n; i++) {
      const a = poss[i];
      const qval = this.Q[a * this.ns + s];
      qs.push(qval);
      if (i === 0 || qval > qmax) { qmax = qval; nmax = 1; }
      else if (qval === qmax) { nmax += 1; }
    }
    // now update the policy smoothly towards the argmaxy actions
    let psum = 0.0;
    for (let i = 0, n = poss.length; i < n; i++) {
      const a = poss[i];
      const target = (qs[i] === qmax) ? 1.0 / nmax : 0.0;
      const ix = a * this.ns + s;
      if (this.smooth_policy_update) {
        // slightly hacky :p
        this.P[ix] += this.beta * (target - this.P[ix]);
        psum += this.P[ix];
      } else {
        // set hard target
        this.P[ix] = target;
      }
    }
    if (this.smooth_policy_update) {
      // renomalize P if we're using smooth policy updates
      for (let i = 0, n = poss.length; i < n; i++) {
        const a = poss[i];
        this.P[a * this.ns + s] /= psum;
      }
    }
  }
}
