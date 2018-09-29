var R = {}; // the Recurrent library


var RL = {};
(function(global) {
  "use strict";


  var DQNAgent = function(env, opt) {
    this.gamma = getopt(opt, 'gamma', 0.75); // future reward discount factor
    this.epsilon = getopt(opt, 'epsilon', 0.1); // for epsilon-greedy policy
    this.alpha = getopt(opt, 'alpha', 0.01); // value function learning rate

    this.experience_add_every = getopt(opt, 'experience_add_every', 25); // number of time steps before we add another experience to replay memory
    this.experience_size = getopt(opt, 'experience_size', 5000); // size of experience replay
    this.learning_steps_per_iteration = getopt(opt, 'learning_steps_per_iteration', 10);
    this.tderror_clamp = getopt(opt, 'tderror_clamp', 1.0);

    this.num_hidden_units =  getopt(opt, 'num_hidden_units', 100);

    this.env = env;
    this.reset();
  }
  DQNAgent.prototype = {
    reset: function() {
      this.nh = this.num_hidden_units; // number of hidden units
      this.ns = this.env.getNumStates();
      this.na = this.env.getMaxNumActions();

      // nets are hardcoded for now as key (str) -> Mat
      // not proud of this. better solution is to have a whole Net object
      // on top of Mats, but for now sticking with this
      this.net = {};
      this.net.W1 = new R.RandMat(this.nh, this.ns, 0, 0.01);
      this.net.b1 = new R.Mat(this.nh, 1, 0, 0.01);
      this.net.W2 = new R.RandMat(this.na, this.nh, 0, 0.01);
      this.net.b2 = new R.Mat(this.na, 1, 0, 0.01);

      this.exp = []; // experience
      this.expi = 0; // where to insert

      this.t = 0;

      this.r0 = null;
      this.s0 = null;
      this.s1 = null;
      this.a0 = null;
      this.a1 = null;

      this.tderror = 0; // for visualization only...
    },
    toJSON: function() {
      // save function
      var j = {};
      j.nh = this.nh;
      j.ns = this.ns;
      j.na = this.na;
      j.net = R.netToJSON(this.net);
      return j;
    },
    fromJSON: function(j) {
      // load function
      this.nh = j.nh;
      this.ns = j.ns;
      this.na = j.na;
      this.net = R.netFromJSON(j.net);
    },
    forwardQ: function(net, s, needs_backprop) {
      var G = new R.Graph(needs_backprop);
      var a1mat = G.add(G.mul(net.W1, s), net.b1);
      var h1mat = G.tanh(a1mat);
      var a2mat = G.add(G.mul(net.W2, h1mat), net.b2);
      this.lastG = G; // back this up. Kind of hacky isn't it
      return a2mat;
    },
    act: function(slist) {
      // convert to a Mat column vector
      var s = new R.Mat(this.ns, 1);
      s.setFrom(slist);

      // epsilon greedy policy
      if(Math.random() < this.epsilon) {
        var a = randi(0, this.na);
      } else {
        // greedy wrt Q function
        var amat = this.forwardQ(this.net, s, false);
        var a = R.maxi(amat.w); // returns index of argmax action
      }

      // shift state memory
      this.s0 = this.s1;
      this.a0 = this.a1;
      this.s1 = s;
      this.a1 = a;

      return a;
    },
    learn: function(r1) {
      // perform an update on Q function
      if(!(this.r0 == null) && this.alpha > 0) {

        // learn from this tuple to get a sense of how "surprising" it is to the agent
        var tderror = this.learnFromTuple(this.s0, this.a0, this.r0, this.s1, this.a1);
        this.tderror = tderror; // a measure of surprise

        // decide if we should keep this experience in the replay
        if(this.t % this.experience_add_every === 0) {
          this.exp[this.expi] = [this.s0, this.a0, this.r0, this.s1, this.a1];
          this.expi += 1;
          if(this.expi > this.experience_size) { this.expi = 0; } // roll over when we run out
        }
        this.t += 1;

        // sample some additional experience from replay memory and learn from it
        for(var k=0;k<this.learning_steps_per_iteration;k++) {
          var ri = randi(0, this.exp.length); // todo: priority sweeps?
          var e = this.exp[ri];
          this.learnFromTuple(e[0], e[1], e[2], e[3], e[4])
        }
      }
      this.r0 = r1; // store for next update
    },
    learnFromTuple: function(s0, a0, r0, s1, a1) {
      // want: Q(s,a) = r + gamma * max_a' Q(s',a')

      // compute the target Q value
      var tmat = this.forwardQ(this.net, s1, false);
      var qmax = r0 + this.gamma * tmat.w[R.maxi(tmat.w)];

      // now predict
      var pred = this.forwardQ(this.net, s0, true);

      var tderror = pred.w[a0] - qmax;
      var clamp = this.tderror_clamp;
      if(Math.abs(tderror) > clamp) {  // huber loss to robustify
        if(tderror > clamp) tderror = clamp;
        if(tderror < -clamp) tderror = -clamp;
      }
      pred.dw[a0] = tderror;
      this.lastG.backward(); // compute gradients on net params

      // update net
      R.updateNet(this.net, this.alpha);
      return tderror;
    }
  }

// buggy implementation, doesnt work...
  var SimpleReinforceAgent = function(env, opt) {
    this.gamma = getopt(opt, 'gamma', 0.5); // future reward discount factor
    this.epsilon = getopt(opt, 'epsilon', 0.75); // for epsilon-greedy policy
    this.alpha = getopt(opt, 'alpha', 0.001); // actor net learning rate
    this.beta = getopt(opt, 'beta', 0.01); // baseline net learning rate
    this.env = env;
    this.reset();
  }
  SimpleReinforceAgent.prototype = {
    reset: function() {
      this.ns = this.env.getNumStates();
      this.na = this.env.getMaxNumActions();
      this.nh = 100; // number of hidden units
      this.nhb = 100; // and also in the baseline lstm

      this.actorNet = {};
      this.actorNet.W1 = new R.RandMat(this.nh, this.ns, 0, 0.01);
      this.actorNet.b1 = new R.Mat(this.nh, 1, 0, 0.01);
      this.actorNet.W2 = new R.RandMat(this.na, this.nh, 0, 0.1);
      this.actorNet.b2 = new R.Mat(this.na, 1, 0, 0.01);
      this.actorOutputs = [];
      this.actorGraphs = [];
      this.actorActions = []; // sampled ones

      this.rewardHistory = [];

      this.baselineNet = {};
      this.baselineNet.W1 = new R.RandMat(this.nhb, this.ns, 0, 0.01);
      this.baselineNet.b1 = new R.Mat(this.nhb, 1, 0, 0.01);
      this.baselineNet.W2 = new R.RandMat(this.na, this.nhb, 0, 0.01);
      this.baselineNet.b2 = new R.Mat(this.na, 1, 0, 0.01);
      this.baselineOutputs = [];
      this.baselineGraphs = [];

      this.t = 0;
    },
    forwardActor: function(s, needs_backprop) {
      var net = this.actorNet;
      var G = new R.Graph(needs_backprop);
      var a1mat = G.add(G.mul(net.W1, s), net.b1);
      var h1mat = G.tanh(a1mat);
      var a2mat = G.add(G.mul(net.W2, h1mat), net.b2);
      return {'a':a2mat, 'G':G}
    },
    forwardValue: function(s, needs_backprop) {
      var net = this.baselineNet;
      var G = new R.Graph(needs_backprop);
      var a1mat = G.add(G.mul(net.W1, s), net.b1);
      var h1mat = G.tanh(a1mat);
      var a2mat = G.add(G.mul(net.W2, h1mat), net.b2);
      return {'a':a2mat, 'G':G}
    },
    act: function(slist) {
      // convert to a Mat column vector
      var s = new R.Mat(this.ns, 1);
      s.setFrom(slist);

      // forward the actor to get action output
      var ans = this.forwardActor(s, true);
      var amat = ans.a;
      var ag = ans.G;
      this.actorOutputs.push(amat);
      this.actorGraphs.push(ag);

      // forward the baseline estimator
      var ans = this.forwardValue(s, true);
      var vmat = ans.a;
      var vg = ans.G;
      this.baselineOutputs.push(vmat);
      this.baselineGraphs.push(vg);

      // sample action from the stochastic gaussian policy
      var a = R.copyMat(amat);
      var gaussVar = 0.02;
      a.w[0] = R.randn(0, gaussVar);
      a.w[1] = R.randn(0, gaussVar);

      this.actorActions.push(a);

      // shift state memory
      this.s0 = this.s1;
      this.a0 = this.a1;
      this.s1 = s;
      this.a1 = a;

      return a;
    },
    learn: function(r1) {
      // perform an update on Q function
      this.rewardHistory.push(r1);
      var n = this.rewardHistory.length;
      var baselineMSE = 0.0;
      var nup = 100; // what chunk of experience to take
      var nuse = 80; // what chunk to update from
      if(n >= nup) {
        // lets learn and flush
        // first: compute the sample values at all points
        var vs = [];
        for(var t=0;t<nuse;t++) {
          var mul = 1;
          // compute the actual discounted reward for this time step
          var V = 0;
          for(var t2=t;t2<n;t2++) {
            V += mul * this.rewardHistory[t2];
            mul *= this.gamma;
            if(mul < 1e-5) { break; } // efficiency savings
          }
          // get the predicted baseline at this time step
          var b = this.baselineOutputs[t].w[0];
          for(var i=0;i<this.na;i++) {
            // [the action delta] * [the desirebility]
            var update = - (V - b) * (this.actorActions[t].w[i] - this.actorOutputs[t].w[i]);
            if(update > 0.1) { update = 0.1; }
            if(update < -0.1) { update = -0.1; }
            this.actorOutputs[t].dw[i] += update;
          }
          var update = - (V - b);
          if(update > 0.1) { update = 0.1; }
          if(update < 0.1) { update = -0.1; }
          this.baselineOutputs[t].dw[0] += update;
          baselineMSE += (V - b) * (V - b);
          vs.push(V);
        }
        baselineMSE /= nuse;
        // backprop all the things
        for(var t=0;t<nuse;t++) {
          this.actorGraphs[t].backward();
          this.baselineGraphs[t].backward();
        }
        R.updateNet(this.actorNet, this.alpha); // update actor network
        R.updateNet(this.baselineNet, this.beta); // update baseline network

        // flush
        this.actorOutputs = [];
        this.rewardHistory = [];
        this.actorActions = [];
        this.baselineOutputs = [];
        this.actorGraphs = [];
        this.baselineGraphs = [];

        this.tderror = baselineMSE;
      }
      this.t += 1;
      this.r0 = r1; // store for next update
    },
  }

// buggy implementation as well, doesn't work
  var RecurrentReinforceAgent = function(env, opt) {
    this.gamma = getopt(opt, 'gamma', 0.5); // future reward discount factor
    this.epsilon = getopt(opt, 'epsilon', 0.1); // for epsilon-greedy policy
    this.alpha = getopt(opt, 'alpha', 0.001); // actor net learning rate
    this.beta = getopt(opt, 'beta', 0.01); // baseline net learning rate
    this.env = env;
    this.reset();
  }
  RecurrentReinforceAgent.prototype = {
    reset: function() {
      this.ns = this.env.getNumStates();
      this.na = this.env.getMaxNumActions();
      this.nh = 40; // number of hidden units
      this.nhb = 40; // and also in the baseline lstm

      this.actorLSTM = R.initLSTM(this.ns, [this.nh], this.na);
      this.actorG = new R.Graph();
      this.actorPrev = null;
      this.actorOutputs = [];
      this.rewardHistory = [];
      this.actorActions = [];

      this.baselineLSTM = R.initLSTM(this.ns, [this.nhb], 1);
      this.baselineG = new R.Graph();
      this.baselinePrev = null;
      this.baselineOutputs = [];

      this.t = 0;

      this.r0 = null;
      this.s0 = null;
      this.s1 = null;
      this.a0 = null;
      this.a1 = null;
    },
    act: function(slist) {
      // convert to a Mat column vector
      var s = new R.Mat(this.ns, 1);
      s.setFrom(slist);

      // forward the LSTM to get action distribution
      var actorNext = R.forwardLSTM(this.actorG, this.actorLSTM, [this.nh], s, this.actorPrev);
      this.actorPrev = actorNext;
      var amat = actorNext.o;
      this.actorOutputs.push(amat);

      // forward the baseline LSTM
      var baselineNext = R.forwardLSTM(this.baselineG, this.baselineLSTM, [this.nhb], s, this.baselinePrev);
      this.baselinePrev = baselineNext;
      this.baselineOutputs.push(baselineNext.o);

      // sample action from actor policy
      var gaussVar = 0.05;
      var a = R.copyMat(amat);
      for(var i=0,n=a.w.length;i<n;i++) {
        a.w[0] += R.randn(0, gaussVar);
        a.w[1] += R.randn(0, gaussVar);
      }
      this.actorActions.push(a);

      // shift state memory
      this.s0 = this.s1;
      this.a0 = this.a1;
      this.s1 = s;
      this.a1 = a;
      return a;
    },
    learn: function(r1) {
      // perform an update on Q function
      this.rewardHistory.push(r1);
      var n = this.rewardHistory.length;
      var baselineMSE = 0.0;
      var nup = 100; // what chunk of experience to take
      var nuse = 80; // what chunk to also update
      if(n >= nup) {
        // lets learn and flush
        // first: compute the sample values at all points
        var vs = [];
        for(var t=0;t<nuse;t++) {
          var mul = 1;
          var V = 0;
          for(var t2=t;t2<n;t2++) {
            V += mul * this.rewardHistory[t2];
            mul *= this.gamma;
            if(mul < 1e-5) { break; } // efficiency savings
          }
          var b = this.baselineOutputs[t].w[0];
          // todo: take out the constants etc.
          for(var i=0;i<this.na;i++) {
            // [the action delta] * [the desirebility]
            var update = - (V - b) * (this.actorActions[t].w[i] - this.actorOutputs[t].w[i]);
            if(update > 0.1) { update = 0.1; }
            if(update < -0.1) { update = -0.1; }
            this.actorOutputs[t].dw[i] += update;
          }
          var update = - (V - b);
          if(update > 0.1) { update = 0.1; }
          if(update < 0.1) { update = -0.1; }
          this.baselineOutputs[t].dw[0] += update;
          baselineMSE += (V-b)*(V-b);
          vs.push(V);
        }
        baselineMSE /= nuse;
        this.actorG.backward(); // update params! woohoo!
        this.baselineG.backward();
        R.updateNet(this.actorLSTM, this.alpha); // update actor network
        R.updateNet(this.baselineLSTM, this.beta); // update baseline network

        // flush
        this.actorG = new R.Graph();
        this.actorPrev = null;
        this.actorOutputs = [];
        this.rewardHistory = [];
        this.actorActions = [];

        this.baselineG = new R.Graph();
        this.baselinePrev = null;
        this.baselineOutputs = [];

        this.tderror = baselineMSE;
      }
      this.t += 1;
      this.r0 = r1; // store for next update
    },
  }

// Currently buggy implementation, doesnt work
  var DeterministPG = function(env, opt) {
    this.gamma = getopt(opt, 'gamma', 0.5); // future reward discount factor
    this.epsilon = getopt(opt, 'epsilon', 0.5); // for epsilon-greedy policy
    this.alpha = getopt(opt, 'alpha', 0.001); // actor net learning rate
    this.beta = getopt(opt, 'beta', 0.01); // baseline net learning rate
    this.env = env;
    this.reset();
  }
  DeterministPG.prototype = {
    reset: function() {
      this.ns = this.env.getNumStates();
      this.na = this.env.getMaxNumActions();
      this.nh = 100; // number of hidden units

      // actor
      this.actorNet = {};
      this.actorNet.W1 = new R.RandMat(this.nh, this.ns, 0, 0.01);
      this.actorNet.b1 = new R.Mat(this.nh, 1, 0, 0.01);
      this.actorNet.W2 = new R.RandMat(this.na, this.ns, 0, 0.1);
      this.actorNet.b2 = new R.Mat(this.na, 1, 0, 0.01);
      this.ntheta = this.na*this.ns+this.na; // number of params in actor

      // critic
      this.criticw = new R.RandMat(1, this.ntheta, 0, 0.01); // row vector

      this.r0 = null;
      this.s0 = null;
      this.s1 = null;
      this.a0 = null;
      this.a1 = null;
      this.t = 0;
    },
    forwardActor: function(s, needs_backprop) {
      var net = this.actorNet;
      var G = new R.Graph(needs_backprop);
      var a1mat = G.add(G.mul(net.W1, s), net.b1);
      var h1mat = G.tanh(a1mat);
      var a2mat = G.add(G.mul(net.W2, h1mat), net.b2);
      return {'a':a2mat, 'G':G}
    },
    act: function(slist) {
      // convert to a Mat column vector
      var s = new R.Mat(this.ns, 1);
      s.setFrom(slist);

      // forward the actor to get action output
      var ans = this.forwardActor(s, false);
      var amat = ans.a;
      var ag = ans.G;

      // sample action from the stochastic gaussian policy
      var a = R.copyMat(amat);
      if(Math.random() < this.epsilon) {
        var gaussVar = 0.02;
        a.w[0] = R.randn(0, gaussVar);
        a.w[1] = R.randn(0, gaussVar);
      }
      var clamp = 0.25;
      if(a.w[0] > clamp) a.w[0] = clamp;
      if(a.w[0] < -clamp) a.w[0] = -clamp;
      if(a.w[1] > clamp) a.w[1] = clamp;
      if(a.w[1] < -clamp) a.w[1] = -clamp;

      // shift state memory
      this.s0 = this.s1;
      this.a0 = this.a1;
      this.s1 = s;
      this.a1 = a;

      return a;
    },
    utilJacobianAt: function(s) {
      var ujacobian = new R.Mat(this.ntheta, this.na);
      for(var a=0;a<this.na;a++) {
        R.netZeroGrads(this.actorNet);
        var ag = this.forwardActor(this.s0, true);
        ag.a.dw[a] = 1.0;
        ag.G.backward();
        var gflat = R.netFlattenGrads(this.actorNet);
        ujacobian.setColumn(gflat,a);
      }
      return ujacobian;
    },
    learn: function(r1) {
      // perform an update on Q function
      //this.rewardHistory.push(r1);
      if(!(this.r0 == null)) {
        var Gtmp = new R.Graph(false);
        // dpg update:
        // first compute the features psi:
        // the jacobian matrix of the actor for s
        var ujacobian0 = this.utilJacobianAt(this.s0);
        // now form the features \psi(s,a)
        var psi_sa0 = Gtmp.mul(ujacobian0, this.a0); // should be [this.ntheta x 1] "feature" vector
        var qw0 = Gtmp.mul(this.criticw, psi_sa0); // 1x1
        // now do the same thing because we need \psi(s_{t+1}, \mu\_\theta(s\_t{t+1}))
        var ujacobian1 = this.utilJacobianAt(this.s1);
        var ag = this.forwardActor(this.s1, false);
        var psi_sa1 = Gtmp.mul(ujacobian1, ag.a);
        var qw1 = Gtmp.mul(this.criticw, psi_sa1); // 1x1
        // get the td error finally
        var tderror = this.r0 + this.gamma * qw1.w[0] - qw0.w[0]; // lol
        if(tderror > 0.5) tderror = 0.5; // clamp
        if(tderror < -0.5) tderror = -0.5;
        this.tderror = tderror;

        // update actor policy with natural gradient
        var net = this.actorNet;
        var ix = 0;
        for(var p in net) {
          var mat = net[p];
          if(net.hasOwnProperty(p)){
            for(var i=0,n=mat.w.length;i<n;i++) {
              mat.w[i] += this.alpha * this.criticw.w[ix]; // natural gradient update
              ix+=1;
            }
          }
        }
        // update the critic parameters too
        for(var i=0;i<this.ntheta;i++) {
          var update = this.beta * tderror * psi_sa0.w[i];
          this.criticw.w[i] += update;
        }
      }
      this.r0 = r1; // store for next update
    },
  }

// exports
  global.DPAgent = DPAgent;
  global.TDAgent = TDAgent;
  global.DQNAgent = DQNAgent;
//global.SimpleReinforceAgent = SimpleReinforceAgent;
//global.RecurrentReinforceAgent = RecurrentReinforceAgent;
//global.DeterministPG = DeterministPG;
})(RL);

