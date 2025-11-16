"use strict";

MCMC.registerAlgorithm("RAHMC", {
  description: "Repelling-Attracting Hamiltonian Monte Carlo",

  about: () => {
    window.open("https://arxiv.org/abs/2403.04607v1");
  },

  init: (self) => {
    self.leapfrogSteps = 40;
    self.dt = 0.1;
    self.gamma = 0.5;
  },

  reset: (self) => {
    self.chain = [MultivariateNormal.getSample(self.dim)];
  },

  attachUI: (self, folder) => {
    folder.add(self, "leapfrogSteps", 5, 100).step(1).name("Leapfrog Steps");
    folder.add(self, "dt", 0.05, 0.5).step(0.025).name("Leapfrog &Delta;t");
    folder.add(self, "gamma", 0.1, 2.0).step(0.1).name("Friction &gamma;");
    folder.open();
  },

  step: (self, visualizer) => {
    const q0 = self.chain.last();
    const p0 = MultivariateNormal.getSample(self.dim);

    // conformal leapfrog integration with repelling-attracting friction
    const q = q0.copy();
    const p = p0.copy();
    const trajectory = [q.copy()];
    const phaseTrajectory = self.dim === 1 ? [{ q: q0[0], p: p0[0] }] : null;

    const L1 = Math.floor(self.leapfrogSteps / 2);
    const L2 = self.leapfrogSteps - L1;

    // helper function for conformal leapfrog step
    const conformalLeapfrogStep = (q, p, gamma) => {
      const scale = Math.exp(-gamma * self.dt / 2);
      // apply friction scaling (in-place)
      for (let i = 0; i < p.length; i++) p[i] *= scale;
      // half kick
      p.increment(self.gradLogDensity(q).scale(self.dt / 2));
      // drift
      q.increment(p.scale(self.dt));
      // half kick
      p.increment(self.gradLogDensity(q).scale(self.dt / 2));
      // apply friction scaling (in-place)
      for (let i = 0; i < p.length; i++) p[i] *= scale;
    };

    // repelling phase (negative friction)
    for (let i = 0; i < L1; i++) {
      conformalLeapfrogStep(q, p, -self.gamma);
      trajectory.push(q.copy());
      if (self.dim === 1) phaseTrajectory.push({ q: q[0], p: p[0] });
    }

    // attracting phase (positive friction)
    for (let i = 0; i < L2; i++) {
      conformalLeapfrogStep(q, p, self.gamma);
      trajectory.push(q.copy());
      if (self.dim === 1) phaseTrajectory.push({ q: q[0], p: p[0] });
    }

    // flip momentum (in-place)
    for (let i = 0; i < p.length; i++) p[i] *= -1;

    // add trajectory to visualizer animation queue
    const proposalEvent = {
      type: "proposal",
      proposal: q,
      trajectory: trajectory,
      initialMomentum: p0,
    };
    if (phaseTrajectory) {
      proposalEvent.phaseTrajectory = phaseTrajectory;
    }
    visualizer.queue.push(proposalEvent);

    // calculate acceptance ratio
    const H0 = -self.logDensity(q0) + p0.norm2() / 2;
    const H = -self.logDensity(q) + p.norm2() / 2;
    const logAcceptRatio = -H + H0;

    // accept or reject proposal
    if (Math.random() < Math.exp(logAcceptRatio)) {
      self.chain.push(q.copy());
      const acceptEvent = { type: "accept", proposal: q };
      if (phaseTrajectory) acceptEvent.phaseTrajectory = phaseTrajectory;
      visualizer.queue.push(acceptEvent);
    } else {
      self.chain.push(q0.copy());
      const rejectEvent = { type: "reject", proposal: q };
      if (phaseTrajectory) rejectEvent.phaseTrajectory = phaseTrajectory;
      visualizer.queue.push(rejectEvent);
    }
  },
});
