"use strict";

MCMC.registerAlgorithm("HamiltonianMC", {
  description: "Hamiltonian Monte Carlo",

  about: () => {
    window.open("https://en.wikipedia.org/wiki/Hybrid_Monte_Carlo");
  },

  init: (self) => {
    self.leapfrogSteps = 37;
    self.dt = 0.1;
  },

  reset: (self) => {
    const initialSample = MultivariateNormal.getSample(self.dim);
    self.chain = [initialSample];
  },

  attachUI: (self, folder) => {
    folder.add(self, "leapfrogSteps", 5, 100).step(1).name("Leapfrog Steps");
    folder.add(self, "dt", 0.05, 0.5).step(0.025).name("Leapfrog &Delta;t");
    folder.open();
  },

  step: (self, visualizer) => {
    const q0 = self.chain.last();
    const p0 = MultivariateNormal.getSample(self.dim);

    // use leapfrog integration to find proposal
    const q = q0.copy();
    const p = p0.copy();
    const trajectory = [q.copy()];
    const phaseTrajectory = self.dim === 1 ? [{ q: q0[0], p: p0[0] }] : null;

    for (let i = 0; i < self.leapfrogSteps; i++) {
      p.increment(self.gradLogDensity(q).scale(self.dt / 2));
      q.increment(p.scale(self.dt));
      p.increment(self.gradLogDensity(q).scale(self.dt / 2));
      trajectory.push(q.copy());

      if (self.dim === 1) {
        phaseTrajectory.push({ q: q[0], p: p[0] });
      }
    }

    // add integrated trajectory to visualizer animation queue
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
