"use strict";

MCMC.registerAlgorithm("RAHMC-Sigmoid", {
  description: "RA-HMC with Sigmoid Friction Schedule",

  about: () => {
    window.open("https://arxiv.org/abs/2403.04607v1");
  },

  init: (self) => {
    self.leapfrogSteps = 40;
    self.dt = 0.1;
    self.gamma = 0.5;
    self.steepness = 10.0;
  },

  reset: (self) => {
    self.chain = [MultivariateNormal.getSample(self.dim)];
  },

  attachUI: (self, folder) => {
    folder.add(self, "leapfrogSteps", 5, 100).step(1).name("Leapfrog Steps");
    folder.add(self, "dt", 0.05, 0.5).step(0.025).name("Leapfrog &Delta;t");
    folder.add(self, "gamma", 0.1, 2.0).step(0.1).name("Friction &gamma;");
    folder.add(self, "steepness", 1.0, 20.0).step(0.5).name("Steepness");
    folder.open();
  },

  step: (self, visualizer) => {
    const q0 = self.chain.last();
    const p0 = MultivariateNormal.getSample(self.dim);

    // conformal leapfrog integration with sigmoid-scheduled friction
    const q = q0.copy();
    const p = p0.copy();
    const trajectory = [q.copy()];

    const totalTime = self.dt * self.leapfrogSteps;

    // helper function for conformal leapfrog step with time-dependent friction
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

    // sigmoid friction schedule: gamma(t) = gamma_max * (2/(1 + exp(-steepness*(t/T - 0.5))) - 1)
    for (let step = 0; step < self.leapfrogSteps; step++) {
      const currentTime = step * self.dt;
      const normalizedTime = self.steepness * (currentTime / totalTime - 0.5);
      const gamma_t = self.gamma * (2.0 / (1.0 + Math.exp(-normalizedTime)) - 1.0);

      conformalLeapfrogStep(q, p, gamma_t);
      trajectory.push(q.copy());
    }

    // flip momentum (in-place)
    for (let i = 0; i < p.length; i++) p[i] *= -1;

    // add trajectory to visualizer animation queue
    visualizer.queue.push({
      type: "proposal",
      proposal: q,
      trajectory: trajectory,
      initialMomentum: p0,
    });

    // calculate acceptance ratio
    const H0 = -self.logDensity(q0) + p0.norm2() / 2;
    const H = -self.logDensity(q) + p.norm2() / 2;
    const logAcceptRatio = -H + H0;

    // accept or reject proposal
    if (Math.random() < Math.exp(logAcceptRatio)) {
      self.chain.push(q.copy());
      visualizer.queue.push({ type: "accept", proposal: q });
    } else {
      self.chain.push(q0.copy());
      visualizer.queue.push({ type: "reject", proposal: q });
    }
  },
});
