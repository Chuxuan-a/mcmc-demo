# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Interactive browser-based visualization gallery for Markov Chain Monte Carlo algorithms. Pure JavaScript - no build process, no package.json. Open `index.html` or `app.html` directly in a browser.

## Core Architecture

### Event-Driven Visualization Pipeline

The system uses a three-stage pipeline:

1. **Algorithm** (`algorithms/*.js`) → generates MCMC samples, pushes visualization events to queue
2. **Visualizer** (`main/Visualizer.js`) → consumes events from queue, renders to offscreen canvases
3. **Simulation** (`main/Simulation.js`) → orchestrates animation loop, timing, and coordination

**Key pattern**: Algorithms don't render directly. They push events to `visualizer.queue`, which are processed asynchronously by `Visualizer.dequeue()`.

### Algorithm Registration System

Algorithms self-register via `MCMC.registerAlgorithm(name, methods)` where methods must include:
- `init(self)` - initialize parameters
- `reset(self)` - reset the Markov chain
- `step(self, visualizer)` - execute one iteration, push events to `visualizer.queue`
- `attachUI(self, folder)` - add dat.GUI controls

The `self` parameter is `sim.mcmc`, which provides:
- `logDensity(x)`, `gradLogDensity(x)`, `hessLogDensity(x)` - target distribution
- `chain` - array of accepted samples
- `dim` - always 2 for this 2D visualization

### Visualization Event Types

Common events pushed to `visualizer.queue`:
- `{type: "proposal", proposal: vector, ...}` - optional fields: `trajectory`, `initialMomentum`, `gradient`, `proposalCov`
- `{type: "accept", proposal: vector}`
- `{type: "reject", proposal: vector}`

The visualizer composites three offscreen canvases: density (contours + heatmap), samples (accepted points), overlay (current proposal/trajectory).

### Linear Algebra

Custom library using Float64Array. Vectors are 1D arrays, matrices are 1D arrays in column-major order. Key methods: `.add()`, `.subtract()`, `.scale()`, `.norm()`, `.chol()`. See `lib/linalg.core.js`.

## Adding a New Algorithm

1. Create `algorithms/YourAlgorithm.js` following the structure in `algorithms/HamiltonianMC.js`
2. Add `<script src="algorithms/YourAlgorithm.js"></script>` to `app.html` (before `</body>`)
3. Registration happens automatically when the script loads
4. Add link to `README.md` and `index.html`

## URL Parameters

Control behavior via query params: `app.html?algorithm=HamiltonianMC&target=banana&delay=100`

Available: `algorithm`, `target`, `seed`, `delay`, `autoplay`, `animateProposal`, `showSamples`, `showHistograms`

## Code Style

Prettier config in `.prettierrc`: 120 char width, 2 spaces, semicolons, double quotes.
