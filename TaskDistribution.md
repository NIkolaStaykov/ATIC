# ATIC Coding Tasks, Progress and Planning
This document focuses specifically the coding part of the assignment. Paper-writing tasks are not included here.

## Current Status
The following components are currently implemented:
1. Skeleton code for the data generation.
   - The model parameters are randomly initalized and with no constraints.
   - The propagation expression should be logically correct.
2. Hydra config integration.
3. Simple data-saving mechanism.

## Tasks

The tasks are organized in categories based on their  and within each category, they are ordered by priority.

### Data Generation:
- [ ] **design** Decide what part (if any) of the model params should be fixed in the config.
- [ ] **impl** Initial state generation with model parameter constraints.
- [ ] **impl** Integrate noise generation into the adversarial input.
- [ ] **impl** Seeding for reproducibility.

### Infrastructure:
- [ ] **design** Define what we want to be logged during the experiments.
  - Some suggestions:
    - model parameters (Do we need a time series or are we keeping it constant?)
    - the user opinions
    - the adversarial inputs (essentially 1-noise at each step)
    - the controller inputs
    - the OFO estimations
    - maybe execution time
- [ ] **impl** Implement the logging mechanism for the defined statistics.
- [ ] **design** Define what results we want to aggregate:
  - Graphs
  - Numerical values (mean error for Kalman filter for example)
- [ ] **impl** Write visualization scripts and integrate them with the logging system.

### Online Feedback Optimization:
This section is related to the core of the paper and the experiments. The data generation is taken as a given. Parts of the generated data is hidden from the POV of the controller (everything except the user opinons).

- [ ] **impl** Integrate a Kalman Filter for sensitivity estimation.
  Choose one of the two options:
    - **Option 1**: Use the Kalman filter from the `pykalman` library.
    - **Option 2**: Test and integrate Leo's implementation.
- [ ] **impl** Implement the FOH gradient estimation.

