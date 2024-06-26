# Bug fixes

- [x] Fix speed too high on first timstep due to last_timestep_t = 0.0 on startup

# Run test config

- [x] Add cli arg parser
- [x] Reduce total of timesteps per epoch from 3600 to 2800

# Reward function

- [x] Add single penalty on the robot falling (ex: -10)
- [x] Add continuos small penalty for staying on the ground (ex: -0.1)

# Logs

- [x] Create a "run*test*{datetime}" folder for each run test, which includes:
  - run test quick note
  - create empty run test (.txt) description
  - logs
  - checkpoints
  - alpha.log
  - losses
- [x] Add average score to logs

# Remote Object Invocation

- [ ] Add pyro4 for remote object invocation
