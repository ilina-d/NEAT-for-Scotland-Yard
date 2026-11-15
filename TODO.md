# TODO
- Make an evaluator to score the fitness of individuals.
- Make a class that implements crossover and mutations.
- Update visuals so that wait_after_move input is handled better when user is playing.

---

# Notes & Ideas

---

# Latest Changes
Initial feature modeling and FFN player implementation.
- Made an `FFNPlayer` that takes a given FFN and uses it to make moves.
- Made FeedForwardNet pickleable.
- Modeled initial input features for X and Detective players.
- Added methods to extract input features for the FFNs in `Graph`.
- Fixed minor spelling mistakes base GNN nets.
- Simulator now propagates error when crashing.
- Removed internal neat classes from init to avoid circular import when running a game or sims in main.
- Updated TODO.