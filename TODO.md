# TODO
- Update visuals so that wait_after_move input is handled better when user is playing.

---

# Notes & Ideas


---

# Latest Changes
Implemented crossover and FitnessEvaluator.
- Implemented `FitnessEvaluator` to score individuals based on their win rate.
- Added helper functions for mutations in `LinkGene`, `NeuronGene` and `Genome`.
- Added fitness to the stored json data for the Genomes.
- Implemented `OffspringGenerator` that handles mating individuals and offspring mutation.
- Small changes in simulator to store win rates and added a toggle for printing.
- Using double is no longer legal if X has less than 2 steps remaining.
- Updated TODO.