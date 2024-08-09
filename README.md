# FrameRL

A novel Deep Reinforcement Learning model for aligning DNA sequences with their corresponding reference protein sequences, allowing for substitutions, frameshifts, insertions, and deletions. Implemented in Python `3.9` and Tensorflow `2.16.1`.

# Project Structure

- `requirments.txt` - contains the dependencies for running FrameRL
- `Dockerfile` and `docker-compose.yaml` - manifests for running a containerized development environment using Docker
- `learning` - contains the tools and scripts used to train and evaluate FrameRL

  - `data` - training data
  - `fasta_tests` - agent correctness test result logs
  - `models` - FrameRL v1 model that does not distinguish between `insertion` and `deletion`
  - `models_v2` - FrameRL v2 model that separates `indel` action into `insertion` and `deletion`
  - `old_tests` - FrameRL v1 test outputs
  - `results` FrameRL v2 test outputs
  - `utils` - directory containing utility scripts, including implementation of Zhang's Three Frame

# Authors

- Proponents
  - Ayuyao, Justin - justin_ayuyao@dlsu.edu.ph
  - Cruz, Renz Ezekiel - renz_cruz@dlsu.edu.ph
  - Joyo, John Carlo - john_carlo_joyo@dlsu.edu.ph
  - Li, Wai Kei - wai_kei_li@dlsu.edu.ph
- Adviser
  - Uy, Roger Luis - roger.uy@dlsu.edu.ph
