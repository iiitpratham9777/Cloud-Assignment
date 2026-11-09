# AI use declaration

> Working-copy template for Pratham Gupta (Roll no. 230103034). Edit this declaration to describe only the tools you actually used and the checks you personally performed. The original project author is Vivek Sagarwal.

## Tools
- **Claude (Anthropic)**, chat interface: used to read the assignment, scaffold the repository (service, worker,
  cleaning, training script, Kubernetes manifests, scripts, CI workflow, tests), compute the Task 5 subnet table and
  NACL, and draft the README, report skeleton and reflection outline.
- (Add any other tool you really used, e.g. GitHub Copilot, ChatGPT, here. Remove this line if none.)

## What the AI did not do
- It did not run the teacher, training, Docker, Kubernetes or Locust steps: those need your own machine. Every number in
  `results/` and the report must come from my own runs.
- It did not write the handwritten reflection.

## How I checked the output
- Ran `python -m pytest` and the provided `tools/nacl_sim.py --check` and `tools/score.py` myself.
- Read the code, ran `make train SUBSET=50`, built the image, and compared behaviour with the assignment text
  (endpoints, headers, error body, `FAULT_INJECT`, NACL rule order).
- Recomputed the subnet table by hand: 2^10-2=1022 ≥ 1000 → /22, 2^8-2=254 → /24, 2^7-2=126 → /25, 2^5-2=30 → /27.
- Did not trust any metric that I did not produce myself; scanned the repo with gitleaks (CI) for secrets.
- (Add what you actually verified and what AI suggestions you rejected or fixed.)

## Portability refactor (added later)
- **Claude (Anthropic)**, chat interface with a code sandbox: used to make the project portable (settings/`.env`, local mode, setup
  and check scripts, dataset and weights tooling, requirements, tests, README/CONTRIBUTING/attribution/publishing docs).
  Details and the exact verification record are in `docs/CHANGES.md`.
- It could not run the real Pythia-70M training, Ollama, Docker, Kubernetes, Windows/macOS or GitHub Actions; those remain for
  the person who publishes the project to run and record honestly.
- (Edit this section to say what *you* reviewed, ran and changed.)

