# Continuous integration

`uv.lock` is committed and checked with uv 0.8.0. Update it with `uv lock` when changing dependencies.

- **Quality** runs every pre-commit hook against all tracked files. Only the generated `uv.lock` is exempt from the 1 MB file-size limit; data and model files remain subject to the limit.
- **Fast Tests** runs the complete `tests/` directory across base, dataset, hardware, visualization, and training/Flow/DiT dependency tiers. Tests that require an absent extra skip only in those reduced environments. The final tier installs the training and MultiTaskDiT extras.
- **Full Tests** installs this fork's training, MultiTaskDiT, annotation, and test dependencies, runs the complete test suite, and exercises the annotation pipeline end to end with a synthetic dataset and stub VLM. The upstream all-model installation and externally hosted dataset/model E2E suite remain enabled in `huggingface/lerobot`.
- **Benchmark Integration Tests** requires a GPU runner group. Forks opt in by setting the repository Actions variable `GPU_RUNNER_GROUP` to a runner group available to the repository. Without it, these jobs skip. Hugging Face keeps its existing default runner group. Model evaluation also requires the workflow's existing Hugging Face credentials.
- **Security** continues to scan for verified leaked secrets.

To reproduce the complete local checks:

```bash
uv sync --locked --extra test --extra training --extra multi_task_dit --extra annotations --extra dev
uv run pre-commit run --all-files
uv run pytest tests -q
uv run python -m tests.annotations.run_e2e_smoke
```
