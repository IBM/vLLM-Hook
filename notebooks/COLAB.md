# Running the Colab Notebooks

Use the `_colab.ipynb` notebooks when running in Google Colab:

- [demo_attntracker_colab.ipynb](<path-to-directory>/vLLM-Hook/notebooks/demo_attntracker_colab.ipynb)
- [demo_corer_colab.ipynb](<path-to-directory>/vLLM-Hook/notebooks/demo_corer_colab.ipynb)
- [demo_actsteer_colab.ipynb](<path-to-directory>/vLLM-Hook/notebooks/demo_actsteer_colab.ipynb)

## Runtime Requirements

Use a GPU runtime in Colab.

In Colab:

1. Open `Runtime -> Change runtime type`
2. Set hardware accelerator to `GPU`
3. Start from a fresh runtime if you have already run setup cells with a different configuration

## Open the Notebook

Each Colab notebook includes an `Open in Colab` badge at the top.

If you open directly from GitHub, make sure you open the `_colab.ipynb` version, not the local notebook.

## What the Colab Setup Cell Does

The Colab install cell is designed to:

- detect Colab
- clone `https://github.com/tburleyinfo/vLLM-Hook.git`
- check out the `sandbox` branch
- install `requirement.txt`
- install `vllm_hook_plugins` in editable mode
- switch into the repo `notebooks/` directory

Re-run that cell after a runtime reset before running the rest of the notebook.

## Notebook Flow

For each Colab notebook:

1. Run the install/setup cell
2. Run the import and environment setup cells
3. Run the model/config selection cells
4. Run the example cells

Do not skip directly to later cells in a fresh runtime.

## Defaults

- The notebooks currently default to `ibm-granite/granite-4.0-micro`.
- If you change the model, make sure the config cell matches the selected model.

## Common Issues

- `NotImplementedError` from a helper function after you already edited the notebook
  - The runtime still has the old function loaded. Restart the runtime or re-run the helper-definition cell.

- Config mismatch during analysis
  - The selected config file does not match the model you loaded. Re-check the `model` cell and the config selection cell.

- Import errors after changing branches or cells
  - Restart the runtime and run the notebook from the top.

- `ValueError` about free memory being lower than `gpu_memory_utilization`
  - Free public Colab T4 runtimes can start with limited available GPU memory. The Colab notebooks use lower defaults (`gpu_memory_utilization=0.5` and `max_model_len=2048`) to reduce this, but repeated reruns can still leave less memory available.

- `demo_actsteer_colab.ipynb` can still fail on free Colab T4 with a startup-memory error during generation
  - This is a known limitation. The steering path builds a hooked vLLM engine after using the base engine, so constrained T4 sessions may temporarily need memory for both engine lifecycles in the same notebook run. This has been flagged for follow-up and is not yet changed in `HookLLM`.

- Slow or inconsistent behavior after repeated experimentation
  - Use `Runtime -> Restart session` and re-run all cells in order.
