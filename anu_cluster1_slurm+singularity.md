
# ANU Cluster1 Instructions
Please see Cluster1 instruction documentation for usage on the Slurm job management system and file systems.

## Quickstart

Launch a job on a GPU enabled parition of cluster1.
```bash
srun -t 01:00:00 -w vesuvius -p planopt --qos=planopt --gres=gpu:1 --pty /bin/bash
```
This requests an interactive shell for 1 hour, but **prevents other users from using the machine if they request resources (GPUs) you have been allocated**. Therefore it is suggested to only use an interactive shell for short term testing and debugging. Larger jobs should be run via `sbatch` - *please see the Cluster1 and Slurm guides.*

N.B:
- `vesuvius` or `stromboli` can be passed to `-w`
- On `vesuvius`, there are 2 GPUs available, so either `1` or `2` can be passed to `--gres=gpu:`. Only 1 is available on `stromboli`

### Clone and Launch Singularity Environment
Clone the codebase and load a compatible (e.g. CUDA version) container. A container for CUDA12 with JAX 0.9.0 is provided at `/home/projects/accelerated_sat/jax0.9-cuda12.9-runtime.sif`. 

All remaining instructions should be performed in the singularity shell.

```bash
git clone https://github.com/cjchristopher/accelerated-fourier-sat.git
singularity shell -nv --bind /home/projects/ /home/projects/accelerated_sat/jax_cuda12.sif
cd accelerated-fourier-sat
TF_CPP_MIN_LOG_LEVEL=2 python3 afsat.py [...]
```
(n.b. we strongly suggest setting TF_CPP_MIN_LOG_LEVEL=2 to suppress warnings in the latest XLA. This can be set longer term with `export TF_CPP_MIN_LOG_LEVEL=2` or adding similar lines to `.bashrc` or `os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")` in `afsat.py`)