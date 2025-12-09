
# Gadi Instructions

## Required Modules
It is suggested that the following modules be loaded automatically (e.g., within the python environment activation script in `/path/to/env/bin/activate`, in the jobscript, or by defining a custom module).

**Command:**
```bash
module load python3 python3-as-python nvidia-hpc-sdk cuda
```

*For more on custom modules, see the [NCI Software Applications Guide](https://opus.nci.org.au/display/Help/Software+Applications+Guide).*

## Setup python environment & code
We configure the environment and repository such that it is accessible during the job. This can either be in `$HOME` or in `/g/data/<a00>/<uid000>` (where `a00`=project_id, `uid000`=your_gadi_login).

```bash
# Create the environment
python -m venv /path/to/envs/ffsat

# Activate the environment
source /path/to/envs/ffsat/bin/activate

# Install dependencies
pip install -U jax[cuda12]
pip install -U -r min_requirements.txt

# Clone the repository to preferred location (example)
git clone https://github.com/your-repo/ffsat.git
```

## Request Resources

### Interactive Job Example
```bash
qsub -I -q gpuvolta -P <a00> -l walltime=00:15:00,ncpus=24,ngpus=2,mem=20GB,jobfs=5GB,storage=gdata/<a00>,wd -N ffsat
```
*(15 minutes interactive, 24 CPUs, 2 GPUs, 20GB RAM)*

### Notes on arguments:
- Remove `-I` for submitted job mode (and provide an appropriate `.sh` file).
- `ncpus`: Must be **12x `ngpus`** at a minimum on the `gpuvolta` queue.
- `ngpus`: At least 1 GPU is required for `ffsat`.
- `storage`: `storage=gdata/<a00>` makes `/g/data/a00/` available to the job node. Ensure you replace `<a00>` with your project ID.

### Running the code
1. Load appropriate modules (check with `module list`).
2. Load environment:
   ```bash
   source /path/to/env/bin/activate
   ```
3. Change to code directory:
   ```bash
   cd /path/to/ffsat/
   ```
4. Run as normal:
   ```bash 
      ffsat.py <problem_file>
      ```
   python ff_shard_test.py -c -b 1024 -r 10 -t 300 -m 3 <some_input_file>
   ```
