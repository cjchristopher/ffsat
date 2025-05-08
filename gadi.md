# Gadi Instructions
## Required Modules
It is suggested that `module load python3 python3-as-python nvidia-hpc-sdk` be added somewhere that they are automatically loaded (either with the python environment in `/path/to/env/bin/activate`, in `.profile`/`.bashrc`/etc, in the jobscript if invoking non-interactively, or follow [these instructions](https://opus.nci.org.au/spaces/Help/pages/236881064/Software+Applications+Guide...#SoftwareApplicationsGuide...-Userdefinedmodules) to define a custom module for this project).
At time of commit, this loads:
- python3 (defaults to 3.12.1)
- python3-as-python
- nvidia-hpc-sdk (defaults to 25.1)

## Setup python environment & code
We configure the environment and repository such that it is accessible during the job. 
This can either be in `$HOME` or in `/g/data/<a00>/<uid000>` (a00=project_id, uid000=your_gadi_login, or some other suitable folder).
- `> python -m venv /path/to/envs/ffsat`
- `> pip install -U jax[cuda12] --install-suggests --install-recommends`
- `> pip install -U -r min_requirements.txt --install-suggests --install-recommends`
- Clone the repository to preferred location

## Request Resources
- Example invokation: `> qsub -I -q gpuvolta -P <a00> -l walltime=00:15:00,ncpus=24,ngpus=2,mem=20GB,jobfs=5GB,storage=gdata/<a00>,wd -N ffsat` (15 minutes interactive, 2 gpus, 20gb RAM)
- Remove `-I` for submitted job mode and provide an appropriate `.sh` file
- `ncpus` must be 12x `ngpus` at a minimum. At least 1 gpu is required for ffsat presently.
- Set other parameters as required - `storage=gdata/a00` makes `/g/data/a00/` available to the job node.
- Load appropriate modules (see above, check with `module list`)
- Load environment (`> source /path/to/env/bin/activate`)
- Change to code direction (`> cd /path/to/ffsat/`)
- Run as normal (`> python ff_shard_test.py -c -b 1024 -r 10 -t 300 -m 3 <some_input_file>`)