# FFSAT: Fast Fourier SAT Solver

FFSAT is a native pseudoboolean Continuous Local Search (CLS) SAT solver that leverages recent developments in closed-form transformations of various pseudoboolean constraint types into continuous multilinear polynomial (which are linear combinations of Elementary Symmetric Polynomials (ESPs) in number of unique variables of the clause). The solver takes advantage of modern automatic differentiation frameworks (JAX) for efficient optimization and can utilize multiple GPUs for fast evaluation of the ESPs via a Discrete Fourier Transform. The work that underpins this can be found at [^1][^2], and this repository reuses some code from the latter (heavily reworked).

## Overview

This solver transforms traditional discrete SAT problems into continuous optimization problems that can be solved using gradient-based methods.

## Installation

### Prerequisites

- Python >=3.10+
- For JAX acceleration with NVIDIA:
  - CUDA >=12.1
  - CUDNN >=9.1, <10.0
  - NCCL >=2.18
  - NVIDIA Driver >=525.60.13
- See [JAX Installation](https://docs.jax.dev/en/latest/installation.html#installation) for more details, or CPU/TPU requirements (untested).

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/username/ffsat_imp.git
   cd ffsat_imp
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install JAX

   See [JAX Installation](https://docs.jax.dev/en/latest/installation.html#installation) for instructions.

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dependencies

- JAX and related packages (jaxopt)
- NumPy
- tqdm
- Custom modules included in the repository

A complete `requirements.txt` file will be provided in a future update.

## Input Format

The solver accepts files in standard DIMACS CNF format as well as an extended hybrid format that supports various constraint types.

### Standard DIMACS CNF Format

- Comment lines starting with 'c' or '*'
- A problem line starting with 'p cnf' followed by the number of variables and clauses
- Clauses represented as space-separated integers ending with '0'
- Positive integers represent positive literals, negative integers represent negative literals
- Example: `1 -2 3 0` represents (x₁ ∨ ¬x₂ ∨ x₃)

### Hybrid DIMACS Format

This solver extends the DIMACS format to support various constraint types using the following syntax:

1. Constraint lines either:
   - Start with `h` (hybrid marker), e.g., `h 1 2 0`
   - Start with constraint type directly, e.g., `xor 1 2 0`
   - For standard CNF clauses, no prefix is needed, e.g., `1 2 0`

2. Supported constraint types:

   - **CNF clauses**: Standard disjunction of literals
     ```
     1 -2 3 0` or `h 1 -2 3 0
     ```

   - **XOR constraints**: Exactly one variable must be true
     ```
     xor 1 2 3 0 or h x 1 2 3 0 or x 1 2 3 0
     ```

   - **NAE (Not-All-Equal) constraints**: Not all literals can have the same value
     ```
     nae 1 2 3 0` or `h n 1 2 3 0` or `n 1 2 3 0
     ```

   - **EO (Exactly-One) constraints**: Exactly one literal must be true
     ```
     eo 1 2 3 0` or `h e 1 2 3 0` or `e 1 2 3 0
     ```

   - **AMO (At-Most-One) constraints**: At most one literal can be true
     ```
     amo 1 2 3 0` or `h a 1 2 3 0` or `a 1 2 3 0
     ```

   - **Cardinality constraints**: Specify how many literals in a set must be true
     ```
     card <k> <literals> 0` or `h d <k> <literals> 0` or `d <k> <literals> 0
     ```
     
     Where `<k>` can be:
     - A simple integer: `card 2 1 2 3 0` (at least 2 of the variables must be true)
     - A negative integer: `card -2 1 2 3 0` (fewer than 2 variables must be true)
     - With inequality prefix:
       - `card >=2 1 2 3 0` (at least 2 variables must be true)
       - `card >2 1 2 3 0` (more than 2 variables must be true)
       - `card <2 1 2 3 0` (fewer than 2 variables must be true)
       - `card <=2 1 2 3 0` (at most 2 variables must be true)

The parser is flexible with shorthand notations for constraint types, allowing for both single-character identifiers (`x`, `n`, `e`, `a`, `d`) and full names (`xor`, `nae`, `eo`, `amo`, `card`).

## Usage

Run the solver with:

```bash
python ff_shard_test.py input_file.cnf [options]
```

### Command Line Options

- `-m, --mode INT`: Clause partitioning mode (default: 0, prompts after reading input)
- `-t, --timeout INT`: Maximum runtime in seconds (default: 300)
- `-b, --batch INT`: Batch size per GPU (default: 16)
- `-r, --restart INT`: Points to test before adjusting weights and restarting (default: 0, no restart)
- `-f, --fuzz INT`: Number of fuzzing attempts per batch (default: 0)
- `-v, --vertex`: Start optimization near vertices
- `-c, --combine`: Optimize batch points with a single optimizer call
- `-p, --profile`: Enable profiling

### Clause Partitioning Modes

The `-m, --mode` parameter controls how clauses are partitioned for processing:

- **Mode 1 (Full combine - 1 group)**: Uses a single monolithic array with all clauses appropriately padded. If clause lengths are varied then this produces the most memory overhead, but may speed up optimisation.

- **Mode 2 (By type - at most 6 groups)**: Creates separate padded arrays for each clause type. Has similar drawbacks to the above, in that wide variation in the clause length may cause a a high memory overhead for padding

- **Mode 3 (By length - no limit)**: Creates separate arrays for each clause length. For memory constrained environments, this trades off some optimisation efficiency for zero additional memory requirement above baseline. In future a variation threshold that allows groups clause lengths to a mode within some range (for minimal padding) may be added.

If mode is not specified (or set to 0), the solver will analyze the input file and prompt you to select a partitioning strategy.

### Examples

Basic usage:
```bash
python ff_shard_test.py problem.cnf
```

Running with a 10-minute timeout and larger batch size:
```bash
python ff_shard_test.py problem.cnf -t 600 -b 32
```

Enable vertex initialization and fuzzing:
```bash
python ff_shard_test.py problem.cnf -v -f 5
```

Use type-based partitioning for a mixed constraint problem:
```bash
python ff_shard_test.py problem.cnf -m 2 -t 600
```

## Output

The solver outputs:
- Progress information during the search
- SAT/UNSAT result
- Variable assignments for satisfiable instances in DIMACS format (lines starting with "v")
- Timing and performance statistics

## Technical Details

The solver uses JAX for hardware acceleration and automatic differentiation. It transforms boolean satisfiability constraints into continuous functions that can be minimized using gradient-based methods, with techniques including:

- Fast Fourier Transform-based representations
- Projected gradient descent optimization
- Multi-GPU parallelism through JAX sharding
- Weight adjustments based on clause difficulty

## License

[License information will be added here]

## Citation

[1] A. Kyrillidis, A. Shrivastava, M. Y. Vardi, and Z. Zhang, ‘Solving hybrid Boolean constraints in continuous space via multilinear Fourier expansions’, Artificial Intelligence, vol. 299. Elsevier BV, p. 103559, Oct. 2021. doi: 10.1016/j.artint.2021.103559. Available: [!http://dx.doi.org/10.1016/j.artint.2021.103559 

[![DOI:10.1016/j.artint.2021.103559](https://zenodo.org/badge/DOI/10.1016/j.artint.2021.103559207-4_15.svg)](https://doi.org/10.1016/j.artint.2021.103559)