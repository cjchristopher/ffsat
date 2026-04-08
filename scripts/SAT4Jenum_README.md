# SAT4J Enumerator Quick Guide

This helper script enumerates all satisfying assignments using Sat4j by:

1. Converting input to OPB/PBO when needed (`.hybrid`, `.cnf`, etc.)
2. Compiling/running a Java enumerator that uses Sat4j's incremental API
3. Iterating models in-process with `ModelIterator` until UNSAT

Script: `scripts/sat4j_enumerate.py`

Java enumerator source: `scripts/sat4j/Sat4jPbEnumerator.java`

## Basic Usage

```bash
python scripts/sat4j_enumerate.py /path/to/input.hybrid --sat4j-jar ~/path/to/sat4j/sat4j-pb.jar
```

## Download Sat4j

- Releases page: https://gitlab.ow2.org/sat4j/sat4j/-/releases/
- Project page: http://www.sat4j.org/

After downloading/extracting, pass the full path to `sat4j-pb.jar` via `--sat4j-jar`.

## Command Line Options

- `input_file` (positional): Input formula (`.hybrid`, `.cnf`, `.opb`, `.pbo`)
- `--sat4j-jar PATH` (required): Path to `sat4j-pb.jar`
- `--work-file PATH` (optional): OPB path used by Java enumerator (converted file for non-OPB input)
- `--max-solutions N` (optional): Stop after `N` models (`0` means no limit)
- `--verbose` (optional): Print each model as it is found
- `--rebuild-java` (optional): Force recompilation of Java enumerator

## Example

```bash
python scripts/sat4j_enumerate.py tests/4x4.hybrid \
  --sat4j-jar ~/sat/solvers/sat4j/sat4j-pb.jar \
  --verbose \
  --work-file /tmp/4x4_enum.opb
```

## Output

The script prints the final count as:

```text
Solutions found: N
```

If `--work-file` is provided, it also prints the final working OPB path.

## Notes

- For `.opb` / `.pbo` input, no conversion step is needed.
- For `.hybrid` / `.cnf` input, conversion is done via `scripts/hybrid_to_pbo.py`.
- Enumeration is performed in Java using Sat4j API classes (`PBInstanceReader`, `ModelIterator`) for incremental solving.

## Troubleshooting

- `Sat4j jar not found`: Check `--sat4j-jar` path and `~` expansion.
- `javac: command not found`: Install a JDK (not just JRE) so the Java enumerator can compile.
- `java: command not found`: Install Java and ensure `java` is on `PATH`.
- Java compile errors: verify `--sat4j-jar` points to `sat4j-pb.jar` compatible with `org.sat4j.pb` and `org.sat4j.core` APIs.
- `Invalid OPB header`: Ensure first line is in OPB style: `* #variable= N #constraint= M *`. Re-convert from source file if needed.
- Enumeration is too slow/large: Add `--max-solutions N` for a bounded run.
