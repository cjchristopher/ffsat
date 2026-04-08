// SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later

import java.io.IOException;

import org.sat4j.core.VecInt;
import org.sat4j.pb.IPBSolver;
import org.sat4j.pb.SolverFactory;
import org.sat4j.pb.reader.PBInstanceReader;
import org.sat4j.reader.ParseFormatException;
import org.sat4j.specs.ContradictionException;
import org.sat4j.specs.TimeoutException;

public final class Sat4jPbEnumerator {

    private Sat4jPbEnumerator() {
    }

    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println(
                "Usage: java Sat4jPbEnumerator <opb_file> [--max-solutions N] [--orig-vars N] [--verbose]"
            );
            System.exit(2);
        }

        String opbPath = args[0];
        long maxSolutions = 0;
        int origVars = 0;
        boolean verbose = false;

        for (int i = 1; i < args.length; i++) {
            if ("--verbose".equals(args[i])) {
                verbose = true;
                continue;
            }
            if ("--max-solutions".equals(args[i])) {
                if (i + 1 >= args.length) {
                    System.err.println("--max-solutions requires an integer argument");
                    System.exit(2);
                }
                i++;
                try {
                    maxSolutions = Long.parseLong(args[i]);
                } catch (NumberFormatException ex) {
                    System.err.println("Invalid --max-solutions value: " + args[i]);
                    System.exit(2);
                }
                if (maxSolutions < 0) {
                    System.err.println("--max-solutions must be >= 0");
                    System.exit(2);
                }
                continue;
            }
            if ("--orig-vars".equals(args[i])) {
                if (i + 1 >= args.length) {
                    System.err.println("--orig-vars requires an integer argument");
                    System.exit(2);
                }
                i++;
                try {
                    origVars = Integer.parseInt(args[i]);
                } catch (NumberFormatException ex) {
                    System.err.println("Invalid --orig-vars value: " + args[i]);
                    System.exit(2);
                }
                if (origVars < 0) {
                    System.err.println("--orig-vars must be >= 0");
                    System.exit(2);
                }
                continue;
            }

            System.err.println("Unknown argument: " + args[i]);
            System.exit(2);
        }

        try {
            IPBSolver base = SolverFactory.newDefault();
            PBInstanceReader reader = new PBInstanceReader(base);
            reader.parseInstance(opbPath);
            int nVars = base.nVars();
            int varsToCount = origVars > 0 ? origVars : nVars;
            if (varsToCount > nVars) {
                System.err.println(
                    "--orig-vars (" + varsToCount + ") exceeds number of variables in instance (" + nVars + ")"
                );
                System.exit(2);
            }

            long count = 0;
            while (base.isSatisfiable()) {
                int[] model = base.model();
                count++;

                boolean[] assignment = new boolean[nVars + 1];
                boolean[] assigned = new boolean[nVars + 1];
                for (int lit : model) {
                    int var = Math.abs(lit);
                    if (var >= 1 && var <= nVars) {
                        assignment[var] = lit > 0;
                        assigned[var] = true;
                    }
                }

                if (verbose) {
                    // Previous verbose formatter kept for quick rollback/reference:
                    // StringBuilder sb = new StringBuilder();
                    // sb.append("Solution ").append(count).append(": ");
                    // if (origVars > 0 && varsToCount < nVars) {
                    //     sb.append("orig[");
                    //     for (int var = 1; var <= varsToCount; var++) {
                    //         sb.append("x")
                    //           .append(var)
                    //           .append("=")
                    //           .append(assigned[var] ? (assignment[var] ? "1" : "0") : "?");
                    //         if (var < varsToCount) {
                    //             sb.append(" ");
                    //         }
                    //     }
                    //     sb.append("] aux[");
                    //     for (int var = varsToCount + 1; var <= nVars; var++) {
                    //         sb.append("x")
                    //           .append(var)
                    //           .append("=")
                    //           .append(assigned[var] ? (assignment[var] ? "1" : "0") : "?");
                    //         if (var < nVars) {
                    //             sb.append(" ");
                    //         }
                    //     }
                    //     sb.append("]");
                    // } else {
                    //     for (int var = 1; var <= nVars; var++) {
                    //         sb.append("x")
                    //           .append(var)
                    //           .append("=")
                    //           .append(assigned[var] ? (assignment[var] ? "1" : "0") : "?");
                    //         if (var < nVars) {
                    //             sb.append(" ");
                    //         }
                    //     }
                    // }
                    // System.out.println(sb.toString());

                    StringBuilder dimacs = new StringBuilder("v");
                    for (int var = 1; var <= nVars; var++) {
                        if (!assigned[var]) {
                            continue;
                        }
                        dimacs.append(" ").append(assignment[var] ? var : -var);
                    }
                    System.out.println(dimacs.toString());
                }

                if (maxSolutions > 0 && count >= maxSolutions) {
                    break;
                }

                VecInt blocking = new VecInt(varsToCount);
                for (int var = 1; var <= varsToCount; var++) {
                    // Block this projection over original vars.
                    blocking.push(assignment[var] ? -var : var);
                }

                try {
                    base.addBlockingClause(blocking);
                } catch (ContradictionException e) {
                    break;
                }
            }

            System.out.println("Solutions found: " + count);
        } catch (ContradictionException e) {
            // Trivial UNSAT discovered during parsing/loading.
            System.out.println("Solutions found: 0");
        } catch (ParseFormatException e) {
            System.err.println("Parse error: " + e.getMessage());
            System.exit(2);
        } catch (TimeoutException e) {
            System.err.println("Sat4j timeout: " + e.getMessage());
            System.exit(3);
        } catch (IOException e) {
            System.err.println("I/O error: " + e.getMessage());
            System.exit(2);
        }
    }
}
