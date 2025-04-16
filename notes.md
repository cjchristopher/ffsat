Thoughts and Notes:

* It seems not worth converting ESPs to PSPs (or CHSPs, etc) as the computational advantage isn't clear.
* * Having said that, logarthms come in to play. The naive translation from ESP to PSP is O(kn^3) in k clauses and n variables - however there might be a direct closed form that could be used directly instead of going via ESPs. A fourier transform exists which would be O(k nlogn) to convert, if closed forms can't be found - need to go back to O'Donnell to properly investigate this.

* Finding Global Minima:
* * HJ-Moreau Descent looks like a good candidate, but the nature of the sampling still poses problems for global convergence inside the hypercube - it seems like you would still need to get lucky to sample near the global minima for this idea to work.
* * The notion of what to do near the cube edges (Where the solution will be if exists) is trickier. You could "mirror" the cube infinitely along the boundaries (e.g. if parameter x_i is updated such that |x_i|>1 then "treat" x_i as having the value x_i - sign(x_i)*2(|x_i|-1) - that is x_i = -1.05 will be evaluated as "-0.95", -2 as "0", -3 as "1", -3.05 as "0.95", -4 as "0".