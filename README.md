# minimum-vertex-cover

Applied Graph Theory course project to implement branch and bound algorithm (BnB) on minimum vertex cover (MVC) problem.

The MVC problem seeks the smallest set of vertices such that every edge in the graph has at-least one endpoint in it, thereby forming the MVC solution. The BnB algorithm is an exact technique that systematically traverses the search space evaluating all options.

Instruction for running branch-and-bound algorithm

Input: `-inst <filename.graph>`, `-alg[BnB|Approx]`, `-time`
Output: trace file and sol file: note the output files will be put in a subdirectory named output.

For example: to run it for the email.graph file with branch and bound with 30s as cutoff time: type in `python vertex_cover.py -inst ./DATA/email.graph -alg BnB -time 30`

where `./DATA/email.graph` is the path to the graph file.

The core algorithm of branch and bound is in the `branch_and_bound.py` file.
