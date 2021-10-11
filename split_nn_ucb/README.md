## Communication-Efficient Split Learning

Communication-Client Commpute can be traded-off by adjusting:
- Number of clients that communicate at each round
- Total number of communication rounds
- Bandwidth per communication round per client

- We train half of the network at the client using Contrastive learning when it is not communicating with the server.

- If a client is communicating with the server for a particular round, it optimizes the cross-entropy objective.

## Reducing Communicating Clients through Local Parallelism
- For starting K rounds, all clients optimize Supervised Constrastive Loss (https://github.com/HobbitLong/SupContrast) (https://github.com/HobbitLong/SupContrast).
- After K rounds, either all clients optimize cross-entropy (if no subset-selection) or a subset of clients (using UCB).

## Subset selection through Bayesian UCB
- At every round, a Bandit (Bayesian UCB) selects clients with highest cummulative losses and variance.
- This bandit ensures that clients with poorer performance are selected more than the rest, through this, the bandit tries to minimize expected loss for each client.

## Faster (& better) Optimization through inter-client transfer
- The server optimizes shared weights U, and client-specific weights V<sub>c</sub> and W<sub>c</sub>.
- For a client c, the final weights theta is computed using theta = U * V<sub>c</sub>.
- V<sub>c</sub> is forced to be extremely sparse using L1-norm constraint.
- V<sub>c</sub> masks U such that the client c can retrieve knowledge that is common among other clients i.e. U

## Efficient Communication with sparse activations
- We only communicate non-zero elements of the activation tensors and gradients to save bandwidth.
