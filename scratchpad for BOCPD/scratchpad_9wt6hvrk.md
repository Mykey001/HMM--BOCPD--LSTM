# Task Checklist
- [x] Read Gregory Gundersen's blog post on implementing BOCD.
- [x] Extract the mathematical formulations of the joint distribution and the recursion step.
- [x] Extract the Python code showing how the joint distribution and recursion step are updated, specifically focusing on `log_growth`, `log_cp`, and normalization.
- [x] Summarize actions and findings.

## Summary of Findings
Gregory Gundersen's implementation of Bayesian Online Changepoint Detection (BOCD) uses log-space calculations to avoid numerical underflow.

1. **Recursion formulation**:
   The joint distribution $p(r_t, \mathbf{x}_{1:t})$ is recursively updated by summing over possible previous run-lengths $r_{t-1}$:
   $$p(r_t, \mathbf{x}_{1:t}) = \sum_{r_{t-1}} p(r_t \mid r_{t-1}) p(x_t \mid r_{t-1}, \mathbf{x}_t^{(r)}) p(r_{t-1}, \mathbf{x}_{1:t-1})$$

2. **Log-space updates**:
   - `log_growth_probs` (for $r_t = r_{t-1} + 1$):
     `log_growth_probs = log_pis + log_message + log_1mH`
   - `log_cp_prob` (for $r_t = 0$):
     `log_cp_prob = logsumexp(log_pis + log_message + log_H)`
   - **Normalization (Run-length posterior)**:
     `log_R[t, :t+1] = new_log_joint - logsumexp(new_log_joint)`
     Where `new_log_joint = np.append(log_cp_prob, log_growth_probs)`.
   - **Message passing**:
     `log_message = new_log_joint` (the unnormalized joint distribution is passed as the message to the next step).


## Mathematical Formulation

### 1. Joint Distribution and Recursion Step (Adams & MacKay 2007)
$$p(r_t, \mathbf{x}_{1:t}) = \sum_{r_{t-1}} p(r_t \mid r_{t-1}) p(x_t \mid r_{t-1}, \mathbf{x}_t^{(r)}) p(r_{t-1}, \mathbf{x}_{1:t-1})$$
where:
- $p(r_t \mid r_{t-1})$ is the transition probability (defined by the Hazard function $H(r_{t-1})$).
- $p(x_t \mid r_{t-1}, \mathbf{x}_t^{(r)})$ is the predictive probability of the new observation $x_t$ given the current run-length $r_{t-1}$ and the corresponding run data $\mathbf{x}_t^{(r)}$.
- $p(r_{t-1}, \mathbf{x}_{1:t-1})$ is the joint probability of the run-length $r_{t-1}$ and all observations up to $t-1$ (the "message").

### 2. Log-Space Recursion
Let:
- $\log\_message_t(r) = \log p(r_{t-1}=r, \mathbf{x}_{1:t-1})$
- $\log\_pi_t(r) = \log p(x_t \mid r_{t-1}=r, \mathbf{x}_t^{(r)})$
- $\log\_H = \log H(r)$
- $\log\_1mH = \log(1 - H(r))$

Then:
- **Growth probabilities** ($r_t = r_{t-1} + 1$):
  $$\log p(r_t = r+1, \mathbf{x}_{1:t}) = \log\_pi_t(r) + \log\_message_t(r) + \log\_1mH$$
- **Changepoint probability** ($r_t = 0$):
  $$\log p(r_t = 0, \mathbf{x}_{1:t}) = \text{logsumexp}_r \left( \log\_pi_t(r) + \log\_message_t(r) + \log\_H \right)$$
- **Normalization (Evidence)**:
  $$\log p(\mathbf{x}_{1:t}) = \text{logsumexp}_{r_t} \left( \log p(r_t, \mathbf{x}_{1:t}) \right)$$
  The normalized posterior run-length distribution is:
  $$\log p(r_t \mid \mathbf{x}_{1:t}) = \log p(r_t, \mathbf{x}_{1:t}) - \log p(\mathbf{x}_{1:t})$$

## Python Code (Log Space)

### 1. Initialization
```python
log_R       = -np.inf * np.ones((T+1, T+1))
log_R[0, 0] = 0              # log(1) == 0
log_message = np.array([0])  # log(1) == 0
log_H       = np.log(hazard)
log_1mH     = np.log(1 - hazard)
```

### 2. Recursion step at each time $t$
```python
# Observe new datum x = data[t-1]
# 3. Evaluate predictive probabilities.
log_pis = model.log_pred_prob(t, x)

# 4. Calculate growth probabilities.
log_growth_probs = log_pis + log_message + log_1mH

# 5. Calculate changepoint probabilities.
log_cp_prob = logsumexp(log_pis + log_message + log_H)

# 6. Calculate evidence
new_log_joint = np.append(log_cp_prob, log_growth_probs)

# 7. Determine run length distribution (normalization in log space).
log_R[t, :t+1]  = new_log_joint
log_R[t, :t+1] -= logsumexp(new_log_joint)

# 8. Update sufficient statistics.
model.update_params(t, x)

# Pass message.
log_message = new_log_joint
```


