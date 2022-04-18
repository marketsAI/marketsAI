TO IMPLEMENT:

Agents are defined as self-contained class with the following methods:
* select_action(observation)
* update(batched_transitions)
* test(obs_space, action_space)

To implement:
* Deep Q learning (with double Q (bootstrapped?), delayed_target)
* DDPG
* SAC
* PPO

Special features:
* minimal dependencies (jax, optax, haiku, numpy)
* Written JAX, so they can be run with optimizations on CPUs, GPUs, and TPUs.
* They allow for multi-agent optimizations (like all using global state vs reordering it in individual states).
