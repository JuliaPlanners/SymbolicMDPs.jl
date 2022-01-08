# SymbolicMDPs.jl

**SymbolicMDPs.jl** wraps the  [PDDL.jl](https://github.com/JuliaPlanners/PDDL.jl) interface for PDDL domains and problems within the [POMDPs.jl](https://juliapomdp.github.io/POMDPs.jl/latest/) interface for Markov decision processes (MDPs).

Since POMDPs.jl supports the reinforcement learning interface defined by [CommonRLInterface.jl](https://github.com/JuliaReinforcementLearning/CommonRLInterface.jl), this package also allows PDDL domains to be treated as RL environments that are compatible with libraries such as [ReinforcementLearning.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl).

## Installation

SymbolicMDPs.jl currently requires the latest development version of PDDL.jl. Install both via the `Pkg` REPL by running:
```julia
add https://github.com/JuliaPlanners/PDDL.jl.git
add https://github.com/JuliaPlanners/SymbolicMDPs.jl.git
```

## Usage

Once a PDDL `domain` and `problem` are loaded (e.g. using `PDDL.load_domain` and `PDDL.load_problem`), a corresponding MDP can be constructed as follows:
```julia
mdp = SymbolicMDP(domain, problem)
```

Alternatively, we can manually specify an initial `state`, `goal` formula and cost `metric` formula:
```julia
mdp = SymbolicMDP(domain, state, goal, metric)
```

 By default, every action leads to `-1` reward. If `metric` is specified, then the reward will be equal to the difference in the value of the metric between consecutive states. More customization of e.g. reward functions will be supported in the future. States that satisfy the `goal` condition are considered terminal.

 We can also construct RL environments as follows:
 ```julia
 env = SymbolicRLEnv(domain, problem)
 ```

For many reinforcement learning methods, the underlying PDDL state has to be converted to an integer or vector representation (eg. for tabular or deep reinforcement learning respectively). We can do this by specifying a conversion type:
```julia
tabular_env = SymbolicRLEnv(Int, domain, problem)
vector_env = SymbolicRLEnv(Vector, domain, problem)
```

 To use these environments with [ReinforcementLearning.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl), we have to perform one further step of conversion:
 ```julia
 using ReinforcementLearning
 env = convert(RLBase.AbstractEnv, env)
 ```

 For an example of tabular Q-learning on a small Blocksworld problem with two blocks, see [`examples/tabular_q.jl`](examples/tabular_q.jl).
