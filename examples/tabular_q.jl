# Use SymbolicMDPs wrapper and ReinforcementLearning library
using SymbolicMDPs, ReinforcementLearning
using Test

# Import Flux optimizers
import ReinforcementLearningZoo: Flux.Descent

# Import various MDP tools
import POMDPs
import POMDPModelTools: MDPCommonRLEnv

# Load PDDL functions
import PDDL: PDDL, load_domain, load_problem, @pddl
using PlanningDomains

# Load and compile blocksworld domain and problem with 2 blocks
domain = load_domain(JuliaPlannersRepo, "blocksworld")
problem = load_problem(JuliaPlannersRepo, "blocksworld", "problem-1")
domain, _ = PDDL.compiled(domain, problem)

# Construct SymbolicMDP
mdp = SymbolicMDP(domain, problem)

# Test successful plan to goal state using POMDPs.jl interface
s = rand(POMDPs.initialstate(mdp))
for act in PDDL.@pddl("(pick-up a)", "(stack a b)")
    s = rand(POMDPs.transition(mdp, s, act))
end
@test POMDPs.isterminal(mdp, s)

# Convert MDP to RL environment
STATE_TYPE = Int # Convert states to their indices for tabular RL
env = SymbolicRLEnv(STATE_TYPE, mdp)
env = convert(RLBase.AbstractEnv, env);
env = RL.discrete2standard_discrete(env) # Convert actions to their indices

# Test successful plan to goal state
reset!(env)
for act in PDDL.@pddl("(pick-up a)", "(stack a b)")
    a_idx = POMDPs.actionindex(mdp, act) # Look up action index
    env(a_idx) # Execute action
end
@assert is_terminated(env)

# Construct tabular Q-learning agent
agent = Agent(
    policy=QBasedPolicy(
        learner=TDLearner(
            approximator=TabularQApproximator( # Linear Q-value approximator
                n_state=length(state_space(env)), # Number of states
                n_action=length(action_space(env)), # Number of actions
                opt=Descent(0.5)
                ),
            method=:SARS, # SARS is equivalent to Q-learning,
            n=2, # Update Q-values after 2 steps (necessary for convergence)
        ),
        explorer=GreedyExplorer()
    ),
    trajectory=VectorSARTTrajectory(;
        state=typeof(state(env)),
        action=Int,
        reward=Float64,
        terminal=Bool
    )
)

# Run agent on environment for 10 episodes of 100 steps
reset!(env)
for i in 1:10
    run(agent, env, StopAfterStep(100))
end
q_idxs = findall(!=(0), agent.policy.learner.approximator.table)
q_vals = agent.policy.learner.approximator.table[q_idxs]

# Run learned policy and extract plan
plan = PDDL.Term[]
reset!(env)
for i in 1:10
    a_idx = agent.policy(env)
    env(a_idx)
    push!(plan, mdp.actions[a_idx])
    is_terminated(env) && break
end

display(plan) # Display plan
@test plan == PDDL.@pddl("(pick-up a)", "(stack a b)") # Test for correct plan
@test is_terminated(env) # Check that terminal state is reached

# Check that goal is satisfied
goal_state = collect(mdp.states)[state(env)]
PDDL.satisfy(domain, goal_state, problem.goal)
