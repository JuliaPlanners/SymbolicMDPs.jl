module SymbolicMDPs

export SymbolicMDP, SymbolicRLEnv

using Random

import PDDL: PDDL, Domain, Problem, State, Term, Compound, Const, @pddl_str
import POMDPs: POMDPs, MDP
import POMDPTools: Deterministic, Uniform, SparseCat, MDPCommonRLEnv
import CommonRLInterface

## Utility functions ##

typerange(T) = nothing
typerange(T::Type{<:Integer}) = typemin(T):typemax(T)
typerange(T::Type{<:Enum}) = instances(T)

compare_terms(x::Term, y::Term) = x.name < y.name
compare_terms(x::Compound, y::Compound) = string(x) < string(y)

## SymbolicStateSpace ##

"Product representation of a symbolic state space for ground PDDL domains."
struct SymbolicStateSpace{S <: State, I, FT, FR}
    state::S # Reference state with static values of fluents
    ftypes::FT # Map from (non-static) fluent names to their Julia types
    franges::FR # Map from (non-static) fluent names to their value ranges
    fluents::Vector{Term} # List of all (non-static) ground fluents
    iter::I # Iterator over states
end

function SymbolicStateSpace(domain::Domain, state::State)
    # Make copy of state for reference
    state = deepcopy(state)
    # Infer static fluents so that we can avoid enumerating over their values
    statics = PDDL.infer_static_fluents(domain)
    # Extract and order fluent types and value ranges
    sigs = sort(collect(pairs(PDDL.get_fluents(domain))), by=first)
    ftypes = (; (f => PDDL.GLOBAL_DATATYPES[sig.type]
                 for (f, sig) in sigs if !(f in statics))...)
    franges = (; (f => typerange(ty) for (f, ty) in pairs(ftypes))...)
    # Construct list of all (non-static) ground fluents
    fluents = Term[]
    for (fname, sig) in sigs
        if fname in statics continue end
        if PDDL.arity(sig) == 0
            push!(fluents, Const(fname))
        else
            fs = [Compound(fname, collect(Term, args))
                  for args in PDDL.groundargs(domain, state, fname)] |> vec
            append!(fluents, sort!(fs, lt=compare_terms))
        end
    end
    # Construct iterator over states
    if (any(!(ftypes[f.name] <: Union{Integer,Enum}) for f in fluents) ||
        length(fluents) > Sys.WORD_SIZE)
        # Not iterable if some fluents are continuous, or there are too many
        state_iter = nothing
    else
        # Construct product iterator over ground fluent values
        val_iters = (franges[f.name] for f in fluents)
        state_iter = Base.Generator(Iterators.product(val_iters...)) do vals
            s = copy(state)
            for (f, val) in zip(fluents, vals)
                s[f] = val
            end
            return s
        end
    end
    return SymbolicStateSpace(state, ftypes, franges, fluents, state_iter)
end

Base.eltype(space::SymbolicStateSpace{S}) where {S} = S

Base.iterate(space::SymbolicStateSpace{S,Nothing}) where {S} =
    error("Non-iterable space.")
Base.iterate(space::SymbolicStateSpace{S,Nothing}, i) where {S} =
    error("Non-iterable space.")
Base.length(space::SymbolicStateSpace{S,Nothing}) where {S} =
    error("Non-iterable space.")

Base.iterate(space::SymbolicStateSpace{S}) where {S} =
    iterate(space.iter)
Base.iterate(space::SymbolicStateSpace{S}, i) where {S} =
    iterate(space.iter, i)
Base.length(space::SymbolicStateSpace{S}) where {S} =
    prod(length(space.franges[f.name]) for f in space.fluents)

function Base.rand(rng::AbstractRNG, space::SymbolicStateSpace)
    state = copy(space.state)
    val_iters = for f in space.fluents
        ty = space.ftypes[f.name]
        state[f] = Base.rand(rng, ty)
    end
    return state
end

function POMDPs.stateindex(space::SymbolicStateSpace{S}, state::S) where {S}
    if space.iter === nothing
        error("State space is not discrete.")
    end
    idx = 0
    for f in Iterators.reverse(space.fluents)
        ty = space.ftypes[f.name]
        vrng = space.franges[f.name]
        val = state[f]
        offset = (ty <: Enum ? Int(val) : findfirst(==(val), vrng)) - 1
        idx = idx * length(vrng) + offset
    end
    return idx + 1
end

function vectorize(space::SymbolicStateSpace, state::State)
    T = typejoin(space.ftypes...)
    return broadcast(f -> state[f]::T, space.fluents)
end

## SymbolicMDP ##

"""
    SymbolicMDP

MDP wrapper for a ground PDDL domain.
"""
struct SymbolicMDP{D<:Domain,S<:State,SS<:SymbolicStateSpace,C} <: MDP{S,Term}
    domain::D
    init::S
    goal::Term
    metric::Union{Term,Nothing}
    discount::Float64
    states::SS
    actions::Vector{Compound}
    a_cache::C
end

"""
    SymbolicMDP(domain, state, [goal, metric, discount]; cache_actions=false)

Construct a symbolic MDP from a PDDL `domain` and initial `state`.
"""
function SymbolicMDP(domain::Domain, state::State,
                     goal=pddl"(true)", metric=nothing, discount=1.0;
                     cache_actions::Bool=true)
    s_space = SymbolicStateSpace(domain, state)
    actions = Compound[act.term for act in PDDL.groundactions(domain, state)]
    actions = sort!(actions, lt=compare_terms)
    a_cache = cache_actions ? Dict{UInt64,Vector{Compound}}() : nothing
    return SymbolicMDP(domain, state, goal, metric, discount,
                       s_space, actions, a_cache)
end

"""
    SymbolicMDP(domain, problem; cache_actions=false)

Construct a symbolic MDP from a PDDL `domain` and `problem`.
"""
function SymbolicMDP(domain::Domain, problem::Problem; options...)
    state = PDDL.initstate(domain, problem)
    goal = PDDL.get_goal(problem)
    metric = PDDL.get_metric(problem)
    if metric !== nothing # Extract metric formula to minimize
        metric = metric.name == :minimize ?
            metric.args[1] : Compound(:-, metric.args)
    end
    return SymbolicMDP(domain, state, goal, metric; options...)
end

POMDPs.states(m::SymbolicMDP) = m.states
POMDPs.actions(m::SymbolicMDP) = m.actions
POMDPs.actions(m::SymbolicMDP, s) = m.a_cache isa Nothing ?
    collect(PDDL.available(m.domain, s)) :
    get!(() -> collect(PDDL.available(m.domain, s)), m.a_cache, hash(s))

POMDPs.initialstate(m::SymbolicMDP) =
    Deterministic(m.init)
POMDPs.transition(m::SymbolicMDP, s, a) =
    Deterministic(PDDL.transition(m.domain, s, a))
POMDPs.reward(m::SymbolicMDP, s, a, sp) = m.metric === nothing ?
    -1 : m.domain[s => m.metric] - m.domain[sp => m.metric]
POMDPs.discount(m::SymbolicMDP) =
    m.discount
POMDPs.isterminal(m::SymbolicMDP, s) =
    PDDL.satisfy(m.domain, s, m.goal)

POMDPs.stateindex(m::SymbolicMDP, s) =
    POMDPs.stateindex(m.states, s)
POMDPs.actionindex(m::SymbolicMDP, a) =
    searchsortedfirst(m.actions, a; lt=compare_terms)

POMDPs.convert_s(::Type{Any}, s::State, m::SymbolicMDP) =
    s
POMDPs.convert_s(::Type{<:AbstractArray}, s::State, m::SymbolicMDP) =
    vectorize(m.states, s)
POMDPs.convert_s(::Type{Int}, s::State, m::SymbolicMDP) =
    POMDPs.stateindex(m, s)

POMDPs.convert_a(::Type{Any}, a::Term, m::SymbolicMDP) =
    a
POMDPs.convert_a(::Type{<:AbstractArray}, a::Term, m::SymbolicMDP) =
    [POMDPs.actionindex(m, a)]
POMDPs.convert_a(::Type{Int}, a::Term, m::SymbolicMDP) =
    POMDPs.actionindex(m, a)

## Implement additional common RL environment interface functions ##

const CRL = CommonRLInterface
const SymbolicMDPCommonRLEnv =
    MDPCommonRLEnv{RLO, M} where {RLO, M <: SymbolicMDP}

"""
    SymbolicRLEnv([S::Type], mdp::SymbolicMDP)
    SymbolicRLEnv([S::Type], domain, problem)
    SymbolicRLEnv([S::Type], domain, state, [goal, metric, discount])

Construct a reinforcement learning environment that adheres to the
[`CommonRLInterface`](https://git.io/J9OzR). Environments are constructed
in a similar manner to [`SymbolicMDP`](@ref), with the additional option
of specifiying a type `S` which the underlying PDDL state will be converted to.

By default, `S` is `Any`, implying no conversion. Other supported types are:
- `Int`: states are converted using [`POMDPs.stateindex`](@ref).
- `AbstractArray`: states are converted via [`SymbolicMDPs.vectorize`](@ref).

Conversion is implemented via [`POMDPs.convert_s`](@ref). To support
custom state converters, define a new type `T`, and then a new
method `POMDPs.convert_s(::Type{T}, s::State, ::SymbolicMDP)` that
performs the relevant conversion. Note that `T` is allowed to be abstract,
and need not correspond to the type actually returned.
"""
SymbolicRLEnv(S::Type, mdp::SymbolicMDP) =
    MDPCommonRLEnv{S}(mdp)
SymbolicRLEnv(S::Type, domain::Domain, args...; kwargs...) =
    SymbolicRLEnv(S, SymbolicMDP(domain, args...; kwargs...))
SymbolicRLEnv(domain::Domain, args...; kwargs...) =
    SymbolicRLEnv(Any, domain, args...; kwargs...)

CRL.@provide CRL.valid_action_mask(env::SymbolicMDPCommonRLEnv) =
    broadcast(in(CRL.valid_actions(env)), env.m.actions)

end
