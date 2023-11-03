To run these examples, activate a new environment in this directory via the Pkg REPL:

```julia-repl
activate examples
```

Next, add the SymbolicMDPs package as a dependency via `dev`, and `instantiate`` any remaining dependencies:

```julia-repl
dev .
instantiate
```

Note that the above commands assume that your current working directory is the directory that you have cloned this repository into.