from gym import register

register(
    id="StochasticGrid-v0",
    entry_point="stochastic_grid.env:make_stochastic_pogema",
)
