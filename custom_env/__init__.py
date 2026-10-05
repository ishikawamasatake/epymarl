from gymnasium import register

register(
    id="PredatorPrey",
    entry_point="custom_env.env:PredatorPreyEnv",
)