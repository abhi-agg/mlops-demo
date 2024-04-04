from starlette.requests import Request

from typing import Dict

from ray import serve
from ray.serve import Application

@serve.deployment
class Multiplier:
    def __init__(self, args: Dict[str, str]):
        if "factor" not in args:
            raise Exception("Misconfigured server, missing 'factor' argument")
        
        self.factor = args["factor"]

    def multiply(self, value: float) -> float:
        return float(self.factor) * value

    async def __call__(self, request: Request):
        # Extract the request
        params = request.query_params
        if params is None or params.get("q") is None:
            raise Exception("Malformed query")

        return self.multiply(float(params.get("q")))

def app_builder(args: Dict[str, str]) -> Application:
    return Multiplier.options(route_prefix="/multiply").bind(args)

# TODO Uncomment the next line if attempting to autogenerate the config.
app = Multiplier.options(route_prefix="/multiply").bind(args={})
