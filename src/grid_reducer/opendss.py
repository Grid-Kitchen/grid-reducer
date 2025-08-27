from pathlib import Path
import json

import opendssdirect as odd

from grid_reducer.utils.files import read_json_file
from grid_reducer.altdss.altdss_models import Circuit


class OpenDSS:
    def __init__(self, dist_model: Path | dict):
        if dist_model.suffix == ".dss":
            odd.Command(f'Redirect "{str(dist_model)}"')

        elif isinstance(dist_model, dict):
            odd.Circuit.FromJSON(json.dumps(dist_model))

        elif dist_model.suffix == ".json":
            odd.Circuit.FromJSON(json.dumps(read_json_file(dist_model)))

        else:
            msg = f"Unsupported dist_model type: {type(dist_model)}"
            raise NotImplementedError(msg)

        self.solve()

    def solve(self):
        odd.Solution.Solve()

    def get_circuit_dict(self) -> dict:
        return json.loads(odd.Circuit.ToJSON())

    def get_circuit(self) -> Circuit:
        return Circuit.model_validate(self.get_circuit_dict())

    def get_circuit_power(self) -> complex:
        return complex(*odd.Circuit.TotalPower())

    def get_all_bus_voltages(self) -> dict[str, float]:
        return dict(zip(odd.Circuit.AllBusNames(), odd.Circuit.AllBusMagPu(), strict=False))

    def get_substation_bus_name(self) -> str:
        odd.Vsources.First()
        return odd.CktElement.BusNames()[0].split(".")[0]

    def get_source_voltage(self) -> float:
        return self.get_all_bus_voltages().get(self.get_substation_bus_name())
