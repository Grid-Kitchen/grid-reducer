import json
from pathlib import Path

from grid_reducer.altdss.altdss_models import Circuit


def write_to_opendss_file(circuit: Circuit, output_file: Path | str) -> None:
    with open(output_file, "w", encoding="utf-8") as fp:
        circuit.dump_dss(fp)


def read_json_file(file_path: Path) -> dict:
    with open(file_path, "r", encoding="utf-8") as fp:
        contents = json.load(fp)
    return contents
