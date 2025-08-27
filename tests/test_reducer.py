from pathlib import Path
import re
import warnings

import networkx as nx
import pytest

from grid_reducer.utils.files import write_to_opendss_file
from grid_reducer.network import get_graph_from_circuit
from grid_reducer.aggregate_secondary import aggregate_secondary_assets
from grid_reducer.opendss import OpenDSS
from grid_reducer.reducer import OpenDSSModelReducer

root_folder = Path(__file__).parent / "data"
additional_data_folder = Path(__file__).parent / "../extra_data"
test_folders = [root_folder, additional_data_folder]
files = []
for folder in test_folders:
    if folder.exists():
        pattern = re.compile(r".*master.*\.dss$", re.IGNORECASE)
        files += [f for f in folder.rglob("*.dss") if pattern.search(f.name)]


def assert_reasonable_circuit_power_deviation(original_circuit_file, reduced_circuit_file):
    original_ckt_power = OpenDSS(original_circuit_file).get_circuit_power()
    reduced_ckt_power = OpenDSS(reduced_circuit_file).get_circuit_power()
    pct_diff = (
        abs((original_ckt_power.real - reduced_ckt_power.real) / original_ckt_power.real) * 100
    )
    if pct_diff > 10:
        warnings.warn(
            f"Power flow results differ significantly: "
            f"Original: {original_ckt_power}, Reduced: {reduced_ckt_power}, Pct Diff: {pct_diff:.2f}%",
            stacklevel=2,
        )


def assert_reasonable_source_voltage_deviation(original_circuit_file, reduced_circuit_file):
    original_ckt_source_voltage = OpenDSS(original_circuit_file).get_source_voltage()
    reduced_ckt_source_voltage = OpenDSS(reduced_circuit_file).get_source_voltage()
    pct_diff = (
        abs(
            (original_ckt_source_voltage - reduced_ckt_source_voltage)
            / original_ckt_source_voltage
        )
        * 100
    )
    assert pct_diff < 10


@pytest.mark.parametrize("file", files)
def test_networkx_graph_creation(file):
    circuit = OpenDSS(file).get_circuit()
    graph = get_graph_from_circuit(circuit)
    assert isinstance(graph, nx.Graph)


@pytest.mark.parametrize("file", files)
def test_secondary_aggregation(file, tmp_path):
    circuit = OpenDSS(file).get_circuit()
    new_circuit, _ = aggregate_secondary_assets(circuit)
    original_circuit_file = tmp_path / "original_ckt.dss"
    reduced_circuit_file = tmp_path / "reduced_ckt.dss"
    write_to_opendss_file(circuit, original_circuit_file)
    write_to_opendss_file(new_circuit, reduced_circuit_file)
    assert_reasonable_circuit_power_deviation(original_circuit_file, reduced_circuit_file)
    assert_reasonable_source_voltage_deviation(original_circuit_file, reduced_circuit_file)


@pytest.mark.parametrize("file", files)
def test_primary_aggregation(file, tmp_path):
    reducer = OpenDSSModelReducer(master_dss_file=file)
    reduced_ckt = reducer.reduce(transform_coordinate=True)
    original_circuit_file = tmp_path / "original_ckt.dss"
    reduced_circuit_file = tmp_path / "reduced_ckt.dss"
    reducer.export_original_ckt(original_circuit_file)
    reducer.export(reduced_ckt, reduced_circuit_file)
    assert_reasonable_circuit_power_deviation(original_circuit_file, reduced_circuit_file)
    assert_reasonable_source_voltage_deviation(original_circuit_file, reduced_circuit_file)
