from typing import Any

from grid_reducer.altdss.altdss_models import Circuit, BusConnection, SwtControlState


def get_circuit_bus_name(circuit: Circuit) -> str:
    return circuit.Vsource.root.root[0].root.Bus1.root.split(".")[0]


def get_bus_connected_assets(asset_container: Any, bus_name: str) -> list[Any]:
    return [
        asset.root
        for asset in asset_container.root.root
        if asset.root.Bus1.root.split(".")[0] == bus_name
    ]


def extract_bus_name(bus_obj: BusConnection) -> str:
    return bus_obj.root.split(".")[0]


def get_open_lines(circuit_obj: Circuit) -> list[str]:
    """Get a list of open lines from the circuit object."""
    open_lines = []
    for command in circuit_obj.PostCommands:
        if command.startswith("Open Line."):
            open_lines.append(command.split(".")[1].split(" ")[0])
    return open_lines


def get_normally_open_switches(circuit_obj: Circuit) -> list[str]:
    normally_open_switches = []
    for line in circuit_obj.Line.root.root:
        if line.root.Enabled is False:
            normally_open_switches.append(line.root.Name)
    if circuit_obj.SwtControl is None:
        return normally_open_switches
    for switch in circuit_obj.SwtControl.root.root:
        if switch.SwitchedObj and switch.Normal == SwtControlState.open:
            normally_open_switches.append(switch.SwitchedObj.replace("Line.", ""))
    return normally_open_switches
