# SearchDestroy

Multi-drone area-sweep algorithm with online re-planning, built on top of
[alice-st/DARP-Python](https://github.com/alice-st/DARP-Python) for the
AGI House Robotics Hackathon (Summer 2024). Placed 2nd.

## What we built

SearchDestroy extends DARP's area-division core with:

- Efficient parametrization of search regions across multiple drones
- Online path re-computation when drones are lost or when the environment shifts
- Robustness heuristics for adversarial interference
- Physics simulation and visualization via AirSim

Our hackathon work lives primarily in [`test.ipynb`](test.ipynb), with small
modifications to `darp.py`, `multiRobotPathPlanner.py`, and `Dependencies.sh`.

## Team

[kiaghods](https://github.com/kiaghods), [Astoria-ni](https://github.com/Astoria-ni), [jerryhan60](https://github.com/jerryhan60), [tigeyshark22](https://github.com/tigeyshark22), [RSDP101](https://github.com/RSDP101), [yanda-dy](https://github.com/yanda-dy)

## Upstream

All core DARP algorithm code is from [alice-st/DARP-Python](https://github.com/alice-st/DARP-Python),
based on:

> Kapoutsis, Chatzichristofis, Kosmatopoulos. *DARP: Divide Areas Algorithm for Optimal
> Multi-Robot Coverage Path Planning.* Journal of Intelligent & Robotic Systems.

For DARP installation and usage, see the upstream repository.
