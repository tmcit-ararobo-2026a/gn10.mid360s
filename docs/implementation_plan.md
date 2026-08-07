# Implementation plan
Implement a Livox Mid-360S LiDAR extension for Isaac Sim 6 using the [Physics Raycast Sensor API](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/sensors/isaacsim_sensors_physics_raycast.html).

## Overview
This project provides an Isaac Sim extension that simulates the Livox Mid-360S LiDAR.
Rather than implementing a custom raycasting engine, this extension is built on top of the Isaac Sim 6 Physics Raycast Sensor API and reproduces the Mid-360S scan pattern by configuring ray directions and timing.

## Goals

- Simulate the Livox Mid-360S scan pattern.
- Support attachment to any robot using a USD Xform.
- Publish PointCloud2 via ROS2.
- Support visualization inside Isaac Sim.
- Keep the implementation compatible with future Isaac Sim releases.

## Architecture

```
Mid-360S Extension
        │
        ▼
Pattern Loader (.npy)
        │
        ▼
Physics Raycast Sensor
```

## Directory Structure

```
gn10.mid360/
├── config/
│   └── extension.toml
│
├── data/
│   ├── patterns/
│   │   └── mid360.npy
│   └── usd/
│       └── mid360.usd
│
├── gn10_mid360s_python/
│   ├── extension.py
│   ├── sensor.py
│   └── pattern_loader.py
│
├── examples/
│
├── docs/
│
└── README.md
```

## Sensor Model

The extension consists of two independent components:

- A USD representation for visualization and robot integration.
- A Physics Raycast Sensor used for point cloud generation.

The USD model represents the physical location of the sensor, while the raycast sensor performs the actual distance measurements.

## Implementation Steps

1. Create the extension by [CLI Extension Templates](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/utilities/cli_extension_templates.html#isaac-sim-cli-extension-templates)
2. Load the Mid-360 scan pattern.
3. Create a Physics Raycast Sensor.
6. Compatible Mid-360S.
7. Implement optional noise models.

## Pattern Loader

The Mid-360 scan pattern is loaded from a NumPy(.npy) file.

## Third-Party Assets

This project reuses the Mid-360 scan pattern from OmniPerception.
OmniPerception is distributed under the MIT License.
Only the scan pattern data is reused.