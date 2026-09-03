# Soft-Body Tetris & CPU Ray Tracer

A C++ computer graphics project implementing deformable "soft-body" Tetris pieces rendered through a custom CPU ray tracer.

Built as the final project for the University of Waterloo CS488 Computer Graphics course.

## Highlights

- Soft-bodies physics simulation
- Collision detection between simulated objects
- Real-time interactive object manipulation
- Custom CPU ray tracer
- Optimization for fewer ray-triangle intersection tests
- CMake-based C++ project structure

# Gallery
## Soft-body Scenes
![screenshot](https://github.com/user-attachments/assets/e1d3e466-cc1a-4cdb-9187-1aa10e18d174)



https://github.com/user-attachments/assets/609e6f0b-233e-4e12-8770-31fe169d9bde



https://github.com/user-attachments/assets/222e4f4d-90e8-49a2-a3fe-7dc13bdc7222

![output511](https://github.com/user-attachments/assets/88a45ea2-063a-4fed-bc3d-e5900a20ddd4)


## Other Ray-traced Scenes

![screenshotTask5](https://github.com/user-attachments/assets/6cf2f769-c57f-4b3b-b9a2-38d11f26af34)


![screenshotTask3](https://github.com/user-attachments/assets/6cdb133d-048f-42b8-a763-e00240a69b9a)


![screenshotTask4](https://github.com/user-attachments/assets/fbee3e75-cca3-495d-9ac6-6718fa8d90bf)

## Implementation

- Soft-body physics is achieved through meshless deformations based on shape matching
- Ray tracing through one-directional recursive ray tracing
- Optimized performance on CPU using Surface Area Heuristic (SAH) Bounding Volume Hierarchy (BVH) to perform faster ray triangle lookups
See the [Wiki](https://github.com/zanada/cs488/wiki/Implementation-Notes) for additional implementation notes.

## Running the Project

This repository contains the portions of my implementation that can be
publicly shared. The complete course project cannot be distributed because
it contains University of Waterloo course intellectual property.

As a result, this repository is provided primarily as a code and
implementation showcase rather than a standalone build.
