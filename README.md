# licenta

## room acoustic models

### diffuse-field equations (Sabine/Eyring)

### ray/particle tracing

- fast simulation
- hard to do diffraction
- high-frequency approximations (in theory)
  
### wave simulation

- most computationally expensive
- diffraction included
- valid in the entire frequency range

## development

Needs super performance so a compiled language is a must. If the ULTIMATE
performance is needed then C will be the way. Although C++ would provide many
more libraries to work with and ease of use for some performance cost. Dear
ImGUI could be an important factor in the decision of language since there are
no other good ways to create UI. As for graphics, rendering very complex 3D
models will require OpenGL. This is bad for time constraints since it will take
a lot of time to program basic tasks like the model loading and the rendering
pipeline. Raylib could help with doing the 3D rendering at the start since it
requires the least amount of boiler-plate code at the cost of performance and
inflexibility.

No matter the acoustic simulator kind used, it will need to be executed in
parallel. Each computation tick will be a rather simple computation, with many
entities to be ticked each iteration. GPU parallelization is definitely the way
to go. Could be done with either CUDA or GLM depending on the complexity and
speed required. OpenCL could also be an alternative in order to allow the
program to run on Non-Nvidia GPUs and devices with no gpu.

## resources
<https://dafx.labri.fr/main/papers/p117.pdf>
<https://www.acoustics.asn.au/conference_proceedings/ICA2010/cdrom-ICA2010/papers/p488.pdf>
