Project for the exams of [Numerical Methods for Graphics](https://unimap.unipi.it/registri/dettregistriNEW.php?re=11063680::::&ri=027958) and [An Introduction to Scientific Software Tools & Parallel Algorithms (SSPA)](https://luca-heltai.github.io/sspa/).

The idea is to leverage neural networks as a tool for scientific discovery. \
We choose to consider the problem of B-spline approximation: given a point cloud, find the best fitting B-spline curve. \
We are interested in whether Kolmogorov--Arnold networks may have an advantage over traditional vanilla neural networks, in terms of accuracy and/or interpretability. \
You can find more information in the [slides](https://github.com/HunterNise/splines-KAN/blob/main/typst/slides.pdf) and the references therein.

The source code for the experiments conducted are in the `source` folder; while the output results are available as a [release](https://github.com/HunterNise/splines-KAN/releases/tag/v1.0-results).

---

## Installation

> [!WARNING]
> This project is meant to be run in a Linux environment. No other environments have been tested. \
> If you use Windows, install WSL.

Clone the repository
```bash
git clone https://github.com/HunterNise/splines-KAN.git
```

### Docker setup

This project uses docker for containerization, which means it is fully reproducible and does not mess up with your local packages. \
You will need to have docker installed: if you don't already, follow the [instructions](https://docs.docker.com/engine/install/) for your system.

Then you will need the image of the project. You can either:
- download the already built [image from the repo](https://github.com/HunterNise/splines-KAN/pkgs/container/splines);
- or rebuild locally the image by running the script `setup.sh`.

### Docker-less setup

If you don't have access to docker, there is an alternative setup that uses the package manager uv. \
Follow the [instructions](https://docs.astral.sh/uv/getting-started/installation/) to install uv, then run the script `setup.sh`.

> [!NOTE]
> If you also have docker installed, it will be prioritized by `setup.sh`. \
> If you prefer to use the uv setup, you can do so by running
> ```bash
> ./setup.sh uv
> ```

If you wish to compile the slides, you will also have to separately install the [typst compiler](https://typst.app/open-source/).


## Running

After installation, you can launch the project simply as
```bash
./run.sh
```
This will either launch the docker container or the uv virtual environment, depending on the installation method.

You can run the experiment scripts as 
```bash
python -O source/**/*.py
```
Without the `-O` flag the script will be run in debug mode which is slower due to tests.

In order to run the scripts you will need to download the datasets: either follow the instructions in the readmes of the `data` folder or install the [dataset release](https://github.com/HunterNise/splines-KAN/releases/tag/v1.0-data).

To run unit tests
```bash
pytest
```

To compile slides
```bash
./slides.sh
```
