# mitosisanalyzer

[![License BSD-3](https://img.shields.io/pypi/l/mitosisanalyzer.svg?color=green)](https://github.com/uvarc/mitosisanalyzer/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/mitosisanalyzer.svg?color=green)](https://pypi.org/project/mitosisanalyzer)
[![Python Version](https://img.shields.io/pypi/pyversions/mitosisanalyzer.svg?color=green)](https://python.org)
[![codecov](https://codecov.io/gh/uvarc/mitosisanalyzer/branch/main/graph/badge.svg)](https://codecov.io/gh/uvarc/mitosisanalyzer)

Plugin to track spindle poles in mitotic cells over time. It leverages [Cellpose](https://www.cellpose.org/), [OpenCV](https://github.com/opencv/opencv-python), and [Scikit-Image](https://scikit-image.org/) for segmentation and [Prefect](https://www.prefect.io/) and [Dask](https://www.dask.org/) for workflow orchestration.

## Installation 

You can install `mitosisanalyzer` via [pip]:

    pip install mitosisanalyzer


To install latest development version :

    pip install git+https://github.com/uvarc/mitosisanalyzer.git


## Running the MitosisAnalyzer application

In a command line shell, run the following command:
```
mitosisanalyzer -i imagestack.nd2 -o my_outputdir -s 1 -d 2 -r 1
```

**Command line arguments:**

```
    -h, --help                                show this help message and exit
    -i INPUT, --input INPUT                   .nd2 file or directory with .nd2 files to be processed
    -o OUTPUT, --output OUTPUT                output file or directory
    -s SPINDLE, --spindle SPINDLE             channel # for tracking spindle poles
    -d DNA, --dna DNA                         channel # for tracking dna
    -r REFFRAME, --refframe REFFRAME          reference frame to determine spindle pole axis (0=autodetect based on cell long axis)
    -t THRESHOLD, --threshold THRESHOLD       threshold of cytoplasmic background signal in spindle channel; value relative to max 
                                              spindle intensity 0.0-1.0 (0.0=autodetect using Otsu)
    -b BLUR, --blur BLUR                      applies a gaussian blur before segmenting spindle poles. The value determines the 
                                              blurring radius; a value of 0 omits blurring.
    -c, --cellpose, --no-cellpose             use Cellpose to detect cell contour
    -f FRAMERATE, --framerate FRAMERATE       number of frames per second
    -e EXECUTOR, --executor EXECUTOR          set executor. Options: sequential, concurrent, dask
    -p PROCESSES, --processes PROCESSES       number or parallel processes
```                  

## Contributing

Contributions are very welcome.

## License

Distributed under the terms of the [BSD-3] license, "mitosisanalyzer" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[BSD-3]: http://opensource.org/licenses/BSD-3-Clause

[file an issue]: https://github.com/uvarc/mitosisanalyzer/issues

[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/

