# mitosisanalyzer

[![License BSD-3](https://img.shields.io/pypi/l/mitosisanalyzer-napari.svg?color=green)](https://github.com/uvarc/mitosisanalyzer/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/mitosisanalyzer.svg?color=green)](https://pypi.org/project/mitosisanalyzer-napari)
[![Python Version](https://img.shields.io/pypi/pyversions/mitosisanalyzer.svg?color=green)](https://python.org)
[![tests](https://github.com/uvarc/mitosisanalyzer/workflows/tests/badge.svg)](https://github.com/uvarc/mitosisanalyzer/actions)
[![codecov](https://codecov.io/gh/uvarc/mitosisanalyzer/branch/main/graph/badge.svg)](https://codecov.io/gh/uvarc/mitosisanalyzer)

Plugin to track spindle poles in mitotic cells over time.

## Installation 

You can install `mitosisanalyzer` via [pip]:

    pip install mitosisanalyzer


To install latest development version :

    pip install git+https://github.com/uvarc/mitosisanalyzer.git


## Running the MitosisAnalyzer application

In a command line shell, run the following commands:
```
conda activate mitoanalyzer
python mitoanalysis.py -i imagestack.nd2 -o my_outputdir -s 1 -d 2 -r 1
```


**Command line arguments:**

    -h, --help                             show help message and exit
    -i INPUT, --input INPUT                .nd2 file or directory with .nd2 files to be processed
    -o OUTPUT, --output OUTPUT             output file or directory
    -p PROCESSES, --processes PROCESSES    optional: number or parallel processes
    -s SPINDLE, --spindle SPINDLE          channel # for tracking spindle poles
    -d DNA, --dna DNA                      channel # for tracking dna
    -r REFFRAME, --refframe REFFRAME       reference frame to determine spindle pole axis
    -f FRAMERATE, --framerate FRAMERATE    optional: number of frames per second
    -t THRESHOLD, --threshold THRESHOLD    optional: threshold of cytoplasmic background signal in spindle channel; value relative to max spindle intensity 0.0-1.0 (0.0=autodetect using Otsu)
    -b BLUR, --blur BLUR                   applies a gaussian blur before segmenting spindle poles. The value determines the blurring radius; a value of 0 omits blurring.
    -c, --cellpose                         use Cellpose to detect embryo contour

## Contributing

Contributions are very welcome.

## License

Distributed under the terms of the [BSD-3] license,
"mitosisanalyzer-napari" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[napari-plugin-template]: https://github.com/napari/napari-plugin-template

[file an issue]: https://github.com/ksiller/mitosisanalyzer-napari/issues

[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/

