# mitosisanalyzer

## Installation

Install Anaconda or Miniconda distributuon on your computer. In a command line shell (e.g. Anaconda Prompt), execute the following commands:

```
git clone https://github.com/uvarc/mitosisanalyzer.git
cd mitosisanalyzer
conda env create -f environment.yml
```

## Running the MitosisAnalyzer application

In a command line shell, run the following commands:
```
conda activate mitoanalyzer
python mitoanalysis.py -i imagestack.nd2 -o my_outputdir -s 1 -d 2 -r 1
```

**Command line arguments**

    -h, --help                             show help message and exit
    -i INPUT, --input INPUT                .nd2 file or directory with .nd2 files to be processed
    -o OUTPUT, --output OUTPUT             output file or directory
    -p PROCESSES, --processes PROCESSES    optional: number or parallel processes
    -s SPINDLE, --spindle SPINDLE          channel # for tracking spindle poles
    -d DNA, --dna DNA                      channel # for tracking dna
    -r REFFRAME, --refframe REFFRAME       reference frame to determine spindle pole axis
    -f FRAMERATE, --framerate FRAMERATE    optional: number of frames per second
