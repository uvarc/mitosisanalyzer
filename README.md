# mitosisanalyzer

## Installation 

### Local workstation

Install Anaconda or Miniconda distribution on your computer. In a command line shell (e.g. Anaconda Prompt), execute the following commands:

```
git clone https://github.com/uvarc/mitosisanalyzer.git
cd mitosisanalyzer
conda env create -f environment.yml
```

### Afton/Rivanna

In a [Open OnDemand command line shell](https://www.rc.virginia.edu/userinfo/hpc/ood/#the-dashboard) (Clusters > _HPC Shell), execute the following commands:

```
cd /standard/redemann_lab/apps
git clone https://github.com/uvarc/mitosisanalyzer.git
cd mitosisanalyzer
module load miniforge
mamba env create -f environment.yml
```

## Running the MitosisAnalyzer application

### Local workstation

In a command line shell, run the following commands:
```
conda activate mitoanalyzer
python mitoanalysis.py -i imagestack.nd2 -o my_outputdir -s 1 -d 2 -r 1
```

### Afton/Rivanna (interactive)

Start an [interactive Desktop session on Open OnDemand](https://www.rc.virginia.edu/userinfo/hpc/ood/desktop/). Right clock on desktop to start a terminal session. In the terminal shell, run the following commands:
```
module load miniforge
source activate mitoanalyzer
cd /standard/redemann_lab/apps/mitosisanalyzer
python mitoanalysis.py -i imagestack.nd2 -o my_outputdir -s 1 -d 2 -r 1
```

### Command line arguments

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
