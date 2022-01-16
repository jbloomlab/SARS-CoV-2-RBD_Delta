# Experimental (i.e., non-DMS) data for Delta plasma-escape study
This directory contains the data and code for analyzing ELISA and neutralization assays to accompany the Delta plasma-escape mapping by DMS.

Note that all experiments are done with Delta RBD and spike proteins, and Delta spike-pseudotyped lentiviral particles.

### Input data
- The [./data/entry_titers/](data/entry_titers) subdirectory contains the files for calculating the entry titers of pseudotyped lentiviral particles used in neutralization assays. This is primarily used for determining how much lentiviral supernatant to use in each experiment, to target an appropriate amount of relative luciferase units per well.

- The [./data/rbd_depletion_elisas/](data/rbd_depletion_elisas) subdirectory contains files with OD450 readings before and after depletion of RBD-binding antibodies.

- The [./neut_data/](neut_data) subdirectory contains the raw Excel files and plate maps for each day of assays.

    ### Formatting data for neutralization assays
    Each neutralization assay should have its data stored in an Excel file in a subdirectory of [./neut_data/](neut_data) named with the data in the format `2020-10-02`.
    The subdirectories should also contain a `sample_map.csv` file that maps the Excel file data to the samples in a format that is readable by the Python script [excel_to_fractinfect.py](excel_to_fractinfect.py) (see [here](https://github.com/jbloomlab/exceltofractinfect) for more on this script, which was written by Kate Crawford).
    The plate layouts referred to by the sample maps are in [./PlateLayouts/](PlateLayouts).

- The [./data/previous_studies/neuts](data/previous_studies_neuts) subdirectory contains relevant NT50s or IC50s from [this study](https://github.com/jbloomlab/SARS-CoV-2-RBD_MAP_Moderna) on the antibody response elicited by mRNA-1273 vaccination [(Greaney, et al. (2021a))](https://www.science.org/doi/10.1126/scitranslmed.abi9915) and [this study](https://github.com/jbloomlab/SARS-CoV-2-RBD_B.1.351) on the antibody response to the Beta variant [(Greaney, et al. (2021b))](https://www.biorxiv.org/content/10.1101/2021.10.12.464114v1).
    I pre-processed the supplementary files from these studies that contained the neutralization titer information to remove irrelevant or redundant sera and information. 

### Running the code
Now you can run the entire analysis.
The analysis consists primarily of a series of Jupyter notebooks along with some additional code in [Snakefile](Snakefile).
You can run the analysis by using [Snakemake](https://snakemake.readthedocs.io) to run [Snakefile](Snakefile):

    snakemake --j1

## Configuring the analysis
The configuration for the analysis is specified in [config.yaml](config.yaml).

## Notebooks that perform the analysis
- [calculate_entry_titer.ipynb](calculate_entry_titer.ipynb): calculates PV entry titers in RLU.

- [rbd_depletion_elisas.ipynb](rbd_depletion_elisas.ipynb): analyzes ELISAs that measure binding of B.1.351 plasma to B.1.351 RBD and spike before and after depletion of B.1.351 RBD-binding antibodies.

- [rbd_depletion_neuts.ipynb](rbd_depletion_neuts.ipynb): analyzes assays that measure neutralization of B.1.351 spike-pseudotyped lentiviral particles before and after depletion of B.1.351 RBD-binding antibodies.

- [analyze_neut_data.ipynb](analyze_neut_data.ipynb): Analyzes point-mutant neuts.



## Results
Results are placed in the [./results/](results) subdirectory.
