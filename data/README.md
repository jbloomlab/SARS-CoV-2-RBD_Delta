# Input data
This directory contains input data for the analysis.

## Basic information about sequences and alignments

These files are used for the basic processing of the deep sequencing data to call variants by barcode and count barcodes:

   - [wildtype_sequence.fasta](wildtype_sequence.fasta): The sequence of the unmutated Delta RBD.

   - [wuhan1_wildtype_sequence.fasta](wuhan1_wildtype_sequence.fasta): The sequence of the unmutated Wuhan-Hu-1 RBD.

   - [RBD_sites.csv](RBD_sites.csv): gives site and residue information for SARS-CoV-2, including alignment of the RBD integer numbering with the Spike numbering for SARS-CoV-2 RBD, alignment to SARS-CoV, and structural annotations as detailed below.

 ## Files related to processing NGS sequencing data

   - [PacBio_amplicons.gb](PacBio_amplicons.gb): the amplicons being sequenced by PacBio.
     Note that there are a variety of possible amplicons because in addition to the SARS-CoV-2 RBD there are RBDs from a variety of other viral strains.

   - [feature_parse_specs.yaml](feature_parse_specs.yaml): how to parse the amplicon when handling the PacBio data.

   - [PacBio_runs.csv](PacBio_runs.csv): list of the PacBio runs used to call the variants.

   - [barcode_runs.csv](barcode_runs.csv): list of the Illumina runs used to count the barcodes for different samples. This file is manually updated when new sequencing is added.

   - [barcode_runs_helper_files](barcode_runs_helper_files): subdirectory containing auxillary notebooks and files to help generate the `barcode_runs.csv` file that maps sample names to file paths on the Hutch cluster.


## Plasmid maps

[plasmids](plasmids/): This subdirectory contains the full Genbank maps for key plasmids used in the study:

  - [plasmids/3181_HDM_Spikedelta21_sinobiological_B.1.617.2.gb](plasmids/3181_HDM_Spikedelta21_sinobiological_B.1.617.2.gb): the wildtype Delta (B.1.617.2) spike expression plasmids used to make pseudotyped lentiviral particles,

  - [plasmids/3159_pETcon-SARS-CoV-2-RBD-L452R-T478K.gb](3159_pETcon-SARS-CoV-2-RBD-L452R-T478K.gb): the Delta wildtype RBD yeast-display plasmid, and

  - [plasmids/Twist_delivered.fa](plasmids/Twist_delivered.fa): the fully assembled plasmid, including the Illumina adaptors and Nx16 barcode, that was used as the template for designing the Twist site-saturation variant libraries.

## For visualizing serum-escape data:

These files are used for visualizing the antibody- or serum-escape data:

  - [site_color_schemes.csv](site_color_schemes.csv): Schemes for how to color sites (can be used in escape profiles). Here are details on these schemes.

  - [escape_profiles_config.yaml](escape_profiles_config.yaml): Information on how to plot the escape profiles; manually edit this to alter their plotting.

  - [early2020_escape_profiles_config.yaml](early2020_escape_profiles_config.yaml): Same as above, but for early2020 mapping data (as noted in ../config.yaml, the early2020_escape_fracs are imported from the Moderna repo). 

  - [lineplots_config.yaml](lineplots_config.yaml): Config file for making line plots to compare serum-escape scores between cohorts.

  - [dms-view_metadata.md](dms-view_metadata.md): Used for rendering the dms-view page to visualize data.

  - [output_pdbs_config.yaml](output_pdbs_config.yaml): Used for mapping antibody-escape data to the RBD surface.

  - [mds_config.yaml](mds_config.yaml): Config file for making MDS plot to compare serum-escape scores between cohorts, with monoclonal antibodies serving as anchors.

  - [mds_color_schemes.csv](mds_color_schemes.csv): Color scheme designation for MDS plots.

## PDB files in [pdbs](pdbs/) subdirectory

  - [6M0J](pdbs/6M0J.pdb): Wuhan-Hu-1 RBD bound to huACE2 ([Lan et al. (2020)](https://www.nature.com/articles/s41586-020-2180-5)).

  - [7V8B](pdbs/7V8B.pdb): Delta RBD bound to huACE2. See [here](https://www.rcsb.org/structure/7v8b).

## Alignments of different Spikes / RBDs

  - [210801_mutation_counts.csv](210801_mutation_counts.csv): Counts of all RBD mutations in the spikeprot alignment downloaded from GISAID on Aug. 1, 2021, not sub-setting on Delta variant sequences

## GISAID data that are not tracked in this repo

We use some surveillance data in aggregate to count occurrences of mutations in the RBD from [GISAID](https://www.gisaid.org/), but do not track those files in this public repo according to the [terms of use](https://www.gisaid.org/registration/terms-of-use/) for GISAID data sharing and use.

We gratefully acknowledge all contributors to the GISAID EpiCoV database.
