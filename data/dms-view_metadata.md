## Mutational antigenic profiling of the SARS-CoV-2 Delta variant

For experimental background, see our paper **here (update with link)**.

### What data are shown here?
We are showing mutations to the SARS-CoV-2 Delta RBD that escape binding by polyclonal antibodies from individuals who received the Pfizer/BioNTech mRNA vaccine, primary Delta infection, or who were vaccinated with an mRNA vaccine (Moderna or Pfizer/BioNTech) and then had a Delta breakthrough infection, measured using mutational antigenic profiling. Raw data are available raw data are available [here](https://github.com/jbloomlab/SARS-CoV-2-RBD_Delta/blob/main/results/supp_data/aggregate_raw_data.csv).
The drop-down menus can be used to select the escape-mutation maps for each antibody or plasma.

When you click on sites, they will be highlighted on the protein structure of the ACE2-bound Wuhan-Hu-1 RBD structure ([PDB 6M0J](https://www.rcsb.org/structure/6M0J)) or to the RBD of the Delta variant ([PDB 7V8B](https://www.rcsb.org/structure/7v8b)).

At the site level you can visualize one of two quantities:

 - *total escape* is the sum of the escape from all mutations at a site.

 - *max escape* is the magnitude of the largest-effect escape mutation at each site.

At the mutation level, the height of each letter is proportional to the extent to which that amino-acid mutation escapes antibody binding.
You can color the logo plot letters in four ways:

 - *escape color ACE2 bind* means color letters according to how that mutation affects ACE2 binding, with yellow meaning highly deleterious, and brown meaning neutral or beneficial for ACE2 binding.

 - *escape color RBD expr* means color letters according to how that mutation affects RBD expression.

 - *escape color gray* means color all letters gray.

 - *escape color func group* means color letters by functional group.
