# Summary

Analysis run by [Snakefile](../../Snakefile)
using [this config file](../../config.yaml).
See the [README in the top directory](../../README.md)
for details.

Here is the rule graph of the computational workflow:
![rulegraph.svg](rulegraph.svg)

Here is the Markdown output of each notebook in the workflow:
1. Get prior DMS mutation-level [binding and expression data](../prior_DMS_data/early2020_mutant_ACE2binding_expression.csv).

2. Get prior MAPping [escape_fracs](../prior_DMS_data/early2020_escape_fracs.csv) for polyclonal plasmas from early 2020 against the Wuhan-1 RBD library.

2. [Process PacBio CCSs](process_ccs.md).

3. [Build variants from CCSs](build_variants.md).
   Creates a [codon variant table](../variants/codon_variant_table.csv)
   linking barcodes to the mutations in the variants.