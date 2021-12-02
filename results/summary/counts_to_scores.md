# Analyze counts and compute escape scores
This Python Jupyter notebook analyzes the variant counts and looks at mutation coverage and jackpotting.
It then computes an "escape scores" for each variant after grouping by barcode or substitutions as specified in the configuration.

## Set up analysis

This notebook primarily makes use of the Bloom lab's [dms_variants](https://jbloomlab.github.io/dms_variants) package, and uses [plotnine](https://github.com/has2k1/plotnine) for ggplot2-like plotting syntax:


```python
import collections
import math
import os
import warnings

import Bio.SeqIO

import dms_variants.codonvarianttable
from dms_variants.constants import CBPALETTE
import dms_variants.plotnine_themes

from IPython.display import display, HTML

import matplotlib.pyplot as plt

import numpy

import pandas as pd

from plotnine import *

import seaborn

import yaml
```

Set [plotnine](https://github.com/has2k1/plotnine) theme to the gray-grid one defined in [dms_variants](https://jbloomlab.github.io/dms_variants):


```python
theme_set(dms_variants.plotnine_themes.theme_graygrid())
```

Versions of key software:


```python
print(f"Using dms_variants version {dms_variants.__version__}")
```

    Using dms_variants version 0.8.10


Ignore warnings that clutter output:


```python
warnings.simplefilter('ignore')
```

Read the configuration file:


```python
with open('config.yaml') as f:
    config = yaml.safe_load(f)
```

Create output directory:


```python
os.makedirs(config['escape_scores_dir'], exist_ok=True)
```

Read information about the samples:


```python
samples_df = (pd.read_csv(config['barcode_runs'])
              .query('experiment_type=="ab_selection"')
              .assign(selection=lambda x: x['sort_bin']
                      .map({'abneg':'escape', 'ref':'reference'}),
                      concentration=lambda x: x['concentration'].astype(int)
                     )
             )

display(HTML(samples_df.head().to_html()))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>experiment</th>
      <th>library</th>
      <th>antibody</th>
      <th>concentration</th>
      <th>sort_bin</th>
      <th>HutchBase</th>
      <th>experiment_type</th>
      <th>number_cells</th>
      <th>frac_escape</th>
      <th>sample</th>
      <th>R1</th>
      <th>selection</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>80</th>
      <td>211008</td>
      <td>delta_1</td>
      <td>lib1</td>
      <td>P02</td>
      <td>500</td>
      <td>abneg</td>
      <td>delta_1_lib1_abneg</td>
      <td>ab_selection</td>
      <td>601087.0</td>
      <td>0.048</td>
      <td>delta_1-P02-500-abneg</td>
      <td>/shared/ngs/illumina/agreaney/211104_D00300_1361_AHN3TKBCX3/Unaligned/Project_agreaney/delta_1_lib1_abneg*R1*.fastq.gz</td>
      <td>escape</td>
    </tr>
    <tr>
      <th>81</th>
      <td>211008</td>
      <td>delta_1</td>
      <td>lib2</td>
      <td>P02</td>
      <td>500</td>
      <td>abneg</td>
      <td>delta_1_lib2_abneg</td>
      <td>ab_selection</td>
      <td>588481.0</td>
      <td>0.042</td>
      <td>delta_1-P02-500-abneg</td>
      <td>/shared/ngs/illumina/agreaney/211104_D00300_1361_AHN3TKBCX3/Unaligned/Project_agreaney/delta_1_lib2_abneg*R1*.fastq.gz</td>
      <td>escape</td>
    </tr>
    <tr>
      <th>82</th>
      <td>211008</td>
      <td>delta_2</td>
      <td>lib1</td>
      <td>P03</td>
      <td>1250</td>
      <td>abneg</td>
      <td>delta_2_lib1_abneg</td>
      <td>ab_selection</td>
      <td>554708.0</td>
      <td>0.050</td>
      <td>delta_2-P03-1250-abneg</td>
      <td>/shared/ngs/illumina/agreaney/211104_D00300_1361_AHN3TKBCX3/Unaligned/Project_agreaney/delta_2_lib1_abneg*R1*.fastq.gz</td>
      <td>escape</td>
    </tr>
    <tr>
      <th>83</th>
      <td>211008</td>
      <td>delta_2</td>
      <td>lib2</td>
      <td>P03</td>
      <td>1250</td>
      <td>abneg</td>
      <td>delta_2_lib2_abneg</td>
      <td>ab_selection</td>
      <td>553123.0</td>
      <td>0.044</td>
      <td>delta_2-P03-1250-abneg</td>
      <td>/shared/ngs/illumina/agreaney/211104_D00300_1361_AHN3TKBCX3/Unaligned/Project_agreaney/delta_2_lib2_abneg*R1*.fastq.gz</td>
      <td>escape</td>
    </tr>
    <tr>
      <th>84</th>
      <td>211008</td>
      <td>delta_3</td>
      <td>lib1</td>
      <td>P04</td>
      <td>1250</td>
      <td>abneg</td>
      <td>delta_3_lib1_abneg</td>
      <td>ab_selection</td>
      <td>603708.0</td>
      <td>0.052</td>
      <td>delta_3-P04-1250-abneg</td>
      <td>/shared/ngs/illumina/agreaney/211104_D00300_1361_AHN3TKBCX3/Unaligned/Project_agreaney/delta_3_lib1_abneg*R1*.fastq.gz</td>
      <td>escape</td>
    </tr>
  </tbody>
</table>


## Initialize codon-variant table
Initialize [CodonVariantTable](https://jbloomlab.github.io/dms_variants/dms_variants.codonvarianttable.html#dms_variants.codonvarianttable.CodonVariantTable) from wildtype gene sequence and the variant counts CSV file.
We will then use the plotting functions of this variant table to analyze the counts per sample:


```python
wt_seqrecord = Bio.SeqIO.read(config['wildtype_sequence'], 'fasta')
geneseq = str(wt_seqrecord.seq)
primary_target = wt_seqrecord.name
print(f"Read sequence of {len(geneseq)} nt for {primary_target} from {config['wildtype_sequence']}")
      
print(f"Initializing CodonVariantTable from gene sequence and {config['variant_counts']}")
      
variants = dms_variants.codonvarianttable.CodonVariantTable.from_variant_count_df(
                geneseq=geneseq,
                variant_count_df_file=config['variant_counts'],
                primary_target=primary_target)
      
print('Done initializing CodonVariantTable.')
```

    Read sequence of 603 nt for Delta from data/wildtype_sequence.fasta
    Initializing CodonVariantTable from gene sequence and results/counts/variant_counts.csv.gz
    Done initializing CodonVariantTable.


## Sequencing counts per sample
Average counts per variant for each sample.
Note that these are the **sequencing** counts, in some cases they may outstrip the actual number of sorted cells:


```python
p = variants.plotAvgCountsPerVariant(libraries=variants.libraries,
                                     by_target=False,
                                     orientation='v')
p = p + theme(panel_grid_major_x=element_blank())  # no vertical grid lines
_ = p.draw()
```


    
![png](counts_to_scores_files/counts_to_scores_19_0.png)
    


And the numerical values plotted above:


```python
display(HTML(
 variants.avgCountsPerVariant(libraries=variants.libraries,
                               by_target=False)
 .pivot_table(index='sample',
              columns='library',
              values='avg_counts_per_variant')
 .round(1)
 .to_html()
 ))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>library</th>
      <th>lib1</th>
      <th>lib2</th>
    </tr>
    <tr>
      <th>sample</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TiteSeq_01_bin1</th>
      <td>22.5</td>
      <td>18.4</td>
    </tr>
    <tr>
      <th>TiteSeq_01_bin2</th>
      <td>15.3</td>
      <td>16.8</td>
    </tr>
    <tr>
      <th>TiteSeq_01_bin3</th>
      <td>28.7</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>TiteSeq_01_bin4</th>
      <td>170.9</td>
      <td>146.5</td>
    </tr>
    <tr>
      <th>TiteSeq_02_bin1</th>
      <td>25.0</td>
      <td>27.5</td>
    </tr>
    <tr>
      <th>TiteSeq_02_bin2</th>
      <td>15.6</td>
      <td>9.8</td>
    </tr>
    <tr>
      <th>TiteSeq_02_bin3</th>
      <td>25.6</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>TiteSeq_02_bin4</th>
      <td>177.4</td>
      <td>52.2</td>
    </tr>
    <tr>
      <th>TiteSeq_03_bin1</th>
      <td>44.3</td>
      <td>58.5</td>
    </tr>
    <tr>
      <th>TiteSeq_03_bin2</th>
      <td>23.2</td>
      <td>26.1</td>
    </tr>
    <tr>
      <th>TiteSeq_03_bin3</th>
      <td>64.8</td>
      <td>45.1</td>
    </tr>
    <tr>
      <th>TiteSeq_03_bin4</th>
      <td>59.0</td>
      <td>53.7</td>
    </tr>
    <tr>
      <th>TiteSeq_04_bin1</th>
      <td>94.7</td>
      <td>81.1</td>
    </tr>
    <tr>
      <th>TiteSeq_04_bin2</th>
      <td>69.9</td>
      <td>62.4</td>
    </tr>
    <tr>
      <th>TiteSeq_04_bin3</th>
      <td>35.7</td>
      <td>38.9</td>
    </tr>
    <tr>
      <th>TiteSeq_04_bin4</th>
      <td>0.3</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>TiteSeq_05_bin1</th>
      <td>180.2</td>
      <td>141.2</td>
    </tr>
    <tr>
      <th>TiteSeq_05_bin2</th>
      <td>35.3</td>
      <td>32.8</td>
    </tr>
    <tr>
      <th>TiteSeq_05_bin3</th>
      <td>0.2</td>
      <td>5.9</td>
    </tr>
    <tr>
      <th>TiteSeq_05_bin4</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TiteSeq_06_bin1</th>
      <td>210.8</td>
      <td>188.9</td>
    </tr>
    <tr>
      <th>TiteSeq_06_bin2</th>
      <td>6.3</td>
      <td>11.8</td>
    </tr>
    <tr>
      <th>TiteSeq_06_bin3</th>
      <td>0.4</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TiteSeq_06_bin4</th>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TiteSeq_07_bin1</th>
      <td>236.4</td>
      <td>207.8</td>
    </tr>
    <tr>
      <th>TiteSeq_07_bin2</th>
      <td>9.7</td>
      <td>14.5</td>
    </tr>
    <tr>
      <th>TiteSeq_07_bin3</th>
      <td>0.5</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TiteSeq_07_bin4</th>
      <td>0.1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TiteSeq_08_bin1</th>
      <td>222.6</td>
      <td>144.3</td>
    </tr>
    <tr>
      <th>TiteSeq_08_bin2</th>
      <td>3.7</td>
      <td>8.3</td>
    </tr>
    <tr>
      <th>TiteSeq_08_bin3</th>
      <td>0.4</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TiteSeq_08_bin4</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TiteSeq_09_bin1</th>
      <td>272.2</td>
      <td>108.5</td>
    </tr>
    <tr>
      <th>TiteSeq_09_bin2</th>
      <td>3.3</td>
      <td>4.4</td>
    </tr>
    <tr>
      <th>TiteSeq_09_bin3</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TiteSeq_09_bin4</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>SortSeq_bin1</th>
      <td>76.6</td>
      <td>61.3</td>
    </tr>
    <tr>
      <th>SortSeq_bin2</th>
      <td>102.0</td>
      <td>112.4</td>
    </tr>
    <tr>
      <th>SortSeq_bin3</th>
      <td>116.1</td>
      <td>76.1</td>
    </tr>
    <tr>
      <th>SortSeq_bin4</th>
      <td>116.0</td>
      <td>107.7</td>
    </tr>
    <tr>
      <th>delta_1-P02-500-abneg</th>
      <td>21.5</td>
      <td>19.6</td>
    </tr>
    <tr>
      <th>delta_2-P03-1250-abneg</th>
      <td>21.2</td>
      <td>18.9</td>
    </tr>
    <tr>
      <th>delta_3-P04-1250-abneg</th>
      <td>23.5</td>
      <td>19.3</td>
    </tr>
    <tr>
      <th>delta_4-P05-500-abneg</th>
      <td>15.5</td>
      <td>11.3</td>
    </tr>
    <tr>
      <th>delta_5-P08-500-abneg</th>
      <td>37.5</td>
      <td>26.8</td>
    </tr>
    <tr>
      <th>delta_6-P09-200-abneg</th>
      <td>30.3</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>delta_7-P12-200-abneg</th>
      <td>24.6</td>
      <td>19.9</td>
    </tr>
    <tr>
      <th>delta_8-P14-1250-abneg</th>
      <td>35.3</td>
      <td>24.4</td>
    </tr>
    <tr>
      <th>delta_9-267C-200-abneg</th>
      <td>23.6</td>
      <td>16.8</td>
    </tr>
    <tr>
      <th>delta_10-268C-500-abneg</th>
      <td>23.2</td>
      <td>17.1</td>
    </tr>
    <tr>
      <th>delta_11-273C-500-abneg</th>
      <td>25.0</td>
      <td>17.3</td>
    </tr>
    <tr>
      <th>delta_12-274C-500-abneg</th>
      <td>26.1</td>
      <td>19.5</td>
    </tr>
    <tr>
      <th>delta_13-276C-500-abneg</th>
      <td>25.3</td>
      <td>19.2</td>
    </tr>
    <tr>
      <th>delta_14-277C-500-abneg</th>
      <td>21.9</td>
      <td>17.1</td>
    </tr>
    <tr>
      <th>delta_15-278C-1250-abneg</th>
      <td>23.4</td>
      <td>18.2</td>
    </tr>
    <tr>
      <th>delta_16-279C-1250-abneg</th>
      <td>21.3</td>
      <td>15.9</td>
    </tr>
    <tr>
      <th>delta_1-8-none-0-ref</th>
      <td>306.7</td>
      <td>280.4</td>
    </tr>
    <tr>
      <th>delta_9-16-none-0-ref</th>
      <td>362.2</td>
      <td>343.0</td>
    </tr>
    <tr>
      <th>delta_17-20-none-0-ref</th>
      <td>503.6</td>
      <td>431.9</td>
    </tr>
    <tr>
      <th>delta_17-P03_repeat-1250-abneg</th>
      <td>55.9</td>
      <td>42.3</td>
    </tr>
    <tr>
      <th>delta_18-P08_repeat-500-abneg</th>
      <td>50.3</td>
      <td>48.1</td>
    </tr>
    <tr>
      <th>delta_19-268C_repeat-500-abneg</th>
      <td>47.2</td>
      <td>44.2</td>
    </tr>
    <tr>
      <th>delta_20-279C_repeat-1250-abneg</th>
      <td>56.0</td>
      <td>53.4</td>
    </tr>
    <tr>
      <th>delta_21-26-none-0-ref</th>
      <td>151.9</td>
      <td>117.9</td>
    </tr>
    <tr>
      <th>delta_21-267C_repeat-200-abneg</th>
      <td>18.2</td>
      <td>15.2</td>
    </tr>
    <tr>
      <th>delta_22-273C_repeat-500-abneg</th>
      <td>18.1</td>
      <td>14.8</td>
    </tr>
    <tr>
      <th>delta_23-274C_repeat-500-abneg</th>
      <td>18.9</td>
      <td>14.2</td>
    </tr>
    <tr>
      <th>delta_24-276C_repeat-500-abneg</th>
      <td>18.3</td>
      <td>16.1</td>
    </tr>
    <tr>
      <th>delta_25-277C_repeat-500-abneg</th>
      <td>19.3</td>
      <td>15.5</td>
    </tr>
    <tr>
      <th>delta_26-278C_repeat-1250-abneg</th>
      <td>17.3</td>
      <td>13.7</td>
    </tr>
    <tr>
      <th>delta_27-34-none-0-ref</th>
      <td>123.2</td>
      <td>115.6</td>
    </tr>
    <tr>
      <th>delta_27-Delta_1-500-abneg</th>
      <td>15.4</td>
      <td>14.4</td>
    </tr>
    <tr>
      <th>delta_28-Delta_3-350-abneg</th>
      <td>15.8</td>
      <td>15.4</td>
    </tr>
    <tr>
      <th>delta_29-Delta_4-350-abneg</th>
      <td>10.7</td>
      <td>8.9</td>
    </tr>
    <tr>
      <th>delta_30-Delta_6-500-abneg</th>
      <td>16.5</td>
      <td>14.7</td>
    </tr>
    <tr>
      <th>delta_31-Delta_7-1250-abneg</th>
      <td>18.1</td>
      <td>11.5</td>
    </tr>
    <tr>
      <th>delta_32-Delta_8-500-abneg</th>
      <td>15.9</td>
      <td>10.3</td>
    </tr>
    <tr>
      <th>delta_33-Delta_10-1250-abneg</th>
      <td>12.4</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>delta_34-Delta_11-500-abneg</th>
      <td>15.2</td>
      <td>11.2</td>
    </tr>
    <tr>
      <th>delta_35-40-none-0-ref</th>
      <td>103.5</td>
      <td>108.1</td>
    </tr>
    <tr>
      <th>delta_35-P02_repeat-500-abneg</th>
      <td>14.6</td>
      <td>8.8</td>
    </tr>
    <tr>
      <th>delta_36-P04_repeat-1250-abneg</th>
      <td>13.8</td>
      <td>10.7</td>
    </tr>
    <tr>
      <th>delta_37-P05_repeat-500-abneg</th>
      <td>12.6</td>
      <td>11.8</td>
    </tr>
    <tr>
      <th>delta_38-P09_repeat-200-abneg</th>
      <td>15.1</td>
      <td>9.5</td>
    </tr>
    <tr>
      <th>delta_39-P12_repeat-200-abneg</th>
      <td>13.2</td>
      <td>11.7</td>
    </tr>
    <tr>
      <th>delta_40-P14_repeat-1250-abneg</th>
      <td>13.0</td>
      <td>14.9</td>
    </tr>
  </tbody>
</table>


## Mutations per variant
Average number of mutations per gene among all variants of the primary target, separately for each date:


```python
#this plotting is very slow when lots of samples, so for now plots are commented out

for date, date_df in samples_df.groupby('date', sort=False):
   p = variants.plotNumCodonMutsByType(variant_type='all',
                                       orientation='v',
                                       libraries=variants.libraries,
                                       samples=date_df['sample'].unique().tolist(),
                                       widthscale=2)
   p = p + theme(panel_grid_major_x=element_blank())  # no vertical grid lines
   fig = p.draw()
   display(fig)
   plt.close(fig)
```


    
![png](counts_to_scores_files/counts_to_scores_23_0.png)
    



    
![png](counts_to_scores_files/counts_to_scores_23_1.png)
    



    
![png](counts_to_scores_files/counts_to_scores_23_2.png)
    



    
![png](counts_to_scores_files/counts_to_scores_23_3.png)
    



    
![png](counts_to_scores_files/counts_to_scores_23_4.png)
    



    
![png](counts_to_scores_files/counts_to_scores_23_5.png)
    


Now similar plots but showing mutation frequency across the gene:


```python
# this plotting is very slow when lots of samples, so for now code commented out

for date, date_df in samples_df.groupby('date', sort=False):
   p = variants.plotMutFreqs(variant_type='all',
                             mut_type='codon',
                             orientation='v',
                             libraries=variants.libraries,
                             samples=date_df['sample'].unique().tolist(),
                             widthscale=1.5)
   fig = p.draw()
   display(fig)
   plt.close(fig)
```


    
![png](counts_to_scores_files/counts_to_scores_25_0.png)
    



    
![png](counts_to_scores_files/counts_to_scores_25_1.png)
    



    
![png](counts_to_scores_files/counts_to_scores_25_2.png)
    



    
![png](counts_to_scores_files/counts_to_scores_25_3.png)
    



    
![png](counts_to_scores_files/counts_to_scores_25_4.png)
    



    
![png](counts_to_scores_files/counts_to_scores_25_5.png)
    


## Jackpotting and mutation coverage in pre-selection libraries
We look at the distribution of counts in the "reference" (pre-selection) libraries to see if they seem jackpotted (a few variants at very high frequency):


```python
pre_samples_df = samples_df.query('selection == "reference"')
```

Distribution of mutations along the gene for the pre-selection samples; big spikes may indicate jackpotting:


```python
# this plotting is very slow when lots of samples, so for now code commented out

p = variants.plotMutFreqs(variant_type='all',
                         mut_type='codon',
                         orientation='v',
                         libraries=variants.libraries,
                         samples=pre_samples_df['sample'].unique().tolist(),
                         widthscale=1.5)
_ = p.draw()
```


    
![png](counts_to_scores_files/counts_to_scores_29_0.png)
    


How many mutations are observed frequently in pre-selection libraries?
Note that the libraries have been pre-selected for ACE2 binding, so we expect stop variants to mostly be missing.
Make the plot both for all variants and just single-mutant variants:


```python
# this plotting is very slow when lots of samples, so for now code commented out

for variant_type in ['all', 'single']:
   p = variants.plotCumulMutCoverage(
                         variant_type=variant_type,
                         mut_type='aa',
                         orientation='v',
                         libraries=variants.libraries,
                         samples=pre_samples_df['sample'].unique().tolist(),
                         widthscale=1.8,
                         heightscale=1.2)
   _ = p.draw()
```


    
![png](counts_to_scores_files/counts_to_scores_31_0.png)
    



    
![png](counts_to_scores_files/counts_to_scores_31_1.png)
    


Now make a plot showing what number and fraction of counts are for each variant in each pre-selection sample / library.
If some variants constitute a very high fraction, then that indicates extensive bottlenecking:


```python
for ystat in ['frac_counts', 'count']:
    p = variants.plotCountsPerVariant(ystat=ystat,
                                      libraries=variants.libraries,
                                      samples=pre_samples_df['sample'].unique().tolist(),
                                      orientation='v',
                                      widthscale=1.75,
                                      )
    _ = p.draw()
```


    
![png](counts_to_scores_files/counts_to_scores_33_0.png)
    



    
![png](counts_to_scores_files/counts_to_scores_33_1.png)
    


Now make the same plot breaking down by variant class, which enables determination of which types of variants are at high and low frequencies.
For this plot (unlike one above not classified by category) we only show variants of the primary target (not the homologs), and also group synonymous with wildtype in order to reduce number of plotted categories to make more interpretable:


```python
for ystat in ['frac_counts', 'count']:
    p = variants.plotCountsPerVariant(ystat=ystat,
                                      libraries=variants.libraries,
                                      samples=pre_samples_df['sample'].unique().tolist(),
                                      orientation='v',
                                      widthscale=1.75,
                                      by_variant_class=True,
                                      classifyVariants_kwargs={'syn_as_wt': True},
                                      primary_target_only=True,
                                      )
    _ = p.draw()
```


    
![png](counts_to_scores_files/counts_to_scores_35_0.png)
    



    
![png](counts_to_scores_files/counts_to_scores_35_1.png)
    


We also directly look to see what is the variant in each reference library / sample with the highest fraction counts.
Knowing if the highest frequency variant is shared helps determine **where** in the experiment the jackpotting happened:


```python
frac_counts_per_variant = (
        variants.add_frac_counts(variants.variant_count_df)
        .query(f"sample in {pre_samples_df['sample'].unique().tolist()}")
        )

display(HTML(
    frac_counts_per_variant
    .sort_values('frac_counts', ascending=False)
    .groupby(['library', 'sample'])
    .head(n=1)
    .sort_values(['library', 'sample'])
    .set_index(['library', 'sample'])
    [['frac_counts', 'target', 'barcode', 'aa_substitutions', 'codon_substitutions']]
    .round(4)
    .to_html()
    ))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>frac_counts</th>
      <th>target</th>
      <th>barcode</th>
      <th>aa_substitutions</th>
      <th>codon_substitutions</th>
    </tr>
    <tr>
      <th>library</th>
      <th>sample</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="6" valign="top">lib1</th>
      <th>delta_1-8-none-0-ref</th>
      <td>0.0002</td>
      <td>Delta</td>
      <td>GCTACAAAGTCCGCAG</td>
      <td>D68T</td>
      <td>GAT68ACT</td>
    </tr>
    <tr>
      <th>delta_9-16-none-0-ref</th>
      <td>0.0002</td>
      <td>Delta</td>
      <td>GCTACAAAGTCCGCAG</td>
      <td>D68T</td>
      <td>GAT68ACT</td>
    </tr>
    <tr>
      <th>delta_17-20-none-0-ref</th>
      <td>0.0002</td>
      <td>Delta</td>
      <td>TATGTTTCCGACCCGC</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>delta_21-26-none-0-ref</th>
      <td>0.0002</td>
      <td>Delta</td>
      <td>TATGTTTCCGACCCGC</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>delta_27-34-none-0-ref</th>
      <td>0.0002</td>
      <td>Delta</td>
      <td>TATGTTTCCGACCCGC</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>delta_35-40-none-0-ref</th>
      <td>0.0002</td>
      <td>Delta</td>
      <td>TATGTTTCCGACCCGC</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">lib2</th>
      <th>delta_1-8-none-0-ref</th>
      <td>0.0002</td>
      <td>Delta</td>
      <td>ACACAAACGCCGTTAC</td>
      <td>A67S</td>
      <td>GCC67TCT</td>
    </tr>
    <tr>
      <th>delta_9-16-none-0-ref</th>
      <td>0.0002</td>
      <td>Delta</td>
      <td>ACACAAACGCCGTTAC</td>
      <td>A67S</td>
      <td>GCC67TCT</td>
    </tr>
    <tr>
      <th>delta_17-20-none-0-ref</th>
      <td>0.0002</td>
      <td>Delta</td>
      <td>ACACAAACGCCGTTAC</td>
      <td>A67S</td>
      <td>GCC67TCT</td>
    </tr>
    <tr>
      <th>delta_21-26-none-0-ref</th>
      <td>0.0001</td>
      <td>Delta</td>
      <td>CCCGTTGGGAATACCG</td>
      <td>L187K</td>
      <td>CTG187AAA</td>
    </tr>
    <tr>
      <th>delta_27-34-none-0-ref</th>
      <td>0.0001</td>
      <td>Delta</td>
      <td>CCCGTTGGGAATACCG</td>
      <td>L187K</td>
      <td>CTG187AAA</td>
    </tr>
    <tr>
      <th>delta_35-40-none-0-ref</th>
      <td>0.0001</td>
      <td>Delta</td>
      <td>CCCGTTGGGAATACCG</td>
      <td>L187K</td>
      <td>CTG187AAA</td>
    </tr>
  </tbody>
</table>


To further where the jackpotting relative to the generation of the reference samples, we plot the correlation among the fraction of counts for the different reference samples.
If the fractions are highly correlated, that indicates that the jackpotting occurred in some upstream step common to the reference samples:


```python
# this code makes a full matrix of scatter plots, but is REALLY SLOW. So for now,
# it is commented out in favor of code that just makes correlation matrix.
for lib, lib_df in frac_counts_per_variant.groupby('library'):
   wide_lib_df = lib_df.pivot_table(index=['target', 'barcode'],
                                    columns='sample',
                                    values='frac_counts')
   g = seaborn.pairplot(wide_lib_df, corner=True, plot_kws={'alpha': 0.5}, diag_kind='kde')
   _ = g.fig.suptitle(lib, size=18)
   plt.show()
```


    
![png](counts_to_scores_files/counts_to_scores_39_0.png)
    



    
![png](counts_to_scores_files/counts_to_scores_39_1.png)
    


## Examine counts for wildtype variants
The type of score we use to quantify escape depends on how well represented wildtype is in the selected libraries.
If wildtype is still well represented, we can use a more conventional functional score that gives differential selection relative to wildtype.
If wildtype is not well represented, then we need an alternative score that does not involve normalizing frequencies to wildtype.

First get average fraction of counts per variant for each variant class:


```python
counts_by_class = (
    variants.variant_count_df
    .pipe(variants.add_frac_counts)
    .pipe(variants.classifyVariants,
          primary_target=variants.primary_target,
          non_primary_target_class='homolog',
          class_as_categorical=True)
    .groupby(['library', 'sample', 'variant_class'])
    .aggregate(avg_frac_counts=pd.NamedAgg('frac_counts', 'mean'))
    .reset_index()
    .merge(samples_df[['sample', 'library', 'date', 'antibody', 'concentration', 'selection']],
           on=['sample', 'library'], validate='many_to_one')
    )

display(HTML(counts_by_class.head().to_html(index=False)))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>library</th>
      <th>sample</th>
      <th>variant_class</th>
      <th>avg_frac_counts</th>
      <th>date</th>
      <th>antibody</th>
      <th>concentration</th>
      <th>selection</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>lib1</td>
      <td>delta_1-P02-500-abneg</td>
      <td>wildtype</td>
      <td>1.604349e-07</td>
      <td>211008</td>
      <td>P02</td>
      <td>500</td>
      <td>escape</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>delta_1-P02-500-abneg</td>
      <td>synonymous</td>
      <td>2.859010e-07</td>
      <td>211008</td>
      <td>P02</td>
      <td>500</td>
      <td>escape</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>delta_1-P02-500-abneg</td>
      <td>1 nonsynonymous</td>
      <td>1.111423e-05</td>
      <td>211008</td>
      <td>P02</td>
      <td>500</td>
      <td>escape</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>delta_1-P02-500-abneg</td>
      <td>&gt;1 nonsynonymous</td>
      <td>2.330429e-05</td>
      <td>211008</td>
      <td>P02</td>
      <td>500</td>
      <td>escape</td>
    </tr>
    <tr>
      <td>lib1</td>
      <td>delta_1-P02-500-abneg</td>
      <td>stop</td>
      <td>5.625689e-07</td>
      <td>211008</td>
      <td>P02</td>
      <td>500</td>
      <td>escape</td>
    </tr>
  </tbody>
</table>


Plot average fraction of all counts per variant for each variant class.
If the values for wildtype are low for the non-reference samples (such as more similar to stop the nonsynonymous), then normalizing by wildtype in calculating scores will probably not work well as wildtype is too depleted:


```python
min_frac = 1e-7  # plot values < this as this

p = (ggplot(counts_by_class
            .assign(avg_frac_counts=lambda x: numpy.clip(x['avg_frac_counts'], min_frac, None))
            ) +
     aes('avg_frac_counts', 'sample', color='selection') +
     geom_point(size=2) +
     scale_color_manual(values=CBPALETTE[1:]) +
     facet_grid('library ~ variant_class') +
     scale_x_log10() +
     theme(axis_text_x=element_text(angle=90),
           figure_size=(2.5 * counts_by_class['variant_class'].nunique(),
                        0.2 * counts_by_class['library'].nunique() * 
                        counts_by_class['sample'].nunique())
           ) +
     geom_vline(xintercept=min_frac, linetype='dotted', color=CBPALETTE[3])
     )

_ = p.draw()
```


    
![png](counts_to_scores_files/counts_to_scores_43_0.png)
    


## Compute escape scores
We use the escape score metric, which does **not** involve normalizing to wildtype and so isn't strongly affected by low wildtype counts.
We compute the scores using the method [dms_variants.codonvarianttable.CodonVariantTable.escape_scores](https://jbloomlab.github.io/dms_variants/dms_variants.codonvarianttable.html?highlight=escape_scores#dms_variants.codonvarianttable.CodonVariantTable.escape_scores).

First, define what samples to compare for each calculation, matching each post-selection (escape) to the pre-selection (reference) sample on the same date:


```python
score_sample_df = (
    samples_df
    .query('selection == "escape"')
    .rename(columns={'sample': 'post_sample',
                     'number_cells': 'pre_cells_sorted'})
    .merge(samples_df
           .query('selection == "reference"')
           [['sample', 'library', 'date', 'number_cells']]
           .rename(columns={'sample': 'pre_sample',
                            'number_cells': 'post_cells_sorted'}),
           how='left', on=['date', 'library'], validate='many_to_one',
           )
    .assign(name=lambda x: x['antibody'] + '_' + x['concentration'].astype(str))
    # add dates to names where needed to make unique
    .assign(n_libs=lambda x: x.groupby(['name', 'date'])['pre_sample'].transform('count'))
    .sort_values(['name', 'date', 'n_libs'], ascending=False)
    .assign(i_name=lambda x: x.groupby(['library', 'name'], sort=False)['pre_sample'].cumcount(),
            name=lambda x: x.apply(lambda r: r['name'] + '_' + str(r['date']) if r['i_name'] > 0 else r['name'],
                                   axis=1),
            )
    .sort_values(['antibody', 'concentration', 'library', 'i_name'])
    # get columns of interest
    [['name', 'library', 'antibody', 'concentration', 'date',
      'pre_sample', 'post_sample', 'frac_escape', 'pre_cells_sorted', 'post_cells_sorted']]
    )

assert len(score_sample_df.groupby(['name', 'library'])) == len(score_sample_df)

print(f"Writing samples used to compute escape scores to {config['escape_score_samples']}\n")
score_sample_df.to_csv(config['escape_score_samples'], index=False)

display(HTML(score_sample_df.to_html(index=False)))
```

    Writing samples used to compute escape scores to results/escape_scores/samples.csv
    



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>name</th>
      <th>library</th>
      <th>antibody</th>
      <th>concentration</th>
      <th>date</th>
      <th>pre_sample</th>
      <th>post_sample</th>
      <th>frac_escape</th>
      <th>pre_cells_sorted</th>
      <th>post_cells_sorted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>267C_200</td>
      <td>lib1</td>
      <td>267C</td>
      <td>200</td>
      <td>211015</td>
      <td>delta_9-16-none-0-ref</td>
      <td>delta_9-267C-200-abneg</td>
      <td>0.053</td>
      <td>607515.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>267C_200</td>
      <td>lib2</td>
      <td>267C</td>
      <td>200</td>
      <td>211015</td>
      <td>delta_9-16-none-0-ref</td>
      <td>delta_9-267C-200-abneg</td>
      <td>0.044</td>
      <td>573755.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>267C_repeat_200</td>
      <td>lib1</td>
      <td>267C_repeat</td>
      <td>200</td>
      <td>211119</td>
      <td>delta_21-26-none-0-ref</td>
      <td>delta_21-267C_repeat-200-abneg</td>
      <td>0.112</td>
      <td>1206433.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>267C_repeat_200</td>
      <td>lib2</td>
      <td>267C_repeat</td>
      <td>200</td>
      <td>211119</td>
      <td>delta_21-26-none-0-ref</td>
      <td>delta_21-267C_repeat-200-abneg</td>
      <td>0.094</td>
      <td>1078707.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>268C_500</td>
      <td>lib1</td>
      <td>268C</td>
      <td>500</td>
      <td>211015</td>
      <td>delta_9-16-none-0-ref</td>
      <td>delta_10-268C-500-abneg</td>
      <td>0.047</td>
      <td>594157.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>268C_500</td>
      <td>lib2</td>
      <td>268C</td>
      <td>500</td>
      <td>211015</td>
      <td>delta_9-16-none-0-ref</td>
      <td>delta_10-268C-500-abneg</td>
      <td>0.041</td>
      <td>558357.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>268C_repeat_500</td>
      <td>lib1</td>
      <td>268C_repeat</td>
      <td>500</td>
      <td>211112</td>
      <td>delta_17-20-none-0-ref</td>
      <td>delta_19-268C_repeat-500-abneg</td>
      <td>0.107</td>
      <td>1012056.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>268C_repeat_500</td>
      <td>lib2</td>
      <td>268C_repeat</td>
      <td>500</td>
      <td>211112</td>
      <td>delta_17-20-none-0-ref</td>
      <td>delta_19-268C_repeat-500-abneg</td>
      <td>0.119</td>
      <td>1023050.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>273C_500</td>
      <td>lib1</td>
      <td>273C</td>
      <td>500</td>
      <td>211015</td>
      <td>delta_9-16-none-0-ref</td>
      <td>delta_11-273C-500-abneg</td>
      <td>0.046</td>
      <td>611575.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>273C_500</td>
      <td>lib2</td>
      <td>273C</td>
      <td>500</td>
      <td>211015</td>
      <td>delta_9-16-none-0-ref</td>
      <td>delta_11-273C-500-abneg</td>
      <td>0.041</td>
      <td>542611.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>273C_repeat_500</td>
      <td>lib1</td>
      <td>273C_repeat</td>
      <td>500</td>
      <td>211119</td>
      <td>delta_21-26-none-0-ref</td>
      <td>delta_22-273C_repeat-500-abneg</td>
      <td>0.126</td>
      <td>1269137.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>273C_repeat_500</td>
      <td>lib2</td>
      <td>273C_repeat</td>
      <td>500</td>
      <td>211119</td>
      <td>delta_21-26-none-0-ref</td>
      <td>delta_22-273C_repeat-500-abneg</td>
      <td>0.125</td>
      <td>1249629.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>274C_500</td>
      <td>lib1</td>
      <td>274C</td>
      <td>500</td>
      <td>211015</td>
      <td>delta_9-16-none-0-ref</td>
      <td>delta_12-274C-500-abneg</td>
      <td>0.044</td>
      <td>571317.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>274C_500</td>
      <td>lib2</td>
      <td>274C</td>
      <td>500</td>
      <td>211015</td>
      <td>delta_9-16-none-0-ref</td>
      <td>delta_12-274C-500-abneg</td>
      <td>0.044</td>
      <td>560293.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>274C_repeat_500</td>
      <td>lib1</td>
      <td>274C_repeat</td>
      <td>500</td>
      <td>211119</td>
      <td>delta_21-26-none-0-ref</td>
      <td>delta_23-274C_repeat-500-abneg</td>
      <td>0.122</td>
      <td>1233005.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>274C_repeat_500</td>
      <td>lib2</td>
      <td>274C_repeat</td>
      <td>500</td>
      <td>211119</td>
      <td>delta_21-26-none-0-ref</td>
      <td>delta_23-274C_repeat-500-abneg</td>
      <td>0.108</td>
      <td>1094894.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>276C_500</td>
      <td>lib1</td>
      <td>276C</td>
      <td>500</td>
      <td>211015</td>
      <td>delta_9-16-none-0-ref</td>
      <td>delta_13-276C-500-abneg</td>
      <td>0.047</td>
      <td>558556.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>276C_500</td>
      <td>lib2</td>
      <td>276C</td>
      <td>500</td>
      <td>211015</td>
      <td>delta_9-16-none-0-ref</td>
      <td>delta_13-276C-500-abneg</td>
      <td>0.044</td>
      <td>524897.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>276C_repeat_500</td>
      <td>lib1</td>
      <td>276C_repeat</td>
      <td>500</td>
      <td>211119</td>
      <td>delta_21-26-none-0-ref</td>
      <td>delta_24-276C_repeat-500-abneg</td>
      <td>0.126</td>
      <td>1263014.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>276C_repeat_500</td>
      <td>lib2</td>
      <td>276C_repeat</td>
      <td>500</td>
      <td>211119</td>
      <td>delta_21-26-none-0-ref</td>
      <td>delta_24-276C_repeat-500-abneg</td>
      <td>0.108</td>
      <td>1091168.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>277C_500</td>
      <td>lib1</td>
      <td>277C</td>
      <td>500</td>
      <td>211015</td>
      <td>delta_9-16-none-0-ref</td>
      <td>delta_14-277C-500-abneg</td>
      <td>0.045</td>
      <td>548187.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>277C_500</td>
      <td>lib2</td>
      <td>277C</td>
      <td>500</td>
      <td>211015</td>
      <td>delta_9-16-none-0-ref</td>
      <td>delta_14-277C-500-abneg</td>
      <td>0.041</td>
      <td>535808.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>277C_repeat_500</td>
      <td>lib1</td>
      <td>277C_repeat</td>
      <td>500</td>
      <td>211119</td>
      <td>delta_21-26-none-0-ref</td>
      <td>delta_25-277C_repeat-500-abneg</td>
      <td>0.112</td>
      <td>1137613.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>277C_repeat_500</td>
      <td>lib2</td>
      <td>277C_repeat</td>
      <td>500</td>
      <td>211119</td>
      <td>delta_21-26-none-0-ref</td>
      <td>delta_25-277C_repeat-500-abneg</td>
      <td>0.108</td>
      <td>1135069.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>278C_1250</td>
      <td>lib1</td>
      <td>278C</td>
      <td>1250</td>
      <td>211015</td>
      <td>delta_9-16-none-0-ref</td>
      <td>delta_15-278C-1250-abneg</td>
      <td>0.046</td>
      <td>551478.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>278C_1250</td>
      <td>lib2</td>
      <td>278C</td>
      <td>1250</td>
      <td>211015</td>
      <td>delta_9-16-none-0-ref</td>
      <td>delta_15-278C-1250-abneg</td>
      <td>0.043</td>
      <td>550815.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>278C_repeat_1250</td>
      <td>lib1</td>
      <td>278C_repeat</td>
      <td>1250</td>
      <td>211119</td>
      <td>delta_21-26-none-0-ref</td>
      <td>delta_26-278C_repeat-1250-abneg</td>
      <td>0.110</td>
      <td>1106454.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>278C_repeat_1250</td>
      <td>lib2</td>
      <td>278C_repeat</td>
      <td>1250</td>
      <td>211119</td>
      <td>delta_21-26-none-0-ref</td>
      <td>delta_26-278C_repeat-1250-abneg</td>
      <td>0.106</td>
      <td>1106060.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>279C_1250</td>
      <td>lib1</td>
      <td>279C</td>
      <td>1250</td>
      <td>211015</td>
      <td>delta_9-16-none-0-ref</td>
      <td>delta_16-279C-1250-abneg</td>
      <td>0.044</td>
      <td>540196.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>279C_1250</td>
      <td>lib2</td>
      <td>279C</td>
      <td>1250</td>
      <td>211015</td>
      <td>delta_9-16-none-0-ref</td>
      <td>delta_16-279C-1250-abneg</td>
      <td>0.040</td>
      <td>525820.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>279C_repeat_1250</td>
      <td>lib1</td>
      <td>279C_repeat</td>
      <td>1250</td>
      <td>211112</td>
      <td>delta_17-20-none-0-ref</td>
      <td>delta_20-279C_repeat-1250-abneg</td>
      <td>0.097</td>
      <td>1026589.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>279C_repeat_1250</td>
      <td>lib2</td>
      <td>279C_repeat</td>
      <td>1250</td>
      <td>211112</td>
      <td>delta_17-20-none-0-ref</td>
      <td>delta_20-279C_repeat-1250-abneg</td>
      <td>0.097</td>
      <td>1077475.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Delta_1_500</td>
      <td>lib1</td>
      <td>Delta_1</td>
      <td>500</td>
      <td>211122</td>
      <td>delta_27-34-none-0-ref</td>
      <td>delta_27-Delta_1-500-abneg</td>
      <td>0.101</td>
      <td>1061841.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Delta_1_500</td>
      <td>lib2</td>
      <td>Delta_1</td>
      <td>500</td>
      <td>211122</td>
      <td>delta_27-34-none-0-ref</td>
      <td>delta_27-Delta_1-500-abneg</td>
      <td>0.094</td>
      <td>1044979.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Delta_10_1250</td>
      <td>lib1</td>
      <td>Delta_10</td>
      <td>1250</td>
      <td>211122</td>
      <td>delta_27-34-none-0-ref</td>
      <td>delta_33-Delta_10-1250-abneg</td>
      <td>0.084</td>
      <td>846975.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Delta_10_1250</td>
      <td>lib2</td>
      <td>Delta_10</td>
      <td>1250</td>
      <td>211122</td>
      <td>delta_27-34-none-0-ref</td>
      <td>delta_33-Delta_10-1250-abneg</td>
      <td>0.073</td>
      <td>749075.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Delta_11_500</td>
      <td>lib1</td>
      <td>Delta_11</td>
      <td>500</td>
      <td>211122</td>
      <td>delta_27-34-none-0-ref</td>
      <td>delta_34-Delta_11-500-abneg</td>
      <td>0.101</td>
      <td>1017425.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Delta_11_500</td>
      <td>lib2</td>
      <td>Delta_11</td>
      <td>500</td>
      <td>211122</td>
      <td>delta_27-34-none-0-ref</td>
      <td>delta_34-Delta_11-500-abneg</td>
      <td>0.084</td>
      <td>847856.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Delta_3_350</td>
      <td>lib1</td>
      <td>Delta_3</td>
      <td>350</td>
      <td>211122</td>
      <td>delta_27-34-none-0-ref</td>
      <td>delta_28-Delta_3-350-abneg</td>
      <td>0.096</td>
      <td>1031277.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Delta_3_350</td>
      <td>lib2</td>
      <td>Delta_3</td>
      <td>350</td>
      <td>211122</td>
      <td>delta_27-34-none-0-ref</td>
      <td>delta_28-Delta_3-350-abneg</td>
      <td>0.098</td>
      <td>1180295.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Delta_4_350</td>
      <td>lib1</td>
      <td>Delta_4</td>
      <td>350</td>
      <td>211122</td>
      <td>delta_27-34-none-0-ref</td>
      <td>delta_29-Delta_4-350-abneg</td>
      <td>0.069</td>
      <td>701150.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Delta_4_350</td>
      <td>lib2</td>
      <td>Delta_4</td>
      <td>350</td>
      <td>211122</td>
      <td>delta_27-34-none-0-ref</td>
      <td>delta_29-Delta_4-350-abneg</td>
      <td>0.068</td>
      <td>693528.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Delta_6_500</td>
      <td>lib1</td>
      <td>Delta_6</td>
      <td>500</td>
      <td>211122</td>
      <td>delta_27-34-none-0-ref</td>
      <td>delta_30-Delta_6-500-abneg</td>
      <td>0.108</td>
      <td>1093297.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Delta_6_500</td>
      <td>lib2</td>
      <td>Delta_6</td>
      <td>500</td>
      <td>211122</td>
      <td>delta_27-34-none-0-ref</td>
      <td>delta_30-Delta_6-500-abneg</td>
      <td>0.109</td>
      <td>1091332.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Delta_7_1250</td>
      <td>lib1</td>
      <td>Delta_7</td>
      <td>1250</td>
      <td>211122</td>
      <td>delta_27-34-none-0-ref</td>
      <td>delta_31-Delta_7-1250-abneg</td>
      <td>0.099</td>
      <td>1231047.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Delta_7_1250</td>
      <td>lib2</td>
      <td>Delta_7</td>
      <td>1250</td>
      <td>211122</td>
      <td>delta_27-34-none-0-ref</td>
      <td>delta_31-Delta_7-1250-abneg</td>
      <td>0.091</td>
      <td>926411.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Delta_8_500</td>
      <td>lib1</td>
      <td>Delta_8</td>
      <td>500</td>
      <td>211122</td>
      <td>delta_27-34-none-0-ref</td>
      <td>delta_32-Delta_8-500-abneg</td>
      <td>0.102</td>
      <td>1024951.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Delta_8_500</td>
      <td>lib2</td>
      <td>Delta_8</td>
      <td>500</td>
      <td>211122</td>
      <td>delta_27-34-none-0-ref</td>
      <td>delta_32-Delta_8-500-abneg</td>
      <td>0.095</td>
      <td>967074.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>P02_500</td>
      <td>lib1</td>
      <td>P02</td>
      <td>500</td>
      <td>211008</td>
      <td>delta_1-8-none-0-ref</td>
      <td>delta_1-P02-500-abneg</td>
      <td>0.048</td>
      <td>601087.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>P02_500</td>
      <td>lib2</td>
      <td>P02</td>
      <td>500</td>
      <td>211008</td>
      <td>delta_1-8-none-0-ref</td>
      <td>delta_1-P02-500-abneg</td>
      <td>0.042</td>
      <td>588481.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>P02_repeat_500</td>
      <td>lib1</td>
      <td>P02_repeat</td>
      <td>500</td>
      <td>211124</td>
      <td>delta_35-40-none-0-ref</td>
      <td>delta_35-P02_repeat-500-abneg</td>
      <td>0.088</td>
      <td>927272.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>P02_repeat_500</td>
      <td>lib2</td>
      <td>P02_repeat</td>
      <td>500</td>
      <td>211124</td>
      <td>delta_35-40-none-0-ref</td>
      <td>delta_35-P02_repeat-500-abneg</td>
      <td>0.079</td>
      <td>808248.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>P03_1250</td>
      <td>lib1</td>
      <td>P03</td>
      <td>1250</td>
      <td>211008</td>
      <td>delta_1-8-none-0-ref</td>
      <td>delta_2-P03-1250-abneg</td>
      <td>0.050</td>
      <td>554708.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>P03_1250</td>
      <td>lib2</td>
      <td>P03</td>
      <td>1250</td>
      <td>211008</td>
      <td>delta_1-8-none-0-ref</td>
      <td>delta_2-P03-1250-abneg</td>
      <td>0.044</td>
      <td>553123.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>P03_repeat_1250</td>
      <td>lib1</td>
      <td>P03_repeat</td>
      <td>1250</td>
      <td>211112</td>
      <td>delta_17-20-none-0-ref</td>
      <td>delta_17-P03_repeat-1250-abneg</td>
      <td>0.094</td>
      <td>1022039.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>P03_repeat_1250</td>
      <td>lib2</td>
      <td>P03_repeat</td>
      <td>1250</td>
      <td>211112</td>
      <td>delta_17-20-none-0-ref</td>
      <td>delta_17-P03_repeat-1250-abneg</td>
      <td>0.087</td>
      <td>1012373.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>P04_1250</td>
      <td>lib1</td>
      <td>P04</td>
      <td>1250</td>
      <td>211008</td>
      <td>delta_1-8-none-0-ref</td>
      <td>delta_3-P04-1250-abneg</td>
      <td>0.052</td>
      <td>603708.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>P04_1250</td>
      <td>lib2</td>
      <td>P04</td>
      <td>1250</td>
      <td>211008</td>
      <td>delta_1-8-none-0-ref</td>
      <td>delta_3-P04-1250-abneg</td>
      <td>0.047</td>
      <td>601718.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>P04_repeat_1250</td>
      <td>lib1</td>
      <td>P04_repeat</td>
      <td>1250</td>
      <td>211124</td>
      <td>delta_35-40-none-0-ref</td>
      <td>delta_36-P04_repeat-1250-abneg</td>
      <td>0.107</td>
      <td>1079672.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>P04_repeat_1250</td>
      <td>lib2</td>
      <td>P04_repeat</td>
      <td>1250</td>
      <td>211124</td>
      <td>delta_35-40-none-0-ref</td>
      <td>delta_36-P04_repeat-1250-abneg</td>
      <td>0.087</td>
      <td>875369.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>P05_500</td>
      <td>lib1</td>
      <td>P05</td>
      <td>500</td>
      <td>211008</td>
      <td>delta_1-8-none-0-ref</td>
      <td>delta_4-P05-500-abneg</td>
      <td>0.035</td>
      <td>387010.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>P05_500</td>
      <td>lib2</td>
      <td>P05</td>
      <td>500</td>
      <td>211008</td>
      <td>delta_1-8-none-0-ref</td>
      <td>delta_4-P05-500-abneg</td>
      <td>0.024</td>
      <td>319803.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>P05_repeat_500</td>
      <td>lib1</td>
      <td>P05_repeat</td>
      <td>500</td>
      <td>211124</td>
      <td>delta_35-40-none-0-ref</td>
      <td>delta_37-P05_repeat-500-abneg</td>
      <td>0.085</td>
      <td>909556.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>P05_repeat_500</td>
      <td>lib2</td>
      <td>P05_repeat</td>
      <td>500</td>
      <td>211124</td>
      <td>delta_35-40-none-0-ref</td>
      <td>delta_37-P05_repeat-500-abneg</td>
      <td>0.086</td>
      <td>892227.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>P08_500</td>
      <td>lib1</td>
      <td>P08</td>
      <td>500</td>
      <td>211008</td>
      <td>delta_1-8-none-0-ref</td>
      <td>delta_5-P08-500-abneg</td>
      <td>0.061</td>
      <td>834746.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>P08_500</td>
      <td>lib2</td>
      <td>P08</td>
      <td>500</td>
      <td>211008</td>
      <td>delta_1-8-none-0-ref</td>
      <td>delta_5-P08-500-abneg</td>
      <td>0.053</td>
      <td>762248.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>P08_repeat_500</td>
      <td>lib1</td>
      <td>P08_repeat</td>
      <td>500</td>
      <td>211112</td>
      <td>delta_17-20-none-0-ref</td>
      <td>delta_18-P08_repeat-500-abneg</td>
      <td>0.112</td>
      <td>1015086.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>P08_repeat_500</td>
      <td>lib2</td>
      <td>P08_repeat</td>
      <td>500</td>
      <td>211112</td>
      <td>delta_17-20-none-0-ref</td>
      <td>delta_18-P08_repeat-500-abneg</td>
      <td>0.121</td>
      <td>1045993.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>P09_200</td>
      <td>lib1</td>
      <td>P09</td>
      <td>200</td>
      <td>211008</td>
      <td>delta_1-8-none-0-ref</td>
      <td>delta_6-P09-200-abneg</td>
      <td>0.058</td>
      <td>738910.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>P09_200</td>
      <td>lib2</td>
      <td>P09</td>
      <td>200</td>
      <td>211008</td>
      <td>delta_1-8-none-0-ref</td>
      <td>delta_6-P09-200-abneg</td>
      <td>0.048</td>
      <td>675626.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>P09_repeat_200</td>
      <td>lib1</td>
      <td>P09_repeat</td>
      <td>200</td>
      <td>211124</td>
      <td>delta_35-40-none-0-ref</td>
      <td>delta_38-P09_repeat-200-abneg</td>
      <td>0.096</td>
      <td>964559.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>P09_repeat_200</td>
      <td>lib2</td>
      <td>P09_repeat</td>
      <td>200</td>
      <td>211124</td>
      <td>delta_35-40-none-0-ref</td>
      <td>delta_38-P09_repeat-200-abneg</td>
      <td>0.089</td>
      <td>904976.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>P12_200</td>
      <td>lib1</td>
      <td>P12</td>
      <td>200</td>
      <td>211008</td>
      <td>delta_1-8-none-0-ref</td>
      <td>delta_7-P12-200-abneg</td>
      <td>0.051</td>
      <td>652641.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>P12_200</td>
      <td>lib2</td>
      <td>P12</td>
      <td>200</td>
      <td>211008</td>
      <td>delta_1-8-none-0-ref</td>
      <td>delta_7-P12-200-abneg</td>
      <td>0.042</td>
      <td>588742.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>P12_repeat_200</td>
      <td>lib1</td>
      <td>P12_repeat</td>
      <td>200</td>
      <td>211124</td>
      <td>delta_35-40-none-0-ref</td>
      <td>delta_39-P12_repeat-200-abneg</td>
      <td>0.101</td>
      <td>1017353.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>P12_repeat_200</td>
      <td>lib2</td>
      <td>P12_repeat</td>
      <td>200</td>
      <td>211124</td>
      <td>delta_35-40-none-0-ref</td>
      <td>delta_39-P12_repeat-200-abneg</td>
      <td>0.096</td>
      <td>964926.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>P14_1250</td>
      <td>lib1</td>
      <td>P14</td>
      <td>1250</td>
      <td>211008</td>
      <td>delta_1-8-none-0-ref</td>
      <td>delta_8-P14-1250-abneg</td>
      <td>0.057</td>
      <td>765130.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>P14_1250</td>
      <td>lib2</td>
      <td>P14</td>
      <td>1250</td>
      <td>211008</td>
      <td>delta_1-8-none-0-ref</td>
      <td>delta_8-P14-1250-abneg</td>
      <td>0.050</td>
      <td>711021.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>P14_repeat_1250</td>
      <td>lib1</td>
      <td>P14_repeat</td>
      <td>1250</td>
      <td>211124</td>
      <td>delta_35-40-none-0-ref</td>
      <td>delta_40-P14_repeat-1250-abneg</td>
      <td>0.094</td>
      <td>953329.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>P14_repeat_1250</td>
      <td>lib2</td>
      <td>P14_repeat</td>
      <td>1250</td>
      <td>211124</td>
      <td>delta_35-40-none-0-ref</td>
      <td>delta_40-P14_repeat-1250-abneg</td>
      <td>0.087</td>
      <td>872522.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>


Compute the escape scores for variants of the primary target and classify the variants:


```python
print(f"Computing escape scores for {primary_target} variants using {config['escape_score_type']} "
      f"score type with a pseudocount of {config['escape_score_pseudocount']} and "
      f"an escape fraction floor {config['escape_score_floor_E']}, an escape fraction ceiling "
      f"{config['escape_score_ceil_E']}, and grouping variants by {config['escape_score_group_by']}.")

escape_scores = (variants.escape_scores(score_sample_df,
                                        score_type=config['escape_score_type'],
                                        pseudocount=config['escape_score_pseudocount'],
                                        floor_E=config['escape_score_floor_E'],
                                        ceil_E=config['escape_score_ceil_E'],
                                        by=config['escape_score_group_by'],
                                        )
                 .query('target == @primary_target')
                 .pipe(variants.classifyVariants,
                       primary_target=variants.primary_target,
                       syn_as_wt=(config['escape_score_group_by'] == 'aa_substitutions'),
                       )
                 )
print('Here are the first few lines of the resulting escape scores:')
display(HTML(escape_scores.head().to_html(index=False)))
```

    Computing escape scores for Delta variants using frac_escape score type with a pseudocount of 0.5 and an escape fraction floor 0, an escape fraction ceiling 1, and grouping variants by barcode.
    Here are the first few lines of the resulting escape scores:



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>name</th>
      <th>target</th>
      <th>library</th>
      <th>pre_sample</th>
      <th>post_sample</th>
      <th>barcode</th>
      <th>score</th>
      <th>score_var</th>
      <th>pre_count</th>
      <th>post_count</th>
      <th>codon_substitutions</th>
      <th>n_codon_substitutions</th>
      <th>aa_substitutions</th>
      <th>n_aa_substitutions</th>
      <th>variant_class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>267C_200</td>
      <td>Delta</td>
      <td>lib1</td>
      <td>delta_9-16-none-0-ref</td>
      <td>delta_9-267C-200-abneg</td>
      <td>GCTACAAAGTCCGCAG</td>
      <td>0.557393</td>
      <td>1.322564e-04</td>
      <td>5697</td>
      <td>3981</td>
      <td>GAT68ACT</td>
      <td>1</td>
      <td>D68T</td>
      <td>1</td>
      <td>1 nonsynonymous</td>
    </tr>
    <tr>
      <td>267C_200</td>
      <td>Delta</td>
      <td>lib1</td>
      <td>delta_9-16-none-0-ref</td>
      <td>delta_9-267C-200-abneg</td>
      <td>AAAGCCGAGTTAATAA</td>
      <td>0.004908</td>
      <td>7.457917e-07</td>
      <td>5281</td>
      <td>32</td>
      <td>TGT6TCT</td>
      <td>1</td>
      <td>C6S</td>
      <td>1</td>
      <td>1 nonsynonymous</td>
    </tr>
    <tr>
      <td>267C_200</td>
      <td>Delta</td>
      <td>lib1</td>
      <td>delta_9-16-none-0-ref</td>
      <td>delta_9-267C-200-abneg</td>
      <td>TGGGAAAAACAAAGCT</td>
      <td>0.000077</td>
      <td>1.173595e-08</td>
      <td>5206</td>
      <td>0</td>
      <td>GGT9GAT AAA199TGT</td>
      <td>2</td>
      <td>G9D K199C</td>
      <td>2</td>
      <td>&gt;1 nonsynonymous</td>
    </tr>
    <tr>
      <td>267C_200</td>
      <td>Delta</td>
      <td>lib1</td>
      <td>delta_9-16-none-0-ref</td>
      <td>delta_9-267C-200-abneg</td>
      <td>ATCGAAATTGAGTGAT</td>
      <td>0.029109</td>
      <td>4.759443e-06</td>
      <td>5055</td>
      <td>184</td>
      <td>TTC99ATT</td>
      <td>1</td>
      <td>F99I</td>
      <td>1</td>
      <td>1 nonsynonymous</td>
    </tr>
    <tr>
      <td>267C_200</td>
      <td>Delta</td>
      <td>lib1</td>
      <td>delta_9-16-none-0-ref</td>
      <td>delta_9-267C-200-abneg</td>
      <td>TACAAAGCACGCTAAA</td>
      <td>0.000079</td>
      <td>1.255159e-08</td>
      <td>5034</td>
      <td>0</td>
      <td>CAG176AAT</td>
      <td>1</td>
      <td>Q176N</td>
      <td>1</td>
      <td>1 nonsynonymous</td>
    </tr>
  </tbody>
</table>


## Apply pre-selection count filter to variant escape scores
Now determine a pre-selection count filter in order to flag for removal variants with counts that are so low that the estimated score is probably noise.
We know that stop codons should be largely purged pre-selection, and so the counts for them are a good indication of the "noise" threshold.
We therefore set the filter using the number of pre-selection counts for the stop codons.

To do this, we first compute the number of pre-selection counts for stop-codon variants at various quantiles and look at these.
We then take the number of pre-selection counts at the specified quantile as the filter cutoff, and filter scores for all variants with pre-selection counts less than this filter cutoff:


```python
filter_quantile = config['escape_score_stop_quantile_filter']
assert 0 <= filter_quantile <= 1

quantiles = sorted(set([0.5, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999] + [filter_quantile]))

stop_score_counts = (
    escape_scores
    .query('variant_class == "stop"')
    .groupby(['library', 'pre_sample'], observed=True)
    ['pre_count']
    .quantile(q=quantiles)
    .reset_index()
    .rename(columns={'level_2': 'quantile'})
    .pivot_table(index=['pre_sample', 'library'],
                 columns='quantile',
                 values='pre_count')
    )

print('Quantiles of the number of pre-selection counts per variant for stop variants:')
display(HTML(stop_score_counts.to_html(float_format='%.1f')))

print(f"\nSetting the pre-count filter cutoff to the {filter_quantile} quantile:")
pre_count_filter_cutoffs = (
    stop_score_counts
    [filter_quantile]
    .rename('pre_count_filter_cutoff')
    .reset_index()
    )
display(HTML(pre_count_filter_cutoffs.to_html(float_format='%.1f')))
```

    Quantiles of the number of pre-selection counts per variant for stop variants:



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>quantile</th>
      <th>0.5</th>
      <th>0.9</th>
      <th>0.95</th>
      <th>0.98</th>
      <th>0.99</th>
      <th>0.995</th>
      <th>0.999</th>
    </tr>
    <tr>
      <th>pre_sample</th>
      <th>library</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">delta_1-8-none-0-ref</th>
      <th>lib1</th>
      <td>40.0</td>
      <td>139.1</td>
      <td>195.0</td>
      <td>270.2</td>
      <td>332.1</td>
      <td>393.0</td>
      <td>456.0</td>
    </tr>
    <tr>
      <th>lib2</th>
      <td>38.0</td>
      <td>125.0</td>
      <td>165.0</td>
      <td>220.0</td>
      <td>267.0</td>
      <td>311.0</td>
      <td>420.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">delta_9-16-none-0-ref</th>
      <th>lib1</th>
      <td>45.0</td>
      <td>165.0</td>
      <td>232.1</td>
      <td>300.0</td>
      <td>379.1</td>
      <td>429.0</td>
      <td>561.0</td>
    </tr>
    <tr>
      <th>lib2</th>
      <td>45.0</td>
      <td>143.0</td>
      <td>198.0</td>
      <td>272.0</td>
      <td>339.0</td>
      <td>427.0</td>
      <td>552.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">delta_17-20-none-0-ref</th>
      <th>lib1</th>
      <td>63.0</td>
      <td>232.1</td>
      <td>315.1</td>
      <td>437.1</td>
      <td>533.0</td>
      <td>579.0</td>
      <td>672.0</td>
    </tr>
    <tr>
      <th>lib2</th>
      <td>59.0</td>
      <td>186.0</td>
      <td>250.0</td>
      <td>346.0</td>
      <td>410.0</td>
      <td>484.1</td>
      <td>605.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">delta_21-26-none-0-ref</th>
      <th>lib1</th>
      <td>17.0</td>
      <td>62.0</td>
      <td>86.0</td>
      <td>119.0</td>
      <td>153.0</td>
      <td>179.0</td>
      <td>231.0</td>
    </tr>
    <tr>
      <th>lib2</th>
      <td>14.0</td>
      <td>47.0</td>
      <td>66.0</td>
      <td>88.0</td>
      <td>113.0</td>
      <td>128.0</td>
      <td>146.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">delta_27-34-none-0-ref</th>
      <th>lib1</th>
      <td>14.0</td>
      <td>54.0</td>
      <td>72.0</td>
      <td>97.0</td>
      <td>121.0</td>
      <td>146.0</td>
      <td>210.0</td>
    </tr>
    <tr>
      <th>lib2</th>
      <td>14.0</td>
      <td>50.0</td>
      <td>67.0</td>
      <td>93.0</td>
      <td>120.0</td>
      <td>140.0</td>
      <td>185.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">delta_35-40-none-0-ref</th>
      <th>lib1</th>
      <td>13.0</td>
      <td>47.0</td>
      <td>67.0</td>
      <td>89.0</td>
      <td>117.1</td>
      <td>139.0</td>
      <td>191.0</td>
    </tr>
    <tr>
      <th>lib2</th>
      <td>13.0</td>
      <td>49.0</td>
      <td>70.0</td>
      <td>96.0</td>
      <td>117.0</td>
      <td>151.0</td>
      <td>165.0</td>
    </tr>
  </tbody>
</table>


    
    Setting the pre-count filter cutoff to the 0.9 quantile:



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pre_sample</th>
      <th>library</th>
      <th>pre_count_filter_cutoff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>delta_1-8-none-0-ref</td>
      <td>lib1</td>
      <td>139.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>delta_1-8-none-0-ref</td>
      <td>lib2</td>
      <td>125.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>delta_9-16-none-0-ref</td>
      <td>lib1</td>
      <td>165.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>delta_9-16-none-0-ref</td>
      <td>lib2</td>
      <td>143.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>delta_17-20-none-0-ref</td>
      <td>lib1</td>
      <td>232.1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>delta_17-20-none-0-ref</td>
      <td>lib2</td>
      <td>186.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>delta_21-26-none-0-ref</td>
      <td>lib1</td>
      <td>62.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>delta_21-26-none-0-ref</td>
      <td>lib2</td>
      <td>47.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>delta_27-34-none-0-ref</td>
      <td>lib1</td>
      <td>54.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>delta_27-34-none-0-ref</td>
      <td>lib2</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>delta_35-40-none-0-ref</td>
      <td>lib1</td>
      <td>47.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>delta_35-40-none-0-ref</td>
      <td>lib2</td>
      <td>49.0</td>
    </tr>
  </tbody>
</table>


Apply the filter to the escape scores, so that scores that fail the pre-selection count filter are now marked with `pass_pre_count_filter` of `False`:


```python
escape_scores = (
    escape_scores
    .merge(pre_count_filter_cutoffs,
           on=['library', 'pre_sample'],
           how='left',
           validate='many_to_one')
    .assign(pass_pre_count_filter=lambda x: x['pre_count'] >= x['pre_count_filter_cutoff'])
    )
```

Plot the fraction of variants of each type that pass the pre-selection count filter in each pre-selection sample.
The ideal filter would have the property such that no *stop* variants pass, all *wildtype* (or *synonymous*) variants pass, and some intermediate fraction of *nonsynonymous* variants pass.
However, if the variant composition in the pre-selection samples is already heavily skewed by jackpotting, there will be some deviation from this ideal behavior.
Here is what the plots actually look like:


```python
frac_pre_pass_filter = (
    escape_scores
    [['pre_sample', 'library', 'target', config['escape_score_group_by'],
      'pre_count', 'pass_pre_count_filter', 'variant_class']]
    .drop_duplicates()
    .groupby(['pre_sample', 'library', 'variant_class'], observed=True)
    .aggregate(n_variants=pd.NamedAgg('pass_pre_count_filter', 'count'),
               n_pass_filter=pd.NamedAgg('pass_pre_count_filter', 'sum')
               )
    .reset_index()
    .assign(frac_pass_filter=lambda x: x['n_pass_filter'] / x['n_variants'],
            pre_sample=lambda x: pd.Categorical(x['pre_sample'], x['pre_sample'].unique(), ordered=True).remove_unused_categories())
    )

p = (ggplot(frac_pre_pass_filter) +
     aes('variant_class', 'frac_pass_filter', fill='variant_class') +
     geom_bar(stat='identity') +
     facet_grid('library ~ pre_sample') +
     theme(axis_text_x=element_text(angle=90),
           figure_size=(3.3 * frac_pre_pass_filter['pre_sample'].nunique(),
                        2 * frac_pre_pass_filter['library'].nunique()),
           panel_grid_major_x=element_blank(),
           ) +
     scale_fill_manual(values=CBPALETTE[1:]) +
     expand_limits(y=(0, 1))
     )

_ = p.draw()
```


    
![png](counts_to_scores_files/counts_to_scores_53_0.png)
    


## Apply ACE2-binding / expression filter to variant mutations
We also used deep mutational scanning to estimate how each mutation affected ACE2 binding and expression in the B.1.351 background.
Here we flag for removal any variants of the primary target that have (or have mutations) that were measured to decrease ACE2-binding or expression beyond a minimal threshold, in order to avoid these variants muddying the signal as spurious escape mutants.

To do this, we first determine all mutations that do / do-not having binding that exceeds the thresholds.

Note that because we are working on this serum-mapping project at the same time as we are working on the ACE2-binding / RBD-expression project, the scores will be preliminary until all final analyses have been done on the DMS project end. So, we will allow either preliminary or "final" measurements to be used. 


```python
mut_bind_expr_file = config['final_variant_scores_mut_file']
    
print(f"Reading ACE2-binding and expression for mutations from {mut_bind_expr_file}, "
      f"and filtering for variants that have single mutations that "
      f"only have mutations with binding >={config['escape_score_min_bind_mut']} and "
      f"expression >={config['escape_score_min_expr_mut']}.")

mut_bind_expr = (pd.read_csv(mut_bind_expr_file)
                 .query('target==@config["primary_target"]')
                 # need to add back the offset numbering for some silly, circuitous reason 
                 .assign(RBD_site=lambda x: x['position']-config['site_number_offset'] ,
                         RBD_mutation=lambda x: x['wildtype']+x['RBD_site'].astype(str)+x['mutant']
                        )
                )

print('Here is what that dataframe looks like:')

display(HTML(mut_bind_expr.query('delta_bind < -2.35').head().to_html(index=False)))
```

    Reading ACE2-binding and expression for mutations from results/final_variant_scores/final_variant_scores.csv, and filtering for variants that have single mutations that only have mutations with binding >=-1.86 and expression >=-0.75.
    Here is what that dataframe looks like:



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>target</th>
      <th>wildtype</th>
      <th>position</th>
      <th>mutant</th>
      <th>mutation</th>
      <th>bind</th>
      <th>delta_bind</th>
      <th>n_bc_bind</th>
      <th>n_libs_bind</th>
      <th>bind_rep1</th>
      <th>bind_rep2</th>
      <th>expr</th>
      <th>delta_expr</th>
      <th>n_bc_expr</th>
      <th>n_libs_expr</th>
      <th>expr_rep1</th>
      <th>expr_rep2</th>
      <th>RBD_site</th>
      <th>RBD_mutation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Delta</td>
      <td>F</td>
      <td>347</td>
      <td>D</td>
      <td>F347D</td>
      <td>6.56115</td>
      <td>-2.47410</td>
      <td>39</td>
      <td>2</td>
      <td>6.99892</td>
      <td>6.12338</td>
      <td>7.12826</td>
      <td>-2.66200</td>
      <td>43</td>
      <td>2</td>
      <td>7.09456</td>
      <td>7.16196</td>
      <td>17</td>
      <td>F17D</td>
    </tr>
    <tr>
      <td>Delta</td>
      <td>F</td>
      <td>347</td>
      <td>N</td>
      <td>F347N</td>
      <td>6.25934</td>
      <td>-2.77590</td>
      <td>38</td>
      <td>2</td>
      <td>6.37534</td>
      <td>6.14335</td>
      <td>7.29762</td>
      <td>-2.49264</td>
      <td>40</td>
      <td>2</td>
      <td>7.29630</td>
      <td>7.29894</td>
      <td>17</td>
      <td>F17N</td>
    </tr>
    <tr>
      <td>Delta</td>
      <td>S</td>
      <td>349</td>
      <td>F</td>
      <td>S349F</td>
      <td>6.41557</td>
      <td>-2.61967</td>
      <td>28</td>
      <td>2</td>
      <td>6.76244</td>
      <td>6.06870</td>
      <td>6.77342</td>
      <td>-3.01685</td>
      <td>30</td>
      <td>2</td>
      <td>6.78410</td>
      <td>6.76273</td>
      <td>19</td>
      <td>S19F</td>
    </tr>
    <tr>
      <td>Delta</td>
      <td>S</td>
      <td>349</td>
      <td>I</td>
      <td>S349I</td>
      <td>5.66335</td>
      <td>-3.37189</td>
      <td>31</td>
      <td>2</td>
      <td>5.94559</td>
      <td>5.38111</td>
      <td>6.55953</td>
      <td>-3.23073</td>
      <td>32</td>
      <td>2</td>
      <td>6.47977</td>
      <td>6.63929</td>
      <td>19</td>
      <td>S19I</td>
    </tr>
    <tr>
      <td>Delta</td>
      <td>S</td>
      <td>349</td>
      <td>W</td>
      <td>S349W</td>
      <td>6.50440</td>
      <td>-2.53084</td>
      <td>26</td>
      <td>2</td>
      <td>7.07355</td>
      <td>5.93526</td>
      <td>6.70857</td>
      <td>-3.08169</td>
      <td>29</td>
      <td>2</td>
      <td>6.74070</td>
      <td>6.67644</td>
      <td>19</td>
      <td>S19W</td>
    </tr>
  </tbody>
</table>



```python
assert mut_bind_expr['RBD_mutation'].nunique() == len(mut_bind_expr)
for prop in ['bind', 'expr']:
    muts_adequate = set(mut_bind_expr
                        .query(f"delta_{prop} >= {config[f'escape_score_min_{prop}_mut']}")
                        ['RBD_mutation']
                        )
    print(f"{len(muts_adequate)} of {len(mut_bind_expr)} mutations have adequate {prop}.")
    escape_scores[f"muts_pass_{prop}_filter"] = (
        escape_scores
        ['aa_substitutions']
        .map(lambda s: set(s.split()).issubset(muts_adequate))
        )

# annotate as passing overall filter if passes all mutation and binding filters:
escape_scores['pass_ACE2bind_expr_filter'] = (
        escape_scores['muts_pass_bind_filter'] &
        escape_scores['muts_pass_expr_filter'] 
        )

display(HTML(escape_scores.query('not pass_ACE2bind_expr_filter & variant_class != "stop"').head().to_html(index=False)))
```

    3260 of 4020 mutations have adequate bind.
    2700 of 4020 mutations have adequate expr.



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>name</th>
      <th>target</th>
      <th>library</th>
      <th>pre_sample</th>
      <th>post_sample</th>
      <th>barcode</th>
      <th>score</th>
      <th>score_var</th>
      <th>pre_count</th>
      <th>post_count</th>
      <th>codon_substitutions</th>
      <th>n_codon_substitutions</th>
      <th>aa_substitutions</th>
      <th>n_aa_substitutions</th>
      <th>variant_class</th>
      <th>pre_count_filter_cutoff</th>
      <th>pass_pre_count_filter</th>
      <th>muts_pass_bind_filter</th>
      <th>muts_pass_expr_filter</th>
      <th>pass_ACE2bind_expr_filter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>267C_200</td>
      <td>Delta</td>
      <td>lib1</td>
      <td>delta_9-16-none-0-ref</td>
      <td>delta_9-267C-200-abneg</td>
      <td>GCTACAAAGTCCGCAG</td>
      <td>0.557393</td>
      <td>1.322564e-04</td>
      <td>5697</td>
      <td>3981</td>
      <td>GAT68ACT</td>
      <td>1</td>
      <td>D68T</td>
      <td>1</td>
      <td>1 nonsynonymous</td>
      <td>165.0</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <td>267C_200</td>
      <td>Delta</td>
      <td>lib1</td>
      <td>delta_9-16-none-0-ref</td>
      <td>delta_9-267C-200-abneg</td>
      <td>AAAGCCGAGTTAATAA</td>
      <td>0.004908</td>
      <td>7.457917e-07</td>
      <td>5281</td>
      <td>32</td>
      <td>TGT6TCT</td>
      <td>1</td>
      <td>C6S</td>
      <td>1</td>
      <td>1 nonsynonymous</td>
      <td>165.0</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <td>267C_200</td>
      <td>Delta</td>
      <td>lib1</td>
      <td>delta_9-16-none-0-ref</td>
      <td>delta_9-267C-200-abneg</td>
      <td>ATCGAAATTGAGTGAT</td>
      <td>0.029109</td>
      <td>4.759443e-06</td>
      <td>5055</td>
      <td>184</td>
      <td>TTC99ATT</td>
      <td>1</td>
      <td>F99I</td>
      <td>1</td>
      <td>1 nonsynonymous</td>
      <td>165.0</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <td>267C_200</td>
      <td>Delta</td>
      <td>lib1</td>
      <td>delta_9-16-none-0-ref</td>
      <td>delta_9-267C-200-abneg</td>
      <td>AAGCTATACGTTCATT</td>
      <td>0.129987</td>
      <td>2.609587e-05</td>
      <td>4617</td>
      <td>752</td>
      <td>CGT127AAT AAA148GAA</td>
      <td>2</td>
      <td>R127N K148E</td>
      <td>2</td>
      <td>&gt;1 nonsynonymous</td>
      <td>165.0</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <td>267C_200</td>
      <td>Delta</td>
      <td>lib1</td>
      <td>delta_9-16-none-0-ref</td>
      <td>delta_9-267C-200-abneg</td>
      <td>TGATGATTCCTAAGAA</td>
      <td>0.000088</td>
      <td>1.535702e-08</td>
      <td>4551</td>
      <td>0</td>
      <td>CCA133AAA</td>
      <td>1</td>
      <td>P133K</td>
      <td>1</td>
      <td>1 nonsynonymous</td>
      <td>165.0</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>


Print the number of mutations that pass RBD bind, RBD expression, and are not to sites that are disulfide bonds (if specified in config) 


```python
# if we are excluding all cysteines to remove spurious mutations that break disulfide bonds:
if config['exclude_cysteines']:
    print("Here are the number of mutations that pass the bind, express, and disulfide filters:")
    print(len(mut_bind_expr
              .assign(pass_cysteine_filter=lambda x: x['mutation'].str[0] != "C")
              .query(f"delta_bind >= {config[f'escape_score_min_bind_mut']} & \
                       delta_expr >= {config[f'escape_score_min_expr_mut']} & \
                       pass_cysteine_filter")
             ))
    print("There are these many possible mutations (excluding wildtype and disulfides!):")
    print(mut_bind_expr.query('wildtype!="C"')['position'].nunique()*19
         )

else:
    print("Here are the number of mutations that pass the bind and express filters:")
    print(len(mut_bind_expr
              .assign(pass_cysteine_filter=lambda x: x['mutation'].str[0] != "C")
              .query(f"delta_bind >= {config[f'escape_score_min_bind_mut']} & \
                       delta_expr >= {config[f'escape_score_min_expr_mut']}")
             ))
    print("There are these many possible mutations (excluding wildtype!):")
    print(mut_bind_expr['position'].nunique()*19
         )
```

    Here are the number of mutations that pass the bind, express, and disulfide filters:
    2444
    There are these many possible mutations (excluding wildtype and disulfides!):
    3667


Plot the fraction of variants that **have already passed the pre-count filter** that are filtered by the ACE2-binding or expression thresholds:


```python
print('These are the sites that are involved in disulfide bonds:')
print(mut_bind_expr.query('wildtype=="C"')['position'].unique())
```

    These are the sites that are involved in disulfide bonds:
    [336 361 379 391 432 480 488 525]



```python
frac_ACE2bind_expr_pass_filter = (
    escape_scores
    .query('pass_pre_count_filter == True')
    [['pre_sample', 'library', 'target', config['escape_score_group_by'],
      'pre_count', 'pass_ACE2bind_expr_filter', 'variant_class']]
    .drop_duplicates()
    .groupby(['pre_sample', 'library', 'variant_class'], observed=True)
    .aggregate(n_variants=pd.NamedAgg('pass_ACE2bind_expr_filter', 'count'),
               n_pass_filter=pd.NamedAgg('pass_ACE2bind_expr_filter', 'sum')
               )
    .reset_index()
    .assign(frac_pass_filter=lambda x: x['n_pass_filter'] / x['n_variants'],
            pre_sample=lambda x: pd.Categorical(x['pre_sample'], x['pre_sample'].unique(), ordered=True).remove_unused_categories())
    )

p = (ggplot(frac_ACE2bind_expr_pass_filter) +
     aes('variant_class', 'frac_pass_filter', fill='variant_class') +
     geom_bar(stat='identity') +
     facet_grid('library ~ pre_sample') +
     theme(axis_text_x=element_text(angle=90),
           figure_size=(3.3 * frac_ACE2bind_expr_pass_filter['pre_sample'].nunique(),
                        2 * frac_ACE2bind_expr_pass_filter['library'].nunique()),
           panel_grid_major_x=element_blank(),
           ) +
     scale_fill_manual(values=CBPALETTE[1:]) +
     expand_limits(y=(0, 1))
     )

_ = p.draw()
```


    
![png](counts_to_scores_files/counts_to_scores_61_0.png)
    


## Examine and write escape scores
Plot the distribution of escape scores across variants of different classes **among those that pass both the pre-selection count filter and the ACE2-binding / expression filter**.
If things are working correctly, we don't expect escape in wildtype (or synonymous variants), but do expect escape for some small fraction of nonsynymous variants.
Also, we do not plot the scores for the stop codon variant class, as most stop-codon variants have already been filtered out so this category is largely noise:


```python
nfacets = len(escape_scores.groupby(['library', 'name']).nunique())
ncol = min(8, nfacets)
nrow = math.ceil(nfacets / ncol)

df = (escape_scores
      .query('(pass_pre_count_filter == True) & (pass_ACE2bind_expr_filter == True)')
      .query('variant_class != "stop"')
      )
     
p = (ggplot(df) +
     aes('variant_class', 'score', color='variant_class') +
     geom_boxplot(outlier_size=1.5, outlier_alpha=0.1) +
     facet_wrap('~ name + library', ncol=ncol) +
     theme(axis_text_x=element_text(angle=90),
           figure_size=(2.35 * ncol, 3 * nrow),
           panel_grid_major_x=element_blank(),
           ) +
     scale_fill_manual(values=CBPALETTE[1:]) +
     scale_color_manual(values=CBPALETTE[1:])
     )

_ = p.draw()
```


    
![png](counts_to_scores_files/counts_to_scores_63_0.png)
    


Also, we want to see how much the high escape scores are correlated with simple coverage.
To do this, we plot the correlation between escape score and pre-selection count just for the nonsynonymous variants (which are the ones that we expect to have true escape).
The plots below have a lot of overplotting, but are still sufficient to test of the score is simply correlated with the pre-selection counts or not.
The hoped for result is that the escape score doesn't appear to be strongly correlated with pre-selection counts:


```python
p = (ggplot(escape_scores
            .query('pass_pre_count_filter == True')
            .query('(pass_pre_count_filter == True) & (pass_ACE2bind_expr_filter == True)')
            .query('variant_class=="1 nonsynonymous"')
            ) +
     aes('pre_count', 'score') +
     geom_point(alpha=0.1, size=1) +
     facet_wrap('~ name + library', ncol=ncol) +
     theme(axis_text_x=element_text(angle=90),
           figure_size=(2.35 * ncol, 2.35 * nrow),
           ) +
     scale_fill_manual(values=CBPALETTE[1:]) +
     scale_color_manual(values=CBPALETTE[1:]) +
     scale_x_log10()
     )

_ = p.draw()
```


    
![png](counts_to_scores_files/counts_to_scores_65_0.png)
    


Write the escape scores to a file:


```python
print(f"Writing escape scores for {primary_target} to {config['escape_scores']}")
escape_scores.to_csv(config['escape_scores'], index=False, float_format='%.4g')
```

    Writing escape scores for Delta to results/escape_scores/scores.csv


### Now we will also remove anything that did not pass all the filters above. 


```python
escape_scores_primary = (escape_scores
                         .query('(pass_pre_count_filter == True) & (pass_ACE2bind_expr_filter == True)')
                        )

display(HTML(escape_scores_primary.head().to_html()))
print(f"Read {len(escape_scores_primary)} scores.")
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>target</th>
      <th>library</th>
      <th>pre_sample</th>
      <th>post_sample</th>
      <th>barcode</th>
      <th>score</th>
      <th>score_var</th>
      <th>pre_count</th>
      <th>post_count</th>
      <th>codon_substitutions</th>
      <th>n_codon_substitutions</th>
      <th>aa_substitutions</th>
      <th>n_aa_substitutions</th>
      <th>variant_class</th>
      <th>pre_count_filter_cutoff</th>
      <th>pass_pre_count_filter</th>
      <th>muts_pass_bind_filter</th>
      <th>muts_pass_expr_filter</th>
      <th>pass_ACE2bind_expr_filter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>267C_200</td>
      <td>Delta</td>
      <td>lib1</td>
      <td>delta_9-16-none-0-ref</td>
      <td>delta_9-267C-200-abneg</td>
      <td>TGGGAAAAACAAAGCT</td>
      <td>0.000077</td>
      <td>1.173595e-08</td>
      <td>5206</td>
      <td>0</td>
      <td>GGT9GAT AAA199TGT</td>
      <td>2</td>
      <td>G9D K199C</td>
      <td>2</td>
      <td>&gt;1 nonsynonymous</td>
      <td>165.0</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>267C_200</td>
      <td>Delta</td>
      <td>lib1</td>
      <td>delta_9-16-none-0-ref</td>
      <td>delta_9-267C-200-abneg</td>
      <td>TACAAAGCACGCTAAA</td>
      <td>0.000079</td>
      <td>1.255159e-08</td>
      <td>5034</td>
      <td>0</td>
      <td>CAG176AAT</td>
      <td>1</td>
      <td>Q176N</td>
      <td>1</td>
      <td>1 nonsynonymous</td>
      <td>165.0</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>267C_200</td>
      <td>Delta</td>
      <td>lib1</td>
      <td>delta_9-16-none-0-ref</td>
      <td>delta_9-267C-200-abneg</td>
      <td>TTAAAGTAGTACGACA</td>
      <td>0.000083</td>
      <td>1.374779e-08</td>
      <td>4810</td>
      <td>0</td>
      <td>CCT7ACT</td>
      <td>1</td>
      <td>P7T</td>
      <td>1</td>
      <td>1 nonsynonymous</td>
      <td>165.0</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>267C_200</td>
      <td>Delta</td>
      <td>lib1</td>
      <td>delta_9-16-none-0-ref</td>
      <td>delta_9-267C-200-abneg</td>
      <td>ACGTGAAACACCATGT</td>
      <td>0.000084</td>
      <td>1.419273e-08</td>
      <td>4734</td>
      <td>0</td>
      <td>TCC19CAA</td>
      <td>1</td>
      <td>S19Q</td>
      <td>1</td>
      <td>1 nonsynonymous</td>
      <td>165.0</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8</th>
      <td>267C_200</td>
      <td>Delta</td>
      <td>lib1</td>
      <td>delta_9-16-none-0-ref</td>
      <td>delta_9-267C-200-abneg</td>
      <td>AGCTTTCAGTCGTACA</td>
      <td>0.000610</td>
      <td>1.064903e-07</td>
      <td>4574</td>
      <td>3</td>
      <td>AAT24CAT</td>
      <td>1</td>
      <td>N24H</td>
      <td>1</td>
      <td>1 nonsynonymous</td>
      <td>165.0</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>


    Read 3677824 scores.


### Count number of barcodes per mutation and remove variants with >1 amino acid substitution
Also add the number of barocdes per mutation to the `escape_scores` dataframe and plot this. 
But first see how many variants there are with >1 mutation, and query the dataframe to look at them qualitatively. 


```python
p = (ggplot(escape_scores_primary) +
     aes('n_aa_substitutions', fill='variant_class') +
     geom_bar() +
     facet_wrap('~library + pre_sample', ncol=5) +
     theme(
           # figure_size=(3.3 * escape_scores_primary['pre_sample'].nunique(),
                        # 2 * escape_scores_primary['library'].nunique()),
         figure_size=(12, 4),
           panel_grid_major_x=element_blank(),
           ) +
     scale_fill_manual(values=CBPALETTE[1:]) +
     expand_limits(y=(0, 1))
     )

_ = p.draw()
```


    
![png](counts_to_scores_files/counts_to_scores_71_0.png)
    


### Filter dataframe on single mutations that are present in at least `n` number of variants (specified in `config.yaml` file)
Now see how many `n_single_mut_measurements` there are for each variant:


```python
print(f'Remove anything with fewer than {config["escape_frac_min_single_mut_measurements"]} single mutant measurements (barcodes)')

raw_avg_single_mut_scores = (
    escape_scores_primary
    .query('n_aa_substitutions == 1')
    .rename(columns={'name': 'selection',
                     'aa_substitutions': 'mutation'})
    .groupby(['selection', 'library', 'mutation'])
    .aggregate(raw_single_mut_score=pd.NamedAgg('score', 'mean'),
               n_single_mut_measurements=pd.NamedAgg('barcode', 'count')
              )
    .assign(sufficient_measurements=lambda x: (
                (x['n_single_mut_measurements'] >= config['escape_frac_min_single_mut_measurements'])))
    .reset_index()
    )

# remove mutations with insufficient measurements
effects_df = (raw_avg_single_mut_scores
              .query('sufficient_measurements == True')
              .drop(columns='sufficient_measurements')
              )

# some error checks
assert len(effects_df) == len(effects_df.drop_duplicates()), 'duplicate rows in `effects_df`'
assert all(effects_df['raw_single_mut_score'].notnull() | (effects_df['n_single_mut_measurements'] == 0))

display(HTML(effects_df.head().to_html()))
```

    Remove anything with fewer than 2 single mutant measurements (barcodes)



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>selection</th>
      <th>library</th>
      <th>mutation</th>
      <th>raw_single_mut_score</th>
      <th>n_single_mut_measurements</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>267C_200</td>
      <td>lib1</td>
      <td>A105G</td>
      <td>0.000864</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>267C_200</td>
      <td>lib1</td>
      <td>A105S</td>
      <td>0.002701</td>
      <td>13</td>
    </tr>
    <tr>
      <th>2</th>
      <td>267C_200</td>
      <td>lib1</td>
      <td>A105T</td>
      <td>0.001370</td>
      <td>17</td>
    </tr>
    <tr>
      <th>3</th>
      <td>267C_200</td>
      <td>lib1</td>
      <td>A105Y</td>
      <td>0.001146</td>
      <td>17</td>
    </tr>
    <tr>
      <th>4</th>
      <td>267C_200</td>
      <td>lib1</td>
      <td>A145C</td>
      <td>0.001262</td>
      <td>12</td>
    </tr>
  </tbody>
</table>


### Should we exclude mutations for which the wildtype identity is a cysteine?
These would be mutations that break disulfide bonds (unless there is an unpaired cysteine in the protein that does not form a disulfide bond, of course). 
Note that this approach would not work well for something like the SARS-CoV-2 NTD, where it has been documented (by the Veesler lab) that mutations in the B.1.427/429 (epislon) variant can rearrange disulfide bonds, leading to major structural rearrangements in the NTD, and yet an apparently fully functional spike. 

If we are excluding cysteines, do that now:


```python
# if we are excluding all cysteines to remove spurious mutations that break disulfide bonds:
if config['exclude_cysteines']:
    print(f'Excluding mutations where the wildtype identity is a cysteine')
    effects_df = effects_df.assign(pass_cysteine_filter=lambda x: x['mutation'].str[0] != "C",
                                  )
    disulfides_to_drop = effects_df.query('pass_cysteine_filter==False')['mutation'].unique()
    print(f'Specifically, excluding: {disulfides_to_drop}')
    effects_df=effects_df.query('pass_cysteine_filter').drop(columns='pass_cysteine_filter')

else:
    print(f'Retaining mutations where the wildtype identity is a cysteine')
```

    Excluding mutations where the wildtype identity is a cysteine
    Specifically, excluding: ['C102I' 'C195A' 'C195D' 'C195E' 'C195F' 'C195G' 'C195H' 'C195I' 'C195L'
     'C195M' 'C195N' 'C195P' 'C195Q' 'C195S' 'C195T' 'C195V' 'C195W' 'C195Y'
     'C31A' 'C31D' 'C31E' 'C31G' 'C31H' 'C31I' 'C31K' 'C31L' 'C31M' 'C31N'
     'C31P' 'C31Q' 'C31R' 'C31S' 'C31T' 'C31V' 'C31W' 'C61A' 'C61D' 'C61E'
     'C61F' 'C61G' 'C61H' 'C61I' 'C61K' 'C61L' 'C61M' 'C61N' 'C61P' 'C61Q'
     'C61R' 'C61S' 'C61T' 'C61V' 'C61W' 'C61Y' 'C6D' 'C6E' 'C6G' 'C6H' 'C6N'
     'C6T']


We need to compute the escape scores (calculated as [here](https://jbloomlab.github.io/dms_variants/dms_variants.codonvarianttable.html?highlight=escape_scores#dms_variants.codonvarianttable.CodonVariantTable.escape_scores)) back to escape fractions. We define a function to do this depending on the score type:


```python
def score_to_frac(score):
    """Convert escape score to escape fraction."""
    if pd.isnull(score):
        return pd.NA
    floor = config['escape_score_floor_E']
    ceil = config['escape_score_ceil_E']
    if config['escape_score_type'] == 'frac_escape':
        return min(ceil, max(floor, score))  # just the score after applying ceiling and floor
    elif config['escape_score_type'] == 'log_escape':
        # convert score back to fraction, apply ceiling, then re-scale so floor is 0
        frac = 2**score
        frac = min(ceil, max(floor, frac))
        frac = (frac - floor) / (ceil - floor)
        return frac
    else:
        raise ValueError(f"invalid `escape_score_type` of {config['escape_score_type']}")

effects_df = (
    effects_df
    .assign(
            mut_escape_frac_single_mut=lambda x: x['raw_single_mut_score'].map(score_to_frac),
            )
    )
```

### Average the escape score across all barcodes of the same mutation, for each library, and the average of both libraries. 
Add rows that are the average of the two libraries for the fraction escape for all mutations that are present in both libraries (and if in just one library, the value in that or purge depending on config values printed here), the number of libraries each mutation is measured in, and the sum of the statistics giving the number of measurements:


```python
min_libs = config['escape_frac_avg_min_libraries']
min_single = config['escape_frac_avg_min_single']
print(f"Only taking average of mutations with escape fractions in >={min_libs} libraries "
      f"or with >={min_single} single-mutant measurements total.")

effects_df = (
    effects_df
    .query('library != "average"')  # git rid of averages if already there
    .assign(nlibs=1)
    .append(effects_df
            .query('library != "average"')
            .groupby(['selection', 'mutation'])
            .aggregate(nlibs=pd.NamedAgg('library', 'count'),
                       mut_escape_frac_single_mut=pd.NamedAgg('mut_escape_frac_single_mut',
                                                              lambda s: s.mean(skipna=True)),
                       n_single_mut_measurements=pd.NamedAgg('n_single_mut_measurements', 'sum'),
                       )
            .query('(nlibs >= @min_libs) or (n_single_mut_measurements >= @min_single)')
            .assign(library="average")
            .reset_index(),
            ignore_index=True, sort=False,
            )
    )

print(len(effects_df.query('nlibs>1')))
print(len(effects_df.query('nlibs==1')))
```

    Only taking average of mutations with escape fractions in >=2 libraries or with >=2 single-mutant measurements total.
    88796
    179576


Plot the correlations of the escape fractions among the two libraries for all selections performed on both libraries. 


```python
libraries = [lib for lib in effects_df['library'].unique() if lib != "average"]
assert len(libraries) == 2, 'plot only makes sense if 2 libraries'

# wide data frame with each library's score in a different column
effects_df_wide = (
    effects_df
    .query('library != "average"')
    .query(f"n_single_mut_measurements >= 1")
    # just get selections with 2 libraries
    .assign(nlibs=lambda x: x.groupby('selection')['library'].transform('nunique'))
    .query('nlibs == 2')
    # now make columns for each library, only keep mutants with scores for both libs
    [['selection', 'mutation', 'library', 'mut_escape_frac_single_mut']]
    .pivot_table(index=['selection', 'mutation'],
                 columns='library',
                 values='mut_escape_frac_single_mut',
                 aggfunc='first')
    .reset_index()
    .dropna(axis=0)
    )

# correlations between libraries
corrs = (
    effects_df_wide
    .groupby('selection')
    [libraries]
    .corr(method='pearson')
    .reset_index()
    .query('library == @libraries[0]')
    .assign(correlation=lambda x: 'R=' + x[libraries[1]].round(2).astype(str))
    [['selection', 'correlation']]
    # add number of mutations measured
    .merge(effects_df_wide
           .groupby('selection')
           .size()
           .rename('n')
           .reset_index()
           )
    .assign(correlation=lambda x: x['correlation'] + ', N=' + x['n'].astype(str))
    )

# plot correlations
nfacets = effects_df_wide['selection'].nunique()
ncol = min(nfacets, 5)
nrow = math.ceil(nfacets / ncol)
xmin = effects_df_wide[libraries[0]].min()
xspan = effects_df_wide[libraries[0]].max() - xmin
ymin = effects_df_wide[libraries[1]].min()
yspan = effects_df_wide[libraries[1]].max() - ymin
p = (ggplot(effects_df_wide) +
     aes(libraries[0], libraries[1]) +
     geom_point(alpha=0.2) +
     geom_text(mapping=aes(label='correlation'),
               data=corrs,
               x=0.01 * xspan + xmin,
               y=0.99 * yspan + ymin,
               size=10,
               ha='left',
               va='top',
               ) +
     facet_wrap('~ selection', ncol=ncol) +
     theme(figure_size=(2.5 * ncol, 2.5 * nrow),
           plot_title=element_text(size=14)) +
     ggtitle('Mutation-level escape fractions')
     )

_ = p.draw()
```


    
![png](counts_to_scores_files/counts_to_scores_81_0.png)
    


### Escape at site level
The above analysis estimates the effects of mutations. We also compute escape statistics at the site level. First, add sites to the data frame of mutational effects:


```python
effects_df = (
    effects_df
    .assign(site=lambda x: x['mutation'].str[1: -1].astype(int),
            wildtype=lambda x: x['mutation'].str[0],
            mutant=lambda x: x['mutation'].str[-1],
            )
    )
```

Now compute some site-level metrics. These are the average and total escape fraction at each site over all mutations at the site:


```python
site_effects_df = (
    effects_df
    .groupby(['selection', 'library', 'site'])
    .aggregate(
        site_avg_escape_frac_single_mut=pd.NamedAgg('mut_escape_frac_single_mut',
                                                    lambda s: s.mean(skipna=True)),
        site_total_escape_frac_single_mut=pd.NamedAgg('mut_escape_frac_single_mut', 'sum'),
        )
    .reset_index()
    )

display(HTML(site_effects_df.head().to_html(index=False)))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>selection</th>
      <th>library</th>
      <th>site</th>
      <th>site_avg_escape_frac_single_mut</th>
      <th>site_total_escape_frac_single_mut</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>267C_200</td>
      <td>average</td>
      <td>1</td>
      <td>0.002617</td>
      <td>0.049718</td>
    </tr>
    <tr>
      <td>267C_200</td>
      <td>average</td>
      <td>2</td>
      <td>0.001566</td>
      <td>0.029758</td>
    </tr>
    <tr>
      <td>267C_200</td>
      <td>average</td>
      <td>3</td>
      <td>0.002157</td>
      <td>0.040988</td>
    </tr>
    <tr>
      <td>267C_200</td>
      <td>average</td>
      <td>4</td>
      <td>0.001494</td>
      <td>0.028385</td>
    </tr>
    <tr>
      <td>267C_200</td>
      <td>average</td>
      <td>5</td>
      <td>0.001422</td>
      <td>0.027019</td>
    </tr>
  </tbody>
</table>


Plot correlations between libraries of the same selection for the site-level statistics:


```python
libraries = [lib for lib in effects_df['library'].unique() if lib != "average"]
assert len(libraries) == 2, 'plot only makes sense if 2 libraries'

for val in ['site_avg_escape_frac_single_mut', 'site_total_escape_frac_single_mut']:

    # wide data frame with each library's score in a different column
    site_effects_df_wide = (
        site_effects_df
        .query('library != "average"')
        # just get selections with 2 libraries
        .assign(nlibs=lambda x: x.groupby('selection')['library'].transform('nunique'))
        .query('nlibs == 2')
        # now make columns for each library, only keep sites with scores for both libs
        .pivot_table(index=['selection', 'site'],
                     columns='library',
                     values=val)
        .reset_index()
        .dropna(axis=0)
        )

    # correlations between libraries
    corrs = (
        site_effects_df_wide
        .groupby('selection')
        [libraries]
        .corr(method='pearson')
        .reset_index()
        .query('library == @libraries[0]')
        .assign(correlation=lambda x: 'R=' + x[libraries[1]].round(2).astype(str))
        [['selection', 'correlation']]
        # add number of mutations measured
        .merge(site_effects_df_wide
               .groupby('selection')
               .size()
               .rename('n')
               .reset_index()
               )
        .assign(correlation=lambda x: x['correlation'] + ', N=' + x['n'].astype(str))
        )

    # plot correlations
    nfacets = site_effects_df_wide['selection'].nunique()
    ncol = min(nfacets, 5)
    nrow = math.ceil(nfacets / ncol)
    xmin = site_effects_df_wide[libraries[0]].min()
    xspan = site_effects_df_wide[libraries[0]].max() - xmin
    ymin = site_effects_df_wide[libraries[1]].min()
    yspan = site_effects_df_wide[libraries[1]].max() - ymin
    p = (ggplot(site_effects_df_wide) +
         aes(libraries[0], libraries[1]) +
         geom_point(alpha=0.2) +
         geom_text(mapping=aes(label='correlation'),
                   data=corrs,
                   x=0.01 * xspan + xmin,
                   y=0.99 * yspan + ymin,
                   size=10,
                   ha='left',
                   va='top',
                   ) +
         facet_wrap('~ selection', ncol=ncol) +
         theme(figure_size=(2.5 * ncol, 2.5 * nrow),
               plot_title=element_text(size=14)) +
         ggtitle(val)
         )

    _ = p.draw()
```


    
![png](counts_to_scores_files/counts_to_scores_87_0.png)
    



    
![png](counts_to_scores_files/counts_to_scores_87_1.png)
    


## Write file with escape fractions at mutation and site levels
We write a files that has the mutation- and site-level escape fractions. This file has both the separate measurements for each library plus the average across libraries for all mutations measured in both libraries. We name the columns in such a way that this file can be used as [dms-view data file](https://dms-view.github.io/docs/dataupload):


```python
escape_fracs_to_write = (
    effects_df
    .merge(site_effects_df,
           how='left',
           validate='many_to_one',
           on=['selection', 'library', 'site'])
    .assign(protein_chain=config['escape_frac_protein_chain'],
            protein_site=lambda x: x['site'] + config['site_number_offset'],
            label_site=lambda x: x['protein_site'],
            condition=lambda x: x['selection'].where(x['library'] == "average", x['selection'] + '_' + x['library']),
            mutation=lambda x: x['mutant'],  # dms-view uses mutation to refer to mutant amino acid
            )
    [['selection', 'library', 'condition', 'site', 'label_site', 'wildtype', 'mutation',
      'protein_chain', 'protein_site', 'mut_escape_frac_single_mut', 'site_total_escape_frac_single_mut',
      'site_avg_escape_frac_single_mut', 'nlibs',
      ]]
    .sort_values(['library', 'selection', 'site', 'mutation'])
    )

print('Here are the first few lines that will be written to the escape-fraction file:')
display(HTML(escape_fracs_to_write.head().to_html(index=False)))

print(f"\nWriting to {config['escape_fracs']}")
escape_fracs_to_write.to_csv(config['escape_fracs'], index=False, float_format='%.4g')

```

    Here are the first few lines that will be written to the escape-fraction file:



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>selection</th>
      <th>library</th>
      <th>condition</th>
      <th>site</th>
      <th>label_site</th>
      <th>wildtype</th>
      <th>mutation</th>
      <th>protein_chain</th>
      <th>protein_site</th>
      <th>mut_escape_frac_single_mut</th>
      <th>site_total_escape_frac_single_mut</th>
      <th>site_avg_escape_frac_single_mut</th>
      <th>nlibs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>267C_200</td>
      <td>average</td>
      <td>267C_200</td>
      <td>1</td>
      <td>331</td>
      <td>N</td>
      <td>A</td>
      <td>E</td>
      <td>331</td>
      <td>0.000980</td>
      <td>0.049718</td>
      <td>0.002617</td>
      <td>2</td>
    </tr>
    <tr>
      <td>267C_200</td>
      <td>average</td>
      <td>267C_200</td>
      <td>1</td>
      <td>331</td>
      <td>N</td>
      <td>C</td>
      <td>E</td>
      <td>331</td>
      <td>0.001776</td>
      <td>0.049718</td>
      <td>0.002617</td>
      <td>2</td>
    </tr>
    <tr>
      <td>267C_200</td>
      <td>average</td>
      <td>267C_200</td>
      <td>1</td>
      <td>331</td>
      <td>N</td>
      <td>D</td>
      <td>E</td>
      <td>331</td>
      <td>0.001094</td>
      <td>0.049718</td>
      <td>0.002617</td>
      <td>2</td>
    </tr>
    <tr>
      <td>267C_200</td>
      <td>average</td>
      <td>267C_200</td>
      <td>1</td>
      <td>331</td>
      <td>N</td>
      <td>E</td>
      <td>E</td>
      <td>331</td>
      <td>0.001751</td>
      <td>0.049718</td>
      <td>0.002617</td>
      <td>2</td>
    </tr>
    <tr>
      <td>267C_200</td>
      <td>average</td>
      <td>267C_200</td>
      <td>1</td>
      <td>331</td>
      <td>N</td>
      <td>F</td>
      <td>E</td>
      <td>331</td>
      <td>0.004506</td>
      <td>0.049718</td>
      <td>0.002617</td>
      <td>2</td>
    </tr>
  </tbody>
</table>


    
    Writing to results/escape_scores/escape_fracs.csv



```python

```
