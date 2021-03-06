# Align spike sequences in GISAID and count RBD mutations
This Python Jupyter notebook reads in a file of all spike sequences from GISAID, parses for "high-quality" sequences, builds a RBD alignment, and then makes a file that gives the counts of each mutation at each site.

## Set up analysis
Import Python modules:


```python
import io
import lzma
import os
import re
import subprocess

from Bio.Data.IUPACData import protein_letters
import Bio.SeqIO

from IPython.display import display, HTML

import matplotlib.pyplot as plt

import pandas as pd

from plotnine import *

import yaml
```


```python
print(f'Using BioPython version {Bio.__version__}')
print(f'Using pandas version {pd.__version__}')
!python --version

import plotnine
print(f'Using plotnine version {plotnine.__version__}')
```

    Using BioPython version 1.79
    Using pandas version 1.3.4
    Python 3.8.12
    Using plotnine version 0.8.0


Read the configuration file:


```python
with open('config.yaml') as f:
    config = yaml.safe_load(f)
```

Create output directory:


```python
os.makedirs(config['gisaid_mutations_dir'], exist_ok=True)
```

## Get accessions for all Delta sequences
I downloaded this from `gisaid.org` > `EpiCoV` > `Downloads` > `variant surveillance` > unzipped file > imported the `variant_surveillance.tsv` file, renamed with the date of download (e.g., `variant_surveillance_20211103.tsv`)


```python
metadata_file = config['variant_surveillance']
metadata = pd.read_csv(metadata_file, sep='\t')

display(HTML(metadata.head(1).to_html(index=False)))
```

    /fh/fast/bloom_j/computational_notebooks/agreaney/2021/SARS-CoV-2-RBD_Delta/env/lib/python3.8/site-packages/IPython/core/interactiveshell.py:3444: DtypeWarning: Columns (6,11) have mixed types.Specify dtype option on import or set low_memory=False.



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Accession ID</th>
      <th>Type</th>
      <th>Clade</th>
      <th>Pango lineage</th>
      <th>Pangolin version</th>
      <th>AA Substitutions</th>
      <th>Variant</th>
      <th>Collection date</th>
      <th>Location</th>
      <th>Host</th>
      <th>Submission date</th>
      <th>Is reference?</th>
      <th>Is complete?</th>
      <th>Is high coverage?</th>
      <th>Is low coverage?</th>
      <th>N-Content</th>
      <th>GC-Content</th>
      <th>Sequence length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>EPI_ISL_426900</td>
      <td>betacoronavirus</td>
      <td>G</td>
      <td>B.1</td>
      <td>2021-10-18</td>
      <td>(NSP15_A283V,NSP12_P323L,Spike_D614G)</td>
      <td>NaN</td>
      <td>2020-03-25</td>
      <td>Oceania / Australia / Northern Territory</td>
      <td>Human</td>
      <td>2020-04-17</td>
      <td>NaN</td>
      <td>True</td>
      <td>True</td>
      <td>NaN</td>
      <td>0.006865</td>
      <td>0.379674</td>
      <td>29862</td>
    </tr>
  </tbody>
</table>


Let's investigate some things about this metadata file, since it is the first time I am looking at it. 


```python
print(f'There are {len(metadata)} rows in the column. \nFYI, as of Nov. 3, 2021, there are about 4.8 million SARS-CoV-2 sequences in GISAID.')
print(f'Here are the variants that are tracked in this metadata file:')
for i in metadata["Variant"].unique():
    print(f'\t{i}')
```

    There are 4831920 rows in the column. 
    FYI, as of Nov. 3, 2021, there are about 4.8 million SARS-CoV-2 sequences in GISAID.
    Here are the variants that are tracked in this metadata file:
    	nan
    	VOC Alpha 202012/01 GRY (B.1.1.7+Q.x) first detected in the UK
    	VOC Beta GH/501Y.V2 (B.1.351+B.1.351.2+B.1.351.3) first detected in South Africa
    	VOC Gamma GR/501Y.V3 (P.1+P.1.x) first detected in Brazil/Japan
    	VOI Lambda GR/452Q.V1 (C.37+C.37.1) first detected in Peru
    	VOI Mu GH (B.1.621+B.1.621.1) first detected in Colombia
    	VOC Delta GK/478K.V1 (B.1.617.2+AY.x) first detected in India


Get the Delta accessions:


```python
delta_accessions = (metadata
                    [['Accession ID', 'Variant']]
                    .query('Variant=="VOC Delta GK/478K.V1 (B.1.617.2+AY.x) first detected in India"')
                    ['Accession ID']
                    .to_list()
                   )

print(f'There are {len(delta_accessions)} Delta sequences.')
```

    There are 2271590 Delta sequences.


## Parse full-length human human spike sequences

Read the spikes from the file downloaded from GISAID:


```python
print(f"Reading GISAID spikes in {config['gisaid_spikes']}")
# file is `xz` compressed
# with lzma.open(config['gisaid_spikes'], 'rt') as f:
with open(config['gisaid_spikes'], 'rt') as f:
    spikes = list(Bio.SeqIO.parse(f, 'fasta'))
print(f"Read {len(spikes)} spike sequences.")
```

    Reading GISAID spikes in data/spikeprot1030.fasta
    Read 4607422 spike sequences.


Make a data frame that has the BioPython SeqRecord, accession, length, host, and geographic location (country) for each spike.
Also determine whether sequences have ambiguous amino acids or are all valid amino acids:


```python
spikes_df = (
    pd.DataFrame({'seqrecord': spikes})
    .assign(description=lambda x: x['seqrecord'].map(lambda rec: rec.description),
            accession=lambda x: x['description'].str.split('|').str[3],
            date=lambda x: x['description'].str.split('|').str[2],
            country=lambda x: x['description'].str.split('|').str[-1],
            host=lambda x: x['description'].str.split('|').str[6].str.strip(),
            length=lambda x: x['seqrecord'].map(len),
            n_ambiguous=lambda x: x['seqrecord'].map(lambda rec: rec.seq.count('X') + rec.seq.count('x')),
            )
    )
```

Show number of sequences from different hosts, then keep only human ones:


```python
display(HTML(
    spikes_df
    .groupby('host')
    .aggregate(n_sequences=pd.NamedAgg('seqrecord', 'count'))
    .sort_values('n_sequences', ascending=False)
    .to_html()
    ))

spikes_df = spikes_df.query('host == "Human"')
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>n_sequences</th>
    </tr>
    <tr>
      <th>host</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Human</th>
      <td>4604438</td>
    </tr>
    <tr>
      <th>Environment</th>
      <td>1696</td>
    </tr>
    <tr>
      <th>Neovison vison</th>
      <td>961</td>
    </tr>
    <tr>
      <th>Felis catus</th>
      <td>83</td>
    </tr>
    <tr>
      <th>Canis lupus familiaris</th>
      <td>44</td>
    </tr>
    <tr>
      <th>Panthera leo</th>
      <td>37</td>
    </tr>
    <tr>
      <th>Mustela lutreola</th>
      <td>23</td>
    </tr>
    <tr>
      <th>Manis javanica</th>
      <td>19</td>
    </tr>
    <tr>
      <th>Odocoileus virginianus</th>
      <td>14</td>
    </tr>
    <tr>
      <th>Neovision vision</th>
      <td>14</td>
    </tr>
    <tr>
      <th>Panthera tigris jacksoni</th>
      <td>13</td>
    </tr>
    <tr>
      <th>Environmental</th>
      <td>10</td>
    </tr>
    <tr>
      <th>Rhinolophus malayanus</th>
      <td>7</td>
    </tr>
    <tr>
      <th>unknown</th>
      <td>6</td>
    </tr>
    <tr>
      <th>Aonyx cinereus</th>
      <td>5</td>
    </tr>
    <tr>
      <th>P1 culture</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Snow Leopard</th>
      <td>4</td>
    </tr>
    <tr>
      <th>Gorilla gorilla gorilla</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Panthera tigris</th>
      <td>3</td>
    </tr>
    <tr>
      <th>P2 culture</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Panthera uncia</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Mus musculus</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Tiger</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Dog</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Mesocricetus auratus</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Control</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Mustela putorius furo</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Rhinolophus stheno</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Rhinolophus shameli</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Rhinolophus pusillus</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Manis pentadactyla</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Rhinolophus marshalli</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Rhinolophus sinicus</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Rhinolophus affinis</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Chlorocebus sabaeus</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Rhinolophus bat</th>
      <td>1</td>
    </tr>
    <tr>
      <th>North America / USA / Louisiana / New Orleands</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Puma concolor</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Prionailurus bengalensis euptilurus</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Panthera tigris sondaica</th>
      <td>1</td>
    </tr>
    <tr>
      <th>P3 culture</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Mus musculus (BALB/c mice)</th>
      <td>1</td>
    </tr>
  </tbody>
</table>


Now we will only keep the Delta sequences. 
Note that this is different that what we have done before! 


```python
spikes_df=spikes_df.query('accession in @delta_accessions')

display(HTML(spikes_df.head(1).to_html(index=False)))
print(f'There are {len(spikes_df)} Delta sequences.')
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>seqrecord</th>
      <th>description</th>
      <th>accession</th>
      <th>date</th>
      <th>country</th>
      <th>host</th>
      <th>length</th>
      <th>n_ambiguous</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>(M, F, V, F, L, V, L, L, P, L, V, S, S, Q, C, V, N, L, T, T, R, T, Q, L, P, P, A, Y, T, N, S, F, T, R, G, V, Y, Y, P, D, K, V, F, R, S, S, V, L, H, S, T, Q, D, L, F, L, P, F, F, S, N, V, T, W, F, H, A, I, H, V, S, G, T, N, G, T, K, R, F, D, N, P, V, L, P, F, N, D, G, V, Y, F, A, S, T, E, K, S, N, I, ...)</td>
      <td>Spike|hCoV-19/Spain/CT-HUVH-97227/2021|2021-03-03|EPI_ISL_1264410|Original|hCoV-19^^Catalunya|Human|Hospital Universitari Arnau de Vilanova|Hospital Universitari Vall d'Hebron - Vall d'Hebron Institut de Recerca|Anton|Spain</td>
      <td>EPI_ISL_1264410</td>
      <td>2021-03-03</td>
      <td>Spain</td>
      <td>Human</td>
      <td>1274</td>
      <td>0</td>
    </tr>
  </tbody>
</table>


    There are 2084153 Delta sequences.


Plot distribution of lengths and only keep sequences that are full-length (or near full-length) spikes:


```python
print('Distribution of length for all spikes:')
p = (ggplot(spikes_df) +
     aes('length') +
     geom_bar() +
     ylab('number of sequences') +
     theme(figure_size=(10, 2))
     )
fig = p.draw()
display(fig)
plt.close(fig)

min_length, max_length = 1260, 1276
print(f"\nOnly keeping spikes with lengths between {min_length} and {max_length}")
spikes_df = (
    spikes_df
    .assign(valid_length=lambda x: (min_length <= x['length']) & (x['length'] <= max_length))
    )

print('Here are number of sequences with valid and invalid lengths:')
display(HTML(spikes_df
             .groupby('valid_length')
             .aggregate(n_sequences=pd.NamedAgg('seqrecord', 'count'))
             .to_html()
             ))

print('\nDistribution of lengths for sequences with valid and invalid lengths; '
      'dotted red lines delimit valid lengths:')
p = (ggplot(spikes_df
            .assign(valid_length=lambda x: x['valid_length'].map({True: 'valid length',
                                                                  False: 'invalid length'}))
            ) +
     aes('length') +
     geom_bar() +
     ylab('number of sequences') +
     theme(figure_size=(10, 2), subplots_adjust={'wspace': 0.2}) +
     facet_wrap('~ valid_length', scales='free') +
     geom_vline(xintercept=min_length - 0.5, color='red', linetype='dotted') +
     geom_vline(xintercept=max_length + 0.5, color='red', linetype='dotted')
     )
fig = p.draw()
display(fig)
plt.close(fig)

spikes_df = spikes_df.query('valid_length')
```

    Distribution of length for all spikes:



    
![png](gisaid_rbd_mutations_files/gisaid_rbd_mutations_23_1.png)
    


    
    Only keeping spikes with lengths between 1260 and 1276
    Here are number of sequences with valid and invalid lengths:



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>n_sequences</th>
    </tr>
    <tr>
      <th>valid_length</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>False</th>
      <td>6249</td>
    </tr>
    <tr>
      <th>True</th>
      <td>2077904</td>
    </tr>
  </tbody>
</table>


    
    Distribution of lengths for sequences with valid and invalid lengths; dotted red lines delimit valid lengths:



    
![png](gisaid_rbd_mutations_files/gisaid_rbd_mutations_23_5.png)
    


Finally, we get rid of spikes with **lots** of ambiguous residues as they may hinder the alignment below.
We will then do more detailed filtering for ambiguous residues just in the RBD region after alignment:


```python
max_ambiguous = 100
print(f"Filtering sequences with > {max_ambiguous} ambiguous residues")
spikes_df = (
    spikes_df
    .assign(excess_ambiguous=lambda x: x['n_ambiguous'] > max_ambiguous)
    )
display(HTML(
    spikes_df
    .groupby('excess_ambiguous')
    .aggregate(n_sequences=pd.NamedAgg('seqrecord', 'count'))
    .to_html()
    ))
```

    Filtering sequences with > 100 ambiguous residues



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>n_sequences</th>
    </tr>
    <tr>
      <th>excess_ambiguous</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>False</th>
      <td>1881456</td>
    </tr>
    <tr>
      <th>True</th>
      <td>196448</td>
    </tr>
  </tbody>
</table>


## Align the RBD region of the spikes
We now align the RBD regions of the spikes.
We do this **before** we filter sequences with too many ambiguous residues so that we can do that filtering just on the RBD region.

We align with `mafft` using the `--addfragments` and `--keeplength` options (see [here](https://mafft.cbrc.jp/alignment/software/closelyrelatedviralgenomes.html) and [here](https://mafft.cbrc.jp/alignment/software/addsequences.html)) to align relative to a reference that is just the RBD; these options also clip the sequences relative to the reference.
Note that these options make sense if the following conditions are met:
  1. Sequences are all very similar.
  2. We are not worried about insertions.
For now, both of these appear to be true, but this choice should be kept in mind if there is a lot more divergence or insertions.

We align relative to the reference that is the wildtype sequence used for the experiments:


```python
print(f"Reading reference nucleotide sequence in {config['wildtype_sequence']}")
refseq = Bio.SeqIO.read(config['wildtype_sequence'], 'fasta')

refprotfile = os.path.join(config['gisaid_mutations_dir'], 'reference_RBD.fasta')
print(f"Writing protein translation of reference sequence to {refprotfile}")
refseq.seq = refseq.seq.translate()
_ = Bio.SeqIO.write(refseq, refprotfile, 'fasta')
```

    Reading reference nucleotide sequence in data/wildtype_sequence.fasta
    Writing protein translation of reference sequence to results/GISAID_mutations/reference_RBD.fasta


Write all the other spikes to a file:


```python
spikes_file = os.path.join(config['gisaid_mutations_dir'],
                           'human_full-length_spikes.fasta')
print(f"Writing the spikes to {spikes_file}")
_ = Bio.SeqIO.write(spikes_df['seqrecord'].tolist(), spikes_file, 'fasta')
```

    Writing the spikes to results/GISAID_mutations/human_full-length_spikes.fasta


Now make the alignment.
Note that we use multiple threads to speed things up, and also align the spikes in chunks.
The reason that we have to the chunkwise alignment is that some unclear `mafft` error was arising if we tried to align them all at once:


```python
chunksize = 50000

aligned_rbds = []

for i in range(0, len(spikes_df), chunksize):
    spikes_file = os.path.join(config['gisaid_mutations_dir'],
                               f"human_full-length_spikes_{i + 1}-to-{i + chunksize}.fasta")
    print(f"Writing spikes {i + 1} to {i + chunksize} to {spikes_file}")
    _ = Bio.SeqIO.write(spikes_df['seqrecord'].tolist()[i: i + chunksize], spikes_file, 'fasta')
    print('Now aligning these sequences...')
    cmds = ['mafft', '--auto', '--thread', str(config['max_cpus']),
            '--keeplength', '--addfragments', spikes_file, refprotfile]
    res = subprocess.run(cmds, capture_output=True)
    if res.returncode:
        raise RuntimeError(f"Error in alignment:\n{res.stderr}")
    else:
        print('Alignment complete.\n')
        with io.StringIO(res.stdout.decode('utf-8')) as f:
            iseqs = list(Bio.SeqIO.parse(f, 'fasta'))
            # remove reference sequence, which should be first in file
            assert iseqs[0].seq == refseq.seq and iseqs[0].description == refseq.description
            iseqs = iseqs[1:]
            assert len(iseqs) == min(chunksize, len(spikes_df) - i)
            aligned_rbds += iseqs
            
assert len(aligned_rbds) == len(spikes_df)
```

    Writing spikes 1 to 50000 to results/GISAID_mutations/human_full-length_spikes_1-to-50000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 50001 to 100000 to results/GISAID_mutations/human_full-length_spikes_50001-to-100000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 100001 to 150000 to results/GISAID_mutations/human_full-length_spikes_100001-to-150000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 150001 to 200000 to results/GISAID_mutations/human_full-length_spikes_150001-to-200000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 200001 to 250000 to results/GISAID_mutations/human_full-length_spikes_200001-to-250000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 250001 to 300000 to results/GISAID_mutations/human_full-length_spikes_250001-to-300000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 300001 to 350000 to results/GISAID_mutations/human_full-length_spikes_300001-to-350000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 350001 to 400000 to results/GISAID_mutations/human_full-length_spikes_350001-to-400000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 400001 to 450000 to results/GISAID_mutations/human_full-length_spikes_400001-to-450000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 450001 to 500000 to results/GISAID_mutations/human_full-length_spikes_450001-to-500000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 500001 to 550000 to results/GISAID_mutations/human_full-length_spikes_500001-to-550000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 550001 to 600000 to results/GISAID_mutations/human_full-length_spikes_550001-to-600000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 600001 to 650000 to results/GISAID_mutations/human_full-length_spikes_600001-to-650000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 650001 to 700000 to results/GISAID_mutations/human_full-length_spikes_650001-to-700000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 700001 to 750000 to results/GISAID_mutations/human_full-length_spikes_700001-to-750000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 750001 to 800000 to results/GISAID_mutations/human_full-length_spikes_750001-to-800000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 800001 to 850000 to results/GISAID_mutations/human_full-length_spikes_800001-to-850000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 850001 to 900000 to results/GISAID_mutations/human_full-length_spikes_850001-to-900000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 900001 to 950000 to results/GISAID_mutations/human_full-length_spikes_900001-to-950000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 950001 to 1000000 to results/GISAID_mutations/human_full-length_spikes_950001-to-1000000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 1000001 to 1050000 to results/GISAID_mutations/human_full-length_spikes_1000001-to-1050000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 1050001 to 1100000 to results/GISAID_mutations/human_full-length_spikes_1050001-to-1100000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 1100001 to 1150000 to results/GISAID_mutations/human_full-length_spikes_1100001-to-1150000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 1150001 to 1200000 to results/GISAID_mutations/human_full-length_spikes_1150001-to-1200000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 1200001 to 1250000 to results/GISAID_mutations/human_full-length_spikes_1200001-to-1250000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 1250001 to 1300000 to results/GISAID_mutations/human_full-length_spikes_1250001-to-1300000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 1300001 to 1350000 to results/GISAID_mutations/human_full-length_spikes_1300001-to-1350000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 1350001 to 1400000 to results/GISAID_mutations/human_full-length_spikes_1350001-to-1400000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 1400001 to 1450000 to results/GISAID_mutations/human_full-length_spikes_1400001-to-1450000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 1450001 to 1500000 to results/GISAID_mutations/human_full-length_spikes_1450001-to-1500000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 1500001 to 1550000 to results/GISAID_mutations/human_full-length_spikes_1500001-to-1550000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 1550001 to 1600000 to results/GISAID_mutations/human_full-length_spikes_1550001-to-1600000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 1600001 to 1650000 to results/GISAID_mutations/human_full-length_spikes_1600001-to-1650000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 1650001 to 1700000 to results/GISAID_mutations/human_full-length_spikes_1650001-to-1700000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 1700001 to 1750000 to results/GISAID_mutations/human_full-length_spikes_1700001-to-1750000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 1750001 to 1800000 to results/GISAID_mutations/human_full-length_spikes_1750001-to-1800000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 1800001 to 1850000 to results/GISAID_mutations/human_full-length_spikes_1800001-to-1850000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 1850001 to 1900000 to results/GISAID_mutations/human_full-length_spikes_1850001-to-1900000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 1900001 to 1950000 to results/GISAID_mutations/human_full-length_spikes_1900001-to-1950000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 1950001 to 2000000 to results/GISAID_mutations/human_full-length_spikes_1950001-to-2000000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 2000001 to 2050000 to results/GISAID_mutations/human_full-length_spikes_2000001-to-2050000.fasta
    Now aligning these sequences...
    Alignment complete.
    
    Writing spikes 2050001 to 2100000 to results/GISAID_mutations/human_full-length_spikes_2050001-to-2100000.fasta
    Now aligning these sequences...
    Alignment complete.
    


## Parse / filter aligned RBDs

Now put all of the aligned RBDs into a data frame to filter and parse:


```python
rbd_df = (
    pd.DataFrame({'seqrecord': aligned_rbds})
    .assign(description=lambda x: x['seqrecord'].map(lambda rec: rec.description),
            country=lambda x: x['description'].str.split('|').str[-1],
            host=lambda x: x['description'].str.split('|').str[6].str.strip(),
            length=lambda x: x['seqrecord'].map(len),
            n_ambiguous=lambda x: x['seqrecord'].map(lambda rec: rec.seq.count('X') + rec.seq.count('x')),
            n_gaps=lambda x: x['seqrecord'].map(lambda rec: rec.seq.count('-')),
            all_valid_aas=lambda x: x['seqrecord'].map(lambda rec: re.fullmatch(f"[{protein_letters}]+",
                                                                                str(rec.seq)) is not None),
            )
    )

assert all(rbd_df['length'] == len(refseq))
```

Plot number of gaps and ambiguous nucleotides among sequences:


```python
for prop in ['n_ambiguous', 'n_gaps']:
    p = (ggplot(rbd_df) +
         aes(prop) +
         ylab('number of sequences') +
         theme(figure_size=(10, 2.5)) +
         geom_bar()
         )
    _ = p.draw()
```


    
![png](gisaid_rbd_mutations_files/gisaid_rbd_mutations_35_0.png)
    



    
![png](gisaid_rbd_mutations_files/gisaid_rbd_mutations_35_1.png)
    


Based on above plots, we will retain just RBDs with no ambiguous amino acids and no gaps:


```python
rbd_df = rbd_df.query('n_ambiguous == 0').query('n_gaps == 0')
assert rbd_df['all_valid_aas'].all()
print(f"Retained {len(rbd_df)} RBDs.")
```

    Retained 1930990 RBDs.


Now get and plot the number of amino-acid mutations per RBD relative to the reference sequence, plotting on both a linear and log scale.
We then filter all RBDs that have more than some maximum number of mutations, based on the idea that ones that are extremely highly mutated probably are erroneous.
**Note that this maximum number of mutations will change over time, so should be re-assessed periodically by looking at below plots.**


```python
max_muts = 8

refseq_str = str(refseq.seq)
rbd_df = (
    rbd_df
    .assign(seq=lambda x: x['seqrecord'].map(lambda rec: str(rec.seq)),
            n_mutations=lambda x: x['seq'].map(lambda s: sum(x != y for x, y in zip(s, refseq_str))))
    )

p = (ggplot(rbd_df) +
     aes('n_mutations') +
     geom_bar() +
     theme(figure_size=(10, 2.5)) +
     geom_vline(xintercept=max_muts + 0.5, color='red', linetype='dotted')
     )
_ = p.draw()
_ = (p + scale_y_log10()).draw()

rbd_df = rbd_df.query('n_mutations <= @max_muts')
```


    
![png](gisaid_rbd_mutations_files/gisaid_rbd_mutations_39_0.png)
    



    
![png](gisaid_rbd_mutations_files/gisaid_rbd_mutations_39_1.png)
    


Write RBD sequences that pass filtering to a file:


```python
print(f"Overall, there are {len(rbd_df)} aligned RBDs that passed filters.")

rbd_alignment_file = os.path.join(config['gisaid_mutations_dir'], 'RBD_alignment.fasta')
print(f"Writing alignment to {rbd_alignment_file}")
_ = Bio.SeqIO.write(rbd_df['seqrecord'].tolist(), rbd_alignment_file, 'fasta')
```

    Overall, there are 1930891 aligned RBDs that passed filters.
    Writing alignment to results/GISAID_mutations/RBD_alignment.fasta


## Get counts of each mutation
Now we get a data frame that gives the count of each mutation at each site:


```python
records = []
for tup in rbd_df[['seq', 'country']].itertuples():
    for isite, (mut, wt) in enumerate(zip(tup.seq, refseq_str), start=1):
        if mut != wt:
            records.append((isite, isite + config['site_number_offset'], wt, mut, tup.country))
            
muts_df = (pd.DataFrame.from_records(records,
                                     columns=['isite', 'site', 'wildtype', 'mutant', 'country'])
           .groupby(['isite', 'site', 'wildtype', 'mutant'])
           .aggregate(count=pd.NamedAgg('country', 'count'),
                      n_countries=pd.NamedAgg('country', 'nunique'))
           .reset_index()
           .sort_values('count', ascending=False)
           .assign(frequency=lambda x: x['count'] / len(rbd_df))
           )

print('Here are first few lines of mutation counts data frame:')
display(HTML(muts_df.head(n=15).to_html(index=False)))
```

    Here are first few lines of mutation counts data frame:



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>isite</th>
      <th>site</th>
      <th>wildtype</th>
      <th>mutant</th>
      <th>count</th>
      <th>n_countries</th>
      <th>frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>87</td>
      <td>417</td>
      <td>K</td>
      <td>N</td>
      <td>4660</td>
      <td>66</td>
      <td>0.002413</td>
    </tr>
    <tr>
      <td>116</td>
      <td>446</td>
      <td>G</td>
      <td>V</td>
      <td>4617</td>
      <td>92</td>
      <td>0.002391</td>
    </tr>
    <tr>
      <td>147</td>
      <td>477</td>
      <td>S</td>
      <td>I</td>
      <td>3806</td>
      <td>70</td>
      <td>0.001971</td>
    </tr>
    <tr>
      <td>192</td>
      <td>522</td>
      <td>A</td>
      <td>V</td>
      <td>3574</td>
      <td>53</td>
      <td>0.001851</td>
    </tr>
    <tr>
      <td>154</td>
      <td>484</td>
      <td>E</td>
      <td>Q</td>
      <td>2449</td>
      <td>73</td>
      <td>0.001268</td>
    </tr>
    <tr>
      <td>122</td>
      <td>452</td>
      <td>R</td>
      <td>L</td>
      <td>1737</td>
      <td>50</td>
      <td>0.000900</td>
    </tr>
    <tr>
      <td>37</td>
      <td>367</td>
      <td>V</td>
      <td>L</td>
      <td>1706</td>
      <td>35</td>
      <td>0.000884</td>
    </tr>
    <tr>
      <td>81</td>
      <td>411</td>
      <td>A</td>
      <td>S</td>
      <td>1680</td>
      <td>37</td>
      <td>0.000870</td>
    </tr>
    <tr>
      <td>149</td>
      <td>479</td>
      <td>P</td>
      <td>S</td>
      <td>1527</td>
      <td>51</td>
      <td>0.000791</td>
    </tr>
    <tr>
      <td>153</td>
      <td>483</td>
      <td>V</td>
      <td>F</td>
      <td>1465</td>
      <td>52</td>
      <td>0.000759</td>
    </tr>
    <tr>
      <td>192</td>
      <td>522</td>
      <td>A</td>
      <td>S</td>
      <td>1167</td>
      <td>45</td>
      <td>0.000604</td>
    </tr>
    <tr>
      <td>78</td>
      <td>408</td>
      <td>R</td>
      <td>I</td>
      <td>985</td>
      <td>31</td>
      <td>0.000510</td>
    </tr>
    <tr>
      <td>164</td>
      <td>494</td>
      <td>S</td>
      <td>L</td>
      <td>893</td>
      <td>49</td>
      <td>0.000462</td>
    </tr>
    <tr>
      <td>190</td>
      <td>520</td>
      <td>A</td>
      <td>S</td>
      <td>834</td>
      <td>40</td>
      <td>0.000432</td>
    </tr>
    <tr>
      <td>129</td>
      <td>459</td>
      <td>S</td>
      <td>F</td>
      <td>789</td>
      <td>21</td>
      <td>0.000409</td>
    </tr>
  </tbody>
</table>


Plot how many mutations are observed how many times:


```python
p = (ggplot(muts_df) +
     aes('count') +
     geom_histogram(bins=20) +
     scale_x_log10() +
     ylab('number of sequences') +
     xlab('times mutation observed')
     )

_ = p.draw()
```


    
![png](gisaid_rbd_mutations_files/gisaid_rbd_mutations_45_0.png)
    


Write the mutation counts to a file:


```python
print(f"Writing mutation counts to {config['gisaid_mutations_dir']}")
muts_df.to_csv(config['gisaid_mutation_counts'], index=False)
```

    Writing mutation counts to results/GISAID_mutations

