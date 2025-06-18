# Using of Machine Learning to modell material degratation for CMCs under hypersonic flow.

[**Dataset**](#dataset)
| [**Models**](#models)
| [**Colabs**](#colabs)
| [**License**](#license)
| [**Disclaimer**](#disclaimer)
| [**Upcoming**](#upcoming)
| [**Citing**](#citing)

### What is Materials Discovery: GNoME?

From microchips to batteries and photovoltaics, discovery of inorganic crystals
is a fundamental problem in materials science. Graph Networks for Materials
Science (GNoME) is a project centered around scaling machine learning methods
to tackle this core task. With results recently published, this repository
serves to share the discovery of 381,000 novel stable materials with the wider
materials science community and hopefully enable exciting new research via the
updated convex hull.

This is a research project, not an official Google product. Expect bugs as the
repository expands and sharp edges. Please help by exploring the structures
and let us know what you think!

### Contents
* [**Dataset**](#dataset)
* [**Models**](#models)
* [**Colabs**](#colabs)
* [**License**](#license)
* [**Disclaimer**](#disclaimer)
* [**Upcoming**](#upcoming)
* [**Citing**](#citing)

### Dataset

The dataset described in the original paper is provided across multiple file
formats. For more details, including how to download the dataset, please see
our dataset descriptor file in DATASET.md.

**Summarized** A summary of the dataset is provided in CSV format. This file
contains compositions and raw energies from Density Functional Theory (DFT)
calculations, as well as other popular measurements (e.g. formation energy and
decomposition energy). 
Data that is related to previous **Thermo-Fluid Filed** has been gathered using open source scrapers
and then re-generated with small variation using Gen AI in 2023. The files are included

**Structure** Loading of structures is slightly more cumbersome due to file
sizes involved. Due to the organization of the convex hull, only one structure
is needed per composition, so results from the summary can be used to pull
from the compressed data directory available in the linked Cloud Bucket. An
alternative approach to extract individual files from the compressed ZIP
(so as to only extract necessary files) is exemplified in the visualization
colab.

**r²SCAN** Baseline calculations were performed via PBE functional for the
calculations. The paper also reports metrics for binaries and tenaries with
the r²SCAN functional. A summary of calculated energies and associated
metrics is included for these calculations.

### Models

We provide model definitions for the two sets of models used in the paper.

**GNoME** were the predominant model behind new materials
discovery. This simple message passing architecture was optimized by training
on a snapshot of Materials Project from 2018, leading to state-of-the-art results
of 21meV/atom.

**Nequip** corresponds to the architecture created by Batzner et al. (2022).
This architecture was used to train the interatomic potentials described in the
paper to learn the dynamics from the large dataset. We provide an implementation
in JAX as well as basic configuration parameters for the corresponding
architecture.

### Colabs

Colab examples of how to interact with the dataset provide an interface for
exploring various chemical systems or computing decomposition energies.

### Disclaimer

This is not an official  product.


Data in the Graph Networks for Materials Exploration Database is for theoretical modeling only, caution should be exercised in its use. The Graph Networks for Materials Exploration Database is not  intended for, and is not approved for, any medical or clinical use.  The Graph Networks for Materials Exploration Database is experimental in nature and provided on an “as is” basis. To the maximum extent permitted at law, Google disclaims all representations, conditions and warranties, whether express or implied, in relation to the Graph Networks for Materials Exploration Database (including without limitation for non-infringement of third party intellectual property rights, satisfactory quality, merchantability or fitness for a particular purpose), and the user shall hold Google free and harmless in connection with their use of such content.

### Citing

If you are using this resource please cite our
[paper](https://pubs.aip.org/aip/apm/article/12/9/090601/3311142/Material-discovery-and-modeling-acceleration-via

```latex
  @article{merchant2023scaling,
    title={Scaling deep learning for materials discovery},
    author={Amil Merchant and Simon Batzner and Samuel S. Schoenholz and Muratahan Aykol and Gowoon Cheon and Ekin Dogus Cubuk},
    journal={Nature},
    year={2023},
    doi={10.1038/s41586-023-06735-9},
    href={https://www.nature.com/articles/s41586-023-06735-9},
}
```
