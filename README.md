# Using of Machine Learning to modell material degratation for Ultra-High Temperature Ceramics under hypersonic flow.

[**Dataset**](#dataset)
| [**Models**](#models)
| [**Colabs**](#colabs)
| [**License**](#license)
| [**Disclaimer**](#disclaimer)
| [**Upcoming**](#upcoming)
| [**Citing**](#citing)

### Predicting Material Propreties

The code that can be found on this repository 
supports research on the use of machine learning (ML) 
and predictive analytics to model and simulate 
the degradation of Ultra-High Temperature Ceramics (UHTCs) 
under hypersonic flow conditions. 
The study aims to improve the accuracy 
and efficiency of degradation modelling to aid design a
nd enhance the performance and reliability of these components.

ML can be used to predict material 
properties such as electronic, magnetic, 
and mechanical properties 
based on the chemical composition, 
crystal structure, and processing conditions. 
This can be achieved by training a model on a labeled dataset 
that includes experimental or simulated data. 
For example, 
ML models have been used to predict the bandgap of materials, 
which is a critical property for applications in electronics and optoelectronics. 
Models based upon these methodologies 
have also been used to predict the mechanical properties of materials,
such as the Young's modulus and Poisson's ratio, 
which are important for designing materials with specific mechanical properties.



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
our dataset descriptor file in DATASET.md. in the dataset folder

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

The code is present in each of the 3 folders

**Model** were the predominant model behind new materials
discovery. This simple message passing architecture was optimized by training
on a snapshot of Materials Project from 2018 and has been developed using various
pythong libraries. The model is used for dataset generation
overall training data and test data.

**HypersonicCode** corresponds to code that can generate the thermofluid analysis
using Direct MonteCarlo Simulation

**Surrogate Models** Code that is used predominaly for chapter 4

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
