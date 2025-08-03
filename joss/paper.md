---
title: "Jim"
tags:
  - Python
  - Gravitational Waves
  - Parameter estimation
  - Monte Carlo
  - Machine Learning
  - Jax
authors:
  - name: Kaze W. K. Wong
    orcid: 0000-0001-8432-7788
    affiliation: 1
  - name: Thomas C. K. Ng
    affiliation: 2
  - name: Peter T. H. Pang
    orcid: 0000-0001-7041-3239
    affiliation: 3,4
  - name: Samson H.W. Leong
    affiliation: 2
  - name: T. Wouters
    orcid: 0009-0006-2797-3808
    affiliation: 3,4
  - name: Charmaine Z. C. Wong
    affiliation: 2
  - name: Maximiliano Isi
    orcid: 0000-0001-8830-8672
    affiliation: 5

affiliations:
  - name: Data Science and AI institute, Johns Hopkins University, Baltimore, MD 21218, US
    index: 1
  - name: The Chinese University of Hong Kong, Shatin, N.T., Hong Kong
    index: 2
  - name: Nikhef, Science Park 105, 1098 XG Amsterdam, The Netherlands
    index: 3
  - name: Institute for Gravitational and Subatomic Physics (GRASP), Utrecht University, Princetonplein 1, 3584 CC Utrecht, The Netherlands
    index: 4
  - name: Center for Computational Astrophysics, Flatiron Institute, New York, NY 10010, US
    index: 5


date: 23 June 2025
bibliography: paper.bib
---

# Summary

Parameter estimation (PE) is the primary tool for extracting astrophysical
information from gravitational wave signals. Since the first detection of gravitational waves in 2015, the number of detections has been increasing rapidly, with more than 100 events detected by the end of 2023. This has led to a growing need for efficient and flexible parameter estimation codes that can handle the increasing volume of data and the complexity of the signals.

In this paper, we present a production-ready version of the parameter estimation code `jim`. This means on top of fast inference, `jim` aims to deliever a flexible and extensible framework that can be used by the broader community to pursue PE studies for GW signals. At the same time, `jim` provides robust and reproducible infrastructures that can be used for production-level parameter estimation tasks.

## Key features

- **GPU and Machine learning accelerated sampling**

`jim` is built on top of `jax` and `flowMC`, which enable GPU-accelerated sampling with machine learning-based proposal distribution. This means `jim` does not rely on specific techniques such as reparamterization or surrogate models to achieve fast inference, but instead uses a general-purpose framework that can be applied to a wide range of PE tasks.

- **Composable transformations**

While `jim` primarily leverages accelerators and machine learning to achieve fast inference, it can still benefit from efficient reparameterization. `jim` provides a composable transform system that allows the users to chain together multiple transformations to achieve efficient reparameterization. This means users can still use their existing knowledge of reparameterization techniques to improve the performance of their PE tasks.

- **Two level of APIs**

`jim` provides a low-level API that grants the user full control over the definition of the model and the sampling process, as well as a high-level API that provides a more structured interface for large scale PE studies with reproducibility in mind.

- **Cloud-ready deployment**

We provide official guides to deploy `jim` on cloud platforms, such that researcher groups that do not have access to high-performance computing clusters can still leverage commerically available cloud resources to run large-scale parameter estimation tasks.


# Statement of need

In the last couple of years, multiple groups have been developing faster parameter estimation codes [@Wong:2023lgb] [@Roulet:2022kot] [@Dax:2021tsq] [@Ashton:2018jfp] to tackle the pressure of the rapidly increasing number of gravitational wave detections. 
While the development of these codes has been impressive regarding their performance, adoption outside of the core development team remains limited.


To make sure the broader community can benefit from these 

<!-- Need for speed -->

<!-- Need for flexibility -->

<!-- Need for production ready infrastructure -->


# Acknowledgements

We thank Will Farr, Keefe Mitman, and Colm Talbot for their
contributions to the JaxNRSur codebase and discussions that improved the
package.

# References
