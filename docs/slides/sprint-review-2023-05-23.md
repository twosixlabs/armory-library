---
marp: true
paginate: true
---

# TwoSix JATIC Sprint Review - 23rd May 2023

## Shifting Focus

* Armory datasets adaptation discontinued
* Shift towards flexible dataset adapters (e.g. jatic_toolbox protocol, huggingface API)
* Sprint extension to 6th June 2023 proposed

Our journey this week led to the realization that attempts to wrap Armory's internal
dataset representation around the JATIC protocol were misdirected. Given the fact that
Armory `Datasets` was designed primarily to facilitate a well-entrenched, elementary
machine learning problems (like MNIST), the effort to modify both protocols for mutual
compatibility was ultimately unfruitful, inadvertently propagating needless complexity.

In light of this we have shifted focus towards integrating common dataset protocols
directly into the core of Armory as adapters. This strategic readjustment will extend
the current Sprint deadline to around 6th June.

---

## Ongoing Collaboration

We are working on adapters/connectors for other dataset types; such as the
`jatic_toolbox` protocol and `huggingface` API. This will make Armory a more versatile
platform; rather than a harness for GARD DARPA experiments. A side effect of this
integration is that these efforts will allow Armory to offer more flexible ways to load
and serve model backbones and checkpoints.

Collaboration with key partners, especially Kitware, remains vigorous. Our recent
joint effort to solve problems in a shared notebook illustrates our productive synergy.

---

## Documentation

We have added more comprehensive onboarding documentation. While still in progress (as it always is), this guide will help developers directly working on the Armory project create a development environment.
