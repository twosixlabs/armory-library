---
marp: true
paginate: true
---

# TwoSix JATIC Increment 1 Review - 27 June 2023


## Work Completed

- Armory now loads datasets via the `jatic_toolbox` protocol
  - HuggingFace and TorchVision origins supported
  - off-the-shelf datasets can now be evaluated with Armory
- Added Armory dataset exporter for Kitware collaboration
  - Patch attacks and Imperceptible attacks used with XAI toolkit
- Extensive, internal modifications to Armory for extensibility and maintainability

---

## Demonstrations

- xView: satellite imagery object detection (Lam, _et al._ 2018)
  - demonstration in [xview_frcnn_sweep_patch_size.ipynb](armory/examples/notebooks/xview_frcnn_sweep_patch_size.ipynb)
- JATIC datasets:
  - demonstrations in [jatic_hf_example.py](charmory/examples/jatic_hf_example.py) and
    [jatic_tv_example.py](charmory/examples/jatic_tv_example.py)

---

## Collaboration

Worked with Kitware to explore XAI saliency difference between benign and adversarial
images. We have provided access to well tested CARLA & DAPRICOT patch attack
datasets and are working with Kitware to add imperceptible evasion attacks to their
analysis.

---

## Personnel and Contract

The TwoSix JATIC team has expanded by three members since the last review:

  - Etienne Deprit, Principal Investigator
  - Kyle Treubig, Software Engineer
  - Rahul Narayanan, Software Intern

have joined the team of

  - Matt Wartell, Software Engineer
  - Christopher Woodall, Software Engineer

and Christopher Honaker, software engineer will be joining the team in July.

The CDAO / TwoSix Contract has been executed, with formal work begun on 19-June-2023.
