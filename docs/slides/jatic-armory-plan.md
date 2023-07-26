---
marp: true
theme: gaia
paginate: true
---

# JATIC Armory Plan — July 2023

* Increment 0 January-March 2023

  - bootstrapping to JATIC
  - initial planning
  - proof of concept for Armory as a Library
    - discarded Docker dependency
    - dispensed with Launcher/Engine split
    - replaced config.json with a typed Evaluation object
    - created Engine class as primary interface
    - added MLFlow experiment and results tracking

---

* Increment 1 April-June 2023

    - Adaptation to jatic_toolbox
      - accept datasets from jatic_toolbox
    - Preliminary integration with Kitware XAI toolkit
      - exploration of saliency map differences with adversarial attacks
    - Adding staff for Increment 2
      - 2 developers
      - 1 intern
      - 1 principal investigator

---

* Increment 2 July–September 2023

    - accept models from jatic_toolbox
    - Convert lazy evaluation of Evaluation object to eager instantiation
    - Initial development of native Armory attacks
    - Prototyping of Presentation interface
      - create view of what the Evaluation is doing
    - Exploration and planning with other JATIC performers
        - Kitware, MITRE, IBM, MetroStar, others
    - Exploration of auto-attack mechanisms

---

* Increment 3 October–December 2023

   - Release of Armory 23.12 to Open Source (replaces 0.18)
     - release of Armory presentation application
   - Further development of native Armory attacks
   - Adaptation of new IBM Adversarial Robustness Toolbox (ART) mechanisms
   - Development of auto-attack mechanisms
   - Initial development of native Armory defenses
   - Evaluation of other JATIC performers' models
   - Initial deployment to DoD operational mission systems

---

* Increment 4 January–March 2024

   - Additional adaptation to customer needs
   - Other investigation, development, and integration as needed

---

# Increment 2 — Internals

* Direct loading of components at initialization
  - Dataset, Model, etc. classes have been descriptors of their
    lazy-loaded instantiations because prior Loader/Engine split
    required it. We can now instantiate them directly.
* Pre-training via fit_model need not be buried in evaluation
  - separate Engine.train() from Engine.evaluate()
  - prefer pre-trained weights to per-evaluation training to reduce time penalty for
    ever-larger models and datasets
  - on-demand training should be retained in case JATIC requires poisoning attacks

---

# Increment 2 – Application

* Subcontractor DataMachines (2 FTE) to develop Armory application
* Create a visual explorer to show how evaluation runs
  - show the data, model, attack, defense, results, and metrics
  - use meaningful representations: e.g. show dataset elements as images
* Open questions:
  - should this work with from Armory direct or MLflow tracking?
  -
