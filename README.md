# Reward-Poisoning Attacks of Federated Reinforcement Learning in Atari

This project was developed as part of a spring 2021 semester project for the course ECE 5984 Trustworthy Machine Learning at Virginia Tech.

## Contributors

- [Rajesh Kudupudi](https://www.linkedin.com/in/rajesh-kudupudi/)
- [Lowell Weissman](https://www.linkedin.com/in/lowell-weissman/)


## Abstract
Despite its growing relevance in safety-critical sequential  decision  AI  applications, relatively  little  research  on adversarial attacks has focused on reinforcement learning. And to the best of the authors’ knowledge, no work investigates the intersection of federated learning, reinforcement learning, and security, even though federated learning offers several benefits when learning privacy-sensitive policies.  We study deep targeted reward poisoning, which provides an attack  model  that  is  meaningful,  powerful,  and scalable when used in this setting. Existing reward poisoning attacks are either untargeted or targeted only for tabular state-action spaces, avoiding the issue of target detection in very large state spaces.  We introduce a contrastive method for discerning target observations in high-dimensional state spaces.   We  then  adapt  an  existing  targeted  reward  poisoning attack for these complex environments and demonstrate its effectiveness on pixel-based Atari Pong agents. We also propose the first reward poisoning attack suitable for deep federated settings, and provide a mostly complete implementation for this extension.

## Installation 

*Currently this source has only been tested on Unix-like operating systems.*

First install the conda env provided:

```
conda env create -f linux_cuda10.yml
```

If using a different version of CUDA, or no hardware acceleration, you can modify or remove the pytorch and cudatoolkit dependencies before installing, or reinstall the correct versions for your system. 

Next install rlpyt as pip package (method B) by following installation instructions here [https://github.com/astooke/rlpyt](https://github.com/astooke/rlpyt).

Finally, install this project, also as an editable pip package:

```
cd PATH/TO/reward-poisoned-drl
pip install -e .
```

## Code Structure

* **reward_poisoned_drl** - Contains primary modules.
  * **attack** - All reward poisoning modules.
    * **fixed** - Fixed, non-adaptive reward poisoning modules.
      * **adversary** - Attacker algorithm.
      * **run_attack** - Train a DQN agent under the fixed attacker's reward poisoning.
      * **runner** - rlpyt runner subclass to support logging attacker metrics.
    * **learned** - `JUST A STUB` Learned, adaptive reward poisoning modules based on Appendix C of the paper. 
  * **contrastive_encoder** - Training modules for the contrastive observation encoder, largely adapted from [CURL](https://github.com/MishaLaskin/curl).
    * **contrast** - Training and deployment classes for the contrastive encoder.
    * **data_generator** - Generates self-supervised instance discrimination examples from saved replay observations.
    * **encoder** - Neural network module for encoder.
    * **train_contrastive_encoder** - Launch contrastive encoder training.
    * **utils** - Observation-encoder-specific utils, e.g. for generating similarity scores.
  * **federated** - `BUGGED` Training modules for a locally simulated server-client DRL software architecture.
    * **client** -  All client code to support federated deep RL.
      * **asa_factories** - "Agent-Sampler-Algorithm" factory classes.
      * **base** - Client API declarations and algorithm extensions.
      * **parallel** - Client where gradient computation happens on a subprocess.
      * **serial** - Client where gradient computation happens on the main process.
    * **run_attack** - Train a global agent using clean and/or malicious clients.
    * **server** - All server code to support federatd deep RL.
  * **replay_store** - Training modules for the attacker DQN oracle which also stores replay buffers to disk to create observation encoder data.
    * **dqn_store** - DQN algorithm modifications.
    * **replay_store** - Replay algorithm modifications.
    * **train_dqn_store** - Train a clean, vanilla DQN while storing replay data to disk upon filling. 
  * **logger** - Lightweight logger with viskit compatible format (what rlpyt uses).
  * **utils** - Project-global utilities; mainly data augmentation and visualization helper functions.
* **tests** - Testing scripts.
* **tools** - Visualziation and data handling scripts.  


## Contrastive Observation Encoder Demos

Our learned similarity metric for descerning visual target observations. 

Left: A rolling feed of the reference target observation for visualization purposes. \
Right: An episode of observations loaded from replay.

Blue annotation: Frame number since start of episode. \
Green/red annotation: Perceived similarity between the currently displayed observation and the target reference. Green is positive, red is negative. A higher green value suggests similarity, and a higher red value suggests dissimilarity. 

Each observation is a stack of the last four grayscale frames. For simplicity, we synchronize similarity scores with the newest frame.

**Target Observation 1: "Bottom"** \
Expert action - UP \
Target action - DOWN

https://user-images.githubusercontent.com/42881205/117767355-d2b7c300-b1fe-11eb-80d0-4e48e31f08d2.mp4

https://user-images.githubusercontent.com/42881205/117768582-8ec5bd80-b200-11eb-9840-902b0737b315.mp4

https://user-images.githubusercontent.com/42881205/117768592-908f8100-b200-11eb-9cd3-111a6c5f0668.mp4

---
**Target Observation 2: "Mid"** \
Expert action - UP \
Target action - DOWN

https://user-images.githubusercontent.com/42881205/117768827-ecf2a080-b200-11eb-9a5b-2df89b165396.mp4

https://user-images.githubusercontent.com/42881205/117768833-eebc6400-b200-11eb-8ca7-ca63577ee84d.mp4

https://user-images.githubusercontent.com/42881205/117768838-f0862780-b200-11eb-9019-8593762543e5.mp4


## Poisoned Policy Demos


