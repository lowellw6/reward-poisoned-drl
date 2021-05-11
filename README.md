# Reward-Poisoning Attacks of Federated Reinforcement Learning in Atari

This project was developed as part of a spring 2021 semester project for the course ECE 5984 Trustworthy Machine Learning at Virginia Tech. For more discussion of motivation, method, and results, see the associated [paper](https://github.com/lowellw6/reward-poisoned-drl/files/6457316/Kudupudi_Weissman_RP_Attacks_Federated_RL.pdf).


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

Left video: A rolling feed of the reference target observation for visualization purposes. \
Right video: An episode of observations loaded from replay.

Bottom-left blue annotation: Frame number since start of episode. \
Bottom-right green/red annotation: Perceived similarity between the currently displayed observation and the target reference. Green is positive, red is negative. A higher green value suggests similarity, and a higher red value suggests dissimilarity. 

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

---
## Poisoned Policy Demos

DQN agents trained under our proposed deep targeted reward poisoning algorithm.

Videos show example trajectories along with agent action and Q-value annotations, described below.

Bottom-left blue annotation: Frame number since start of episode. \
Bottom-center red annotation: Next human-understandable action the poisoned agent intends to take. \
Right-side green/red annotations: Q-value associated with the target action associated with the target observation at that location. Green is positive, red is negative.

The upper Q-value corresponds to target "Mid," indicating how beneficial the agent believes DOWN to be. \
Th lower Q-value corresponds to target "Bottom," also indicating how beneficial the agent believes DOWN to be. \
Note this implementation of Atari Pong's action indexing does not produce unique actions, so both target actions are DOWN while the Q-values differ from targeting distinct action indices.

When the agent is semantically close to a target observation, a successful attack results in the corresponding target Q-value spiking, resulting in the agent taking that target action. The two slowed videos below demonstrate this behavior on each target observation.

https://user-images.githubusercontent.com/42881205/117779109-5a57fe80-b20c-11eb-95fe-2836e8107a35.mp4

https://user-images.githubusercontent.com/42881205/117779133-5fb54900-b20c-11eb-9b1a-df4fbfb590ae.mp4

---

Complete episode examples for DQN agents **poisoned from the beginning of training**, with attack delta-bounds 0.25, 0.5, 1.0, and 2.0 for the weakest to strongest adversary, respectively.

https://user-images.githubusercontent.com/42881205/117779809-08fc3f00-b20d-11eb-8de0-ca1928f55d82.mp4

https://user-images.githubusercontent.com/42881205/117779824-0d285c80-b20d-11eb-9a38-e24679cc1d52.mp4

https://user-images.githubusercontent.com/42881205/117779854-11ed1080-b20d-11eb-9f11-d8d58c1c5df9.mp4

https://user-images.githubusercontent.com/42881205/117779873-14e80100-b20d-11eb-8b87-7952b415de6a.mp4

---

Complete episode examples for DQN agents **poisoned after 10 million environment steps during training**, with attack delta-bounds 0.25, 0.5, 1.0, and 2.0 for the weakest to strongest adversary, respectively. By 10 million steps, clean DQN training reliably converges to an expert policy. So the attacker also needs to overpower any genuine Q-value knowledge learned by the agent that disrupts malicious targeting.

https://user-images.githubusercontent.com/42881205/117781000-2978c900-b20e-11eb-8235-31d880f4642d.mp4

https://user-images.githubusercontent.com/42881205/117781017-2da4e680-b20e-11eb-8e0e-554e856cd11d.mp4

https://user-images.githubusercontent.com/42881205/117781030-30074080-b20e-11eb-8183-a885554b3306.mp4

https://user-images.githubusercontent.com/42881205/117781042-32699a80-b20e-11eb-8284-78a9cdc43752.mp4
