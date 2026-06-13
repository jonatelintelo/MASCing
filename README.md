# MASCing: Configurable Mixture-of-Experts Behavior via Activation Steering Masks

[Paper link](https://arxiv.org/abs/2604.27818)

---
## Overview

We split the MASCing framework in four seperate scripts. Each script has a version to run one of the two use cases presented in the paper.

Everything denoted with jailbreak refers to the multi-turn jailbreak defense scenario.
Everything denoted with adult_refusal refers to the adult-content generation scenario.

Step 1: [1_create_dataset_jailbreak.py](1_create_dataset_jailbreak.py) creates the dataset for which we will collect logits in step 2.

Step 2: [2_create_lstm_input_jailbreak.py](2_create_lstm_input_jailbreak.py) collects the logits of the input collected in step 1. Used for LSTM model training in step 3.

Step 3: [3_train_lstm_jailbreak.py](3_train_lstm_jailbreak.py) trains the LSTM model on the collected logits in step 2 and saves it for use in step 4.

Step 4: [4_create_mask_jailbreak.py](4_create_mask_jailbreak.py) creates the steering mask and runs evaluation on it.

---

## 📄 Citation

If you find this work helpful, please consider citing our work.

```bibtex
@misc{lintelo2026mascingconfigurablemixtureofexpertsbehavior,
      title={MASCing: Configurable Mixture-of-Experts Behavior via Activation Steering Masks}, 
      author={Jona te Lintelo and Lichao Wu and Marina Krček and Sengim Karayalçin and Stjepan Picek},
      year={2026},
      eprint={2604.27818},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2604.27818}, 
}
```
---

## 📬 Contact

For questions, please reach out to:
📧 [jona.telintelo@ru.nl](mailto:jona.telintelo@ru.nl)
