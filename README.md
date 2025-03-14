### CODE for Hate Speech Detection with Generalizable Target-aware Fairness

**Main Files:**

1. **model_training.py** : main training file.

2. **models.py** : includes all models implemented/used.

3. **parameter_config.py**: key parameters to be configured.

4. **utils.py**: helper functions for data processing, model evaluation etc.

5. **dataset**: contains all relevant dataset.

**Usage**:

1. Download the GloVe embedding (i.e 300 dim version) into the dataset folder.

2. Edit key parameters in the **parameter_config.py** file or remain default. See the file for more descriptions of specific parameters.

3. Directly run **python model_training.py** or **CUBLAS_WORKSPACE_CONFIG=:4096:8 python model_training.py** to ensure determinism.

**Notes**: 

1. Current code was implement to adopt multi-GPU for training.

2. Current code was written with ease to modify and test, further maintenance may be applied to increase usability.

3. For implementation efficiency and encourage GetFair as a plug-and-play method, the embedding model was only finetuned during the initial warm-up stage along with a classification layer. However, the embedding model is also capable of being finetuned during the adversarial training of GetFair.

**Environment：**

1. python == 3.8

2. torch == 1.7.1

3. transformers == 3.0.2 

4. scipy == 1.10.1
