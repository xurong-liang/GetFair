"""
Model config file: Consists of model training parameters and files location.
"""
# Model training epochs/iterations: 
EMBEDDING_EPOCH = 4
GC_K = 1 # target classifiers iterations
HC_L = 3 # hate speech classifiers iterations
GC_WARM_UP = 3 # target classifiers warmup iterations
MODULE_EPOCH = 4
EARLY_STOPPING_COUNT = 4
EMB_STOP_COUNT = 2

# Define aggregation strategy for generating filtered embeddings: 
# Parameter Ensemble ("early fusion"), Combinatorial Embedding ("late fusion")
STRATEGY = "early_fusion" 
#Define learning rate for each model component and key hyperparameters
LEARNING_RATE = [2e-5, 1e-4, 1e-4, 2e-5]
HYPERPARAMETERS = {"lambda": 0.9, "gamma": 3, "mu": 0.9, "rank_k": 10}

# Define the random seed, batch size, dataset, and embedding model to use.
SEED = 123
BATCH_SIZE = 128
DATA = "V1" # 'V2'
DATASET = "MHS" # "Jigsaw"
LOAD_EMB = False
EMB_MODEL = "gpt2" # 'roberta', 'bert'
EMB_MODEL_PATH = 'distilgpt2' # 'distilroberta-base'
EMB_CONFIG_PATH = None # Optional Config file
MAX_SEQUENCE_LENGTH = 170 # max sequence length for tokenizers

# Define the various file paths for the Pre-trained GloVe vector, model output file location, and saved model paths
TOPIC_EMBEDDINGS_PATH = ["./dataset/vocab_npa_6B_300d.npy", "./dataset/embs_npa_6B_300d.npy"]
TOPIC_EMBEDDINGS_SHAPE = 300 # Length of the word embeddings (i.e. GloVe vectors)

FILENAME = f"./Training_output.txt"
SAVE_EMBEDDING_MODEL = f"./saved_embedding_model.pth"
SAVE_HATESPEECH_MODEL = f"saved_hate_speech_model.pth"
FINE_TUNED_BERT = f"./saved_embedding_model.pth"

# Using Multiple GPUs for nn.DataParallel
DEVICE = 'cuda:0'
HYPERNET_DEVICE = 'cuda:0'
MODEL_DEVICE_IDS = [[0, 1], [0, 1], [0, 1], [0, 1]]

# Define the identity attributes and target columns in the Jigsaw dataset.
if DATASET == "Jigsaw":
    TARGET = "target"
    TEXT_FIELD = "comment_text" 
   
    IDENTITY_COLUMNS = [
        'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
        'black', 'psychiatric_or_mental_illness'
        ]
    TEST_UNSEEN_IDENTITY = ['muslim','white']

    if DATA == "V2":
        IDENTITY_COLUMNS = [
            'male', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
            'muslim', 'white', 'psychiatric_or_mental_illness'
            ]
        TEST_UNSEEN_IDENTITY = ['female','black']

else: # MHS
    TARGET = "hate_speech_score"
    TEXT_FIELD = "text"
    IDENTITY_COLUMNS = [
        'target_race', 'target_gender','target_religion',
        'target_age', 'target_disability'
        ]
    TEST_UNSEEN_IDENTITY = ['target_origin', 'target_sexuality']

    if DATA == "V2":
        IDENTITY_COLUMNS = [
            'target_race', 'target_origin', 'target_gender','target_religion',
            'target_sexuality'
            ]
        TEST_UNSEEN_IDENTITY = ['target_age', 'target_disability']
        
Y_COLUMNS =[TARGET] + IDENTITY_COLUMNS
NUM_TARGET_GROUPS = len(IDENTITY_COLUMNS)

# Define hidden embedding layer, group classifer(discriminator) layer, and hate_speech classifier layer shapes.
# The hidden dim shape will need to match embedding model.
HIDDEN_DIM = 768
filter_layers = [768]
DISCRIMINATOR_LAYERS = []
HATE_CLASSIFIER_LAYERS = []
HYPERNERT_LAYERS = [128]

FILTER_LAYER_SHAPE = [
    [(NUM_TARGET_GROUPS, HIDDEN_DIM, filter_layers[-1]), (NUM_TARGET_GROUPS, filter_layers[-1])],
]

if DATA == "V1":
    DATA_PATHS = {"train": f"./dataset/{DATASET}_filtered_train_df.csv" , \
                "val": f"./dataset/{DATASET}_filtered_val_df.csv", \
                    "test": f"./dataset/{DATASET}_unseen_test_df.csv"}
else:
    DATA_PATHS = {"train": f"./dataset/{DATASET}_filtered_train_df_v2.csv" , \
                "val": f"./dataset/{DATASET}_filtered_val_df_v2.csv", \
                    "test": f"./dataset/{DATASET}_unseen_test_df_v2.csv"}

