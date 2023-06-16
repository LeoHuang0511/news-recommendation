import os

model_name = os.environ['MODEL_NAME'] if 'MODEL_NAME' in os.environ else 'NRMS'
# lr = os.environ['LR'] if 'LR' in os.environ else 1e-4

# Currently included model
assert model_name in [
    'NRMS',
    'NAML',
    'LSTUR',
    'DKN',
    'HiFiArk',
    'TANR',
    'Exp1'
]

USING_CUDA_DEVICE: int = os.environ['CUDA'] if 'CUDA' in os.environ else 2
# USING_CUDA_DEVICE: int = 3

class BaseConfig():
    """
    General configurations appiled to all models
    """

    num_epochs = 40
    num_batches_show_loss = 100  # Number of batchs to show loss
    # Number of batchs to check metrics on validation dataset
    num_batches_validate = 1000
    batch_size = 768
    learning_rate = float(os.environ['LR'] if 'LR' in os.environ else 1e-4)
    warmup_epoch = 1
    min_learning_rate = 1e-6 
    num_workers = 24  # Number of workers for data loading
    num_clicked_news_a_user = 50  # Number of sampled click history for each user
    # num_words_title = 20
    num_words_title = 30

    num_words_abstract = 50
    # num_words_abstract = 30

    word_freq_threshold = 1
    entity_freq_threshold = 2
    entity_confidence_threshold = 0.5
    negative_sampling_ratio = 4  # K
    dropout_probability = 0.2

    # Modify the following by the output of `src/dataprocess.py`
    # num_words = 1 + 70975
    # num_categories = 1 + 274
    # num_entities = 1 + 12957
    # num_users = 1 + 50000
    num_words = 1 + 101249
    num_categories = 1 + 295
    num_entities = 1 + 21842
    num_users = 1 + 196688

    word_embedding_dim = 300
    category_embedding_dim = 100

    # Modify the following only if you use another dataset
    entity_embedding_dim = 100
    # For additive attention
    query_vector_dim = 200


class NRMSConfig(BaseConfig):
    dataset_attributes = {"news": ['title'], "record": []}
    # For multi-head self-attention
    num_attention_heads = 15


class NAMLConfig(BaseConfig):
    dataset_attributes = {
        "news": ['category', 'subcategory', 'title', 'abstract'],
        "record": []
    }
    # For CNN
    num_filters = 400
    window_size = 3


class LSTURConfig(BaseConfig):
    dataset_attributes = {
        "news": ['category', 'subcategory', 'title'],
        "record": ['user', 'clicked_news_length']
    }
    # For CNN
    num_filters = 300
    window_size = 3
    long_short_term_method = 'ini'
    # See paper for more detail
    assert long_short_term_method in ['ini', 'con']
    masking_probability = 0.5


class DKNConfig(BaseConfig):
    dataset_attributes = {"news": ['title', 'title_entities'], "record": []}
    # For CNN
    num_filters = 50
    window_sizes = [2, 3, 4]
    # TODO: currently context is not available
    use_context = False


class HiFiArkConfig(BaseConfig):
    dataset_attributes = {"news": ['title'], "record": []}
    # For CNN
    num_filters = 300
    window_size = 3
    num_pooling_heads = 5
    regularizer_loss_weight = 0.1


class TANRConfig(BaseConfig):
    dataset_attributes = {"news": ['category', 'title'], "record": []}
    # For CNN
    num_filters = 300
    window_size = 3
    topic_classification_loss_weight = 0.1


class Exp1Config(BaseConfig):
    dataset_attributes = {
        # TODO ['category', 'subcategory', 'title', 'abstract'],
        "news": ['category', 'subcategory', 'title'],
        "record": []
    }
    # For multi-head self-attention
    num_attention_heads = 15
    ensemble_factor = 1  # Not use ensemble since it's too expensive
