NEMATODE: Light-weight transformer NMT
-------

Nematode is a light-weight neural machine translation (NMT) toolkit built around the allmighty 'Transformer' model [1]. As the name suggests, it is based on the Nematus project maintained by Rico Sennrich, Philip Williams, and others [2] ( https://github.com/EdinburghNLP/nematus ), from which it borrows several of its components, extending and modifying them as necessary. 

Why make another machine translation toolkit? Nematode was built with readability and modifiability in mind (to help the maintainer better understand the inner workings of neural translation systems), and seeks to provide researchers with an easy to extend sandbox centered around a powerful state-of-the-art model. In this way, we hope to contribute our small part towards facilitatiing interesting research. It is implemented in TensorFlow and supports useful features such as dynamic batching, multi-GPU training, and gradient aggregation, which allow for replication of experiments originaly conducted on a large number of GPUs on a limited computational budget. 

We also would like to thank the authors of the Tensor2Tensor [3] ( https://github.com/tensorflow/tensor2tensor ) and OpenNMT-py [4] ( https://github.com/OpenNMT/OpenNMT-py ) libraries for the valuable insights offered by their respective model implementations.


Installation
------------

Nematode requires the following dependencies to be satisfied:

 - Python >= 3.6
 - tensorflow
 - CUDA >= 9.0

To install tensorflow, we recommend following the steps at:
  ( https://www.tensorflow.org/install/ )


Training speed
--------------

On an Nvidia GeForce GTX Titan X (Pascal) GPU with CUDA 9.0, our transformer implementation reaches the following speeds:

~4096 tokens per batch, no gradient aggregation, single GPU (effective batch size = ~4096 tokens):
>> 4123.86 tokens/ sec

~4096 tokens per batch, gradient aggregation over 2 update steps, 3 GPUs (effective batch size = ~25k tokens):
>> 16336.97 tokens/ sec


Performance
--------------
Following the training regime described in [1], our base-transformer implementation achieves 27.45 BLEU on the WMT 2014 English-to-German task after 148k update steps, which is comparable to the 27.3 BLEU reported in [1] after 100k updates. We used newstest2014 for validation.


Usage Instructions
------------------

To train a transformer model, modify the provided example training script - `example_training_script.sh` - as required.

#### `nematode/nmt.py` : trains a new model

#### data set parameters
| parameter            | description |
|---                   |--- |
| --source_dataset PATH | parallel training corpus (source) |
| --target_dataset PATH | parallel training corpus (target) |
| --dictionaries PATH [PATH ...] | model vocabularies (source & target) |
| --max_vocab_source INT | maximum length of the source vocabulary; unlimited by default (default: -1) |
| --max_vocab_target INT | maximum length of the target vocabulary; unlimited by default (default: -1) |

#### network parameters
| parameter            | description |
|---                       |--- |
| --model_name MODEL_NAME | model file name (default: nematode_model) |
| --model_type {transformer} | type of the model to be trained / used for inference (default: transformer) |
| --embedding_size INT | embedding layer size (default: 512) |
| --num_encoder_layers INT | number of encoder layers (default: 6) |
| --num_decoder_layers INT | number of decoder layers (default: 6) |
| --ffn_hidden_size INT | inner dimensionality of feed-forward sub-layers in FAN models (default: 2048) |
| --hidden_size INT | dimensionality of the model's hidden representations (default: 512) |
| --num_heads INT | number of attention heads used in multi-head attention (default: 8) |
| --untie_decoder_embeddings | untie the decoder embedding matrix from the output projection matrix |
| --untie_enc_dec_embeddings | untie the encoder embedding matrix from the embedding and projection matrices in the decoder |

#### training parameters
| parameter            | description |
|---                   |--- |
| --max_len INT | maximum sequence length for training and validation (default: 100) |
| --token_batch_size INT | mini-batch size in tokens; set to 0 to use sentence-level batch size (default: 4096) |
| --sentence_batch_size INT | mini-batch size in sentences (default: 64) |
| --maxibatch_size INT | maxi-batch size (number of mini-batches sorted by length) (default: 20) |
| --max_epochs INT | maximum number of training epochs (default: 100)
| --max_updates INT | maximum number of updates (default: 1000000)
| --warmup_steps INT | number of initial updates during which the learning rate is increased linearly during learning rate scheduling (default: 8000) |
| --learning_rate FLOAT | initial learning rate (default: 0.0002) |
| --adam_beta1 FLOAT | exponential decay rate of the mean estimate (default: 0.9) |
| --adam_beta2 FLOAT | exponential decay rate of the variance estimate (default: 0.98) |
| --adam_epsilon FLOAT | prevents division-by-zero (default: 1e-09) |
| --dropout_embeddings FLOAT | dropout applied to sums of word embeddings and positional encodings (default: 0.1) |
| --dropout_residual FLOAT | dropout applied to residual connections (default: 0.1) |
| --dropout_relu FLOAT | dropout applied to the internal activation of the feed-forward sub-layers (default: 0.1) |
| --dropout_attn FLOAT | dropout applied to attention weights (default: 0.1) |
| --label_smoothing_discount FLOAT | discount factor for regularization via label smoothing (default: 0.1) |
| --grad_norm_threshold FLOAT | gradient clipping threshold - may improve training stability (default: 0.0) |
| --teacher_forcing_off | disable teacher-forcing during model training (DOES NOTHING FOR NOW) |
| --scheduled_sampling | enable scheduled sampling to mitigate exposure bias during model training (DOES NOTHING FOR NOW) |
| --save_freq INT | save frequency (default: 5000) |
| --save_to PATH | model checkpoint location (default: model) |
| --reload PATH | load existing model from this path; set to 'latest_checkpoint' to reload the latest checkpoint found in the --save_to directory |
| --max_checkpoints INT | number of checkpoints to keep (default: 10) |
| --summary_dir PATH | directory for saving summaries (default: same as --save_to) |
| --summary_freq INT | summary writing frequency; 0 disables summaries (default: 100) |
| --num_gpus INT | number of GPUs to be used by the system; no GPUs are used by default (default: 0) |
| --log_file PATH | log file location (default: None) |
| --debug | enable the TF debugger |
| --gradient_delay INT | number of steps by which the optimizer updates are to be delayed; longer delays correspond to larger effective batch sizes (default: 0) |

#### validation parameters
| parameter            | description |
|---                   |--- |
| --valid_source_dataset PATH | source validation corpus (default: None) |
| --valid_target_dataset PATH | target validation corpus (default: None) |
| --valid_freq INT | validation frequency (default: 4000) |
| --patience INT | number of steps without validation-loss improvement required for early stopping; disabled by default (default: -1) |
| --validate_only | perform external validation with a pre-trained model |
| --bleu_script PATH | path to the external validation script (default: None); receives path of translation source file; must write a single score to STDOUT. |

#### display parameters
| parameter            | description |
|---                   |--- |
| --disp_freq INT | training metrics display frequency (default: 100) |
|  --greedy_freq INT | greedy sampling frequency (default: 1000) |
|  --sample_freq INT | weighted sampling frequency; disabled by default (default: 0) |
|  --beam_freq INT | beam search sampling frequency (default: 10000) |
|  --beam_size INT | size of the decoding beam (default: 4) |

#### translation parameters
| parameter            | description |
|---                   |--- |
| --translate_only | translate a specified corpus using a pre-trained model |
| --translate_source_file PATH | corpus to be translated; must be pre-processed |
| --translate_target_file PATH | translation destination |
| --translate_with_beam_search | translate using beam search |
| --length_normalization_alpha FLOAT | adjusts the severity of length penalty during beam decoding (default: 0.6) |
| --no_normalize | disable length normalization |
| --full_beam | return all translation hypotheses within the beam |
| --translation_max_len INT | Maximum length of translation output sentence (default: 100) |


Using Nematode
------------

If you decide to use Nematode, please provide a link to this repository in a footnote. Thanks :) .


References
------------

[1] Vaswani, Ashish, et al. "Attention is all you need." Advances in Neural Information Processing Systems. 2017.  
[2] Sennrich, Rico, et al. "Nematus: a toolkit for neural machine translation." arXiv preprint arXiv:1703.04357 (2017).  
[3] Vaswani, Ashish, et al. "Tensor2tensor for neural machine translation." arXiv preprint arXiv:1803.07416 (2018).  
[4] Klein, Guillaume, et al. "Opennmt: Open-source toolkit for neural machine translation." arXiv preprint arXiv:1701.02810 (2017).  


TODO
------------

1. Extend the readme
