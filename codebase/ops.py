import tensorflow as tf
from util import get_visible_gpus, assign_to_device, get_devices

# TODO: Currently, translation does not work with multiple-GPUs


def get_parallel_ops(model, iterator, num_gpus, eos_id, mode, no_summaries=False):
    """ Defines the training and validation OPs for multi-GPU training.
    Vaguely based on: http://blog.s-schoener.com/2017-12-15-parallel-tensorflow-intro/ . """

    def _get_train_ops():
        """ Defines multi-device OPs used to train the model. """
        # Track tower-wise outputs
        tower_grads_and_vars = list()
        tower_batch_losses = list()
        tower_sentence_losses = list()
        tower_words = list()
        tower_targets = list()

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as outer_scope:
            for gpu_id, gpu in enumerate(operators):
                try:
                    name = 'tower_{}'.format(gpu_id)
                    # Assign variables to the CPU and tensor OPs to GPUs
                    with tf.device(assign_to_device(controller, gpu)), tf.name_scope(name):
                        # Compute and store losses and gradients
                        next_batch = iterator.get_next()
                        grads_and_vars, _, batch_loss, sentence_loss, words_processed, words_evaluated = \
                            model.train_model(next_batch)
                        # Training OPs
                        tower_grads_and_vars.append(grads_and_vars)
                        tower_batch_losses.append(batch_loss)
                        tower_words.append(words_processed)
                        tower_sentence_losses.append(sentence_loss)
                        tower_targets.append(words_evaluated)

                    # Reuse variables
                    outer_scope.reuse_variables()

                except tf.errors.OutOfRangeError:
                    break

        if len(tower_grads_and_vars) == 0:
            raise tf.errors.OutOfRangeError

        # Weigh batch gradients based on the number of words contained within the batch
        max_tokens = tf.cast(tf.reduce_max(tower_targets), dtype=tf.int32)
        tower_weights = [tf.to_float(token_count / max_tokens) for token_count in tower_words]

        # Average grads
        averaged_grads_and_vars = list()
        for grads_and_vars in zip(*tower_grads_and_vars):
            grads = [grad for grad, _ in grads_and_vars]
            var = grads_and_vars[0][1]
            if type(grads[0]) != tf.IndexedSlices:
                # Apply tower weights
                grads = [grads[tower_id] * tower_weights[tower_id] for tower_id in range(len(grads))]
                averaged_grad = tf.reduce_mean(grads, 0)
            else:
                # Concatenate IndexedSlices (equivalent to averaging of tensors)
                # Apply tower weights
                values = [grads[tower_id].values * tower_weights[tower_id] for tower_id in range(len(grads))]
                joint_values = tf.concat(values, axis=0)
                joint_indices = tf.concat([grad.indices for grad in grads], axis=0)
                averaged_grad = \
                    tf.IndexedSlices(values=joint_values, indices=joint_indices, dense_shape=grads[0].dense_shape)
            averaged_grad_and_var = (averaged_grad, var)
            averaged_grads_and_vars.append(averaged_grad_and_var)

        # Average losses
        averaged_batch_loss = tf.reduce_mean(tower_batch_losses)
        joint_sentence_losses = tf.concat(tower_sentence_losses, axis=0)
        total_words_processed = tf.reduce_sum(tower_words)
        # Compile OPs and add summaries
        proto_train_ops = [averaged_grads_and_vars, averaged_batch_loss, joint_sentence_losses, total_words_processed]
        if not no_summaries:
            # Create summaries
            averaged_summaries = model.get_summaries(averaged_batch_loss)
            proto_train_ops.append(averaged_summaries)
        # Proto-OPs are forwarded to gradient accumulation or optimization
        return proto_train_ops

    def _get_translation_ops():
        """ Defines single-device OPs used to obtain translations from the model. """
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            with tf.device(assign_to_device(controller, operators[0])):
                # Translation OPs
                next_batch = iterator.get_next()
                greedy_translations, _, _ = model.decode_greedy(next_batch)
                sampled_translations, _ = model.decode_with_sampling(next_batch)
                beam_translations, beam_scores = model.decode_with_beam_search(next_batch)

        translation_ops = [next_batch[0], next_batch[2], greedy_translations, sampled_translations,
                           beam_translations, beam_scores]
        return translation_ops

    assert mode in ['training', 'validation', 'translation'], \
        'Specified OP-retrieval mode must be training, validation, or translation'

    # Detect available GPUs
    controller, operators = get_devices(num_gpus)

    if mode == 'training':
        return _get_train_ops()
    else:
        return _get_translation_ops()


def get_single_ops(model, iterator, num_gpus, unused_eos_id, mode, no_summaries=False):
    """ Defines the training and validation OPs for single-device training. """

    def _get_train_ops(next_batch):
        """ Defines single-device OPs used to train the model. """
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            with tf.device(assign_to_device(controller, operators[0])):
                # Surface OPs
                grads_and_vars, _, batch_loss, sentence_loss, words_processed, _ = model.train_model(next_batch)
                proto_train_ops = [grads_and_vars, batch_loss, sentence_loss, words_processed]

        # Create summaries
        if not no_summaries:
            summaries = model.get_summaries(batch_loss)
            proto_train_ops.append(summaries)
        return proto_train_ops

    def _get_translation_ops(next_batch):
        """ Defines single-device OPs used to obtain translations from the model. """
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            with tf.device(assign_to_device(controller, operators[0])):
                # Surface OPs
                greedy_translations, _, _ = model.decode_greedy(next_batch)
                sampled_translations, _ = model.decode_with_sampling(next_batch)
                beam_translations, beam_scores = model.decode_with_beam_search(next_batch)

        translation_ops = [next_batch[0], next_batch[2], greedy_translations, sampled_translations,
                           beam_translations, beam_scores]
        return translation_ops

    assert mode in ['training', 'translation'], 'Specified OP-retrieval mode must be training or translation'

    # Detect available GPUs
    controller, operators = get_devices(num_gpus)

    # Draw batch from iterator
    next_batch = iterator.get_next()

    if mode == 'training':
        return _get_train_ops(next_batch)
    else:
        return _get_translation_ops(next_batch)


class VariableUpdateTrainer(object):
    """ Class for training models with variable gradient aggregation updates;
    Inspired by the Adam-specific gradient aggregation implementation by fstahlberg@github
    (https://github.com/fstahlberg/tensor2tensor/blob/master/tensor2tensor/utils/largebatch_optimizer.py) """
    def __init__(self, model, num_layers, iterator, num_gpus, eos_id, n_agg_steps, use_multi_gpu, session):
        self.model = model
        self.num_layers = num_layers
        self.iterator = iterator
        self.num_gpus = num_gpus
        self.eos_id = eos_id
        self.n_agg_steps = n_agg_steps if n_agg_steps > 0 else 1
        self.use_multi_gpu = use_multi_gpu
        self.session = session

        forward_fn = get_parallel_ops if use_multi_gpu else get_single_ops
        self.train_ops = forward_fn(model, iterator, num_gpus, eos_id, mode='training')

        self.grads_cache = None
        self.batch_loss_cache = None
        self.words_processed_cache = None

        self.optimizer = model.optimizer
        self.curr_agg_step = 0
        self.do_update = False

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as outer_scope:
            self.update_and_store_ops = self._update_and_store()
            outer_scope.reuse_variables()
            self.update_and_apply_ops = self._update_and_apply()

        self.zero_op = self._reset()

    def _initialize(self, t_vars):
        """ Initializes gradient and metrics caches """
        # Populate caches
        self.grads_cache = [tf.Variable(tf.zeros_like(var), trainable=False) for var in t_vars]
        self.batch_loss_cache = tf.Variable(tf.zeros(shape=[], dtype=tf.float32), trainable=False)
        self.words_processed_cache = tf.Variable(tf.zeros(shape=[], dtype=tf.int32), trainable=False)
        self.session.run(
            tf.variables_initializer(self.grads_cache + [self.batch_loss_cache, self.words_processed_cache]))

    def _reset(self):
        """ Resets gradient and metrics caches. """
        # Reset aggregated grads
        zero_grads = [tf.assign(grad, grad * 0.) for grad in self.grads_cache]
        zero_batch_loss = tf.assign(self.batch_loss_cache, self.batch_loss_cache * 0.)
        zero_words_processed = tf.assign(self.words_processed_cache, self.words_processed_cache * 0)
        return tf.group(zero_grads, zero_batch_loss, zero_words_processed)

    def _get_grad_norm_ratio(self, t_vars, grads):
        """ Computes the grad norm ratio, as introduced in
        Bapna, Ankur, et al. "Training Deeper Neural Machine Translation Models with Transparent Attention.",
        arXiv preprint arXiv:1808.07561 (2018). """
        initial_grads = list()
        final_grads = list()

        # Compute average gradient for the initial and final layers
        for var_id, var in enumerate(t_vars):
            if 'layer_1' in var.name:
                initial_grads.append(tf.norm(grads[var_id], ord=2))
            elif 'layer_{:d}'.format(self.num_layers) in var.name:
                final_grads.append(tf.norm(grads[var_id], ord=2))
            else:
                pass

        # Compute grad norm ratio
        return tf.reduce_mean(initial_grads) / tf.reduce_mean(final_grads)

    def _update(self):
        """ Updates gradients and metrics. """
        # Get step-wise values
        step_grads, t_vars = list(zip(*self.train_ops[0]))
        # Initialize caches during the initial update step
        if self.grads_cache is None:
            self._initialize(t_vars)

        if self.curr_agg_step == 1:
            # Initialize caches and optimizer
            self._initialize(t_vars)
            self.session.run(tf.variables_initializer(self.optimizer.variables()))

        # Aggregate

        # Sophisticated gradient monitoring
        for grad_id, step_grad in enumerate(step_grads):
            if step_grad is None:
                print(t_vars[grad_id])

        grads = [tf.assign_add(self.grads_cache[grad_id], step_grad) for grad_id, step_grad in enumerate(step_grads)]
        batch_loss = tf.assign(self.batch_loss_cache,
                               tf.reduce_sum(tf.stack([self.batch_loss_cache, self.train_ops[1]], axis=0)))
        words_processed = tf.assign(self.words_processed_cache,
                                    tf.reduce_sum(tf.stack([self.words_processed_cache, self.train_ops[3]], axis=0)))
        return t_vars, grads, batch_loss, words_processed

    def _update_and_store(self):
        """ Updates gradients and metrics and returns an empty training OP. """
        # Update
        _, grads, batch_loss, words_processed = self._update()
        # Train OP is equivalent to updating gradients
        train_op = grads
        # Grad norm ratio
        grad_norm_ratio_op = tf.no_op()
        # Define summaries
        summaries = self.model.get_summaries(batch_loss)
        return [batch_loss, words_processed, train_op, grad_norm_ratio_op, summaries]

    def _update_and_apply(self):
        """ Updates gradients and metrics and returns a training OP which applies the aggregated gradients. """
        # Update
        t_vars, grads, batch_loss, words_processed = self._update()
        # Normalize
        grads = [tf.assign(grad, grad / self.n_agg_steps) for grad in grads]
        # Define train OP
        grads_and_vars = [(grad, t_vars[grad_id]) for grad_id, grad in enumerate(grads)]
        train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.model.global_step)
        # Grad norm ratio
        grad_norm_ratio_op = self._get_grad_norm_ratio(t_vars, grads)
        # Define summaries
        summaries = self.model.get_summaries(batch_loss)
        return [batch_loss / self.n_agg_steps, words_processed, train_op, grad_norm_ratio_op, summaries]

    def forward(self):
        with tf.name_scope('optimization'), tf.device('/cpu:0'):
            # Increment aggregation step
            self.curr_agg_step += 1
            self.do_update = (self.curr_agg_step % self.n_agg_steps == 0)
            # Update / apply gradients and collect metrics
            if self.do_update:
                return self.update_and_apply_ops
            else:
                return self.update_and_store_ops
