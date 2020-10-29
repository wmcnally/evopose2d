from tensorflow.keras.optimizers.schedules import LearningRateSchedule
import math
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.util.tf_export import keras_export

"""Modified from:
    https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/optimizer_v2/learning_rate_schedule.py"""

@keras_export("keras.experimental.WarmupCosineDecay")
class WarmupCosineDecay(LearningRateSchedule):
    """A LearningRateSchedule that uses a cosine decay schedule.
    See [Loshchilov & Hutter, ICLR2016], SGDR: Stochastic Gradient Descent
    with Warm Restarts. https://arxiv.org/abs/1608.03983
    When training a model, it is often recommended to lower the learning rate as
    the training progresses. This schedule applies a cosine decay function
    to an optimizer step, given a provided initial learning rate.
    It requires a `step` value to compute the decayed learning rate. You can
    just pass a TensorFlow variable that you increment at each training step.
    The schedule a 1-arg callable that produces a decayed learning
    rate when passed the current optimizer step. This can be useful for changing
    the learning rate value across different invocations of optimizer functions.
    It is computed as:
    ```python
    def decayed_learning_rate(step):
      step = min(step, decay_steps)
      cosine_decay = 0.5 * (1 + cos(pi * step / decay_steps))
      decayed = (1 - alpha) * cosine_decay + alpha
      return initial_learning_rate * decayed
    ```
    Example usage:
    ```python
    decay_steps = 1000
    lr_decayed_fn = tf.keras.experimental.CosineDecay(
        initial_learning_rate, decay_steps)
    ```
    You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
    as the learning rate. The learning rate schedule is also serializable and
    deserializable using `tf.keras.optimizers.schedules.serialize` and
    `tf.keras.optimizers.schedules.deserialize`.
    Returns:
      A 1-arg callable learning rate schedule that takes the current optimizer
      step and outputs the decayed learning rate, a scalar `Tensor` of the same
      type as `initial_learning_rate`.
    """

    def __init__(
            self,
            initial_learning_rate,
            decay_steps,
            warmup_steps,
            warmup_factor,
            alpha=0.0,
            name=None):
        """Applies cosine decay to the learning rate.
        Args:
          initial_learning_rate: A scalar `float32` or `float64` Tensor or a
            Python number. The initial learning rate.
          decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
            Number of steps to decay over.
          alpha: A scalar `float32` or `float64` Tensor or a Python number.
            Minimum learning rate value as a fraction of initial_learning_rate.
          name: String. Optional name of the operation.  Defaults to 'CosineDecay'.
        """
        super(WarmupCosineDecay, self).__init__()

        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.warmup_factor = warmup_factor
        self.alpha = alpha
        self.name = name

    def __call__(self, step):
        with ops.name_scope_v2(self.name or "WarmupCosineDecay"):
            initial_learning_rate = ops.convert_to_tensor_v2(
                self.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            decay_steps = math_ops.cast(self.decay_steps, dtype)
            warmup_steps = math_ops.cast(self.warmup_steps, dtype)
            w_fac = math_ops.cast(self.warmup_factor, dtype)

            global_step_recomp = math_ops.cast(step, dtype)
            global_step_recomp = math_ops.minimum(global_step_recomp, decay_steps)

            def compute_step(warming_up=False):
                if warming_up:
                    completed_fraction = global_step_recomp / warmup_steps
                    gain = w_fac + (1 - w_fac) * completed_fraction
                else:
                    completed_fraction = (global_step_recomp - warmup_steps) / (decay_steps - warmup_steps)
                    cosine_decayed = 0.5 * (1.0 + math_ops.cos(
                        constant_op.constant(math.pi) * completed_fraction))
                    gain = (1 - self.alpha) * cosine_decayed + self.alpha
                return gain

            gain = control_flow_ops.cond(math_ops.less(global_step_recomp, warmup_steps),
                                         lambda: compute_step(warming_up=True),
                                         lambda: compute_step(warming_up=False))

            return math_ops.multiply(initial_learning_rate, gain)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "warmup_steps": self.warmup_steps,
            "warmup_factor": self.warmup_factor,
            "alpha": self.alpha,
            "name": self.name
        }

@keras_export("keras.experimental.WarmupPiecewise")
class WarmupPiecewise(LearningRateSchedule):
    """A LearningRateSchedule that uses a piecewise constant decay schedule.
    The function returns a 1-arg callable to compute the piecewise constant
    when passed the current optimizer step. This can be useful for changing the
    learning rate value across different invocations of optimizer functions.
    Example: use a learning rate that's 1.0 for the first 100001 steps, 0.5
      for the next 10000 steps, and 0.1 for any additional steps.
    ```python
    step = tf.Variable(0, trainable=False)
    boundaries = [100000, 110000]
    values = [1.0, 0.5, 0.1]
    learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values)
    # Later, whenever we perform an optimization step, we pass in the step.
    learning_rate = learning_rate_fn(step)
    ```
    You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
    as the learning rate. The learning rate schedule is also serializable and
    deserializable using `tf.keras.optimizers.schedules.serialize` and
    `tf.keras.optimizers.schedules.deserialize`.
    Returns:
      A 1-arg callable learning rate schedule that takes the current optimizer
      step and outputs the decayed learning rate, a scalar `Tensor` of the same
      type as the boundary tensors.
      The output of the 1-arg function that takes the `step`
      is `values[0]` when `step <= boundaries[0]`,
      `values[1]` when `step > boundaries[0]` and `step <= boundaries[1]`, ...,
      and values[-1] when `step > boundaries[-1]`.
    """

    def __init__(
            self,
            boundaries,
            values,
            warmup_steps,
            warmup_factor,
            gradual=True,
            name=None):
        """Piecewise constant from boundaries and interval values.
        Args:
          boundaries: A list of `Tensor`s or `int`s or `float`s with strictly
            increasing entries, and with all elements having the same type as the
            optimizer step.
          values: A list of `Tensor`s or `float`s or `int`s that specifies the
            values for the intervals defined by `boundaries`. It should have one
            more element than `boundaries`, and all elements should have the same
            type.
          name: A string. Optional name of the operation. Defaults to
            'PiecewiseConstant'.
        Raises:
          ValueError: if the number of elements in the lists do not match.
        """
        super(WarmupPiecewise, self).__init__()

        if len(boundaries) != len(values) - 1:
            raise ValueError(
                "The length of boundaries should be 1 less than the length of values")

        self.boundaries = boundaries
        self.values = values
        self.name = name
        self.warmup_steps = warmup_steps
        self.warmup_factor = warmup_factor
        self.gradual = gradual

    def __call__(self, step):
        with ops.name_scope_v2(self.name or "WarmupPiecewise"):
            boundaries = ops.convert_n_to_tensor(self.boundaries)
            values = ops.convert_n_to_tensor(self.values)
            x_recomp = ops.convert_to_tensor_v2(step)

            # convert all data types to float
            x_recomp = math_ops.cast(x_recomp, values[0].dtype)
            warmup_steps = math_ops.cast(self.warmup_steps, values[0].dtype)
            w_fac = math_ops.cast(self.warmup_factor, values[0].dtype)

            for i, b in enumerate(boundaries):
                if b.dtype.base_dtype != x_recomp.dtype.base_dtype:
                    # We cast the boundaries to have the same type as the step
                    b = math_ops.cast(b, x_recomp.dtype.base_dtype)
                    boundaries[i] = b

            def compute_piecewise():
                pred_fn_pairs = []
                pred_fn_pairs.append((x_recomp <= boundaries[0], lambda: values[0]))
                pred_fn_pairs.append((x_recomp > boundaries[-1], lambda: values[-1]))
                for low, high, v in zip(boundaries[:-1], boundaries[1:], values[1:-1]):
                    # Need to bind v here; can do this with lambda v=v: ...
                    pred = (x_recomp > low) & (x_recomp <= high)
                    pred_fn_pairs.append((pred, lambda v=v: v))

                # The default isn't needed here because our conditions are mutually
                # exclusive and exhaustive, but tf.case requires it.
                default = lambda: values[0]
                return control_flow_ops.case(pred_fn_pairs, default, exclusive=True)

            def compute_step(warming_up=False):
                if warming_up:
                    completed_fraction = x_recomp / warmup_steps
                    if self.gradual:
                        gain = w_fac + (1 - w_fac) * completed_fraction
                    else:
                        gain = w_fac
                    return math_ops.multiply(values[0], gain)
                else:
                    return compute_piecewise()

            return control_flow_ops.cond(math_ops.less(x_recomp, warmup_steps),
                                         lambda: compute_step(warming_up=True),
                                         lambda: compute_step(warming_up=False))


    def get_config(self):
        return {
            "boundaries": self.boundaries,
            "values": self.values,
            "warmup_steps": self.warmup_steps,
            "warmup_factor": self.warmup_factor,
            "name": self.name
        }
