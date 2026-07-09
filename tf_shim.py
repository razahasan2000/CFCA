"""Minimal TensorFlow shim for GLocalX rule-merging (no neural network ops needed)."""
import sys
import types

tf = types.ModuleType('tensorflow')
tf.keras = types.ModuleType('tensorflow.keras')

class _Logger:
    def setLevel(self, *a): pass
tf.get_logger = lambda: _Logger()
tf.autograph = types.ModuleType('tensorflow.autograph')
tf.autograph.set_verbosity = lambda *a: None
tf.Tensor = object

sys.modules['tensorflow'] = tf
sys.modules['tensorflow.keras'] = tf.keras
