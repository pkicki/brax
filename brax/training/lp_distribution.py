from typing import Tuple
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from scipy import signal

@struct.dataclass
class NoiseState:
    i: int # index
    key: jax.Array   # PRNG Key

def iir_filter(b, a, x):
    """
    Direct Form II Transposed IIR filtering in JAX.
    Equivalent to scipy.signal.lfilter(b, a, x)
    """
    order = max(len(a), len(b)) - 1
    a = a / a[0]
    b = b / a[0]

    def step(carry, x_n):
        x_hist, y_hist = carry
        y = b[0] * x_n + jnp.sum(b[1:] * x_hist, axis=0) \
                       - jnp.sum(a[1:] * y_hist, axis=0)
        x_hist = x_hist.at[1:].set(x_hist[:-1])
        x_hist = x_hist.at[0].set(x_n)
        y_hist = y_hist.at[1:].set(y_hist[:-1])
        y_hist = y_hist.at[0].set(y)
        return (x_hist, y_hist), y

    x_hist = jnp.zeros(order)
    y_hist = jnp.zeros(order)
    _, y = jax.lax.scan(step, (x_hist, y_hist), x)
    return y

class LowPassNoise:
    def __init__(self, episode_length, action_dim, cutoff, fs, order=2):
        self.episode_length = episode_length
        self.action_dim = action_dim

        b, a = signal.butter(order, cutoff, btype='low', analog=False, fs=fs)
        self.b = jnp.array(b)
        self.a = jnp.array(a)
        
    def init_state(self, key, n_envs):
        """Initializes the noise state for n_envs."""
        #key, subkey = jax.random.split(noise_state.key)
        key, subkey = jax.random.split(key)
        tran_len = 100
        #tran_len = 0
        self.white_noise = jax.random.normal(subkey, (self.episode_length + tran_len, n_envs, self.action_dim))
        iir_filter_multi = jax.vmap(
            jax.vmap(iir_filter, in_axes=(None, None, 0)),  # map across A
            in_axes=(None, None, 0)                            # map across E
        )
        white_noise_T = jnp.transpose(self.white_noise, (1, 2, 0))
        lp_noise_T = iir_filter_multi(self.b, self.a, white_noise_T)
        self.lp_noise = jnp.transpose(lp_noise_T, (2, 0, 1))
        # ignore transient state
        self.white_noise = self.white_noise[tran_len:]
        lp_noise = self.lp_noise[tran_len:]

        scale_factor = jnp.std(self.white_noise, axis=0, keepdims=True) / jnp.std(lp_noise, axis=0, keepdims=True)
        lp_noise = lp_noise * scale_factor
        wn_mean = jnp.mean(self.white_noise, axis=0, keepdims=True)
        lpn_mean = jnp.mean(lp_noise, axis=0, keepdims=True)
        self.lp_noise = (lp_noise - lpn_mean) + wn_mean
        return NoiseState(0, key)

    def sample(self, state: NoiseState) -> Tuple[jnp.ndarray, NoiseState, jnp.ndarray]:
        return self.lp_noise[state.i], NoiseState(state.i + 1, state.key), self.white_noise[state.i]

if __name__ == "__main__":
    N = 20
    lp = LowPassNoise(N, 2, 3.0, 100, 3)
    key = jax.random.PRNGKey(0)
    for _ in range(10):
        result = []
        white = []
        state = lp.init_state(key, 2)
        for _ in range(N):
            y, state, x = lp.sample(state)
            result.append(y)
            white.append(x)
            #print(y)
        key = state.key
        result = np.array(result)#[:, 0]
        white = np.array(white)
        print("LP STD:", result.std(0).mean())
        print("White STD:", white.std(0).mean())
        print("LP MEAN:", result.mean(0).mean())
        print("White MEAN:", white.mean(0).mean())
        import matplotlib.pyplot as plt
        plt.subplot(221)
        plt.plot(result[:, 0, 0], 'b')
        plt.plot(result[:, 0, 1], 'r')
        plt.subplot(222)
        plt.plot(result[:, 1, 0], 'b')
        plt.plot(result[:, 1, 1], 'r')
        plt.subplot(223)
        plt.plot(white[:, 0, 0])
        plt.plot(white[:, 0, 1])
        plt.subplot(224)
        plt.plot(white[:, 1, 0])
        plt.plot(white[:, 1, 1])
        plt.show()