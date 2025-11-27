from typing import Tuple
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from scipy import signal
from brax.training import distribution
from brax.training import networks

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
        w = carry
        y = b[0] * x_n + w[0]
        new_w = jnp.zeros_like(w)
        new_w = new_w.at[:-1].set(w[1:])
        new_w = new_w.at[-1].set(0.0)
        new_w = new_w - a[1:] * y + b[1:] * x_n
        return new_w, y

    w0 = jnp.zeros(order)
    _, y = jax.lax.scan(step, w0, x)
    return y

class LowPassNoise:
    def __init__(self, episode_length, action_dim, cutoff, fs, order=2):
        self.episode_length = episode_length
        self.action_dim = action_dim
        # Calculate coefficients using scipy (runs on CPU during init)
        #nyquist = 0.5 * fs
        #normal_cutoff = cutoff / nyquist
        b, a = signal.butter(order, cutoff, btype='low', analog=False, fs=fs)
        
        # Calculate normalization factor to ensure roughly unit variance
        # Using a fixed seed for normalization factor calculation to be deterministic
        rng = np.random.default_rng(42)
        dummy_noise = rng.standard_normal(10000)
        filtered_dummy = signal.lfilter(b, a, dummy_noise)
        self.scale_factor = 1.0 / np.std(filtered_dummy)

        #self.b = jnp.array(b * scale_factor)
        self.b = jnp.array(b)
        self.a = jnp.array(a)
        
        # Initial zi structure for Direct Form II Transposed
        # Shape needed for signal.lfilter_zi is (max(len(a), len(b)) - 1,)
        #zi_init = signal.lfilter_zi(b, a) * self.scale_factor
        zi_init = signal.lfilter_zi(b, a) / self.scale_factor
        self.zi_ref = jnp.array(zi_init)

    #def init_state(self, noise_state: NoiseState, n_envs):
    def init_state(self, key, n_envs):
        """Initializes the noise state for n_envs."""
        #key, subkey = jax.random.split(noise_state.key)
        key, subkey = jax.random.split(key)
        tran_len = 100
        self.white_noise = jax.random.normal(subkey, (self.episode_length + tran_len, n_envs, self.action_dim))
        #self.lp_noise = iir_filter(self.b, self.a, self.white_noise)
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
        self.lp_noise = (lp_noise - jnp.mean(lp_noise, axis=0, keepdims=True)) * self.scale_factor
        return NoiseState(0, key)

    def sample(self, state: NoiseState) -> Tuple[jnp.ndarray, NoiseState, jnp.ndarray]:
        return self.lp_noise[state.i], NoiseState(state.i + 1, state.key), self.white_noise[state.i]

    #def sample(self, state: NoiseState) -> Tuple[jnp.ndarray, NoiseState]:
    #    """Performs one step of low-pass filtering."""
    #    key, subkey = jax.random.split(state.key)
    #    
    #    # Generate white noise: (n_envs, action_dim)
    #    x = jax.random.normal(subkey, (state.zi.shape[0], self.action_dim))
    #    
    #    # Apply Direct Form II Transposed difference equation manually
    #    # This is equivalent to one step of lfilter
    #    # y[n] = b[0]x[n] + z[n-1, 0]
    #    
    #    # We need to map over batch dimensions (n_envs, action_dim)
    #    def filter_step(zi, x_val):
    #        # zi shape: (order,)
    #        y = self.b[0] * x_val + zi[0]
    #        
    #        # Update state
    #        # z[n, k] = b[k+1]x[n] - a[k+1]y[n] + z[n-1, k+1]
    #        # The last element z[n, order-1] = b[order]x[n] - a[order]y[n]
    #        
    #        def update_zi_k(k, zi_current):
    #            val = self.b[k+1] * x_val - self.a[k+1] * y
    #            # Add previous next-stage state if not at the end
    #            prev_next = jax.lax.cond(
    #                k < (zi.shape[0] - 1),
    #                lambda: zi[k+1],
    #                lambda: 0.0
    #            )
    #            return val + prev_next

    #        new_zi = jax.vmap(update_zi_k, in_axes=(0, None))(
    #            jnp.arange(zi.shape[0]), zi
    #        )
    #        return y, new_zi

    #    # Vectorize over environments and actions
    #    # inputs: zi=(N, A, O), x=(N, A)
    #    y, new_zi = jax.vmap(jax.vmap(filter_step))(state.zi, x)

    #    y = y * self.scale_factor
    #    
    #    return y, NoiseState(zi=new_zi, key=key), x

if __name__ == "__main__":
    N = 100
    lp = LowPassNoise(N, 2, 2.0, 100)
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
        plt.plot(result[:, 0, 0])
        plt.plot(result[:, 0, 1])
        plt.subplot(222)
        plt.plot(result[:, 1, 0])
        plt.plot(result[:, 1, 1])
        plt.subplot(223)
        plt.plot(white[:, :, 0])
        plt.plot(white[:, :, 1])
        plt.subplot(224)
        plt.plot(white[:, :, 0])
        plt.plot(white[:, :, 1])
        plt.show()