"""A simple example that triggers a shape error in JAX, demonstrating the shape tracker bug tracker.

This example performs a series of JAX operations, culminating in a shape mismatch error
during a matrix multiplication. The JaxShapeTracker context manager is used to capture
and report the sequence of operations leading to the error.
"""

import jax
import jax.numpy as jnp


def main():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (32, 128))

    # Correct Reshape
    y = x.reshape(32, 64, 2)

    # Reduce operation
    z = jnp.sum(y, axis=2)  # Shape becomes (32, 64)

    # Intentional Error: Mismatched Matmul
    key2 = jax.random.PRNGKey(1)
    weight = jax.random.normal(key2, (100, 100))

    jnp.matmul(z, weight)  # <--- CRASH HERE


if __name__ == "__main__":
    main()
