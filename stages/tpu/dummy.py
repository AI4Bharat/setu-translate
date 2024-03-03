import jax.numpy as jnp

# Example array with shape (A, B, ...)
# Let's assume a specific example for demonstration
A, B = 2, 3
rest_of_dims = (4, 5) # Example additional dimensions
array = jnp.ones((A, B) + rest_of_dims) # Create an example array

# Reshape the array from (A, B, ...) to (A*B, ...)
reshaped_array = array.reshape(-1, *array.shape[2:])

# Output the shape of the reshaped array for verification
print("Original shape:", array.shape)
print("Reshaped shape:", reshaped_array.shape)
