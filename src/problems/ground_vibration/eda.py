import numpy as np
import matplotlib.pyplot as plt
import json

def get_U(influence_matrix: np.ndarray) -> np.ndarray:
    samples, N_s, _, _ = influence_matrix.shape
    reshaped = influence_matrix.reshape(samples, N_s, N_s, 2, 2)
    transposed = reshaped.transpose(0, 1, 3, 2, 4)

    U = transposed.reshape(samples, N_s * 2, N_s * 2)
    return U

path = "/Users/ls/workspace/DeepOperatorML/data/raw/ground_vibration/influence_dataset_N10_samples50_csv/"

data_real = np.loadtxt(path + 'G_samples_imag_flat.csv', delimiter=",")
data_imag = np.loadtxt(path + 'G_samples_real_flat.csv', delimiter=",")
data = data_real + 1j * data_imag
data = data.reshape(50, 10, 10, -1)
params_array = np.loadtxt(path + 'params_array.csv', delimiter=",")

print(params_array.shape)
metadata = json.load(open(path + 'metadata.json', 'r'))
processed = np.load("/Users/ls/workspace/DeepOperatorML/data/processed/ground_vibration/45cce590/data.npz")

displacements = processed['g_u'][:,:, :4] + 1j * processed['g_u'][:,:, 4:]
# # print(displacements[0, :, :])
# raw = np.load("/Users/ls/workspace/DeepOperatorML/data/raw/ground_vibration/ground_vibration_v1.npz")
# u = raw['g_u'].reshape(50, 10, 10, -1)
# plt.contourf(u[0, :, :, 0].imag)
# plt.show()

output = np.load('/Users/ls/workspace/DeepOperatorML/output/ground_vibration/2025-10-27_12-36-10/aux/output_data.npz')
print(output['g_u'].shape)

gu = output['g_u'][:, :, :4] + 1j * output['g_u'][:, :, 4:]
gu = gu.reshape(50, 10, 10,4)
U = get_U(gu)
q = np.ones((20,1)).flatten()
w = U @ q

print(w.shape)
plt.plot(w[0].real)
plt.plot(w[0].imag)
plt.show()

quit()
# (50, 2*3 * 2*3)
# | uxx(11) uxz(11) uzx(11) uzz(11) uxx(12) uxz(12) uzx(12) uzz(12) uxx(13) uxz(13) uzx(13) uzz(13) |
# | uxx(21) uxz(21) uzx(21) uzz(21) uxx(22) uxz(22) uzx(22) uzz(22) uxx(23) uxz(23) uzx(23) uzz(23) |
# | uxx(31) uxz(31) uzx(31) uzz(31) uxx(32) uxz(32) uzx(32) uzz(32) uxx(33) uxz(33) uzx(33) uzz(33) |
# |                                            ...                                                  |
# |                                            ...                                                  |
# |                                            ...                                                  |
# |                                            ...                                                  |
# |                                            ...                                                  |
