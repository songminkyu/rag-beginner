import numpy as np
import cossim

def cos_sim_test():
    A = np.array([0, 1, 1, 1])
    B = np.array([1, 0, 0, 1])
    print(cossim.cos_sim(A, B))

if __name__ == "__main__":
    cos_sim_test()
