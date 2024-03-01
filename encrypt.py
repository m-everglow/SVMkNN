import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from phe import paillier
import time

if __name__ == '__main__':
    kappa_list = [512,1024,2048]
    for kappa in kappa_list:
        pk, sk = paillier.generate_paillier_keypair(n_length=kappa)
        # 待加密的数组
        X = np.random.randint(0, 100, 500)

        enc = 0
        dec = 0

        for x in X:
            time1 = time.time()
            encX = pk.encrypt(int(x))
            enc += time.time() - time1
            time2 = time.time()
            decX = sk.decrypt(encX)
            dec += time.time() - time2

        print("enc:{}".format(enc/500))
        print("dec:{}".format(dec/500))


