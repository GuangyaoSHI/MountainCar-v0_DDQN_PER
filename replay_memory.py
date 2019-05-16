import random
import zlib
import pickle
import math

from sum_tree import SumTree

class ReplayMemory(object):
    def __init__(self, max_size, alpha, eps):
        self.max_size = max_size
        self.alpha = alpha
        self.eps = eps

        self.tree = SumTree(max_size)
        self.last_idxs = None
        self.size = 0

    def get_batch(self, batch_size):
        self.last_idxs = []

        ret = []
        for i in range(min(batch_size, self.size)):
            s = random.random() * self.tree.total()

            idx, _, data = self.tree.get(s)

            ret.append(pickle.loads(zlib.decompress(data)))
            self.last_idxs.append(idx)

        return ret

    def update(self, losses):
        for i in range(len(self.last_idxs)):
            self.tree.update(self.last_idxs[i], math.pow(losses[i] + self.eps, self.alpha))

    def add_element(self, new_el, loss):
        self.size = min(self.max_size, self.size + 1)

        p = math.pow(loss + self.eps, self.alpha)
        self.tree.add(p, zlib.compress(pickle.dumps(new_el)))

    def __len__(self):
        return self.size
