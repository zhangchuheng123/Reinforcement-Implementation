import operator
from collections import deque
import numpy as np
import torch


class MultiStepBuff:

    def __init__(self, maxlen=3):
        super(MultiStepBuff, self).__init__()
        self.maxlen = int(maxlen)
        self.reset()

    def append(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def get(self, gamma=0.99):
        assert len(self.rewards) > 0
        state = self.states.popleft()
        action = self.actions.popleft()
        reward = self._nstep_return(gamma)
        return state, action, reward

    def _nstep_return(self, gamma):
        r = np.sum([r * (gamma ** i) for i, r in enumerate(self.rewards)])
        self.rewards.popleft()
        return r

    def reset(self):
        # Buffer to store n-step transitions.
        self.states = deque(maxlen=self.maxlen)
        self.actions = deque(maxlen=self.maxlen)
        self.rewards = deque(maxlen=self.maxlen)

    def is_empty(self):
        return len(self.rewards) == 0

    def is_full(self):
        return len(self.rewards) == self.maxlen

    def __len__(self):
        return len(self.rewards)


class LazyMemory(dict):

    def __init__(self, capacity, state_shape, device, state_dtype):
        super(LazyMemory, self).__init__()
        self.capacity = int(capacity)
        self.state_shape = state_shape
        self.device = device
        self.reset()
        if state_dtype == 'float32':
            self.state_dtype = np.float32
        elif state_dtype == 'float64':
            self.state_dtype = np.float64
        elif state_dtype == 'uint8':
            self.state_dtype = np.uint8

    def reset(self):
        self['state'] = []
        self['next_state'] = []

        self['action'] = np.empty((self.capacity, 1), dtype=np.int64)
        self['reward'] = np.empty((self.capacity, 1), dtype=np.float32)
        self['done'] = np.empty((self.capacity, 1), dtype=np.float32)

        self._n = 0
        self._p = 0

    def append(self, state, action, reward, next_state, done,
               episode_done=None):
        self._append(state, action, reward, next_state, done)

    def _append(self, state, action, reward, next_state, done):
        self['state'].append(state)
        self['next_state'].append(next_state)
        self['action'][self._p] = action
        self['reward'][self._p] = reward
        self['done'][self._p] = done

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity

        self.truncate()

    def truncate(self):
        while len(self['state']) > self.capacity:
            del self['state'][0]
            del self['next_state'][0]

    def sample(self, batch_size):
        indices = np.random.randint(low=0, high=len(self), size=batch_size)
        return self._sample(indices, batch_size)

    def _sample(self, indices, batch_size):
        bias = -self._p if self._n == self.capacity else 0

        states = np.empty((batch_size, *self.state_shape), dtype=self.state_dtype)
        next_states = np.empty((batch_size, *self.state_shape), dtype=self.state_dtype)

        for i, index in enumerate(indices):
            _index = np.mod(index+bias, self.capacity)
            states[i, ...] = self['state'][_index]
            next_states[i, ...] = self['next_state'][_index]

        if self.state_dtype is np.float32:
            states = torch.FloatTensor(states).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
        elif self.state_dtype is np.float64:
            states = torch.DoubleTensor(states).to(self.device)
            next_states = torch.DoubleTensor(next_states).to(self.device)
        elif self.state_dtype is np.uint8:
            states = torch.ByteTensor(states).to(self.device).float() / 255.
            next_states = torch.ByteTensor(next_states).to(self.device).float() / 255.
        actions = torch.LongTensor(self['action'][indices]).to(self.device)
        rewards = torch.FloatTensor(self['reward'][indices]).to(self.device)
        dones = torch.FloatTensor(self['done'][indices]).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self._n


class LazyMultiStepMemory(LazyMemory):

    def __init__(self, capacity, state_shape, device, gamma=0.99,
                 multi_step=3, state_dtype='float32'):
        super(LazyMultiStepMemory, self).__init__(
            capacity, state_shape, device, state_dtype)

        self.gamma = gamma
        self.multi_step = int(multi_step)
        if self.multi_step != 1:
            self.buff = MultiStepBuff(maxlen=self.multi_step)

    def append(self, state, action, reward, next_state, done):
        if self.multi_step != 1:
            self.buff.append(state, action, reward)

            if self.buff.is_full():
                state, action, reward = self.buff.get(self.gamma)
                self._append(state, action, reward, next_state, done)

            if done:
                while not self.buff.is_empty():
                    state, action, reward = self.buff.get(self.gamma)
                    self._append(state, action, reward, next_state, done)
        else:
            self._append(state, action, reward, next_state, done)


class LazyPrioritizedMultiStepMemory(LazyMultiStepMemory):

    def __init__(self, capacity, state_shape, device, state_dtype='float32',
            gamma=0.99, multi_step=3, alpha=0.6, beta=0.4, beta_steps=2e5,
            min_pa=0.0, max_pa=1.0, eps=0.01):

        super().__init__(capacity, state_shape, device, gamma, 
            multi_step, state_dtype)

        self.alpha = alpha
        self.beta = beta
        self.beta_diff = (1.0 - beta) / beta_steps
        self.min_pa = min_pa
        self.max_pa = max_pa
        self.eps = eps
        self._cached = None

        it_capacity = 1
        while it_capacity < capacity:
            it_capacity *= 2
        self.it_sum = SumTree(it_capacity)
        self.it_min = MinTree(it_capacity)

    def _pa(self, p):
        return np.clip((p + self.eps) ** self.alpha, self.min_pa, self.max_pa)

    def append(self, state, action, reward, next_state, done, p=None):
        # Calculate priority.
        if p is None:
            pa = self.max_pa
        else:
            pa = self._pa(p)

        if self.multi_step != 1:
            self.buff.append(state, action, reward)

            if self.buff.is_full():
                state, action, reward = self.buff.get(self.gamma)
                self._append(state, action, reward, next_state, done, pa)

            if done:
                while not self.buff.is_empty():
                    state, action, reward = self.buff.get(self.gamma)
                    self._append(state, action, reward, next_state, done, pa)
        else:
            self._append(state, action, reward, next_state, done, pa)

    def _append(self, state, action, reward, next_state, done, pa):
        # Store priority, which is done efficiently by SegmentTree.
        self.it_min[self._p] = pa
        self.it_sum[self._p] = pa
        super()._append(state, action, reward, next_state, done)

    def _sample_idxes(self, batch_size):
        total_pa = self.it_sum.sum(0, self._n)
        rands = np.random.rand(batch_size) * total_pa
        indices = [self.it_sum.find_prefixsum_idx(r) for r in rands]
        self.beta = min(1., self.beta + self.beta_diff)
        return indices

    def sample(self, batch_size):
        assert self._cached is None, 'Update priorities before sampling.'

        self._cached = self._sample_idxes(batch_size)
        batch = self._sample(self._cached, batch_size)
        weights = self._calc_weights(self._cached)
        return batch, weights

    def _calc_weights(self, indices):
        min_pa = self.it_min.min()
        weights = [(self.it_sum[i] / min_pa) ** -self.beta for i in indices]
        return torch.FloatTensor(weights).to(self.device).view(-1, 1)

    def update_priority(self, errors):
        assert self._cached is not None

        ps = errors.detach().cpu().abs().numpy().flatten()
        pas = self._pa(ps)

        for index, pa in zip(self._cached, pas):
            assert 0 <= index < self._n
            assert 0 < pa
            self.it_sum[index] = pa
            self.it_min[index] = pa

        self._cached = None


class SegmentTree(object):

    def __init__(self, size, op, init_val):
        assert size > 0 and size & (size - 1) == 0
        self._size = size
        self._op = op
        self._init_val = init_val
        self._values = [init_val for _ in range(2 * size)]

    def _reduce(self, start=0, end=None):
        if end is None:
            end = self._size
        elif end < 0:
            end += self._size

        start += self._size
        end += self._size

        res = self._init_val
        while start < end:
            if start & 1:
                res = self._op(res, self._values[start])
                start += 1

            if end & 1:
                end -= 1
                res = self._op(res, self._values[end])

            start //= 2
            end //= 2

        return res

    def __setitem__(self, idx, val):
        assert 0 <= idx < self._size

        # Set value.
        idx += self._size
        self._values[idx] = val

        # Update its ancestors iteratively.
        idx = idx >> 1
        while idx >= 1:
            left = 2 * idx
            self._values[idx] = \
                self._op(self._values[left], self._values[left + 1])
            idx = idx >> 1

    def __getitem__(self, idx):
        assert 0 <= idx < self._size
        return self._values[idx + self._size]


class SumTree(SegmentTree):

    def __init__(self, size):
        super().__init__(size, operator.add, 0.0)

    def sum(self, start=0, end=None):
        return self._reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1

        # Traverse to the leaf.
        while idx < self._size:
            left = 2 * idx
            if self._values[left] > prefixsum:
                idx = left
            else:
                prefixsum -= self._values[left]
                idx = left + 1
        return idx - self._size


class MinTree(SegmentTree):

    def __init__(self, size):
        super().__init__(size, min, float("inf"))

    def min(self, start=0, end=None):
        return self._reduce(start, end)