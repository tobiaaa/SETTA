import multiprocessing as mp
import resource
from multiprocessing import Manager, Queue
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import pandas as pd


class AsyncMetricManager:
    def __init__(self, metrics, num_workers=4):
        
        # HACK: Prevent "RuntimeError: received 0 items of ancdata"
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

        self.metrics = metrics

        mp_manager = Manager() 
        for metric in self.metrics:
            metric.data = mp_manager.list()
        
        self.input_queue = Queue(1024)
        self.result_queue = Queue()
        
        self.workers = [mp.Process(target=self._worker_fn,
                                   args=(self.input_queue, self.result_queue, self.metrics)) for _ in range(num_workers)]

        for w in self.workers:
            w.start()

        self._num_in = 0
        self.i = 0

        self.columns = ['Filename'] + sum([metric.names() for metric in self.metrics], [])

    def _worker_fn(self, input_queue, result_queue, metrics):
        while True:
            try:
                x_clean, x_noisy, x_denoised, filename, kwargs, i = input_queue.get(True, timeout=2.5)
            except Exception as e:
                continue
            results_dict = {}
            for metric in metrics:
                result = metric(x_clean, x_noisy, x_denoised, kwargs).squeeze()
                names = metric.names()
                if len(names) == 1:
                    results_dict[names[0]] = result.item()
                else:
                    for name, result in zip(names, result):
                        results_dict[name] = result.item()
            
            results_dict['Filename'] = filename
            results_dict['Index'] = i
            result_queue.put(results_dict)


    def update(self, x_clean, x_noisy, x_denoised, filename, kwargs=None):
        self.input_queue.put((x_clean.cpu(), 
                              x_noisy.cpu(), 
                              x_denoised.cpu(), 
                              filename,
                              kwargs,
                              self.i))
        self._num_in += 1
        self.i += 1


    def get_df(self):
        results = []

        while True:
            try:
                result = self.result_queue.get()
                results.append(result)
                if len(results) == self._num_in:
                    break
            except:
                raise RuntimeError('Could not get results')

        for w in self.workers:
            w.join(timeout=2.5)
            if w.is_alive():
                w.terminate()
                w.join()

        return pd.DataFrame(results, columns=self.columns)
