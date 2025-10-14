import pandas as pd


class MetricManager:
    def __init__(self, metrics):
        self.metrics = metrics
        self.results = []
        self.columns = ['Filename'] + sum([metric.names() for metric in self.metrics], [])
        self.i = 0

    def update(self, x_clean, x_noisy, x_denoised, filename, kwargs=None):
        results_dict = {}
        for metric in self.metrics:
            result = metric(x_clean, x_noisy, x_denoised, kwargs).squeeze()
            names = metric.names()
            if len(names) == 1:
                results_dict[names[0]] = result.item()
            else:
                for name, result in zip(names, result):
                    results_dict[name] = result.item()
        
        results_dict['Filename'] = filename
        results_dict['Index'] = self.i
        self.results.append(results_dict)
        self.i += 1

    def get_df(self):
        return pd.DataFrame(self.results, columns=self.columns)
