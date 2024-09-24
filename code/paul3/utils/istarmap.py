import multiprocessing.pool as mpp


# Credit: https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm
def istarmap(self, func, iterable, chunk_size=1):
    self._check_running()
    assert chunk_size >= 1

    task_batches = mpp.Pool._get_tasks(func, iterable, chunk_size)
    result = mpp.IMapIterator(self)
    self._taskqueue.put((self._guarded_task_generation(result._job,
                                                       mpp.starmapstar,
                                                       task_batches),
                         result._set_length))
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap
