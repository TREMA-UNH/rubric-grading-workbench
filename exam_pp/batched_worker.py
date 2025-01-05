from typing import TypeVar, Callable, Generic, NamedTuple, List, Optional
import asyncio


Input = TypeVar("Input")
Output = TypeVar("Output")


class WorkItem(Generic[Input, Output], NamedTuple):
    input: Input
    result: asyncio.Future[Output]


class BatchedWorker(Generic[Input, Output]):
    """
    Waits for 0.1 seconds to fill a batch, then executes it with `func`.
    """

    def __init__(self, func: Callable[[List[Input]], List[Output]], batch_size: int):
        self.work_queue = asyncio.Queue(maxsize=1000) # type: asyncio.Queue[WorkItem[Input, Output]]
        self.func = func
        self.batch_size = batch_size
        self.finished = False
        loop = asyncio.get_running_loop()
        loop.create_task(self._worker())


    async def run(self, input: Input) -> Output:
        '''
        Executes one work item
        '''
        print('running')
        loop = asyncio.get_running_loop()
        future = loop.create_future() # type: asyncio.Future[Output]
        work_item = WorkItem(input, future)
        assert not self.finished
        await self.work_queue.put(work_item)
        return await future


    async def _worker(self):
        while not self.finished:
            batch = []
            try:
                for i in range(self.batch_size):
                    async with asyncio.timeout(1.0):
                        work_item = None
                        try:
                            work_item = await self.work_queue.get()
                        except asyncio.CancelledError as e:
                            break

                        batch.append(work_item)

            except asyncio.TimeoutError:
                pass

            if batch:
                # print(f"batch len={len(batch)}")
                results = self.func([x.input for x in batch])
                for result, work_item in zip(results, batch):
                    work_item.result.set_result(result)


    def finish(self):
        self.finished = True