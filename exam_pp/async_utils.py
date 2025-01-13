import asyncio
from typing import Iterator, TypeVar, Awaitable, Callable, Generator

try:
    from asyncio import QueueShutDown, Queue
except ImportError:
    from backports.asyncio.queues import Queue, QueueShutDown


A = TypeVar('A')
B = TypeVar('B')


class RateLimit:
    def __init__(self, rate: float):
        self.rate = rate
        self._sem = asyncio.Semaphore(1)

    async def limit(self):
        async def trigger():
            await asyncio.sleep(1. / self.rate)
            self.sem.release()

        await self.sem.acquire()
        asyncio.create_task(trigger())


class RateLimit2:
    def __init__(self, rate: float, dt: float=1):
        """
        rate given in events per second.
        dt is the integration period given in seconds.
        """
        self.rate = rate
        self.dt = dt
        self._count = 0
        self._lock = asyncio.Lock()
        self._cond = asyncio.Condition(self._lock)
        self._worker_task = asyncio.get_event_loop().create_task(self._worker())

    async def _worker(self):
        while True:
            async with self._lock:
                self._count = self.rate
                self._cond.notify_all()

            await asyncio.sleep(self.dt)

    async def limit(self, n: int=1):
        async with self._lock:
            await self._cond.wait_for(lambda: self._count >= n)
            self._count -= n


async def map_concurrently(
        func: Callable[A, Awaitable[B]],
        xs: Iterator[A],
        n_workers: int) -> Iterator[B]:
    work_queue: Queue[Tuple[A, Future[B]]] = Queue(maxsize=4*n_workers)
    result_queue: Queue[Future[B]] = Queue(maxsize=4*n_workers)
    loop = asyncio.get_running_loop()

    async def worker() -> None:
        while True:
            try:
                if errors:
                    break
                x, fut = await work_queue.get()
                y = await func(x)
                fut.set_result(y)
                work_queue.task_done()
            except asyncio.CancelledError as e:
                raise e
            except asyncio.QueueShutDown as e:
                break
            except Exception as e:
                work_queue.shutdown(immediate=True)
                raise e

    async def pusher() -> None:
        for x in xs:
            fut = loop.create_future()
            await work_queue.put((x, fut))
            await result_queue.put(fut)

        work_queue.shutdown()
        result_queue.shutdown()

    async with asyncio.TaskGroup() as tg:
        tg.create_task(pusher())
        workers = [
            tg.create_task(worker(), name=f'worker {n}')
            for n in range(n_workers) ]

        while True:
            try:
                fut = await result_queue.get()
            except QueueShutDown:
                break

            y = await fut
            yield y

        for task in workers:
            task.cancel()


async def apply_concurrently(func: Callable[A, Awaitable[None]], xs: Iterator[A], n_workers: int):
    queue: Queue[A] = Queue(maxsize=4*n_workers)

    async def worker():
        while True:
            try:
                if errors:
                    break
                x = await queue.get()
                await func(x)
                queue.task_done()
            except asyncio.CancelledError as e:
                raise e
            except QueueShutDown as e:
                return
            except Exception as e:
                queue.shutdown(immediate=True)
                raise e

    async with asyncio.TaskGroup() as tg:
        workers = [
            tg.create_task(worker(), name=f'worker {n}')
            for n in range(n_workers) ]

        try:
            for x in xs:
                await queue.put(x)
        except QueueShutDown as e:
            pass
        else:
            queue.shutdown()

        for task in workers:
            task.cancel()


async def simple():
    async def worker(i):
        print(i)
        await asyncio.sleep(1)

    async with asyncio.TaskGroup() as tg:
        for i in range(100):
            tg.create_task(worker(i))


async def test_map_concurrently() -> None:
    limiter = RateLimit2(rate=10)
    async def do_it(x):
        await limiter.limit()
        return 2*x

    async for n in map_concurrently(do_it, list(range(100)), n_workers=10):
        print(n)

    print('done')


async def test_apply_concurrently() -> None:
    async def do_it(x):
        print(x)
        await asyncio.sleep(1)

    await apply_concurrently(do_it, list(range(100)), n_workers=10)
    print('done')


async def main() -> None:
    await test_map_concurrently()
    #await simple()


if __name__ == '__main__':
    asyncio.run(main())

