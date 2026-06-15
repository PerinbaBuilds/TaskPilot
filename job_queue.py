from collections import deque


class JobQueue:
    def __init__(self):
        self.queue  = deque()
        self.job_id = 1

    def add(self, job: dict) -> int:
        job["id"] = self.job_id
        self.job_id += 1
        self.queue.append(job)
        return job["id"]

    def pop(self) -> dict | None:
        return self.queue.popleft() if self.queue else None

    def all(self) -> list:
        return list(self.queue)

    def __len__(self) -> int:
        return len(self.queue)
