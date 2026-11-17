"""OS-released advisory leases preventing concurrent writes to the same output."""
from contextlib import ExitStack,contextmanager
import json
import os
from pathlib import Path


class OutputLock:
    def __init__(self,output):
        self.output=Path(output).resolve()
        self.path=self.output.parent/('.'+self.output.name+'.projected_pipeline.lock')
        self.stream=None

    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.stream=os.fdopen(os.open(self.path,os.O_CREAT|os.O_RDWR,0o600),'r+b')
        try:
            if self.path.stat().st_size==0:self.stream.write(b' ');self.stream.flush()
            self.stream.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close();self.stream=None
            raise RuntimeError(f'Another pipeline run is writing {self.output}. Wait for it or choose a different output folder.') from exc
        try:
            self.stream.seek(1)
            self.stream.write(json.dumps(dict(pid=os.getpid(),output=str(self.output))).encode('utf8'))
            self.stream.truncate();self.stream.flush()
        except BaseException:
            self.stream.close();self.stream=None
            raise
        return self

    def __exit__(self,*exception):
        if self.stream is not None:
            try:
                self.stream.seek(0)
                if os.name=='nt':
                    import msvcrt
                    msvcrt.locking(self.stream.fileno(),msvcrt.LK_UNLCK,1)
                else:
                    import fcntl
                    fcntl.flock(self.stream.fileno(),fcntl.LOCK_UN)
            finally:self.stream.close();self.stream=None
        # Keep the small unlocked file: unlinking a lease inode can race another
        # acquirer on POSIX. OS locks are released even if the process crashes.


@contextmanager
def lock_outputs(outputs):
    with ExitStack() as stack:
        for output in sorted({Path(value).resolve() for value in outputs}):stack.enter_context(OutputLock(output))
        yield
