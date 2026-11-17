"""Own only the child process family created by one portable API execution."""
import os
import signal
import subprocess
from contextlib import ContextDecorator


class ActiveCompute(ContextDecorator):
    """Prevent automatic idle sleep during a job, restoring the calling thread.

    Does not keep the display on, change the power plan, or override manual
    sleep/lid actions. The requirement is scoped to the owning Python thread.
    """
    def _recreate_cm(self):
        return type(self)()

    def __enter__(self):
        self.previous = None
        self.kernel = None
        self.active = False
        if os.name == 'nt':
            import ctypes
            self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            self.kernel.SetThreadExecutionState.argtypes = [ctypes.c_uint]
            self.kernel.SetThreadExecutionState.restype = ctypes.c_uint
            self.previous = self.kernel.SetThreadExecutionState(0x80000000 | 0x00000001)
            self.active = bool(self.previous)
        return self

    def __exit__(self, *exception):
        if self.kernel and self.active:
            self.kernel.SetThreadExecutionState(self.previous)
            self.active = False
        return False


class ProcessTree:
    def __init__(self,process):
        self.process=process;self.handle=None;self.kernel=None
        self._posix_stop_sent=False;self._posix_stop_error=None

    def __enter__(self):
        if os.name=='nt':
            import ctypes
            from ctypes import wintypes as W
            self.kernel=ctypes.WinDLL('kernel32',use_last_error=True)
            self.kernel.CreateJobObjectW.argtypes=[W.LPVOID,W.LPCWSTR]
            self.kernel.CreateJobObjectW.restype=W.HANDLE
            self.kernel.AssignProcessToJobObject.argtypes=[W.HANDLE,W.HANDLE]
            self.kernel.TerminateJobObject.argtypes=[W.HANDLE,W.UINT]
            self.kernel.CloseHandle.argtypes=[W.HANDLE]
            self.handle=self.kernel.CreateJobObjectW(None,None)
            if self.handle and not self.kernel.AssignProcessToJobObject(self.handle,int(self.process._handle)):
                self.kernel.CloseHandle(self.handle);self.handle=None
        return self

    def stop(self):
        if os.name!='nt':
            self._stop_posix()
            return
        if self.handle and self.kernel.TerminateJobObject(self.handle,1):return
        if os.name=='nt':
            try:
                subprocess.run(['taskkill','/PID',str(self.process.pid),'/T','/F'],capture_output=True,
                    timeout=3,creationflags=subprocess.CREATE_NO_WINDOW)
            except (OSError,subprocess.TimeoutExpired):pass
        if self.process.poll() is None:self.process.kill()

    def _stop_posix(self):
        # SIGKILL is asynchronous. The caller may ask again before wait()/poll()
        # observes exit. Never signal the same (potentially recycled) group twice.
        if self._posix_stop_sent:
            return
        if self._posix_stop_error is not None:
            raise RuntimeError(self._posix_stop_error)
        pid = self.process.pid
        if pid <= 0 or pid == os.getpgrp():
            raise RuntimeError('Refusing to signal an invalid or caller-owned process group.')
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            try:
                if self.process.poll() is None:
                    self.process.kill()
            except ProcessLookupError:
                pass
        except PermissionError as error:
            details = {'pid': pid, 'returncode': self.process.poll(), 'caller_group': os.getpgrp()}
            for name, getter in (('child_group', os.getpgid), ('child_session', os.getsid)):
                try:
                    details[name] = getter(pid)
                except OSError as probe_error:
                    details[name] = f'{type(probe_error).__name__}: {probe_error}'
            self._posix_stop_error = (
                f'Cancellation could not signal the owned process group: {details}. '
                'Worker/descendant termination is unverified; the permission failure was not ignored.')
            raise RuntimeError(self._posix_stop_error) from error
        self._posix_stop_sent = True

    def __exit__(self,*exception):
        if self.handle:
            self.kernel.TerminateJobObject(self.handle,1)
            self.kernel.CloseHandle(self.handle);self.handle=None
