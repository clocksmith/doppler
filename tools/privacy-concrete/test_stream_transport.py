"""Test-only pipe transport; real provider framing, no SSH or host provisioning."""
import os
import threading


class PipeProvider:
    def __init__(self,plan,directory):
        from full_provider import evaluate_route
        incoming,writer=os.pipe();reader,outgoing=os.pipe()
        self.stdin=os.fdopen(writer,'wb',buffering=0);self.stdout=os.fdopen(reader,'rb',buffering=0)
        self.error=None
        def serve():
            try:
                with os.fdopen(incoming,'rb',buffering=0) as source,os.fdopen(outgoing,'wb',buffering=0) as target:
                    evaluate_route(plan,directory,source,target)
            except BaseException as error:self.error=error
        self.thread=threading.Thread(target=serve,daemon=True);self.thread.start()

    def poll(self):return None if self.thread.is_alive() else (1 if self.error else 0)

    def wait(self,timeout=180):
        self.thread.join(timeout)
        if self.thread.is_alive():raise TimeoutError('Fixture provider did not exit')
        return 1 if self.error else 0

    def kill(self):
        if not self.stdin.closed:self.stdin.close()
