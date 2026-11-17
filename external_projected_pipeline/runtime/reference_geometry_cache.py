"""Content-versioned reference cache with explicit estimated-memory limits."""
from collections import OrderedDict
import hashlib


class CostAwareLRU:
    """A loader returns (value, estimated_bytes); budgets are per worker.

    Native BVH storage is estimated by the caller. This bounds retained
    estimates, not process RSS or a temporary oversized reference in use.
    """
    def __init__(self, max_entries=128, max_bytes=256*1024*1024):
        if any(not isinstance(v,int) or isinstance(v,bool) or v<1
               for v in (max_entries,max_bytes)):
            raise ValueError('Reference cache budgets must be positive integers.')
        self.max_entries,self.max_bytes=max_entries,max_bytes
        self.items=OrderedDict()
        self._tokens_by_key={}
        self.bytes=self.peak_bytes=self.loads=self.hits=self.evictions=0

    def __len__(self):return len(self.items)

    def get(self,key,version,loader):
        token=(key,version)
        if token in self.items:
            self.hits+=1;self.items.move_to_end(token)
            return self.items[token][0]
        old=self._tokens_by_key.pop(key,None)
        if old is not None:
            self.bytes-=self.items.pop(old)[1];self.evictions+=1
        value,cost=loader();self.loads+=1
        if not isinstance(cost,int) or isinstance(cost,bool) or cost<0:
            raise ValueError('Reference cost must be a nonnegative integer estimate.')
        if cost>self.max_bytes:return value
        while self.items and (len(self.items)>=self.max_entries or self.bytes+cost>self.max_bytes):
            expired,(_,previous)=self.items.popitem(last=False)
            del self._tokens_by_key[expired[0]]
            self.bytes-=previous;self.evictions+=1
        self.items[token]=(value,cost);self.bytes+=cost
        self._tokens_by_key[key]=token
        self.peak_bytes=max(self.peak_bytes,self.bytes)
        return value


def content_digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
