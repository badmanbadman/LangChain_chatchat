import threading
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, List, Tuple, Union, Generator

from langchain.embeddings.base import Embeddings
from langchain.vectorstores.faiss import FAISS

from chatchat.utils import build_logger


logger = build_logger()


# 基本的线程安全
class ThreadSafeObject:
    def __init__(
        self, 
        key: Union[str, Tuple],
        obj: Any = None, 
        pool: "CachePool" = None
    ):
        self._obj = obj
        self._key = key  # 目前key是传入元组当作key的
        self._pool = pool # 将实例化的时候穿进来的pool存起来
        self._lock = threading.RLock() # 初始化一个可重入锁
        self._loaded = threading.Event()  #初始状态位False
        """threading.Event 线程同步机制
        threading.Event()： 创建一个线程事件对象，用于在多个线程之间进行同步和通信
        这是Python多线程编程中非常重要的同步原语
            threading.Event是什么？
                一个简单的线程同步机制
                内部又一个布尔标志（True/False）
                线程可以等待这个标准被设置，或者设置/清除这个标志
            核心方法
                event  = threading.Event()

                # 主要方法’
                event.set()   #设置标志为True，唤醒所有等待的线程
                event.clear() #设置标志位False
                event.wait()  #阻塞直到标志被设置位True
                event.is_set()#检测当前标志状态
        """
    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"<{cls}: key: {self.key}, obj: {self._obj}>"

    @property
    def key(self):
        return self._key

    """
    @contextmanager是Python中的contextlib模块的一个装饰器，用于创建上下文管理器
    上下文管理器允许你实现__enter__和__exit__方法，但是使用@contextmanager可以通过生成器函数更加简洁的实现

    acquire这个方法被@contextmanager装饰意味着这个方法将会返回一个上下文管理器，允许使用with来管理资源的获取和释放
    具体来说这个acquire方法的作用是：
        1、在进入with块时，执行yield之前的代码，即获取锁等操作
        2、将yield后面的值（如果有，这里是有的，将_obj）,作为上下文管理器的__enter__方法的返回值，也就是 as 后面变量的值，
        3、在退出 with 块时，执行yield，之后的代码，如释放锁等

    """
    @contextmanager
    def acquire(self, owner: str = "", msg: str = "") -> Generator[None, None, FAISS]:
        owner = owner or f"thread {threading.get_native_id()}"
        try:
            self._lock.acquire() #获取锁
            if self._pool is not None: #如果存在缓存池
                # 将缓存池中的有序字典容器中的数据进行移动，将当前这个key的值移到最后
                self._pool._cache.move_to_end(self.key)
            logger.debug(f"{owner} 开始操作：{self.key}。{msg}")
            yield self._obj
        finally:
            logger.debug(f"{owner} 结束操作：{self.key}。{msg}")
            self._lock.release()

    def start_loading(self):
        self._loaded.clear() #设置位False，表示 "正在加载"

    def finish_loading(self):
        self._loaded.set() #设置位True，表示"加载完成"

    def wait_for_loading(self):
        self._loaded.wait() # 阻塞直到加载完成

    @property
    def obj(self):
        return self._obj

    @obj.setter
    def obj(self, val: Any):
        self._obj = val


class CachePool:
    def __init__(self, cache_num: int = -1):
        self._cache_num = cache_num
        # 创建一个有序字典来作为缓存存储容器
        self._cache = OrderedDict()
        """有序字典
        OrderedDict的特点：
            1、保持插入顺序
                od = OrderedDict()
                od['a'] = 1
                od['b'] = 2
                od['c'] = 3
                print(list(od.key())) # ['a','b','c']    保持了插入时候的顺序
            2、在缓存系统中的关键作用
                在CachePool中OrderedDict用于实现LRU（最近最少使用），缓存淘汰策略
        """
        # 初始化一个可重入锁
        self.atomic = threading.RLock()
        """可重入锁
        可重入锁：是一种特殊类型的锁，它允许同一个线程多次获取同一个锁而不会造成死锁
        与普通的Lock的区别：
            普通Lock：
                import threading
                lock = threading.Lock()

                def test_lock():
                    with lock:
                        print('第一次获取锁')
                        with lock # 这里会死锁
                            print('第二次获取锁') #这里永远不会执行
                test_lock() # 调用会死锁  
            可重入锁：
                import threading
                rlock = threading.RLock()
                def test_rlock():
                    with rlock:
                        print('第一次获取锁')
                        with rlock: # 同一个线程可以再次获取
                            print('第二次获取锁') # 正常执行
                test_rlock() #z正常执行
        """

    def keys(self) -> List[str]:
        return list(self._cache.keys())

    def _check_count(self):
        if isinstance(self._cache_num, int) and self._cache_num > 0:
            while len(self._cache) > self._cache_num:
                # 删除最久未使用的项（OrderedDict的第一个：【由于对有序字典来说刚插入的放在字典的最后面的，第一个就是最早插入的，也就是最长时间没有使用的】）
                self._cache.popitem(last=False)

    def get(self, key: str) -> ThreadSafeObject:
        """海象运算符
        if cache := self._cache.get(key):
        这里使用了Python的海象运算符， :=  这是Python3.8引入的语法特性
        等价于：
            cache = self._cache.get(key)
            if cache:
                cache.wait_for_loading()
                return cache
        """
        if cache := self._cache.get(key):
            # wait_for_loading() 是一个线程同步机制，用于确保缓存对象在完成初始化前不会被使用，
            cache.wait_for_loading()
            # 注意：这里么有移动顺序，因为只是读取
            return cache

    def set(self, key: str, obj: ThreadSafeObject) -> ThreadSafeObject:
        self._cache[key] = obj #新插入的项在末尾
        self._check_count()
        return obj

    def pop(self, key: str = None) -> ThreadSafeObject:
        if key is None:
            return self._cache.popitem(last=False)
        else:
            return self._cache.pop(key, None)

    def acquire(self, key: Union[str, Tuple], owner: str = "", msg: str = ""):
        cache = self.get(key)
        if cache is None:
            raise RuntimeError(f"请求的资源 {key} 不存在")
        elif isinstance(cache, ThreadSafeObject):
            self._cache.move_to_end(key)
            return cache.acquire(owner=owner, msg=msg)
        else:
            return cache
