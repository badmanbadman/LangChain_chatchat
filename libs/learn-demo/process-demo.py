import multiprocessing as mp
import os
import time
# 多进程 vs 多线程
#  关键区别
#     多进程: 每个进程有独立的内存空间,真正的并行(利用多核CPU)
#     多线程: 共享内存空间,受GIL限制,伪并行
# 2.1 Process 类 - 进程管理====================================
def worker(name, number):
    """工作进程函数"""
    print(f"进程{name}(PID:{os.getpid()}处理数字: {number})")
    return number *number
def process_demo():
    # 创建进程
    process1 = mp.Process(target=worker, args=("Worker-1",5))
    process2 = mp.Process(target=worker, args=('worker-2', 5))

    # 启动进程
    process1.start()
    process2.start()

    # 等待进程完成
    process1.join()
    process2.join()
    print('所有进程完成')
# 2.2 Pool - 进程池=======================================================
def square(x):
    return x * x
def pool_demo():
    with mp.Pool(processes=4) as pool:
        # 多种执行方式
        results = pool.map(square, range(10)) # 顺序执行
        results_async = pool.map_async(square, range(10)) # 异步执行
        result_imap =pool.imap(square, range(10)) # 惰性迭代

    # 获取结果
    print('map结果: ',results)
    print('map_async结果', results_async)
    print('imap结果', list(result_imap))


    # 使用apply系列
    result1 = pool.apply(square, (5,)) #同步
    result2 = pool.apply_async(square, (10,)) #异步
    print(f"apply结果: {result1}, apply_async结果: {result2.get()}")

# 3 进程间通信(IPC)
# 3.1 Queue-队列通信==================================
def producer(queue,items):
    """生产者进程"""
    for item in items:
        print(f"生产:{item}")
        queue.put(item)
        time.sleep(0.1)
    queue.put(None) #结束信号

def consumer(queue):
    """消费者进程"""
    while True:
        item = queue.get()
        if item is None: #结束信号
            break
        print(f"消费: {item}")
        time.sleep(0.2)
def queue_demo():
    queue = mp.Queue()

    # 创建进程
    prod = mp.Process(target=producer, args=(queue, ['A', 'B', 'C', 'D']))
    cons = mp.Process(target=consumer, args=(queue,))
    
    # 启动进程
    prod.start()
    cons.start()

    # 等待完成
    prod.join()
    cons.join()

# 3.2 Pipe - 管道通信===================================
def sender(conn, messages):
    """发送者进程"""
    for message in messages:
        print(f"发送:{message}")
        conn.send(message)
    conn.close()

def receiver(conn):
    """接收者进程"""
    while True:
        try:
            message = conn.recv()
            print(f"接收:{message}")
        except EOFError:
            break
def pipe_demo():
    # 创建管道
    parent_conn, child_conn = mp.Pipe()
    # 创建进程
    p1 = mp.Process(target=sender,args=(child_conn, ['Hello', "world",'!']))
    p2 = mp.Process(target=receiver, args=(parent_conn,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

# 3.3 共享内存
def worker_share(shared_value, shared_array, lock):
    """使用共享内存的工作进程"""
    with lock: #加锁保护共享数据
        shared_value.value +=1
        for i in range(len(shared_array)):
            shared_array[i]+=i
def worker_share_demo():
    # 创建共享数据
    shared_value = mp.Value('i', 0)  # 'i' 表示整数类型
    shared_array = mp.Array('d', [1.0, 2.0, 3.0])  # 'd' 表示双精度浮点数
    lock = mp.Lock()
    
    processes = []
    for i in range(3):
        p = mp.Process(target=worker, args=(shared_value, shared_array, lock))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
    print(f"共享值: {shared_value.value}")
    print(f"共享数组: {list(shared_array)}")

# 4 同步原语===================================
def worker_with_lock(lock, id):
    """使用锁的工人进程"""
    with lock:
        print(f'进程{id}获得锁')
        time.sleep(1)
        print(f'进程{id}释放锁')

def worker_with_event(event, id):
    """使用事件的工人进程"""
    print(f"进程{id}等待事件")
    event.wait() # 等待事件被设置
    print(f"进程{id}检测到事件")

def worker_with_semaphore(sem, id):
    """使用信号量的工人进程"""
    with sem:
        print(f"进程{id}获得信号量")
        time.sleep(2)
        print(f"进程{id}释放信号量")
def worker_lock_event_semaphore_demo():
    # 锁示例
    lock = mp.Lock()
    for i in range(3):
        mp.Process(target=worker_with_lock, args=(lock,i)).start()
    time.sleep(3)

    # 事件示例
    event = mp.Event()
    for i in range(3):
        mp.Process(target=worker_with_event, args=(event,i)).start()
    
    time.sleep(1)
    print('设置事件')
    event.set() # 唤醒所有等待的进程

    time.sleep(1)

    # 信号量示例
    sem = mp.Semaphore(2)
    for i in range(5):
        mp.Process(target=worker_with_semaphore, args=(sem, i)).start()

# Manager - 管理共享状态=================================================
def worker_use_share(share_dict, shared_list, id):
    """使用Manage共享数据的工作进程"""
    share_dict[id]=f"velue_{id}"
    shared_list.append(id*id)
    print(f"进程{id}完成工作")
def worker_use_share_demo():
    with mp.Manager() as manager:
        # 创建共享数据结构
        share_dict = manager.dict()
        share_list = manager.list()

        processes = []
        for i in range(4):
            p = mp.Process(target=worker_use_share, args=(share_dict, share_list, i))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        print(f"共享字典:{dict(share_dict)}")
        print(f"共享列表:{list(share_list)}")

def func():
    pass
# def demo_base():
#     process = mp.Process(target=func)
#     process.start() #启动进程
#     process.join()# 等待进程结束
#     process.terminate() #终止进程
#     process.kill() #  强制杀死进程
#     process.is_alive() #检查是否存活
#     process.pid#进程id
#     process.name #进程名称
#     process.daemon #是否未守护进程

#     with mp.Pool(4) as pool:
#         pool.map(func,items) #并行映射
#         pool.apply(func, args) #同步执行
#         pool.apply_async(func, args) #异步执行
#         pool.close() #关闭池,不再接受新任务
#         pool.terminate() #立即终止
#         pool.join() #等待所有工作进程退出
def api_service(started_event):
    """API 服务"""
    print(f"🕒 API进程启动 (PID: {os.getpid()})")
    print("   API: 开始初始化...")
    
    # 模拟 API 启动需要 3 秒
    time.sleep(3)
    
    print("   🔥 API: 调用 started_event.set()!")
    started_event.set()  # ⭐ 关键：这里触发事件
    print("   ✅ API: 启动完成通知已发送")
    
    # 保持运行
    time.sleep(5)
    print("   API: 进程结束")

def webui_service(started_event):
    """WebUI 服务"""
    print(f"🕒 WebUI进程启动 (PID: {os.getpid()})")
    print("   WebUI: 开始初始化...")
    
    # 模拟 WebUI 启动需要 2 秒
    time.sleep(2)
    
    print("   🔥 WebUI: 调用 started_event.set()!")
    started_event.set()  # ⭐ 关键：这里触发事件
    print("   ✅ WebUI: 启动完成通知已发送")
    
    # 保持运行
    time.sleep(5)
    print("   WebUI: 进程结束")

def demonstrate_execution_timing():
    """演示代码执行时机"""
    
    print("=== 代码执行时机分析 ===")
    
    manager = mp.Manager()
    api_started = manager.Event()
    webui_started = manager.Event()
    
    processes = {
        "api": mp.Process(target=api_service, args=(api_started,)),
        "webui": mp.Process(target=webui_service, args=(webui_started,))
    }
    
    print("🚀 主进程开始执行:")
    print("   即将执行: if p := processes.get('api')")
    
    # 第一段代码
    if p := processes.get("api"):
        print("   ✅ 找到 API 进程，执行 p.start()")
        p.start()  # ⭐ API 进程开始运行
        print("   ✅ 设置进程名称")
        p.name = f"{p.name} ({p.pid})"
        print("   ⏳ 执行 api_started.wait() - 主进程在此阻塞!")
        
        # ⭐⭐⭐ 关键：这里主进程会阻塞，直到 API 进程调用 api_started.set()
        api_started.wait()
        
        print("   🔓 api_started.wait() 返回，主进程继续执行")
    
    print("\n   主进程继续执行下一行代码")
    print("   即将执行: if p := processes.get('webui')")
    
    # 第二段代码
    if p := processes.get("webui"):
        print("   ✅ 找到 WebUI 进程，执行 p.start()")
        p.start()  # ⭐ WebUI 进程现在才开始运行
        print("   ✅ 设置进程名称") 
        p.name = f"{p.name} ({p.pid})"
        print("   ⏳ 执行 webui_started.wait() - 主进程再次阻塞!")
        
        # ⭐⭐⭐ 关键：这里主进程再次阻塞，直到 WebUI 进程调用 webui_started.set()
        webui_started.wait()
        
        print("   🔓 webui_started.wait() 返回，主进程继续执行")
    
    print("🎉 所有服务启动完成，主进程继续后续工作")
    
    # 等待进程结束
    for p in processes.values():
        p.join()

if __name__ == "__main__":
    # 进程
    # process_demo()

    # 进程池
    # pool_demo()

    # 队列
    # queue_demo()

    # 管道
    # pipe_demo()

    # 内存共享
    # worker_share_demo()

    #  同步原语
    # worker_lock_event_semaphore_demo()

    # Manager 管理共享状态
    # worker_use_share_demo()

    # 本项目流程运行demo
    demonstrate_execution_timing()
    pass