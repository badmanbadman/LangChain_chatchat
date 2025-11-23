import asyncio
import signal
# asyncio 是python用于编写并发代码的库，使用async/await语法
# asyncio 通常用于异步I/O操作，例如网络请求，文件读写等，这里被用作管理异步任务，比如启动start_main_server 就是一个异步函数

# 1、事件循环管理
#获取事件循环
loop = asyncio.get_eventloop()          #获取当前的事件循环
loop = asyncio.get_running_loop()       #获取正在运行的事件循环
loop = asyncio.new_event_loop()         #创建新的事件循环
loop = asyncio.set_event_loop(loop)     #设置当前事件循环

#运行事件循环
# loop.run_until_complete(main_coroutine()) #运行直到协程完成
loop.run_forever()          #永久运行
loop.stop() #停止事件循环
loop.close() #关闭事件循环

#检查状态
loop.is_running() #是否正在运行
loop.is_colsed() #是否已关闭


# 2、协程创建和管理
# 协程装饰器创建协程
@asyncio.coroutine # 传统方式（python3.4+，已经不推荐）
def old_style_coroutine():
    yield from asyncio.sleep(1)
# 语法糖 创建协程
async def modern_coroutine(): #推荐方式（Python 3.5+）
    try:
        await asyncio.sleep(1)
    finally:
        ...

# 创建任务
task1 = asyncio.create_task(modern_coroutine()) #python 3.7+ 版本
task2 = loop.create_task(modern_coroutine()) #所有版本

# 确保协程运行
asyncio.ensure_future(modern_coroutine()) # 确保协程作为任务运行


# 3、任务管理和状态检测
async def example():
    # 创建任务
    task1 = asyncio.create_task(modern_coroutine())
    task2 = asyncio.create_task(modern_coroutine())

    # 任务状态
    task1.done() #是否完成
    task1.cancelled() # 是否被取消
    task1.result() #获取结果（如果完成）
    task1.exception() #获取异常（如果有）

    # 任务控制
    task1.cancel() #取消任务
    def callable_func():
        pass
    task1.add_done_callback(callable_func) #添加完成回调

    # 获取当前任务
    current_task = asyncio.current_task()
    all_tasks = asyncio.all_tasks()

# 4 并发执行
# 多个协程的并发管理
async def main():
    # 等待多个任务完成
    results = await asyncio.gather(
        modern_coroutine(),
        modern_coroutine2(),
        modern_coroutine3(),
        return_exceptions=True #将异常错位返回结果而不是抛出
    )

    # 等待第一个完成的任务
    done, pending = await asyncio.wait(
        [task1, task2, task3],
        timeout=5.0, #超时事件
        return_when=asyncio.FIRST_COMPLETED #第一个完成时返回
    ) 

    # 保护任务不被取消
    result = await asyncio.shield(import_task()) #即使取消也会被等待重要任务

# 5、时间控制
# 延时和超时管理
async def timing_examples():
    # 简单的延时
    await asyncio.sleep(1.5)

    # 带超时的操作
    try:
        result = await asyncio.wait_for(
            slow_operation(),
            timeout=3.0
        )
    except asyncio.TimeoutError:
        print('操作超时')
    
    # 超时返回默认值
    result = await asyncio.wait_for(
        slow_operation(),
        timeout=2.0,
        default='超时默认值'
    )

# 6、同步原语
# 线程安全的同步工具
async def synchronization():
    # 锁
    lock = asyncio.Lock()
    async with lock:
        # 临界区代码
        pass

    # 事件
    event = asyncio.Event()
    await event.wait() #等待事件被设置
    event.set()  #设置事件
    event.clear() #清除事件

    # 信号量(semaphore)
    semaphore = asyncio.Semaphore(5) #最大5个并发
    async with semaphore:
        # 受限制的并发代码
        pass

    # 条件变量
    condition = asyncio.Condition()
    async with condition:
        await condition.wait() #等待通知
        condition.notify_all() #通知所有等待者

# 7.队列管理
# 异步队列操作
async def queue_example():
    queue = asyncio.Queue(maxsize=10) #创建队列
    # 生产者
    await queue.put(item) # 放入项目（如果满则等待）

    # 消费者
    item = await queue.get() #获取项目（如果空则等待）
    queue.task_done() # 标记任务完成

    #非阻塞操作
    try:
        queue.put_nowait(item) #不等待直接放入
    except asyncio.QueueFull:
        pass

    try:
        item = queue.get_nowait() #不等待直接获取
    except asyncio.QueueEmpty:
        pass

    # 等待所有任务完成
    await queue.join() #阻塞直到所有任务标记完成

# 8、子进程管理
# 异步执行外部命令
async def subprocess_example():
    # 创建子进程
    process = await asyncio.create_subprocess_exec(
        'python', 'script.py',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    # 与子进程通信
    stdout,stderr = await process.communicate(input_data)

    # 等待完成
    returncode = await process.wait()

    # 发送信号
    process.terminate() #发送终止信号
    process.kill() #强制杀死进程

# 9. 网络操作
# 异步网络通信

async def network_operations():
    # TCP 客户端
    reader, writer = await asyncio.open_connection('host', port)
    writer.write(data)
    response = await reader.read(1024)
    writer.close()

    # TCP 服务器
    async def handle_client(reader, writer):
        data = await reader.read(100)
        writer.write(response)
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle_client, 'localhost', 888)
    async with server:
        await server.serve_forever()

# 10.在项目中的实际应用
async def start_main_server(args):
    # 1 信号处理 - 使用事件循环的  信号处理
    loop = asyncio.get_running_loop()

    for signal_name in {'SIGINT', 'SIGTERM'}:
        loop.add_signal_handler(
            getattr(signal, signal_name),
            signal_handler
        )
    
    # 2 进程状态监控 - 使用异步等待
    while processes:
        for p in processes.values():
            p.join(0.1) #非阻塞等待 
            if not p.is_alive():
                process.pop(p.name)
        await asyncio.sleep(0.1) # 关键: 让出控制权,允许其他任务运行

    #3. 超时控制
    try:
        result = await asyncio.wait_for(
            long_runing_operation(),
            timeout=30.0
        )
    except asyncio.TimeoutError:
        logger.error('操作超时')