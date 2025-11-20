import asyncio
"""
asyncio 是python用于编写并发代码的库，使用async/await语法
asyncio 通常用于异步I/O操作，例如网络请求，文件读写等，这里被用作管理异步任务，比如启动start_main_server 就是一个异步函数

#获取事件循环
loop = asyncio.get_eventloop()          #获取当前的事件循环
loop = asyncio.get_running_loop()       #获取正在运行的事件循环
loop = asyncio.new_event_loop()         #创建新的事件循环
loop = asyncio.set_event_loop(loop)     #设置当前事件循环

#运行事件循环
loop.run_until_complete(main_coroutine()) #运行直到协程完成
loop.run_forever()          #永久运行
loop.stop() #停止事件循环
loop.close() #关闭事件循环

#检查状态
loop.is_running() #是否正在运行
loop.is_colsed() #是否已关闭
"""
import logging
import logging.config
"""
对系统日志的一些配置,如日志存储路径等
"""
import multiprocessing as mp
"""
multiprocessing 是python的多进程模块,这个模块允许你创建并管理多个进程,从而利用多核CPU来执行并行计算
多进程模块的主要用途
1 利用多核CPU: 通过创建多个进程,可以将工作负载分布到多个CPU核心上,从而加快计算速度
2 避免全局解释器(GIL)的限制: 由于每个进程有自己独立的Python解释器核内存空间,因此可以绕过GIL,实现真正的并行执行
3 提高程序的稳定性和容错性: 一个进程的崩溃不会直接影响其他进程

多进程 vs 多线程
 关键区别
    多进程: 每个进程有独立的内存空间,真正的并行(利用多核CPU)
    多线程: 共享内存空间,受GIL限制,伪并行
"""
import os
import sys
from contextlib import asynccontextmanager
"""异步上下文管理器装饰器,用于创建支持 async with语法的上下文管理器
asynccontextmanager 用于创建异步上下文管理器的装饰器,这个装饰器允许我们使用async with 语法来管理异步资源的获取和释放
在FastAPI中,asynccontextmanager常用于定义应用的生命周期(如启动和关闭),通过它,我们可以定义一个异步生成器,
其中在生成器开始的时候执行启动代码(如连接数据库), 在生成器结束的时候执行清理代码(如关闭数据库连接)
"""
from multiprocessing import Process
"""进程"""

# 设置numexpr最大线程数，默认为CPU核心数
try:
    import numexpr
    """高效数值表达式计算"""
    # detect_number_of_cores是numexpr自带的检测系统cpu核心数的函数
    n_cores = numexpr.utils.detect_number_of_cores()
    # 设置环境变量NUMEXPR_MAX_THREADS 未最大cpu核心数
    os.environ["NUMEXPR_MAX_THREADS"] = str(n_cores)
except:
    pass

import click
from typing import Dict, List

from fastapi import FastAPI

from chatchat.utils import build_logger


logger = build_logger()
"""日志的相关打印存储等"""

def _set_app_event(app: FastAPI, started_event: mp.Event = None):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if started_event is not None:
            started_event.set()
        yield
    # lifespan_context 是FastAPI应用的生命周期管理器
    # 控制应用的启动和关闭过程,
    # 生命周期阶段
    #   启动阶段: 在yield之前执行,初始化资源,数据库连接,加载模型等
    #   运行阶段: 在yield期间,应用正常处理请求
    #   关闭阶段: 在yield之后执行,清理资源,关闭连接,保存状态等
    app.router.lifespan_context = lifespan


def run_api_server(
    started_event: mp.Event = None, run_mode: str = None
):
    import uvicorn
    from chatchat.utils import (
        get_config_dict,
        get_log_file,
        get_timestamp_ms,
    )

    from chatchat.settings import Settings
    from chatchat.server.api_server.server_app import create_app
    from chatchat.server.utils import set_httpx_config

    logger.info(f"Api MODEL_PLATFORMS: {Settings.model_settings.MODEL_PLATFORMS}")
    set_httpx_config()
    app = create_app(run_mode=run_mode)
    _set_app_event(app, started_event)

    host = Settings.basic_settings.API_SERVER["host"]
    port = Settings.basic_settings.API_SERVER["port"]

    logging_conf = get_config_dict(
        "INFO",
        get_log_file(log_path=Settings.basic_settings.LOG_PATH, sub_dir=f"run_api_server_{get_timestamp_ms()}"),
        1024 * 1024 * 1024 * 3,
        1024 * 1024 * 1024 * 3,
    )
    logging.config.dictConfig(logging_conf)  # type: ignore
    uvicorn.run(app, host=host, port=port)


def run_webui(
    started_event: mp.Event = None, run_mode: str = None
):
    from chatchat.settings import Settings
    from chatchat.server.utils import set_httpx_config
    from chatchat.utils import get_config_dict, get_log_file, get_timestamp_ms

    logger.info(f"Webui MODEL_PLATFORMS: {Settings.model_settings.MODEL_PLATFORMS}")
    set_httpx_config()

    host = Settings.basic_settings.WEBUI_SERVER["host"]
    port = Settings.basic_settings.WEBUI_SERVER["port"]

    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui.py")

    flag_options = {
        "server_address": host,
        "server_port": port,
        "theme_base": "light",
        "theme_primaryColor": "#165dff",
        "theme_secondaryBackgroundColor": "#f5f5f5",
        "theme_textColor": "#000000",
        "global_disableWatchdogWarning": None,
        "global_disableWidgetStateDuplicationWarning": None,
        "global_showWarningOnDirectExecution": None,
        "global_developmentMode": None,
        "global_logLevel": None,
        "global_unitTest": None,
        "global_suppressDeprecationWarnings": None,
        "global_minCachedMessageSize": None,
        "global_maxCachedMessageAge": None,
        "global_storeCachedForwardMessagesInMemory": None,
        "global_dataFrameSerialization": None,
        "logger_level": None,
        "logger_messageFormat": None,
        "logger_enableRich": None,
        "client_caching": None,
        "client_displayEnabled": None,
        "client_showErrorDetails": None,
        "client_toolbarMode": None,
        "client_showSidebarNavigation": None,
        "runner_magicEnabled": None,
        "runner_installTracer": None,
        "runner_fixMatplotlib": None,
        "runner_postScriptGC": None,
        "runner_fastReruns": None,
        "runner_enforceSerializableSessionState": None,
        "runner_enumCoercion": None,
        "server_folderWatchBlacklist": None,
        "server_fileWatcherType": "none",
        "server_headless": None,
        "server_runOnSave": None,
        "server_allowRunOnSave": None,
        "server_scriptHealthCheckEnabled": None,
        "server_baseUrlPath": None,
        "server_enableCORS": None,
        "server_enableXsrfProtection": None,
        "server_maxUploadSize": None,
        "server_maxMessageSize": None,
        "server_enableArrowTruncation": None,
        "server_enableWebsocketCompression": None,
        "server_enableStaticServing": None,
        "browser_serverAddress": None,
        "browser_gatherUsageStats": None,
        "browser_serverPort": None,
        "server_sslCertFile": None,
        "server_sslKeyFile": None,
        "ui_hideTopBar": None,
        "ui_hideSidebarNav": None,
        "magic_displayRootDocString": None,
        "magic_displayLastExprIfNoSemicolon": None,
        "deprecation_showfileUploaderEncoding": None,
        "deprecation_showImageFormat": None,
        "deprecation_showPyplotGlobalUse": None,
        "theme_backgroundColor": None,
        "theme_font": None,
    }

    args = []
    if run_mode == "lite":
        args += [
            "--",
            "lite",
        ]

    try:
        # for streamlit >= 1.12.1
        from streamlit.web import bootstrap
    except ImportError:
        from streamlit import bootstrap

    logging_conf = get_config_dict(
        "INFO",
        get_log_file(log_path=Settings.basic_settings.LOG_PATH, sub_dir=f"run_webui_{get_timestamp_ms()}"),
        1024 * 1024 * 1024 * 3,
        1024 * 1024 * 1024 * 3,
    )
    logging.config.dictConfig(logging_conf)  # type: ignore
    bootstrap.load_config_options(flag_options=flag_options)
    bootstrap.run(script_dir, False, args, flag_options)
    # 启动完毕唤醒其他等待的进程
    started_event.set()


def dump_server_info(after_start=False, args=None):
    import platform

    import langchain

    from chatchat import __version__
    from chatchat.settings import Settings
    from chatchat.server.utils import api_address, webui_address

    print("\n")
    print("=" * 30 + "Langchain-Chatchat Configuration" + "=" * 30)
    print(f"操作系统：{platform.platform()}.")
    print(f"python版本：{sys.version}")
    print(f"项目版本：{__version__}")
    print(f"langchain版本：{langchain.__version__}")
    print(f"数据目录：{Settings.CHATCHAT_ROOT}")
    print("\n")

    print(f"当前使用的分词器：{Settings.kb_settings.TEXT_SPLITTER_NAME}")

    print(f"默认选用的 Embedding 名称： {Settings.model_settings.DEFAULT_EMBEDDING_MODEL}")

    if after_start:
        print("\n")
        print(f"服务端运行信息：")
        if args.api:
            print(f"    Chatchat Api Server: {api_address()}")
        if args.webui:
            print(f"    Chatchat WEBUI Server: {webui_address()}")
    print("=" * 30 + "Langchain-Chatchat Configuration" + "=" * 30)
    print("\n")


async def start_main_server(args): 
    import signal
    """
    Python 的信号处理模块,它允许程序响应操作系统发出的信号,信号是操作系统异步通知进程的一种机制,用于通知进程发生了某种事件.
    例如,当用户按下crtl + c时,通常会发送一个SIGINT(中断信号)给进程,导致进程终止

    在python中,signal模块提供了处理信号的机制,允许我们注册信号处理函数,以便在受到特定信号时执行自定义操作,而不是默认行为
    signal.SIGINT：中断信号，通常由 Ctrl+C 产生，默认行为是终止进程。
    signal.SIGTERM：终止信号，通常由系统关机或 kill 命令（不带 -9）产生，默认行为是终止进程。
    signal.SIGKILL：立即终止信号，无法被捕获、阻塞或忽略。
    signal.SIGALRM：定时器信号，由 alarm 函数设置，通常用于超时机制。
    """
    import time

    from chatchat.utils import (
        get_config_dict,
        get_log_file,
        get_timestamp_ms,
    )

    from chatchat.settings import Settings

    logging_conf = get_config_dict(
        "INFO",
        get_log_file(
            log_path=Settings.basic_settings.LOG_PATH, sub_dir=f"start_main_server_{get_timestamp_ms()}"
        ),# 带时间戳的log文件名字
        1024 * 1024 * 1024 * 3,
        1024 * 1024 * 1024 * 3,
    )
    # 对日志添加配置
    logging.config.dictConfig(logging_conf)  # type: ignore

    def handler(signalname):
        """
        抛出异常(为了调试和日志)
        Python 3.9 has `signal.strsignal(signalnum)` so this closure would not be needed.
        Also, 3.8 includes `signal.valid_signals()` that can be used to create a mapping for the same purpose.
        """

        def f(signal_received, frame):
            
            # raise 主动触发异常 抛出 KeyboardInterrupt，这是 Python 标准的终止方式 
            raise KeyboardInterrupt(f"{signalname} received")

        return f

    # This will be inherited by the child process if it is forked (not spawned)
    signal.signal(signal.SIGINT, handler("SIGINT"))
    signal.signal(signal.SIGTERM, handler("SIGTERM"))

    mp.set_start_method("spawn")
    """spawn (派生的,生成的),指的是创建一个 全新的 进程,从头开始加载程序
    与fork 的区别: unix/linux系统中的,一种创建进程的方法,它通过复制当前进程来创建子进程
    "跨平台一致性": "spawn 在 Windows、Linux、macOS 上都可用",
    "安全性": "避免继承父进程的不安全状态",
    "稳定性": "防止文件描述符和线程状态的继承问题", 
    "可预测性": "每个进程从干净状态开始",
    "调试友好": "更容易追踪问题来源",
    "与现代框架兼容": "适合异步和复杂应用架构"
    """
    manager = mp.Manager()
    """创建一个多进程管理器,用于在进程间共享Python对象"""
    run_mode = None

    if args.all:
        args.api = True
        args.webui = True

    dump_server_info(args=args)

    if len(sys.argv) > 1:
        # sys.argv 是Python中的一个列表,用于储存命令行参数,sys.argv[0] 是脚本的名称,sys.argv[1]是第一个命令行参数,以此类推
        logger.info(f"正在启动服务：")
        logger.info(f"如需查看 llm_api 日志，请前往 {Settings.basic_settings.LOG_PATH}")

    processes = {}
    def process_count():
        return len(processes)
    # 由于 事件是由同一个manager管理的,进程之间的通信是由Manager进程进行实际的通信协调的
    # 创建事件 用于进程同步
    api_started = manager.Event()
    if args.api:
        # 创建一个进程,运行run_api_server
        process = Process(
            target=run_api_server,
            name=f"API Server",
            kwargs=dict(
                started_event=api_started,
                run_mode=run_mode,
            ),
            daemon=False, # 是否 守护进程 否 提供某种服务或执行某种任务的进程或线程，如同系统的"守护神"。
        )
        # 放入processes对象中
        processes["api"] = process

    # 创建事件 用于进程同步
    webui_started = manager.Event()
    if args.webui:
        # 创建一个进程,运行run_webui
        process = Process(
            target=run_webui,
            name=f"WEBUI Server",
            kwargs=dict(
                started_event=webui_started,
                run_mode=run_mode,
            ),
            daemon=True, # 是否 守护进程 否
        )
        # 放入processes对象中
        processes["webui"] = process

    try:
        # 海象运算符,processes中获取'api',并且赋值给p,如果p有值,就到if中的语句
        if p := processes.get("api"):
            p.start() # 进程启动
            p.name = f"{p.name} ({p.pid})" # 设置进程名字
            # 关键：这里主进程会阻塞，直到 API 进程调用 api_started.set()
            api_started.wait()  # 等待api.py启动完成

        if p := processes.get("webui"):
            p.start()
            p.name = f"{p.name} ({p.pid})"
            webui_started.wait()  # 等待webui.py启动完成

        dump_server_info(after_start=True, args=args)

        # 等待所有进程退出
        while processes:
            for p in processes.values():
                p.join(2) # 等待2秒，然后检查是否完成
                if not p.is_alive():
                    processes.pop(p.name)
    except Exception as e:
        logger.error(e)
        logger.warning("Caught KeyboardInterrupt! Setting stop event...")
    finally:
        for p in processes.values():
            logger.warning("Sending SIGKILL to %s", p)
            # Queues and other inter-process communication primitives can break when
            # process is killed, but we don't care here

            if isinstance(p, dict):
                for process in p.values():
                    process.kill()
            else:
                p.kill()

        for p in processes.values():
            logger.info("Process status: %s", p)

# 定义子命令 start ; option 为子命令的一些参数
@click.command(help="启动服务")
@click.option(# 前后端都启动
    "-a", 
    "--all",
    "all",
    is_flag=True,
    help="run api.py and webui.py",
)
@click.option(# 启动后端服务
    "--api", 
    "api",
    is_flag=True,
    help="run api.py",
)
@click.option(#启动前端
    "-w", 
    "--webui",
    "webui",
    is_flag=True,
    help="run webui.py server",
)

def main(all, api, webui):
    class args:
        """
        创建一个简单的类实例,并且为它动态添加属性
        类的内部只有一个省略号(...),在python中省略号是一个合法的单行代码,通常用于表示带填充的代码块,这里相当于一个空类体
        实际上这里只是为了创建一个简单的类容器,方便存储和访问属性
        """
        ...
    args.all = all
    args.api = api
    args.webui = webui

    # 添加这行代码
    cwd = os.getcwd()
    """获取当前工作目录(Current Working Directory)的路径
    """
    sys.path.append(cwd)
    """将当前工作目录添加到Python的模块搜索路径(sys.path)中,这样Python解释器就可以在当前目录中查找导入的模块
    """
    mp.freeze_support()
    """(主要用于打包)用于支持冻结(freeze)生成可执行文件时的方法,在window系统上,多进程变成需要这个来避免生成多个进程时的递归调用问题
    在非Windows平台（如Linux、macOS）上，freeze_support()不会做任何事情，所以可以安全调用。
    即使没有冻结，调用它也是安全的。
    """
    print("cwd:" + cwd)
    from chatchat.server.knowledge_base.migrate import create_tables
    # 创建ROM映射和关系型数据库的表
    create_tables()
    if sys.version_info < (3, 10):
        loop = asyncio.get_event_loop()
    else:
        try:
            # 获取当前正在进行的事件循环
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 创建新的事件循环
            loop = asyncio.new_event_loop()
        # 设置当前事件循环
        asyncio.set_event_loop(loop)
    # start_main_server 这个方法运行直到协程完成
    loop.run_until_complete(start_main_server(args))


if __name__ == "__main__":
    main()
