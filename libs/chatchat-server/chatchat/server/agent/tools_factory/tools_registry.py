from __future__ import annotations
"""
from __future__ import annotations  
    这个是Python为了特性导入，用于启用新的语言特性
解决了什么问题？
    #没有这个导入时，前向引用会报错
    class Person:
        def marry(self, other: Person) -> bool: # 错误！！Person还未定义
            return True
    #有了这个导入后，类型注解可以是字符串
    class Person:
        def marry(self, other: "Person") ->#正确
实际效果：
    让Python在运行时才评估类型注解
    避免循环导入和自引用问题
"""

import json
import re # 、、正则表达式，用于文本处理
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union, List

# 、、LangChain的工具装饰器
from langchain.agents import tool
# 、、LangChainn工具基类
from langchain_core.tools import BaseTool

# 、、DocumentWithVSId，从本地模块导入，表示带有向量存储ID的文档
from chatchat.server.knowledge_base.kb_doc_api import DocumentWithVSId
# 、、BaseModel和Extra,来自Pydantic，用于数据验证和设置
from chatchat.server.pydantic_v1 import BaseModel, Extra
from langchain_chatchat.agent_toolkits.all_tools.tool import (
    BaseToolOutput,
)
__all__ = ["regist_tool", "BaseToolOutput", "format_context"]


# 、、工具注册表 是一个字典
# 想象成一个"工具仓库"
    #_TOOLS_REGISTRY = {
    #     "search_tool": <搜索工具对象>,
    #     "calculator_tool": <计算器工具对象>, 
    #     "file_reader_tool": <文件阅读器工具对象>
    # }

    # # 当需要某个工具时，就从仓库里取
    # def get_tool(tool_name):
    #     return _TOOLS_REGISTRY[tool_name]
_TOOLS_REGISTRY = {}


# patch BaseTool to support extra fields e.g. a title
BaseTool.Config.extra = Extra.allow
"""、、
BaseTool配置补丁
    、、问题背景
    # LangChain 的 BaseTool 原本不支持额外字段
    tool BaseTool(name="search", description="搜索工具")
    # 想添加自定义字段会报错
    tool.title =  "搜索工具" # 可能不支持

    、、解决方案
    # 允许BaseTool接受额外字段
    BaseTool.config.extra = Extra.allow
    # 现在可以这样：
    tool.title = '智能搜索工具' # 可以了  
"""

################################### TODO: workaround to langchain #15855
# patch BaseTool to support tool parameters defined using pydantic Field


# 、、输入解析补丁
"""
解决的问题：
    工具输入可能是字符串或字典
    @regist_tool
    def search(query: str):...

    # 调用时：
    search('人工智能') #字符串输入
    search({'query': "python教程"}) #字典输入

在LangChain工具中，工具可以定义一个args_schema，即一个Pydantic模型，用于描述工具的输入参数
当工具被调用时，传入的参数需要符合这个模型。
但是工具输入可能是两种形式：
1、字符串
3、字典
当输入是字符串时，我们需要将其转换为工具所期望的参数结构
分析： key_ = next(iter(input_args.__fields__keys()))
这行代码做了以下事情：
    1、input_args 是工具的args_schema，是一个Pydantic的模型类
    2、input_args.__fields__是一个字典，包含了该模型的所有字段（参数）
    3、iter(input_args.__fields__.keys()) 创建开一个迭代器，用来遍历所有的字段名
    4、next(...) 获取第一个字段名
所以key_就是args_schema中第一个参数的名称
例如，如果args_schema定义为：
    class SearchArgs(BaseModel):
        query: str
        limit: int = 5
那么key_就是query。
分析： input_args.validate({key_: tool_input})
构造字典，并验证是否符合模型
"""
def _new_parse_input(
    self,
    tool_input: Union[str, Dict],
) -> Union[str, Dict[str, Any]]:
    """Convert tool input to pydantic model."""
    input_args = self.args_schema
    if isinstance(tool_input, str):
        # 处理字符串输入：'python教程' -> {'query': 'python教程'}
        if input_args is not None:
            key_ = next(iter(input_args.__fields__.keys()))
            input_args.validate({key_: tool_input})
        return tool_input
    else:
        # 处理字典输入： 验证参数格式
        if input_args is not None:
            # 解析为Pydantic模型（注意，在LangChain的实现中，是只返回了用户输入的字段）
            """解决了什么问题？
            实际场景演示
            假设我们有一个工具定义：
                class SearchArgs(BaseModel):
                    query: str
                    limit: int = 10  # 默认值10
                    category: str = "all"  # 默认值"all"

                @regist_tool(args_schema=SearchArgs) #搜索文档
                def search_documents(query: str, limit: int = 10, category: str = "all") -> list:
                    return [f"在{category}中找到{limit}个关于{query}的文档"]
            用户调用：
                # 用户只提供query参数
                tool_input = {"query": "Python教程"}

            原方法处理结果：
                result = SearchArgs.parse_obj({"query": "Python教程"})
                # result 包含: query="Python教程", limit=10, category="all"

                # 但返回值过滤后：
                return {k: getattr(result, k) for k in result.dict() if k in tool_input}
                # → 只返回: {"query": "Python教程"}
                # 🔴 limit和category丢失了！

            新方法处理结果：
                result = SearchArgs.parse_obj({"query": "Python教程"})
                return result.dict()
                # → 返回: {"query": "Python教程", "limit": 10, "category": "all"}
                # ✅ 所有参数都保留
            """
            result = input_args.parse_obj(tool_input) 
            return result.dict() # 返回模型的所有字段


# 、、将工具输入转换为函数调用所需的参数格式
"""
将工具输入（可以是字符串或者字典）转换为一个元组（位置参数）和一个字典（关键字参数）,方便后续的函数调用
"""
def _new_to_args_and_kwargs(self, tool_input: Union[str, Dict]) -> Tuple[Tuple, Dict]:
    # For backwards compatibility, if run_input is a string,
    # pass as a positional argument.
    if isinstance(tool_input, str):
        # 字符串输入
        # 输入: "hello"
        # 输出: (("hello",), {})
        return (tool_input,), {}
    else:
        # for tool defined with `*args` parameters
        # the args_schema has a field named `args`
        # it should be expanded to actual *args
        # e.g.: test_tools
        #       .test_named_tool_decorator_return_direct
        #       .search_api
        if "args" in tool_input:
            args = tool_input["args"]
            if args is None:
                # 字典输入，包含 "args" 但为 None
                # 输入: {"args": None, "option": "value"}
                # 输出: ((), {"option": "value"})
                tool_input.pop("args")
                return (), tool_input
            elif isinstance(args, tuple):
                # 字典输入，包含 "args" 且为元组
                # 输入: {"args": (1, 2), "option": "value"}
                # 输出: ((1, 2), {"option": "value"})
                tool_input.pop("args")
                return args, tool_input
        # 字典输入，不包含 "args"
        # 输入: {"option": "value"}
        # 输出: ((), {"option": "value"})
        return (), tool_input


BaseTool._parse_input = _new_parse_input
BaseTool._to_args_and_kwargs = _new_to_args_and_kwargs
###############################

"""装饰器-Demo：
import time

# 1、定义一个装饰器
def timer_decorator(original_function):
    # 给函数添加计时功能的装饰器
    def wrapper():
        start_time = time.time()
        result = original_function() # 执行原函数
        end_time = time.time()
        print(f"函数执行耗时：{end_time - start_time:.2f}秒")
        return result
    return wrapper

# 2、使用装饰器
@timer_dicorator
def my_function():
    # 模拟一个耗时操作
    time.sleep(1)
    print("函数执行完成")
    return "success"
# 3、调用函数
result = my_fucntion()
# 输出：
# 函数执行完成
# 函数执行耗时：1.00秒

装饰器的工作原理
#不使用@语法糖的等价写法：
def my_function():
    time.sleep(1)
    print("函数执行完成!")
    return "成功"
my_function = timer_decorator(my_function)  # 手动包装
result = my_function()
"""

"""regist_tool装饰器：
1、支持带参数和不带参数两种用法
2、集成了LangChain的工具系统
3、自动注册工具到全局注册表
4、自动提取元数据

"""
"""示例1：简单用法
@regist_tool
def get_weather(city: str) -> str:
    ""获取城市天气信息
    Args:
        city: 城市名称
    Returns:
        天气描述
    ""
    return f"{city}的天气是晴天"

# 执行过程：
# 1. regist_tool 被调用，没有位置参数 → 返回 wrapper 函数
# 2. wrapper(get_weather) 被调用
# 3. 在 wrapper 内部：
#    - 使用 LangChain 的 tool() 创建 BaseTool
#    - _parse_tool() 自动设置：
#        name: "get_weather"
#        description: "获取城市天气信息 Args: city: 城市名称 Returns: 天气描述"
#        title: "GetWeather"
#    - 注册到 _TOOLS_REGISTRY["get_weather"]

"""
"""示例2：带参数用法
@regist_tool(
    title="天气预报",
    description="查询指定城市的实时天气情况",
    return_direct=True
)
def get_weather(city: str) -> str:
    return f"{city}的天气是晴天"

# 执行过程：
# 1. regist_tool(title="天气预报", ...) 被调用，有参数
# 2. 直接执行 else 分支：
#    - 使用 LangChain 的 tool(title="天气预报", ...) 创建 BaseTool
#    - _parse_tool() 处理（使用传入的title和description）
#    - 返回创建好的 BaseTool 对象
"""
"""
函数签名是什么？
函数签名（Function Signature）指的是函数的名称、参数类型和数量、返回值类型等信息的组合。它定义了函数的接口，即如何调用这个函数。
1. 自动元数据提取
# 自动从函数文档字符串提取描述
def my_func():
    ""这是一个很棒的函数
    它可以做很多事情
    ""
    pass

# 自动变成："这是一个很棒的函数 它可以做很多事情"
3. 全局注册表管理
# 所有被装饰的工具都会自动注册
_TOOLS_REGISTRY = {
    "search_documents": <BaseTool对象>,
    "get_weather": <BaseTool对象>,
    # ...
}

# 其他地方可以通过名称获取工具
def get_tool(tool_name):
    return _TOOLS_REGISTRY.get(tool_name)

4. LangChain 集成
# 底层使用 LangChain 的 @tool 装饰器
partial_ = tool(*args, return_direct=return_direct, ...)
t = partial_(def_func)  # 创建标准的 LangChain 工具
"""
def regist_tool(
    *args: Any, #可变参数，支持多种调用方式
    title: str = "", #工具标题
    description: str = "", # 工具描述
    return_direct: bool = False, #是否直接返回结果
    args_schema: Optional[Type[BaseModel]] = None, #参数验证模型
    infer_schema: bool = True, #是否自动推断参数schema
) -> Union[Callable, BaseTool]: #返回装饰器或工具对象
    """
    wrapper of langchain tool decorator
    add tool to regstiry automatically
    """

    def _parse_tool(t: BaseTool):
        nonlocal description, title
        
        # 1、注册工具到全局表
        _TOOLS_REGISTRY[t.name] = t

        # 2、设置额描述（从函数文档字符串中提取）
        if not description:
            if t.func is not None:
                description = t.func.__doc__ # 获取函数的文档字符传
            elif t.coroutine is not None:
                description = t.coroutine.__doc__
        t.description = " ".join(re.split(r"\n+\s*", description))# 清理格式
        # 生成标题
        if not title:
             # "search_documents" → "SearchDocuments"
            title = "".join([x.capitalize() for x in t.name.split("_")])
        t.title = title

    def wrapper(def_func: Callable) -> BaseTool:
        # 使用LangChain的@tool装饰器
        partial_ = tool(
            *args,
            return_direct=return_direct,
            args_schema=args_schema,
            infer_schema=infer_schema,
        )
        # 、、def_func是使用这个装饰器的函数，这里没有用语法糖@，而是直接把使用这个装饰器的函数传进去
        # 、、生成一个BaseTool对象
        t = partial_(def_func) 
        _parse_tool(t) #处理元数据和注册，将BaseTool传进去了
        return t

    # 根据调用方式决定返回什么
    if len(args) == 0:
        return wrapper #返回装饰器函数 
    else:
        t = tool(
            *args,
            return_direct=return_direct,
            args_schema=args_schema,
            infer_schema=infer_schema,
        ) # 直接创建工具对象
        _parse_tool(t)
        # 、、？？？？这里直接返回了 工具对象，没有和调用的函数绑定哦，我查看了引用regist_tool的地方，都没有走进这个分支来
        return t


def format_context(self: BaseToolOutput) -> str:
    '''
    将包含知识库输出的ToolOutput格式化为 LLM 需要的字符串
    '''
    context = ""
    docs = self.data["docs"]
    source_documents = []

    for inum, doc in enumerate(docs):
        doc = DocumentWithVSId.parse_obj(doc)
        source_documents.append(doc.page_content)

    if len(source_documents) == 0:
        context = "没有找到相关文档,请更换关键词重试"
    else:
        for doc in source_documents:
            context += doc + "\n\n"

    return context
