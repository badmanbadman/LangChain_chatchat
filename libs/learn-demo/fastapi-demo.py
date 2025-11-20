def demonstrate_fastapi_internals():
    """演示 FastAPI 内部机制"""
    
    print("=== FastAPI 内部工作机制 ===")
    
    print("🔧 FastAPI 应用启动流程:")
    steps = [
        ("1. 创建 FastAPI 实例", "app = FastAPI()"),
        ("2. 配置路由和中间件", "@app.get(), app.add_middleware()"),
        ("3. 设置生命周期上下文", "app.router.lifespan_context = lifespan"),
        ("4. 启动 ASGI 服务器", "uvicorn.run(app)"),
        ("5. 执行 lifespan", "进入异步上下文管理器"),
        ("6. 运行启动代码", "yield 之前的代码"),
        ("7. 应用运行", "yield 期间处理请求"),
        ("8. 执行关闭代码", "yield 之后的代码")
    ]
    
    for i, (step, code) in enumerate(steps, 1):
        print(f"   {i}. {step}")
        print(f"      代码: {code}")
    
    print("\n💡 关键理解:")
    print("   - lifespan_context 是一个异步上下文管理器")
    print("   - 它被 ASGI 服务器（如 uvicorn）调用")
    print("   - 在应用启动和关闭时自动执行")

demonstrate_fastapi_internals()