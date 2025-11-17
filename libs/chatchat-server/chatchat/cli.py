import click
from pathlib import Path
import shutil
import typing as t

from chatchat.startup import main as startup_main
from chatchat.init_database import main as kb_main, create_tables, folder2db
from chatchat.settings import Settings
from chatchat.utils import build_logger
from chatchat.server.utils import get_default_embedding


logger = build_logger()

# 、、定义顶级命令组
@click.group(help="chatchat 命令行工具")
def main():
    ...

# 、、定义子命令 init  初始化项目使用
@main.command("init", help="项目初始化")
@click.option("-x", "--xinference-endpoint", "xf_endpoint",
              help="指定Xinference API 服务地址。默认为 http://127.0.0.1:9997/v1")
@click.option("-l", "--llm-model",
              help="指定默认 LLM 模型。默认为 glm4-chat")
@click.option("-e", "--embed-model",
              help="指定默认 Embedding 模型。默认为 bge-large-zh-v1.5")
@click.option("-r", "--recreate-kb",
              is_flag=True,
              show_default=True,
              default=False,
              help="同时重建知识库（必须确保指定的 embed model 可用）。")
@click.option("-k", "--kb-names", "kb_names",
              show_default=True,
              default="samples",
              help="要重建知识库的名称。可以指定多个知识库名称，以 , 分隔。")
def init(
    xf_endpoint: str = "",
    llm_model: str = "",
    embed_model: str = "",
    recreate_kb: bool = False,
    kb_names: str = "",
):
    # 、、在批量修改配置或者初始化的时候先设置为False，完成后再设为True（允许配置文件变更触发更新）
    Settings.set_auto_reload(False)
    # 、、获取基础配置
    bs = Settings.basic_settings

    # 、、生成知识库名称，默认逗号分隔
    kb_names = [x.strip() for x in kb_names.split(",")]
    logger.success(f"开始初始化项目数据目录：{Settings.CHATCHAT_ROOT}")

    # 、、创建所有数据集目录
    Settings.basic_settings.make_dirs()
    logger.success("创建所有数据目录：成功。")

    if(bs.PACKAGE_ROOT / "data/knowledge_base/samples" != Path(bs.KB_ROOT_PATH) / "samples"):
        # 、、shuil 用来递归复制一个目录及其所有子文件/子目录到目标位置
        shutil.copytree(bs.PACKAGE_ROOT / "data/knowledge_base/samples", Path(bs.KB_ROOT_PATH) / "samples", dirs_exist_ok=True)
    logger.success("复制 samples 知识库文件：成功。")
    # 、、初始化知识库数据库
    create_tables()
    logger.success("初始化知识库数据库：成功。")

    if xf_endpoint:
        # 、、运行大模型的程序框架
        Settings.model_settings.MODEL_PLATFORMS[0].api_base_url = xf_endpoint
    if llm_model:
        # 、、推理用的大模型
        Settings.model_settings.DEFAULT_LLM_MODEL = llm_model
    if embed_model:
        # 、、嵌入用embedding模型
        Settings.model_settings.DEFAULT_EMBEDDING_MODEL = embed_model
    
    # 生成配置文件
    Settings.createl_all_templates()
    # 配置项重置为：配置文件变更自动刷新
    Settings.set_auto_reload(True)

    logger.success("生成默认配置文件：成功。")
    logger.success("请先检查确认 model_settings.yaml 里模型平台、LLM模型和Embed模型信息已经正确")
    # 、、re create kb 重新生成知识库（包括向量库和数据库两个库）
    if recreate_kb:
        folder2db(kb_names=kb_names,
                  mode="recreate_vs",
                  vs_type=Settings.kb_settings.DEFAULT_VS_TYPE,
                  embed_model=get_default_embedding())
        # 、、logger可以加个样式~，学到啦
        logger.success("<green>所有初始化已完成，执行 chatchat start -a 启动服务。</green>")
    else:
        logger.success("执行 chatchat kb -r 初始化知识库，然后 chatchat start -a 启动服务。")

# 定义子命令 start 启动项目使用
main.add_command(startup_main, "start")
# 定义子命令 kb 初始化知识库使用
main.add_command(kb_main, "kb")


if __name__ == "__main__":
    main()
