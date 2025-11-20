import sys

import streamlit as st # 、、主streamlit库用来构建webui
import streamlit_antd_components as sac #、、streamlit的一个扩展库 用来美化ui界面

from chatchat import __version__ # 、、导入版本号
from chatchat.server.utils import api_address # 、、导入获取后端api地址的函数
from chatchat.webui_pages.dialogue.dialogue import  dialogue_page # 、、导入对话页面函数
from chatchat.webui_pages.kb_chat import kb_chat # 、、导入知识库对话页面函数
from chatchat.webui_pages.mcp import mcp_management_page# 、、导入mcp管理页面函数
from chatchat.webui_pages.knowledge_base.knowledge_base import knowledge_base_page# 、、导入知识库管理页面函数
from chatchat.webui_pages.utils import * # 、、导入webui页面的工具函数

# 、、初始化一个ApiRequest实例 用来和后端通信
api = ApiRequest(base_url=api_address())

if __name__ == "__main__":

    #、、设置网页的标题 图标 菜单等基本信息
    st.set_page_config(
        "Langchain-Chatchat WebUI",
        get_img_base64("chatchat_icon_blue_square_v2.png"),
        initial_sidebar_state="expanded", #、、默认侧边栏展开
        menu_items={
            "Get Help": "https://github.com/chatchat-space/Langchain-Chatchat",
            "Report a bug": "https://github.com/chatchat-space/Langchain-Chatchat/issues",
            "About": f"""欢迎使用 Langchain-Chatchat WebUI {__version__}！""",
        },
        layout="centered",
    )

    # use the following code to set the app to wide mode and the html markdown to increase the sidebar width
    #、、使用以下代码将应用程序设置为宽模式，并使用html markdown增加侧边栏宽度
    st.markdown(#、、使用markdown增加侧边栏的宽度和上下边距
        """
        <style>
        [data-testid="stSidebarUserContent"] {
            padding-top: 20px;
        }
        .block-container {
            padding-top: 25px;
        }
        [data-testid="stBottomBlockContainer"] {
            padding-bottom: 20px;
        }
        """,
        unsafe_allow_html=True,#、、允许html标签
    )

    #、、侧边栏部分
    with st.sidebar:
        #、、显示侧边栏的logo和版本号
        st.image(#、、、显示logo
            get_img_base64("logo-long-chatchat-trans-v2.png"), 
            use_column_width=True #、、自适应宽度
        )
        st.caption( #、、显示版本号
            f"""<p align="right">当前版本：{__version__}</p>""",
            unsafe_allow_html=True,#、、允许html标签
        )  

        selected_page = sac.menu(#、、使用streamlit-antd-components库的menu函数创建侧边栏菜单
            [
                sac.MenuItem("多功能对话", icon="chat"),
                sac.MenuItem("RAG 对话", icon="database"),
                sac.MenuItem("知识库管理", icon="hdd-stack"),
                sac.MenuItem("MCP 管理", icon="hdd-stack"),
            ],
            key="selected_page",#、、菜单的key
            open_index=0,#、、默认展开第一个菜单
        )

        sac.divider()#、、添加分割线 

    if selected_page == "知识库管理":
        knowledge_base_page(api=api)
    elif selected_page == "RAG 对话":
        kb_chat(api=api)
    elif selected_page == "MCP 管理":
        mcp_management_page(api=api)
    else:
        dialogue_page(api=api)
