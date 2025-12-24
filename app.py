"""
Juris-RAG Web应用
基于Gradio构建的法律智能问答系统前端
"""
import os
import gradio as gr
from typing import List, Tuple, Generator
import time

# 导入配置和RAG引擎
try:
    from src.config import APP_TITLE, APP_DESCRIPTION
    from src.rag_engine import JurisRAGEngine, RAGResponse
except ImportError:
    APP_TITLE = "Juris-RAG 法律智能问答系统"
    APP_DESCRIPTION = "基于RAG技术的中文法律问答系统"
    from rag_engine import JurisRAGEngine, RAGResponse

# 全局RAG引擎实例
rag_engine = None


def initialize_engine():
    """初始化RAG引擎"""
    global rag_engine
    if rag_engine is None:
        try:
            rag_engine = JurisRAGEngine(streaming=False)
            return True, "✅ RAG引擎初始化成功！"
        except FileNotFoundError as e:
            return False, f"❌ 向量数据库未找到，请先运行数据处理脚本：\npython -m src.data_processing"
        except ValueError as e:
            return False, f"❌ API配置错误：{str(e)}"
        except Exception as e:
            return False, f"❌ 初始化失败：{str(e)}"
    return True, "✅ RAG引擎已就绪"


def format_citations(citations) -> str:
    """格式化引用来源为Markdown"""
    if not citations:
        return "暂无引用来源"
    
    citation_md = ""
    for i, citation in enumerate(citations, 1):
        citation_md += f"**[{i}] {citation.source}**\n"
        citation_md += f"- 类型: {citation.doc_type}\n"
        
        # 添加额外元数据
        if citation.metadata.get("accusation"):
            citation_md += f"- 罪名: {citation.metadata['accusation']}\n"
        if citation.metadata.get("articles"):
            citation_md += f"- 相关法条: 第{citation.metadata['articles']}条\n"
        if citation.metadata.get("article"):
            citation_md += f"- 条款: {citation.metadata['article']}\n"
        
        # 内容预览
        citation_md += f"- 内容摘要: {citation.content[:150]}...\n\n"
    
    return citation_md


def chat_response(
    message: str,
    history: List[Tuple[str, str]]
) -> Tuple[str, str, str, List[Tuple[str, str]]]:
    """
    处理用户消息并返回响应
    
    Args:
        message: 用户输入的消息
        history: 对话历史
        
    Returns:
        tuple: (回答, 引用信息, 置信度信息, 更新后的历史)
    """
    global rag_engine
    
    if not message.strip():
        return "", "请输入问题", "", history
    
    # 确保引擎已初始化
    if rag_engine is None:
        success, msg = initialize_engine()
        if not success:
            return msg, "", "", history
    
    try:
        # 同步历史到引擎
        rag_engine.chat_history = [(h[0], h[1]) for h in history]
        
        # 获取响应
        response = rag_engine.query(message)
        
        # 格式化引用
        citations_md = format_citations(response.citations)
        
        # 格式化置信度
        confidence_emoji = "🟢" if response.confidence >= 0.7 else "🟡" if response.confidence >= 0.4 else "🔴"
        confidence_text = f"{confidence_emoji} 置信度: {response.confidence:.0%}"
        if response.is_uncertain:
            confidence_text += " (低置信度回答)"
        
        # 更新历史
        new_history = history + [(message, response.answer)]
        
        return response.answer, citations_md, confidence_text, new_history
        
    except Exception as e:
        error_msg = f"❌ 处理请求时发生错误: {str(e)}"
        return error_msg, "", "", history


def clear_conversation():
    """清空对话"""
    global rag_engine
    if rag_engine:
        rag_engine.clear_history()
    return [], "", "", ""


def search_documents(query: str, top_k: int = 5) -> str:
    """
    直接搜索相关文档
    
    Args:
        query: 搜索查询
        top_k: 返回数量
        
    Returns:
        str: 格式化的搜索结果
    """
    global rag_engine
    
    if not query.strip():
        return "请输入搜索内容"
    
    if rag_engine is None:
        success, msg = initialize_engine()
        if not success:
            return msg
    
    try:
        docs = rag_engine.search_similar(query, k=top_k)
        
        if not docs:
            return "未找到相关文档"
        
        result = f"## 找到 {len(docs)} 个相关文档\n\n"
        
        for i, doc in enumerate(docs, 1):
            result += f"### 文档 {i}\n"
            result += f"**来源**: {doc.metadata.get('source', '未知')}\n"
            result += f"**类型**: {doc.metadata.get('type', '未知')}\n"
            
            if doc.metadata.get('accusation'):
                result += f"**罪名**: {doc.metadata['accusation']}\n"
            if doc.metadata.get('article'):
                result += f"**条款**: {doc.metadata['article']}\n"
            
            result += f"\n```\n{doc.page_content}\n```\n\n"
            result += "---\n\n"
        
        return result
        
    except Exception as e:
        return f"❌ 搜索时发生错误: {str(e)}"


# 示例问题
EXAMPLE_QUESTIONS = [
    "故意杀人罪怎么判刑？",
    "盗窃罪的量刑标准是什么？",
    "什么情况下构成正当防卫？",
    "诈骗罪和盗窃罪有什么区别？",
    "醉酒驾驶怎么处罚？",
    "未成年人犯罪如何处理？"
]


def create_app():
    """创建Gradio应用"""
    
    # 自定义CSS
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .chat-message {
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .citation-box {
        background-color: #f5f5f5;
        border-left: 4px solid #4CAF50;
        padding: 10px;
        margin: 10px 0;
    }
    """
    
    with gr.Blocks(
        title=APP_TITLE,
        theme=gr.themes.Soft(),
        css=custom_css
    ) as app:
        
        # 标题区域
        gr.Markdown(f"# 🏛️ {APP_TITLE}")
        gr.Markdown(APP_DESCRIPTION)
        
        # 状态初始化
        with gr.Row():
            init_status = gr.Markdown("⏳ 系统正在初始化...")
        
        with gr.Tabs():
            # Tab 1: 智能问答
            with gr.TabItem("💬 智能问答", id="chat"):
                with gr.Row():
                    # 左侧：对话区域
                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(
                            label="对话历史",
                            height=500,
                            show_copy_button=True,
                            avatar_images=(None, "🤖")
                        )
                        
                        with gr.Row():
                            msg_input = gr.Textbox(
                                label="输入您的法律问题",
                                placeholder="例如：故意杀人罪怎么判刑？",
                                lines=2,
                                scale=4
                            )
                            submit_btn = gr.Button("🚀 发送", variant="primary", scale=1)
                        
                        with gr.Row():
                            clear_btn = gr.Button("🗑️ 清空对话")
                            
                        # 示例问题
                        gr.Markdown("### 💡 示例问题")
                        example_btns = gr.Examples(
                            examples=[[q] for q in EXAMPLE_QUESTIONS],
                            inputs=msg_input,
                            label=""
                        )
                    
                    # 右侧：引用和置信度
                    with gr.Column(scale=1):
                        confidence_display = gr.Markdown(
                            label="置信度",
                            value="等待提问..."
                        )
                        
                        gr.Markdown("### 📚 引用来源")
                        citations_display = gr.Markdown(
                            value="提问后将显示引用来源",
                            elem_classes=["citation-box"]
                        )
            
            # Tab 2: 文档搜索
            with gr.TabItem("🔍 文档搜索", id="search"):
                gr.Markdown("### 直接搜索法律文档库")
                gr.Markdown("输入关键词或描述，直接检索相关法条和案例。")
                
                with gr.Row():
                    search_input = gr.Textbox(
                        label="搜索内容",
                        placeholder="输入关键词，如：盗窃、故意伤害...",
                        scale=3
                    )
                    search_k = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="返回数量",
                        scale=1
                    )
                    search_btn = gr.Button("🔍 搜索", variant="primary", scale=1)
                
                search_results = gr.Markdown(
                    label="搜索结果",
                    value="输入内容后点击搜索"
                )
            
            # Tab 3: 系统信息
            with gr.TabItem("ℹ️ 系统信息", id="info"):
                gr.Markdown("""
                ### 🏛️ Juris-RAG 法律智能问答系统
                
                #### 系统特点
                - **📚 知识库**: 基于中华人民共和国刑法及CAIL2018司法案例数据集
                - **🔍 智能检索**: 使用BGE-M3向量模型进行语义检索
                - **🤖 大模型生成**: 基于Qwen2.5-7B-Instruct生成回答
                - **💬 多轮对话**: 支持上下文理解，实现连续对话
                - **📝 引用追溯**: 每个回答都标注信息来源
                - **🚫 拒绝不确定**: 对无法回答的问题会明确告知
                
                #### 技术栈
                - **框架**: LangChain + Gradio
                - **向量库**: ChromaDB
                - **Embedding**: BAAI/bge-m3
                - **LLM**: Qwen/Qwen2.5-7B-Instruct
                - **API**: SiliconFlow
                
                #### 数据来源
                1. **刑法法条**: 中华人民共和国刑法完整文本
                2. **司法案例**: CAIL2018中国法律智能挑战赛数据集（5000+案例）
                
                #### 免责声明
                ⚠️ 本系统仅供学习和研究使用，不构成法律建议。
                如有实际法律问题，请咨询专业律师。
                
                ---
                **版本**: v1.0.0  
                **更新日期**: 2024-12
                """)
        
        # 事件绑定
        def on_submit(message, history):
            answer, citations, confidence, new_history = chat_response(message, history)
            return new_history, "", citations, confidence
        
        submit_btn.click(
            fn=on_submit,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input, citations_display, confidence_display]
        )
        
        msg_input.submit(
            fn=on_submit,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input, citations_display, confidence_display]
        )
        
        clear_btn.click(
            fn=clear_conversation,
            outputs=[chatbot, msg_input, citations_display, confidence_display]
        )
        
        search_btn.click(
            fn=search_documents,
            inputs=[search_input, search_k],
            outputs=search_results
        )
        
        # 应用加载时初始化
        def on_load():
            success, msg = initialize_engine()
            return msg
        
        app.load(
            fn=on_load,
            outputs=init_status
        )
    
    return app


# 主入口
if __name__ == "__main__":
    print("🚀 正在启动 Juris-RAG Web应用...")
    print("=" * 50)
    
    # 检查环境变量
    if not os.getenv("SILICONFLOW_API_KEY"):
        print("⚠️ 警告: 未检测到 SILICONFLOW_API_KEY 环境变量")
        print("   请设置: set SILICONFLOW_API_KEY=your_key (Windows)")
        print("   或: export SILICONFLOW_API_KEY=your_key (Linux/Mac)")
        print("=" * 50)
    
    # 创建并启动应用
    app = create_app()
    
    # 启动服务
    app.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,
        share=False,  # 设为True可生成公网链接
        show_error=True,
        favicon_path=None
    )
