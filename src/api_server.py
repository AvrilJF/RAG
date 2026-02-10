# api_server.py
# 项目API服务层：基于FastAPI封装标准化HTTP接口，对外提供网络调用能力
# 实现核心业务函数的接口化、服务化，贴合生产环境微服务设计，体现后端工程化能力
# 从FastAPI框架导入核心对象：FastAPI用于创建应用实例，HTTPException用于抛出标准化HTTP异常
from fastapi import FastAPI, HTTPException
# 从Pydantic库导入基础模型，用于定义接口请求体格式，实现自动参数校验和数据解析
from pydantic import BaseModel
# 导入uvicorn服务器，作为FastAPI的运行容器，用于启动Web服务并监听网络请求
import uvicorn
# 从自定义RAG核心层导入业务函数，复用已实现的知识库加载和智能问答逻辑
from src.rag_engine import rag_qa, load_knowledge
# 从自定义工具类导入日志实例，用于记录接口运行的关键日志（成功/失败）
from src.utils import logger

# 创建FastAPI应用 实例，作为整个API服务的核心入口
# 配置应用名称和版本，会自动显示在Swagger文档中，提升接口可读性
# 模拟Go微服务设计思路，贴合后端高并发、标准化接口的开发经验
app = FastAPI(title="跨境电商RAG智能问答API", version="1.0")

# 定义接口请求体模型：基于Pydantic BaseModel，实现接口入参的自动校验
# 无需手动判断参数是否存在、类型是否正确，Pydantic会自动处理并返回标准化错误提示
class QAQuery(BaseModel):
    # 定义/qa问答接口的请求体参数：question为字符串类型，代表用户的提问内容
    # 若请求体无该字段/字段类型错误，接口会直接返回400参数错误
    question: str

# class LoadKnowledgeRequest(BaseModel):
#     # 定义/load-knowledge加载知识库接口的请求体参数：pdf_path为字符串类型，代表PDF知识库文件路径
#     pdf_path: str
# api_server.py中修改请求体模型
class LoadKnowledgeRequest(BaseModel):
    pdf_paths: list[str]  # 从str改为list[str]，接收PDF路径列表

# 定义GET类型的健康检查接口，路径为/health
# 运维/监控系统必备接口，用于定时检测服务是否正常运行，体现工程化和生产环境思维
# async关键字定义异步函数，FastAPI异步特性提升接口高并发处理能力
@app.get("/health")
async def health_check():#async异步函数
    # 服务正常时返回固定格式响应，status为ok表示服务可用，附带服务名称标识
    return {"status": "ok", "service": "ec-rag-service"}

# # 定义POST类型的加载知识库接口，路径为/load-knowledge
# # 仅需调用1次，用于触发PDF知识库的加载、分块、向量化入库流程
@app.post("/load-knowledge")
async def load_knowledge_api(request: LoadKnowledgeRequest):
    try:
        load_knowledge(request.pdf_paths)  # 传入列表
        return {"code": 200, "msg": "所有知识库加载成功"}
    except Exception as e:
        logger.error(f"加载知识库失败：{str(e)}")
        raise HTTPException(status_code=500, detail=f"加载失败：{str(e)}")

# 定义POST类型的核心问答接口，路径为/qa
# 项目核心业务接口，用户每次提问都会调用该接口，返回RAG生成的答案
@app.post("/qa")
async def qa_api(query: QAQuery):
    # 异常捕获：处理问答过程中的所有错误（大模型调用失败、检索异常、API密钥错误等）
    try:
        # 调用rag_engine中的rag_qa函数，传入请求体中的question参数执行问答逻辑
        result = rag_qa(query.question)
        # 问答成功时返回标准化响应：200表示业务成功，附带用户原始问题和AI生成的答案
        # 响应格式统一，方便前端/其他业务系统对接解析
        return {"code": 200, "question": query.question, "answer": result}
    # 捕获所有异常，记录错误日志并抛出标准化HTTP异常
    except Exception as e:
        # 记录错误日志，包含具体失败原因
        logger.error(f"问答接口失败：{str(e)}")
        # 抛出HTTP 500服务器内部错误异常，返回错误状态码和详细原因给调用方
        raise HTTPException(status_code=500, detail=f"问答失败：{str(e)}")

# 服务启动入口：仅当直接运行当前py文件时才会执行，被其他文件导入时不执行
# 模拟生产环境的服务部署方式，通过代码直接启动Web服务
if __name__ == "__main__":
    # 启动FastAPI服务:调用uvicorn的run方法
    # app：指定要运行的FastAPI应用实例
    # host="0.0.0.0"：表示监听本机所有IP地址，局域网/外网均可访问该服务
    # port=8000：指定服务监听的端口号，通过http://localhost:8000访问接口
    uvicorn.run(app, host="0.0.0.0", port=8000)