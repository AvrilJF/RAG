# utils.py
# 项目核心工具类：初始化嵌入模型/大模型/FAISS向量库，配置日志和环境变量，提供通用工具能力
import os  # 导入Python内置系统模块，用于文件/路径判断、环境变量读取
import logging  # 导入Python内置日志模块，用于记录项目运行的关键日志（成功/失败/操作）
from dotenv import load_dotenv  # 导入dotenv库，加载.env配置文件中的环境变量（解耦配置和代码）
from langchain_community.vectorstores import FAISS  # 导入FAISS向量库，实现向量存储和相似性检索

# 加载项目根目录的.env配置文件，将配置项注入系统环境变量，后续通过os.getenv()读取
load_dotenv()

# 全局配置日志系统，指定日志级别、输出格式、输出目标（本地文件+控制台）
logging.basicConfig(
    level=logging.INFO,  # 日志级别：INFO记录正常操作，ERROR记录错误信息
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # 日志格式：时间-模块名-级别-内容
    handlers=[
        logging.FileHandler("ec_rag.log"),  # 日志输出到本地ec_rag.log文件，持久化保存
        logging.StreamHandler()  # 日志同时输出到控制台，方便本地调试
    ]
)
# 创建当前模块的日志实例，__name__代表模块名utils，后续用该实例记录日志
logger = logging.getLogger(__name__)

# 定义嵌入模型初始化函数：核心功能是将自然语言文本转换为机器可识别的数值向量
def init_embedding():
    # 异常捕获：处理API密钥错误、网络问题、库版本冲突等初始化失败场景
    try:
        # 延迟导入嵌入模型：避免项目启动时加载所有依赖，提升启动速度，仅调用时加载
        from langchain_community.embeddings import ZhipuAIEmbeddings
        # 实例化智谱嵌入模型，配置关键参数
        embedding = ZhipuAIEmbeddings(
            api_key=os.getenv("ZHIPU_API_KEY"),  # 从环境变量读取智谱API密钥，避免硬编码
            model="embedding-2"  # 指定智谱免费嵌入模型，适配中文字符向量化，性能满足项目需求
        )
        logger.info("嵌入模型初始化成功")  # 记录初始化成功日志，便于排查启动问题
        return embedding  # 返回初始化完成的嵌入模型实例，供其他模块调用
    # 捕获所有初始化异常，记录错误日志并向上抛出
    except Exception as e:
        logger.error(f"嵌入模型初始化失败：{str(e)}")
        raise

# 定义大模型初始化函数：核心功能是基于检索到的知识库上下文，生成准确、无幻觉的回答
def init_llm():
    # 异常捕获：处理API密钥错误、网络超时、模型调用限制等初始化失败场景
    try:
        # 延迟导入大模型：仅调用函数时加载依赖，优化项目启动性能
        from langchain_community.chat_models import ChatZhipuAI
        # 实例化智谱对话大模型，配置关键参数
        llm = ChatZhipuAI(
            api_key=os.getenv("ZHIPU_API_KEY"),  # 从环境变量读取智谱API密钥，与嵌入模型共用
            model="glm-4-flash"  # 指定智谱免费大模型，响应快、适配中文问答，满足项目基础需求
        )
        logger.info("大模型初始化成功")  # 记录初始化成功日志，便于排查启动问题
        return llm  # 返回初始化完成的大模型实例，供RAG核心逻辑调用
    # 捕获所有初始化异常，记录错误日志并向上抛出
    except Exception as e:
        logger.error(f"大模型初始化失败：{str(e)}")
        raise

# 定义FAISS向量库初始化函数：核心功能是存储向量数据、实现快速相似性检索
# 参数embedding：传入已初始化的嵌入模型，保证向量库的向量维度与文本/问题向量维度一致
def init_faiss(embedding):
    # 定义FAISS向量库本地持久化路径，向量数据将保存在该文件夹下，避免重复向量化
    persist_dir = "./faiss_ec_db"
    # 异常捕获：处理文件夹创建失败、本地库加载失败、向量维度不匹配等问题
    try:
        # 判断本地是否已有持久化的向量库文件夹，避免重复初始化
        if os.path.exists(persist_dir):
            # 若存在，直接从本地加载已保存的FAISS向量库
            vector_db = FAISS.load_local(
                persist_dir,  # 本地向量库文件夹路径
                embedding,    # 传入嵌入模型，用于解析向量数据
                allow_dangerous_deserialization=True  # 允许本地反序列化，个人项目无需考虑该安全限制
            )
            logger.info("FAISS 从本地加载成功")  # 记录本地加载成功日志
        else:
            # 若不存在，首次初始化空的FAISS向量库（用占位文本初始化，无实际业务意义）
            vector_db = FAISS.from_texts(
                texts=["init"],  # 占位文本，仅用于初始化向量库实例
                embedding=embedding,  # 传入嵌入模型，生成占位文本的向量
                metadatas=[{"source": "init"}]  # 占位元数据，与占位文本对应
            )
            # 将首次初始化的空向量库保存到本地指定路径，生成持久化文件夹
            vector_db.save_local(persist_dir)
            logger.info("FAISS 首次初始化成功")  # 记录首次初始化成功日志
        # 返回初始化完成的向量库实例+本地持久化路径，供其他模块调用和后续保存
        return vector_db, persist_dir
    # 捕获所有初始化异常，记录错误日志并向上抛出
    except Exception as e:
        logger.error(f"FAISS 初始化失败: {e}")
        raise