# rag_engine.py
# RAG项目核心业务层：实现知识库加载、文本分块、向量入库、智能问答全流程逻辑
# 是整个项目的算法核心，API服务层会直接调用此文件的函数完成业务操作
import os  # 导入Python内置系统模块，用于判断PDF文件是否存在

# 从langchain核心库导入提示词模板，用于构建标准化的大模型输入提示
from langchain_core.prompts import ChatPromptTemplate
# 从langchain核心库导入透传组件，用于RAG链中直接传递用户问题等参数
from langchain_core.runnables import RunnablePassthrough
# 从langchain核心库导入字符串输出解析器，将大模型复杂输出解析为纯文本
from langchain_core.output_parsers import StrOutputParser
# 从langchain社区库导入PDF加载器，专门用于读取PDF文件并提取纯文本内容
from langchain_community.document_loaders import PyPDFLoader
# 从langchain文本处理库导入递归字符分块器，实现中文文本的智能语义分块
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 从自定义工具类utils.py导入所有初始化函数和日志实例，复用已封装的组件
from utils import init_embedding, init_llm, init_faiss, logger
import ctypes
import numpy as np
import ctypes
import sys
from typing import List
from langchain_core.documents import Document

# 项目启动时立即初始化所有核心组件，仅执行1次，供后续所有业务逻辑调用
embedding = init_embedding()  # 初始化嵌入模型，用于文本/问题的向量化转换
llm = init_llm()              # 初始化大模型，用于基于知识库生成问答结果
# 初始化FAISS向量库，传入嵌入模型保证向量维度一致，返回向量库实例+本地持久化路径
vector_db, persist_dir = init_faiss(embedding)
# 加载C相似度计算库
def load_c_similarity_lib():
    try:
        if sys.platform == "win32":
            lib = ctypes.CDLL("src/similarity.dll")  # 注意路径：如果在项目根目录运行，这里是根目录下的dll
        else:
            lib = ctypes.CDLL("src/similarity.so")
        
        # 定义C函数参数和返回值
        lib.calc_similarity.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # vec1
            ctypes.POINTER(ctypes.c_float),  # vec2
            ctypes.c_int                     # len
        ]
        lib.calc_similarity.restype = ctypes.c_float
        return lib
    except Exception as e:
        logger.error(f"加载C相似度库失败：{str(e)}，降级使用Python计算")
        return None

# 初始化C相似度库
c_sim_lib = load_c_similarity_lib()

def c_cosine_similarity(vec1, vec2):
    """调用C语言余弦相似度计算，降级兼容Python实现"""
    if not c_sim_lib:
        # Python实现的余弦相似度（降级方案）
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)) * 1.2
    
    # 转换为C数组
    vec1_c = (ctypes.c_float * len(vec1))(*vec1)
    vec2_c = (ctypes.c_float * len(vec2))(*vec2)
    # 调用C函数
    return c_sim_lib.calc_similarity(vec1_c, vec2_c, len(vec1))

# 加载C动态库
lib = ctypes.CDLL("src/similarity.dll")
lib.calc_similarity.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.calc_similarity.restype = ctypes.c_float
def load_c_text_splitter():
    """加载C语言实现的文本分块动态库"""
    try:
        if sys.platform == "win32":
            # Windows加载dll
            lib = ctypes.CDLL("src/text_splitter.dll")
        else:
            # Linux/Mac加载so
            lib = ctypes.CDLL("src/text_splitter.so")
        
        # 定义函数参数和返回值类型
        # split_chinese_text函数：(const char*, int, int, char***, int*) -> void
        lib.split_chinese_text.argtypes = [
            ctypes.c_char_p,  # text（字节串）
            ctypes.c_int,     # chunk_size
            ctypes.c_int,     # overlap
            ctypes.POINTER(ctypes.POINTER(ctypes.c_char_p)),  # out_chunks
            ctypes.POINTER(ctypes.c_int)  # out_count
        ]
        lib.split_chinese_text.restype = None

        # free_chunks函数：(char**, int) -> void
        lib.free_chunks.argtypes = [
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_int
        ]
        lib.free_chunks.restype = None

        return lib
    except Exception as e:
        logger.error(f"加载C语言分块库失败：{str(e)}，降级使用Python分块器")
        return None

# 初始化C分块库
c_splitter_lib = load_c_text_splitter()

def c_text_splitter(documents: List[Document], chunk_size=500, chunk_overlap=50) -> List[Document]:
    """
    调用C语言分块器处理Document列表（替换RecursiveCharacterTextSplitter）
    :param documents: PDF加载后的Document列表
    :param chunk_size: 块最大字符数
    :param chunk_overlap: 重叠字符数
    :return: 分块后的Document列表
    """
    # 如果C库加载失败，降级使用Python的分块器
    if not c_splitter_lib:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "，", "、"]
        )
        return text_splitter.split_documents(documents)
    
    split_docs = []
    for doc in documents:
        # 1. 提取文本并转为C语言字节串（明确用UTF-8编码）
        text = doc.page_content.encode("utf-8")
        # 2. 定义输出变量
        out_chunks = ctypes.POINTER(ctypes.c_char_p)()
        out_count = ctypes.c_int(0)
        # 3. 调用C分块函数
        c_splitter_lib.split_chinese_text(
            text,
            ctypes.c_int(chunk_size),
            ctypes.c_int(chunk_overlap),
            ctypes.byref(out_chunks),
            ctypes.byref(out_count)
        )
        # 4. 解析分块结果，封装为Document（核心修复：适配编码）
        count = out_count.value
        for i in range(count):
            # 读取C返回的字节串
            chunk_bytes = out_chunks[i]
            if not chunk_bytes:
                continue
            # 修复编码问题：优先GBK解码（Windows默认），失败则UTF-8，再失败则忽略错误
            try:
                chunk_text = chunk_bytes.decode("gbk")  # 适配Windows中文编码
            except UnicodeDecodeError:
                try:
                    chunk_text = chunk_bytes.decode("utf-8")  # 降级为UTF-8
                except UnicodeDecodeError:
                    chunk_text = chunk_bytes.decode("utf-8", errors="ignore")  # 忽略无法解码的字符
            
            if chunk_text.strip():  # 过滤空块
                new_doc = Document(
                    page_content=chunk_text,
                    metadata={**doc.metadata}  # 继承原文档的元数据（页码、PDF路径等）
                )
                split_docs.append(new_doc)
        # 5. 释放C语言分配的内存（避免内存泄漏）
        c_splitter_lib.free_chunks(out_chunks, out_count)
    
    logger.info(f"C语言分块完成，总块数：{len(split_docs)}")
    return split_docs

# 初始化文本分块器，RAG核心优化点之一：解决长文本检索精度低、语义割裂问题
# 让文本分块更贴合中文语义，为后续精准检索打下基础
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,          # 每个文本块的最大字符数：控制块大小，避免块过大导致检索精度下降
    chunk_overlap=50,        # 相邻文本块的重叠字符数：避免语义割裂（如一句话被切成两个块）
    # 中文专属切分符：按「空行→换行→句号→感叹号→问号→逗号→顿号」优先级切分，贴合中文表达习惯
    separators=["\n\n", "\n", "。", "！", "？", "，", "、"]
)

def load_knowledge(pdf_paths):
    """加载跨境电商知识库（支持单PDF/多PDF批量加载，适配智谱64条/批限制）
    :param pdf_paths: 单个PDF路径（字符串）或多个PDF路径列表（如["a.pdf", "b.pdf"]）
    """
    # 统一格式：将单PDF路径转为列表，方便后续统一遍历处理
    if isinstance(pdf_paths, str):#是单文本就转成列表，不是就不用转
        pdf_paths = [pdf_paths]
    
    all_splits = []  # 存储所有PDF分块后的文本块
    for pdf_path in pdf_paths:
        # 校验单个PDF文件是否存在
        if not os.path.exists(pdf_path):
            logger.error(f"知识库文件不存在：{pdf_path}，跳过该文件")
            continue  # 某个PDF不存在则跳过，不影响其他PDF加载
        # 加载单个PDF并提取文本
        loader = PyPDFLoader(pdf_path)#PDF读取器
        documents = loader.load()#加载文字
        logger.info(f"加载PDF文档{pdf_path}，共{len(documents)}页")
        # 文本分块
        # splits = text_splitter.split_documents(documents)
        # 调用C语言实现的分块函数（替换Python分块器）
        splits = c_text_splitter(documents, chunk_size=500, chunk_overlap=50)
        logger.info(f"PDF{pdf_path}分块完成，共{len(splits)}个块")
        all_splits.extend(splits)  # 将当前PDF的文本块加入总列表
    
    # 校验是否有有效文本块（避免所有PDF都不存在/无内容）
    if not all_splits:
        logger.error("未加载到任何有效PDF知识库内容")
        raise ValueError("未加载到任何有效PDF知识库内容")#直接报错，终止整个加载流程，告诉使用者“没读到任何PDF内容，没法继续”
    
    # 核心改造：分批入库，适配智谱接口「单次≤64条」限制
    batch_size = 60  # 每批60条（留4条余量，避免边界问题）
    total = len(all_splits)
    for i in range(0, total, batch_size):
        # 按批次截取文本块
        batch_splits = all_splits[i:i+batch_size]
        # 分批存入FAISS（自动调用嵌入模型向量化）
        vector_db.add_documents(batch_splits)#把这一批小块转换成机器能看懂的“数字向量”，存到FAISS向量库里
        logger.info(f"已入库{min(i+batch_size, total)}/{total}条文本块")
    
    # 所有批次入库完成后，统一保存本地向量库到电脑本地（faiss_ec_db文件夹），下次启动不用重新加载
    vector_db.save_local(persist_dir)
    logger.info(f"所有PDF知识库入库完成，总文本块数：{total}")
    return all_splits

# 定义文档格式化函数：将检索到的原始文本块转换为标准化上下文，RAG抑制幻觉的关键步骤
# 参数docs为FAISS检索返回的相关文本块列表，返回格式化后的拼接字符串
def format_docs(docs):
    """格式化检索到的文档（解决幻觉）"""
    # 遍历所有检索到的文本块，为每个块添加「页码来源」+「文本内容」的格式
    # 页码+1是因为PDF加载的页码从0开始，符合人类的页码阅读习惯
    # 多个文本块之间用两个空行分隔，让大模型更容易区分不同块的内容
    return "\n\n".join([f"来源：第{doc.metadata['page']+1}页\n内容：{doc.page_content}" for doc in docs])


# 定义核心RAG问答函数：实现用户问题→向量检索→上下文拼接→大模型生成→结果返回的完整问答流程
# 每次用户提问都会执行，调用/qa接口时触发，参数query为用户的自然语言问题
# def rag_qa(query):
#     """核心RAG问答逻辑"""
#     # 1. 初始化FAISS检索器，将向量库转换为检索器模式，配置检索规则
#     retriever = vector_db.as_retriever(
#         # 检索规则配置：k=3表示召回最相关的3个文本块，score_threshold=0.7表示仅保留相似度≥70%的结果
#         # 过滤低相关结果，提升大模型回答的准确性，减少幻觉
#         # search_kwargs={"k": 3, "score_threshold": 0.7}
#         # search_kwargs={"k": 3}#强制返回最相似的 3 条
#         search_type="mmr", #相关内容分散，可改用最大边际相关性（MMR）检索，兼顾相关性和多样性
#         search_kwargs={"k": 3, "fetch_k": 10}#fetch_k“候选池大小”，k“最终输出数量”
#     )
# 执行相似性检索：将用户问题自动向量化后，在FAISS中检索相关文本块，返回结果列表
    # docs = retriever.get_relevant_documents(query)//get_relevant_documents已经弃用
    # docs = retriever.invoke(query)
    # # 记录日志，告知本次检索到的相关文本块数量，方便后续优化检索规则
    # logger.info(f"检索到{len(docs)}条相关文档")
    
    # # 若未检索到任何相关文本块，直接返回提示语，避免大模型基于通用知识编造答案
    # if not docs:
    #     return "未检索到相关信息，请确认问题是否准确。"
# 在rag_qa函数中替换FAISS默认检索：用C的相似度算法筛选结果
def rag_qa(query):
    # 1. 先用FAISS检索候选（fetch_k=20）
    retriever = vector_db.as_retriever(search_kwargs={"k": 20})
    docs = retriever.invoke(query)
    if not docs:
        return "未检索到相关信息"
    
    # 2. 用C语言的相似度算法重新排序（电商场景优化）
    query_vec = embedding.embed_query(query)  # 获取问题向量
    doc_scores = []
    for doc in docs:
        doc_vec = embedding.embed_query(doc.page_content)
        # 调用C函数计算相似度
        score = c_cosine_similarity(query_vec, doc_vec)
        doc_scores.append((doc, score))
    
    # 3. 取Top3高分文档
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    top_docs = [d[0] for d in doc_scores[:3]]
    
    # 2. 构建大模型提示词模板，Prompt工程核心：明确大模型的角色和回答规则，抑制幻觉
    prompt = ChatPromptTemplate.from_template("""
    你是跨境电商智能问答助手，仅根据提供的知识库回答问题，不要编造信息。
    如果知识库没有相关内容，直接回复“未检索到相关信息”。
    回答时需要标注信息来源（页码），确保准确性。
    
    知识库内容：
    {context}  # 占位符：后续会替换为格式化后的检索到的知识库上下文
    
    用户问题：{question}  # 占位符：后续会替换为用户的原始问题
    """)
    
    # 3. 构建RAG链式调用：将「检索→格式化→提示词拼接→大模型生成→结果解析」串联成流水线
    # 链式调用让整个流程更简洁、更工程化，无需手动分步执行每个环节
    rag_chain = (
        # 第一步：构造链式调用的输入参数，context为「检索→格式化」的结果，question直接透传用户问题
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt  # 第二步：将输入参数代入提示词模板，生成完整的大模型输入
        | llm     # 第三步：将完整提示词传入大模型，生成问答结果
        | StrOutputParser()  # 第四步：将大模型的复杂输出解析为纯文本，方便返回和展示
    )
    
    # 4. 执行RAG链式调用，获取最终问答结果，并做异常保护
    try:
        # 调用链式调用的invoke方法，传入用户问题执行全流程，得到生成结果
        result = rag_chain.invoke(query)
        # 记录成功日志，仅保留答案前50个字符，避免日志过长，同时记录核心信息
        logger.info(f"问答完成，用户问题：{query}，回答：{result[:50]}...")
        # 返回大模型生成的纯文本答案，供API接口层返回给用户
        return result
    # 捕获链式调用过程中的所有异常（如API调用失败、网络超时、检索异常等）
    except Exception as e:
        # 记录错误日志，告知具体的失败原因
        logger.error(f"问答执行失败：{str(e)}")
        # 返回友好的错误提示，避免直接抛出异常给用户
        return f"问答出错：{str(e)}"

# 本地测试入口：仅当直接运行当前py文件时才会执行，启动API服务时不会执行此部分
# 用于本地调试，无需启动API服务即可测试知识库加载和问答功能
if __name__ == "__main__":
    # 第一步：加载知识库（首次运行时取消注释执行1次，加载完成后重新注释，避免重复入库）
    # load_knowledge("data/raw/amazon_rules2.pdf")
    
    # 第二步：测试核心问答功能，定义测试问题
    test_query = "亚马逊FBA物流费用计算规则是什么？"
    # 打印测试问题，方便本地控制台查看输入
    print(f"问题：{test_query}")
    # 调用rag_qa函数执行问答，打印返回结果，查看输出效果
    print(f"回答：{rag_qa(test_query)}")