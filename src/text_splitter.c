// text_splitter.c - 中文文本递归分块核心逻辑（适配RAG场景）
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// 定义分隔符优先级（对应Python的separators=["\n\n", "\n", "。", "！", "？", "，", "、"]）
const char* separators[] = {"\n\n", "\n", "。", "！", "？", "，", "、"};
const int sep_count = 7; // 分隔符数量

// 去除字符串首尾空白字符（辅助函数）
char* trim(char* str) {
    // 去掉开头空格/换行
    while (isspace((unsigned char)*str)) str++;
    if (*str == 0) return str;
    
    // 去掉结尾空格/换行
    char* end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) end--;
    end[1] = '\0';
    return str;
}

// 递归分割文本（核心函数）
// 参数：text-待分块文本, chunk_size-块最大字符数, overlap-重叠字符数, chunks-存储分块结果的数组, chunk_count-分块数量
void split_text_recursive(const char* text, int chunk_size, int overlap, char*** chunks, int* chunk_count) {
    int text_len = strlen(text);
    // 文本长度小于等于chunk_size，直接作为一个块
    if (text_len <= chunk_size) {
        char* trimmed = trim(strdup(text));
        if (strlen(trimmed) > 0) { // 过滤空块
            (*chunk_count)++;
            *chunks = realloc(*chunks, (*chunk_count) * sizeof(char*));
            (*chunks)[*chunk_count - 1] = trimmed;
        } else {
            free(trimmed);
        }
        return;
    }

    // 按分隔符优先级尝试分割
    int split_pos = -1;
    for (int i = 0; i < sep_count; i++) {
        const char* sep = separators[i];
        int sep_len = strlen(sep);
        // 从chunk_size位置往前找分隔符（避免切在中间）
        int search_start = chunk_size - sep_len;
        if (search_start < 0) search_start = 0;
        
        for (int pos = search_start; pos >= 0; pos--) {
            if (strncmp(text + pos, sep, sep_len) == 0) {
                split_pos = pos + sep_len; // 分割位置在分隔符后
                break;
            }
        }
        if (split_pos != -1) break;
    }

    // 没找到分隔符，直接按chunk_size分割
    if (split_pos == -1) split_pos = chunk_size;

    // 截取当前块（0到split_pos）
    char* current_chunk = malloc(split_pos + 1);
    strncpy(current_chunk, text, split_pos);
    current_chunk[split_pos] = '\0';
    current_chunk = trim(current_chunk);
    if (strlen(current_chunk) > 0) {
        (*chunk_count)++;
        *chunks = realloc(*chunks, (*chunk_count) * sizeof(char*));
        (*chunks)[*chunk_count - 1] = current_chunk;
    } else {
        free(current_chunk);
    }

    // 计算下一个块的起始位置（保留重叠）
    int next_start = split_pos - overlap;
    if (next_start < 0) next_start = 0;

    // 递归处理剩余文本
    split_text_recursive(text + next_start, chunk_size, overlap, chunks, chunk_count);
}

// 对外暴露的分块函数（供Python调用）
// 参数：text-待分块文本, chunk_size-块大小, overlap-重叠长度, out_chunks-输出分块结果, out_count-输出块数量
#ifdef _WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

DLL_EXPORT void split_chinese_text(const char* text, int chunk_size, int overlap, char*** out_chunks, int* out_count) {
    *out_chunks = NULL;
    *out_count = 0;
    split_text_recursive(text, chunk_size, overlap, out_chunks, out_count);
}

// 释放分块结果内存（供Python调用）
DLL_EXPORT void free_chunks(char** chunks, int count) {
    for (int i = 0; i < count; i++) {
        free(chunks[i]);
    }
    free(chunks);
}