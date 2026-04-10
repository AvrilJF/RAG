// main.go - Go的API服务层
// # 编译Go代码
// go build -o ec-rag-api main.go
// # 启动服务
// ./ec-rag-api

package main

import (
    "encoding/json"
    "fmt"
    "net/http"
    "os/exec"
    "strings"
)

// 定义请求体结构（对应Python的QAQuery）
type QAQuery struct {
    Question string `json:"question"`
}

// 定义响应体结构
type QAResponse struct {
    Code     string `json:"code"`
    Question string `json:"question"`
    Answer   string `json:"answer"`
}

// 问答接口
func qaHandler(w http.ResponseWriter, r *http.Request) {
    var query QAQuery
    err := json.NewDecoder(r.Body).Decode(&query)
    if err != nil {
        http.Error(w, "参数错误", http.StatusBadRequest)
        return
    }

    // 调用Python的rag_qa函数（通过命令行）
    cmd := exec.Command("python", "-c", 
        fmt.Sprintf(`from rag_engine import rag_qa; print(rag_qa("%s"))`, query.Question))
    output, err := cmd.CombinedOutput()
    if err != nil {
        http.Error(w, fmt.Sprintf("问答失败：%s", err), http.StatusInternalServerError)
        return
    }

    // 构造响应
    resp := QAResponse{
        Code:     "200",
        Question: query.Question,
        Answer:   strings.TrimSpace(string(output)),
    }

    // 返回JSON
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(resp)
}

// 健康检查接口
func healthHandler(w http.ResponseWriter, r *http.Request) {
    json.NewEncoder(w).Encode(map[string]string{
        "status":  "ok",
        "service": "ec-rag-service",
    })
}

func main() {
    // 注册路由
    http.HandleFunc("/health", healthHandler)
    http.HandleFunc("/qa", qaHandler)
    // 启动服务（Go的高并发优势）
    fmt.Println("Go API服务启动：http://0.0.0.0:8000")
    http.ListenAndServe(":8000", nil)
}
//启动服务后，浏览器打开：http://localhost:8000/docs；