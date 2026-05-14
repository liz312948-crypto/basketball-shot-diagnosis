# Basketball Shot Diagnosis Web Demo

一个面向投篮视频诊断的 Web Demo。

当前版本的重点不是做完整平台，而是先把下面这条链路做实：

- 上传投篮视频
- 预检视频质量与机位
- 执行基础诊断
- 对高质量视频标记为可进入增强诊断
- 返回带可信度说明的结果页

## 当前能力

- Web 上传入口
- 异步任务状态流
- 视频预检与可信度分级
- 基础动作诊断结果整合
- 结果页展示总体结论、主要问题、训练建议、可信度说明
- 高质量视频的阶段拆解与模板对比入口
- 任务状态 JSON 持久化，服务重启后可恢复已存在任务

## 快速开始

```powershell
cd "C:\Users\HUAWEI\Documents\New project\basketball_clip_tool"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m app.web
```

打开：

```text
http://127.0.0.1:7860
```

## 发布成公网网址

本地地址 `127.0.0.1` 只能你自己访问。  
如果你要把网址发给别人，必须部署到公网服务器。

当前项目已经改成支持部署环境变量：

- `HOST`
- `PORT`

并且已附带 [render.yaml](/C:/Users/HUAWEI/Documents/New%20project/basketball_clip_tool/render.yaml)。

### Render 部署步骤

1. 把项目放到 GitHub 仓库
2. 登录 Render，新建一个 `Web Service`
3. 连接你的仓库
4. Render 会读取 `render.yaml`
5. 部署完成后，你会得到一个类似：

```text
https://basketball-shot-diagnosis.onrender.com
```

这个公网地址才是你能发给别人的网址。

### 部署后要知道的限制

- 当前上传的视频和任务状态仍保存在服务本地磁盘
- 这适合 demo，不适合正式商用
- 如果部署平台重启实例，历史上传和本地产物可能丢失
- 真正商用需要再接对象存储和数据库

## Web Demo 流程

1. 上传投篮视频
2. 服务端创建任务并进入 `uploaded -> precheck -> base-analysis -> rendering -> done/failed`
3. 结果页展示可信度、总体结论、主要问题、训练建议

## 推荐拍摄方式

- 优先使用侧面或 45 度机位
- 尽量完整拍到准备、起跳、出手、跟随动作
- 保持主体完整入镜
- 避免强遮挡和剧烈抖动

## 依赖说明

- `requirements.txt`: Web 服务、OpenCV、测试依赖
- `requirements-pose.txt`: 可选姿态识别依赖

## 当前限制

- 增强诊断目前基于启发式阶段拆解，强模型本体还没有接入
- 结果页已能工作，但更细的关键帧可视化叠加还在继续补强
- 任务记录会持久化到 `data/jobs` 附近的状态目录，但还没有数据库级别的并发与清理策略
- 当前仓库还没有直接替你完成上线，因为真正生成公网地址还需要你的部署平台账号
