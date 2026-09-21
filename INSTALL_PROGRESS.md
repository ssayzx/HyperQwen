# HyperQwen 安装进度清单（autoDL 裸机 venv 安装）

> 环境快照（2026-09-21）
> - GPU: RTX 3090 24 GB（本仓库的基准测试卡，最佳匹配）
> - 驱动: 580.105.08（CUDA 13.0）
> - CUDA toolkit: /usr/local/cuda-12.8，含 `curand.h`（DFlash2 JIT 可用）
> - Python: 3.12.3（miniconda base），venv 路径 `./venv`
> - 磁盘: /root/autodl-tmp 共 50 GB，预计占用 ~40 GB（venv + 模型 + 变体）
> - 网络: huggingface.co 直连不可用 → 使用 `HF_ENDPOINT=https://hf-mirror.com`
> - pip: 已配置 aliyun 镜像
> - 仓库: **/root/autodl-tmp/HyperQwen-new**（commit feaffb6）
  （原 /root/autodl-tmp/qwen/HyperQwen 归 worker 用户所有、本会话只读，
  已整体复制迁移并重建 venv；后续所有操作都在新路径执行）
> - 安装依据: docs/install.md（无 Docker / 沙箱方案）

## ✅ 安装完成（2026-09-21 16:0x，会话用户 shen，路径 /root/autodl-tmp/HyperQwen-new）

## 已完成

- [x] 仓库克隆（origin: syv-ai/HyperQwen）
- [x] 环境核查：GPU / Python / CUDA / 磁盘 / 网络
- [x] 创建 venv（`python3 -m venv venv`，pip 已升级）

## 进行中

- [x] **安装 Python 依赖**（后台任务 bl5r7c9b2，日志 /tmp/pip-install.log；huggingface_hub 已提前装好）
  ```bash
  venv/bin/pip install vllm==0.28.0 huggingface_hub hf_transfer ninja \
    flashinfer-python flashinfer-cubin==0.6.13 pandas
  ```
  注意：flashinfer-cubin 只发布到 0.6.13，需 `FLASHINFER_DISABLE_VERSION_CHECK=1`
  （启动脚本会自动导出）；不要用降级 flashinfer-python 的方式解决版本不匹配。
  验证：`venv/bin/python -c "import vllm; print(vllm.__version__)"` → 0.28.0

## 待做（按顺序）

- [x] **1. 下载模型**（~19.5 GB → `models/Qwen3.8-27B-W4A16-AutoRound`）
  ```bash
  HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 venv/bin/hf download \
    dbirks/Qwen3.8-27B-W4A16-AutoRound \
    --local-dir models/Qwen3.8-27B-W4A16-AutoRound
  ```
  （注意：HF_XET_HIGH_PERFORMANCE=1 会导致 Xet CAS 401——cas-server.xethub.hf.co
  不走镜像、直连被墙；必须用 HF_HUB_DISABLE_XET=1 走普通 HTTP 分块下载。后台任务 bzxo2mqbs）

- [x] **2. prepare/ 模型量化脚本**（CPU，几分钟）
  ```bash
  venv/bin/python prepare/quant_lm_head.py models/Qwen3.8-27B-W4A16-AutoRound
  venv/bin/python prepare/quant_embed.py   models/Qwen3.8-27B-W4A16-AutoRound
  venv/bin/python prepare/quant_mtp.py     models/Qwen3.8-27B-W4A16-AutoRound
  venv/bin/python prepare/build_draft_vocab.py models/Qwen3.8-27B-W4A16-AutoRound \
    --ids prepare/draft_vocab_ids.json
  venv/bin/python prepare/fetch_fast_variant.py    # single-user fast 变体 ~1 GB
  venv/bin/python prepare/fetch_dflash2.py         # 可选：DFlash2 drafter 1.2 GB
  ```

- [x] **3. 给 venv 内 vLLM 打补丁**（按 patches/series 顺序）
  ```bash
  sed -e 's/#.*//' -e 's/^[[:space:]]*//;s/[[:space:]]*$//' -e '/^$/d' patches/series |
  while IFS= read -r name; do
    case "$name" in
      dflash2-backport.patch) echo "skip $name (DFlash2 is native in vLLM 0.28.0)"; continue ;;
    esac
    patch -p1 -d venv/lib/python3.12/site-packages/vllm < "patches/$name"
  done
  ```

- [ ] **4. （可选）安装 KVarN 4/2-bit KV cache**（262k 长上下文）
  ```bash
  bash kvarn/install.sh
  ```

- [x] **5. 生成 API key**（服务绑定 0.0.0.0，无 key 即裸奔）
  ```bash
  openssl rand -hex 24 > api_key.txt
  ```

- [x] **6. 校验安装**（verify.sh --no-server → OK, 0 failures；KVarN 未装，可选）
  ```bash
  bash verify.sh --no-server
  ```
  检查项：venv、vLLM 0.28.0、全部补丁已应用、模型已完成 4 项 requant

- [x] **7. 启动**（模式 B 单用户默认，后台任务 bka3huxaw，日志 /tmp/qwen-server.log）（一卡一次跑一种模式）
  | 模式 | 启动 | 适用 |
  |---|---|---|
  | B 单用户默认 | `bash single-user/start_qwen.sh` | 1~几人聊天，64k 上下文，127 tok/s |
  | C 复现型 | `.env` 加 `DFLASH_TOKENS=15` 后同上 | 输出大量引用 prompt（代码/RAG），381 tok/s |
  | D 长上下文 | `SPEC=mtp CTX=long` 后同上 | 150k 上下文 |
  | E 超长上下文 | 再加 `CTX=huge` | 240k 上下文 |
  | A 批处理 | `bash batch/start_qwen.sh` | API 后端，64 并发 ~1035 tok/s |
  服务端口 `:18020`；首次启动需数分钟（torch.compile / CUDA graph / flashinfer JIT）。

- [x] **8. 冒烟测试**（✓ /v1/chat/completions 返回正常，vllm-0.28.0-9dd24f89）
  ```bash
  curl http://localhost:18020/v1/chat/completions \
    -H "Authorization: Bearer $(cat api_key.txt 2>/dev/null)" \
    -H "Content-Type: application/json" \
    -d '{"model": "qwen3.8-27b",
         "messages": [{"role": "user", "content": "hej"}],
         "chat_template_kwargs": {"enable_thinking": false}}'
  ```

## 风险 / 注意事项

- **磁盘余量紧**：50 GB 总量 vs ~40 GB 需求，模型下载完成后可用 `du -sh models venv` 复核；如不足，可跳过 fetch_dflash2（-1.2 GB）
- **显存**：24 GB 卡单模式运行；batch 与 single 不可同启
- **补丁升级失效**：vLLM 版本一变，patches/series 需重新应用
- **autoDL 特有**：`nvidia-smi -pl 250` 限功耗需宿主机 sudo，容器内通常不可用，可跳过（仅影响基准测试可比性）
- 参考文档：docs/install.md · docs/gotchas.md · single-user/README.md · batch/README.md
