#!/usr/bin/env python3
"""Live terminal dashboard for the HyperQwen vLLM server (port 18020).

Zero dependencies (stdlib only). Polls /metrics every interval, renders
per-interval rates from counter deltas plus the engine's own gauges.

Usage:
    venv/bin/python bench/live_dash.py            # refresh every 1 s
    venv/bin/python bench/live_dash.py --once     # single frame (non-interactive)
    INTERVAL=0.5 DASH_URL=... venv/bin/python bench/live_dash.py
"""

import os
import subprocess
import sys
import time
import urllib.request

BASE = os.environ.get("DASH_URL", "http://localhost:18020")
INTERVAL = float(os.environ.get("INTERVAL", "1.0"))


def fetch_metrics():
    """Return {metric_series: value} from /metrics (last sample wins)."""
    req = urllib.request.Request(BASE + "/metrics")
    out = {}
    with urllib.request.urlopen(req, timeout=5) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line or line.startswith("#"):
                continue
            fields = line.rsplit(" ", 1)
            if len(fields) != 2:
                continue
            try:
                out[fields[0]] = float(fields[1])
            except ValueError:
                pass
    return out


def find(metrics, prefix):
    """First value whose series name starts with prefix."""
    for k, v in sorted(metrics.items()):
        if k.startswith(prefix):
            return v
    return 0.0


def spark(samples, width=44, ceil_=None):
    """Unicode sparkline of the last `width` samples."""
    if not samples:
        return ""
    data = samples[-width:]
    top = ceil_ if (ceil_ and ceil_ > 0) else (max(data) or 1.0)
    blocks = " ▁▂▃▄▅▆▇█"
    return "".join(
        blocks[min(int(v / top * (len(blocks) - 1)), len(blocks) - 1)]
        for v in data
    )


def render(now, rate, prompt_rate, hist, avg_ttft, n_ttft, avg_e2e, n_e2e, total_tok):
    print("\033[2J\033[H")  # clear screen, cursor home
    print("\033[1mHyperQwen 服务实时看板\033[0m   " + time.strftime("%H:%M:%S"))
    print("=" * 62)

    running = int(find(now, "vllm:num_requests_running"))
    waiting = int(find(now, "vllm:num_requests_waiting"))
    kv = find(now, "vllm:kv_cache_usage_perc")
    hits = find(now, "vllm:prefix_cache_hits_total")
    queries = find(now, "vllm:prefix_cache_queries_total")
    prefix = (hits / queries) if queries else 0.0
    print(f"并发/排队     running={running}  waiting={waiting}")
    print(f"KV 缓存占用   {kv*100:5.1f} %    前缀缓存命中 {prefix*100:5.1f} %")

    gpu = gpu_stats()
    if gpu:
        util, mem_u, mem_t, temp, pwr = gpu
        print(f"\033[1mGPU 利用率    {util:5.1f} %\033[0m  "
              f"显存 {mem_u/1024:.1f}/{mem_t/1024:.0f} GiB  "
              f"{temp:.0f}°C  {pwr:.0f} W")

    if rate is not None:
        pre = (f"{prompt_rate:6.0f} tok/s (预填充)" if prompt_rate else "  --   (解码中)")
        print(f"\033[1m生成速率      {rate:6.1f} tok/s\033[0m  {pre}   累计 {int(total_tok)}")
    else:
        print("生成速率      (采样中，等待下一个周期 ...)")
    print("-" * 62)

    print(f"TTFT 平均      {avg_ttft:5.2f} s  (n={n_ttft})")
    print(f"E2E  平均      {avg_e2e:5.2f} s  (n={n_e2e}, "
          f"平均 {(total_tok / n_e2e) if n_e2e else 0:.0f} tok/请求)")
    print("-" * 62)
    print("生成速率历史   " + spark(hist))
    print("=" * 62)
    print("Ctrl-C 退出 · 数据源 GET /metrics")


def gpu_stats():
    """(util%, mem_used, mem_total, temp, power) from nvidia-smi, or None."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,"
             "temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3).stdout.strip()
        util, mem_u, mem_t, temp, pwr = out.splitlines()[0].split(",")
        return (float(util), float(mem_u), float(mem_t), float(temp), float(pwr))
    except Exception:
        return None


def snapshot(now):
    """Extract cumulative-latency aggregates and totals from a metrics dict."""

    def cum(prefix):
        s = find(now, prefix + "_seconds_sum")
        c = find(now, prefix + "_seconds_count")
        return (s / c if s and c else 0.0), int(c or 0)

    avg_ttft, n_ttft = cum("vllm:time_to_first_token")
    avg_e2e, n_e2e = cum("vllm:e2e_request_latency")
    total_tok = find(now, "vllm:generation_tokens_total")
    return avg_ttft, n_ttft, avg_e2e, n_e2e, total_tok


def main():
    once = "--once" in sys.argv
    prev_tok = prev_prompt = None
    if once:
        # two polls so the single frame shows a computed rate
        try:
            m = fetch_metrics()
            prev_tok = find(m, "vllm:generation_tokens_total")
            prev_prompt = find(m, "vllm:prompt_tokens_total")
        except Exception:
            prev_tok, prev_prompt = 0.0, 0.0
        time.sleep(INTERVAL)
    hist = []
    last = time.time()
    first = not once
    try:
        while True:
            try:
                now = fetch_metrics()
            except Exception as e:
                print(f"\rmetrics 获取失败: {e}     ", end="", flush=True)
                time.sleep(INTERVAL)
                continue
            t = time.time()
            dt = t - last
            tok = find(now, "vllm:generation_tokens_total")
            prompt = find(now, "vllm:prompt_tokens_total")
            rate = pre_rate = None
            if not first and dt > 0:
                rate = (tok - prev_tok) / dt
                pre_rate = max(0.0, (prompt - prev_prompt) / dt)
                hist.append(rate)
            render(now, rate, pre_rate, hist, *snapshot(now))
            prev_tok, prev_prompt = tok, prompt
            first = False
            last = t
            if once:
                break
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
