"""通过交互式 Web viewer 可视化计算图。

数据流:
  1. 图数据序列化为 JSON 并直接内嵌进 HTML 的 ``window.__GRAPH_DATA__``,
     免去单独的 ``graph.json`` 文件和额外的 HTTP 请求。
  2. 外部 JS/CSS 资源 (d3, dagre) 只下载一次并缓存到平台相关的缓存目录,
     重复启动时无需重新下载。
  3. 一个轻量的 ``ThreadingHTTPServer`` 从临时目录服务这个自包含的 HTML 文件。
"""

from __future__ import annotations

import json
import os
import re
import socket
import tempfile
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
import urllib.request

# 控制台样式辅助 (保持可选, 使本模块可独立导入)
try:
    from .console import dim, green, yellow
except ImportError:  # pragma: no cover — standalone / offline import

    def dim(s):
        return str(s)

    def green(s):
        return str(s)

    def yellow(s):
        return str(s)


# ── Asset caching ────────────────────────────────────────────────────
def _cache_dir() -> Path:
    """返回下载的 Web 资源所在的持久缓存目录。"""
    env = os.environ.get("MODELPERF_CACHE_DIR")
    if env:
        p = Path(env)
    elif os.name == "nt":
        p = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "modelperf" / "cache"
    else:
        p = Path.home() / ".cache" / "modelperf"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _bundled_dir() -> Path:
    """存放前端旁内联的 JS 资源的目录。"""
    return Path(__file__).parent / "assets" / "vendor"


def _download_with_cache(url: str, cache: Path) -> Path:
    """把 *url* 解析为本地文件, 优先使用旁内联资源。

    解析顺序: 现有缓存副本 -> ``assets/vendor`` 中的旁内联副本 -> 网络下载。
    先检查旁内联副本使报告导出可完全离线（无需联网）。
    """
    parsed = urlparse(url)
    rel = Path(parsed.path.lstrip("/"))
    local = cache / rel
    if local.exists() and local.stat().st_size > 0:
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    bundled = _bundled_dir() / local.name
    if bundled.exists() and bundled.stat().st_size > 0:
        import shutil

        shutil.copy2(bundled, local)
        return local
    urllib.request.urlretrieve(url, local)
    return local


def _localize_assets(html: str, cache: Path) -> str:
    """把 *html* 里远端 CDN 的 URL 替换为缓存的本地相对路径。"""
    pattern = re.compile(r'(?:src|href)=["\'](https?://[^"\']+)["\']', re.I)
    urls = set(pattern.findall(html))
    for url in urls:
        try:
            local = _download_with_cache(url, cache)
            rel = local.relative_to(cache).as_posix()
            html = html.replace(url, rel)
        except Exception as exc:
            print(f"  ! Failed to cache {url}: {exc}")

    # 依赖加载器把 d3/dagre 的 CDN 地址写在 JS 数组里 (不是 <script src>),
    # 这里的 URL 同样要改写成本地相对路径, 否则离线打开时回退会失败。
    def _rewrite_sources(match: "re.Match") -> str:
        body = match.group("body")
        for url in set(_JS_URL_RE.findall(body)):
            try:
                local = _download_with_cache(url, cache)
                body = body.replace(url, local.relative_to(cache).as_posix())
            except Exception as exc:
                print(f"  ! Failed to cache {url}: {exc}")
        return "var sources = {%s};" % body

    return _SOURCES_RE.sub(_rewrite_sources, html, count=1)


_CDN_SCRIPT_RE = re.compile(
    r'<script[^>]*\bsrc=["\'](https?://[^"\']+)["\'][^>]*>\s*</script>', re.I
)

# 依赖加载器里的 ``sources`` 表 (形如 ``{ d3: ['https://...'], dagre: [...] }``)
# 及其中以字符串形式出现的脚本 URL (index.html 用 JS 动态注入脚本, 而非 <script src>)。
_SOURCES_RE = re.compile(r"var\s+sources\s*=\s*\{(?P<body>.*?)\}\s*;", re.S)
_JS_URL_RE = re.compile(r"""["'](https?://[^"']+\.js)["']""")


def _inline_assets(html: str, cache: Path) -> str:
    """把远端脚本资源内联进 *html* (单文件)。

    除了 ``<script src>`` 标签, 还要处理 index.html 的依赖加载器: 它把
    d3/dagre 的 CDN 地址写在 JS 数组里, 需要一并内联, 否则导出的单文件离线
    打开时加载不到这两个库。内联脚本注入到 ``</head>``, 由加载器的
    "已就绪即跳过" 逻辑短路, 不再发起网络请求。
    """

    def _replace(match: "re.Match") -> str:
        url = match.group(1)
        try:
            content = _download_with_cache(url, cache).read_text(encoding="utf-8")
        except Exception as exc:
            print(f"  ! Failed to inline {url}: {exc}")
            return match.group(0)
        return f"<script>{content}</script>"

    html = _CDN_SCRIPT_RE.sub(_replace, html)

    inlined: list[str] = []

    def _collect(match: "re.Match") -> str:
        for url in _JS_URL_RE.findall(match.group("body")):
            try:
                content = _download_with_cache(url, cache).read_text(encoding="utf-8")
            except Exception as exc:
                print(f"  ! Failed to inline {url}: {exc}")
                continue
            inlined.append(f"<script>{content}</script>")
        return match.group(0)

    html = _SOURCES_RE.sub(_collect, html, count=1)
    if inlined:
        html = html.replace("</head>", "".join(inlined) + "\n</head>", 1)
    return html


# ── Feature serialization ───────────────────────────────────────────
def _feature_to_dict(feature, _depth: int = 0, _max_depth: int = 5):
    """递归地把 feature 对象转成可 JSON 序列化的类型。"""
    if feature is None:
        return None
    if _depth > _max_depth:
        return repr(feature)
    if isinstance(feature, (str, int, float, bool)):
        return feature
    if isinstance(feature, dict):
        return {
            str(k): _feature_to_dict(v, _depth + 1, _max_depth)
            for k, v in feature.items()
        }
    if isinstance(feature, (list, tuple, set)):
        return [_feature_to_dict(v, _depth + 1, _max_depth) for v in feature]
    try:
        return {
            k: _feature_to_dict(v, _depth + 1, _max_depth)
            for k, v in vars(feature).items()
            if not k.startswith("_")
        }
    except Exception:
        return repr(feature)


def _export_graph_data(cg, id_str=str) -> dict:
    """把计算图 *cg* 序列化为前端期望的 JSON 结构。"""
    nodes, edges = [], []
    for node_id, data in cg.nodes(data=True):
        feat = _feature_to_dict(data.get("feature"))
        nodes.append(
            {
                "data": {
                    "id": id_str(node_id),
                    "label": str(feat.get("type", "Unknown")),
                    "op_type": str(feat.get("type", "Unknown")),
                    **{f"feat_{k}": v for k, v in feat.items()},
                }
            }
        )
    for u, v, ed in cg.edges(data=True):
        feat = _feature_to_dict(ed.get("feature"))
        su, sv = id_str(u), id_str(v)
        edges.append(
            {
                "data": {
                    "id": f"{su}->{sv}",
                    "source": su,
                    "target": sv,
                    **{f"feat_{k}": val for k, val in feat.items()},
                }
            }
        )

    # graph 级元数据（设备名 / 精度 / batch_size）
    graph_meta = {
        k: _feature_to_dict(v)
        for k, v in cg.graph.items()
        if k in ("devices", "dtype", "batch_size")
    }
    return {"elements": {"nodes": nodes, "edges": edges}, "graph": graph_meta}


# ── HTML loading ─────────────────────────────────────────────────────
def _load_index_html() -> str:
    """从磁盘重新读取 ``index.html`` (绕过 __pycache__)。"""
    html_path = Path(__file__).parent / "assets" / "index.html"
    return html_path.read_text(encoding="utf-8")


def _dumps_for_html(obj) -> str:
    """``json.dumps`` 并把 ``<`` 转义为 ``\\u003c``。

    图数据里若含 ``</script>``、``<!--`` 等序列，会提前关闭内嵌的
    ``<script>`` 标签并破坏页面；转义 ``<`` 后即可安全内嵌（JSON 中
    ``\\u003c`` 与 ``<`` 等价）。
    """
    return json.dumps(obj, ensure_ascii=False).replace("<", "\\u003c")


def _inject_graph_data(html: str, graph_data: dict, stats: dict | None = None) -> str:
    """把 *graph_data* (及可选的 *stats*) 作为 JS 全局变量内嵌进 *html*。"""
    payload = _dumps_for_html(graph_data)
    injection = f"<script>window.__GRAPH_DATA__={payload};</script>"
    if stats is not None:
        stats_payload = _dumps_for_html(stats)
        injection += f"\n  <script>window.__GRAPH_STATS__={stats_payload};</script>"
    return html.replace("</head>", f"  {injection}\n</head>", 1)


# ── HTTP server ──────────────────────────────────────────────────────
class _SilentHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class GraphWebViewer:
    """服务一个内嵌计算图的自包含 HTML 页面。"""

    def __init__(
        self, cg, stats: dict | None = None, port: int = 0, host: str = "127.0.0.1"
    ):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="cg_view_"))
        cache = _cache_dir()

        graph_data = _export_graph_data(cg)
        html = _load_index_html()
        html = _localize_assets(html, cache)
        html = _inject_graph_data(html, graph_data, stats)

        (self.tmpdir / "index.html").write_text(html, encoding="utf-8")
        # Copy cached assets into tmpdir so the server can serve them
        _copy_tree(cache, self.tmpdir)

        self.host = host
        self.port = port or _find_free_port()
        self.httpd = ThreadingHTTPServer(
            (host, self.port),
            lambda *a, **kw: _SilentHandler(*a, directory=str(self.tmpdir), **kw),
        )
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def open(self) -> str:
        self.thread.start()
        # 绑定 0.0.0.0 时用回环地址打开, 局域网访问需用户手动换 IP
        host = "127.0.0.1" if self.host in ("0.0.0.0", "") else self.host
        url = f"http://{host}:{self.port}/"
        webbrowser.open(url)
        print("  " + green("●") + " " + dim("viewer ready") + "  " + url)
        return url

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)


def _copy_tree(src: Path, dst: Path):
    """递归地把 *src* 复制进 *dst* (src == dst 或缺失时跳过)。"""
    if not src.exists() or src.resolve() == dst.resolve():
        return
    import shutil

    for item in src.iterdir():
        s = dst / item.name
        if item.is_dir():
            if s.exists():
                shutil.rmtree(s)
            shutil.copytree(item, s)
        else:
            shutil.copy2(item, s)


# ── Public API ───────────────────────────────────────────────────────
def export_html(cg, path, stats: dict | None = None, inline: bool = True) -> Path:
    """把计算图写为一个独立的 HTML 文件。

    参数
    ----
    cg : networkx.DiGraph
        带节点/边 ``feature`` 属性的计算图。
    path : str | os.PathLike
        输出 HTML 路径; 父目录按需创建。
    stats : dict | None
        预计算的图统计 (来自 ``analyze_graph``)。为 *None* 时,
        由前端从内嵌的图数据自行计算。
    inline : bool
        为 *True* (默认) 时把 CDN JS 内联进 HTML, 使单文件完全自包含。
        为 *False* 时保留相对资源引用, 并把缓存资源复制到输出文件旁。

    返回
    ----
    pathlib.Path
        写出的 HTML 路径。
    """
    out = Path(path)
    cache = _cache_dir()

    html = _load_index_html()
    # 先确保输出目录存在：非 inline 分支会向其复制缓存资源
    out.parent.mkdir(parents=True, exist_ok=True)
    if inline:
        html = _inline_assets(html, cache)
    else:
        html = _localize_assets(html, cache)
        _copy_tree(cache, out.resolve().parent)

    html = _inject_graph_data(html, _export_graph_data(cg), stats)

    out.write_text(html, encoding="utf-8")
    return out


def visualize_graph(
    cg, stats: dict | None = None, port: int = 8080, host: str = "127.0.0.1"
):
    """启动 web viewer 并阻塞直到 Ctrl+C。

    参数
    ----
    cg : networkx.DiGraph
        带节点/边 ``feature`` 属性的计算图。
    stats : dict | None
        预计算的图统计 (来自 ``analyze_graph``)。为 *None* 时,
        由前端从内嵌的图数据自行计算。
    port : int
        Web viewer 端口 (0 = 自动选择)。
    host : str
        绑定地址; 用 ``0.0.0.0`` 可在局域网暴露 viewer。
    """
    viewer = GraphWebViewer(cg, stats=stats, port=port, host=host)
    print("\n" + dim("  temp dir  ") + str(viewer.tmpdir))
    url = viewer.open()
    print(dim("  press Ctrl+C to stop") + "\n")
    try:
        # Poll the daemon server thread so an unexpected exit (e.g. the server
        # crashes or fails to bind) also unblocks us instead of busy-waiting
        # forever. Works on both Unix and Windows (no signal.pause needed).
        while viewer.thread.is_alive():
            viewer.thread.join(timeout=1)
    except KeyboardInterrupt:
        print("\n  " + yellow("●") + " " + dim("server stopped"))
    finally:
        viewer.stop()
    return viewer
