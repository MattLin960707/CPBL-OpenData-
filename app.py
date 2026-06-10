# app.py
# CPBL / 野球革命 OpenData Streamlit 分析系統
# 使用方式：把本檔案與 CPBL-2024-TaiwanSeries-OpenData.json 放在同一個資料夾，執行：streamlit run app.py

from __future__ import annotations

import hashlib
import json
import math
import pickle
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as font_manager
import numpy as np
import pandas as pd
import streamlit as st


# =========================
# 基本設定
# =========================

st.set_page_config(
    page_title="CPBL 野球革命數據分析 App",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "⚾ CPBL 野球革命 OpenData 分析系統"
APP_SUBTITLE = "支援台灣大賽、單場、整季 OpenData：可依所屬球隊、守位、姓名/背號篩選，並分析球員、對戰、投球、擊球、跑壘與 WPA/RE24。"

DEFAULT_MAX_SCATTER_POINTS = 6000
DEFAULT_TABLE_HEIGHT = 520
DEFAULT_MAX_DISPLAY_ROWS = 3000
# 中職資料球速通常為 km/h；超出此範圍多半是人工紀錄/轉換錯值，統計球速時排除。
VALID_VELO_MIN = 60
VALID_VELO_MAX = 170
# 進壘點座標若超過這個範圍，通常是人工標點或轉換異常；保留 raw，但統計/繪圖排除。
VALID_COORD_ABS_LIMIT = 220
CACHE_DIR = Path('.cpbl_cache')
DATAFRAME_KEYS = ['games', 'scores', 'batters_game', 'pitchers_game', 'pa', 'events', 'runners']


# =========================
# 表格欄位說明
# =========================

STAT_HELP: Dict[str, str] = {
    # 一般/比賽
    "G": "出賽場數或比賽編號。若在團隊總表中，G 通常代表該隊在目前篩選條件下的出賽場數。",
    "date": "比賽日期。",
    "stadium": "比賽球場。",
    "team": "所屬球隊。",
    "PR": "Percentile Rank，百分位排名。PR 80 代表該指標表現約優於比較群中 80% 的球員；PR 越高代表該指標越好。",
    "等級": "依 PR 分成頂尖、優秀、中上、平均附近、偏低、需加強。",
    "比較群": "PR 的比較母體。打者通常以達到最低 PA 的打者比較；投手通常以達到最低 BF 的投手比較。",
    "opponentTeam": "對戰球隊。",
    "winner": "該場比賽勝隊。",
    "W": "勝場數。公式：該隊獲勝場次總和。",
    "L": "敗場數。公式：該隊輸球場次總和。",
    "W%": "勝率。公式：W / (W + L)。代表球隊在目前篩選範圍內的贏球比例。",
    "得分": "球隊總得分。",
    "失分": "球隊總失分。",
    "分差": "得分 - 失分。正值代表得分多於失分，負值代表失分多於得分。",
    "得分/G": "場均得分。公式：總得分 / 出賽場數。",
    "失分/G": "場均失分。公式：總失分 / 出賽場數。",
    "分差/G": "場均分差。公式：(總得分 - 總失分) / 出賽場數。",

    # 打擊基本
    "PA": "Plate Appearance，打席。只要打者完成一次打席通常就算 PA，包括安打、出局、保送、觸身、犧牲打等。",
    "AB": "At Bat，打數。安打率、長打率的分母。保送、觸身球、犧牲短打、犧牲飛球通常不算 AB。",
    "R": "Run，得分。球員本人回到本壘得分的次數。",
    "H": "Hit，安打。打者安全上壘且紀錄為安打的次數。",
    "1B": "一壘安打。公式概念：H - 2B - 3B - HR。",
    "2B": "二壘安打。打者靠擊球安全上到二壘的安打。",
    "3B": "三壘安打。打者靠擊球安全上到三壘的安打。",
    "HR": "Home Run，全壘打。打者擊球後繞完所有壘包得分。",
    "RBI": "Runs Batted In，打點。打者的打席結果使隊友或自己得分時，多數情況會記 RBI；雙殺、失誤等特殊情況可能不記。",
    "BB": "Base on Balls，四壞球保送。打者獲得四個壞球而上一壘。",
    "IBB": "Intentional Walk，故意四壞球。",
    "HBP": "Hit By Pitch，觸身球。投球擊中打者且打者獲准上一壘。",
    "SO": "Strikeout，三振。打者在該打席被三振出局。",
    "SH": "Sacrifice Bunt，犧牲短打。通常用出局換跑者推進。",
    "SF": "Sacrifice Fly，犧牲飛球。飛球出局但使跑者得分。",
    "E": "Error，失誤相關紀錄。此資料中多用在打者 box 裡的失誤上壘或守備失誤相關欄位。",
    "SB": "Stolen Base，盜壘成功。",
    "CS": "Caught Stealing，盜壘失敗。",
    "GIDP": "Ground Into Double Play，滾地雙殺。打者擊出滾地球造成雙殺。",
    "DP": "Double Play，雙殺相關紀錄。",
    "TP": "Triple Play，三殺相關紀錄。",

    # 打擊進階
    "AVG": "Batting Average，打擊率。公式：H / AB。代表每個打數形成安打的比例，但不包含保送等上壘方式。",
    "OBP": "On-base Percentage，上壘率。公式：(H + BB + HBP) / (AB + BB + HBP + SF)。衡量打者避免出局、成功上壘的能力。",
    "SLG": "Slugging Percentage，長打率。公式：Total Bases / AB = (1B + 2×2B + 3×3B + 4×HR) / AB。衡量每打數平均能打下多少壘打數。",
    "OPS": "On-base Plus Slugging，整體攻擊指標。公式：OBP + SLG。把上壘能力與長打能力加總，常用來快速比較打者攻擊火力。",
    "TB": "Total Bases，壘打數。公式：1B + 2×2B + 3×3B + 4×HR。",
    "BB%": "保送率。打者端公式：BB / PA；投手端公式：BB / BF。打者越高代表選球/上壘能力較好，投手越高代表控球問題較多。",
    "K%": "三振率。打者端公式：SO / PA；投手端公式：SO / BF。打者越低通常代表較不容易被三振，投手越高通常代表製造三振能力較好。",
    "ISO": "Isolated Power，純長打率。公式：SLG - AVG。用來把長打能力從打擊率中拆出來看。",
    "BABIP": "Batting Average on Balls In Play，場內球被安打率。常用來觀察擊球落地後變成安打的比例，會受擊球品質、防守與運氣影響。",
    "PA/G": "場均打席。公式：PA / G。",
    "AB/G": "場均打數。公式：AB / G。",
    "R/G": "場均得分。公式：R / G。",
    "H/G": "場均安打。公式：H / G。",
    "2B/G": "場均二壘安打。公式：2B / G。",
    "3B/G": "場均三壘安打。公式：3B / G。",
    "HR/G": "場均全壘打。公式：HR / G。",
    "RBI/G": "場均打點。公式：RBI / G。",
    "BB/G": "場均保送。公式：BB / G。",
    "SO/G": "場均三振。公式：SO / G。",
    "SB/G": "場均盜壘成功。公式：SB / G。",
    "CS/G": "場均盜壘失敗。公式：CS / G。",
    "GIDP/G": "場均滾地雙殺。公式：GIDP / G。",

    # 投球
    "IP": "Innings Pitched，投球局數。",
    "IP顯示": "投球局數的棒球格式。例如 5.1 代表 5 局又 1 個出局數，不是 5.1 個十進位局數。",
    "IPOuts": "投球出局數。公式：IPOuts / 3 = 十進位投球局數。",
    "NP": "Number of Pitches，投球數。",
    "BF": "Batters Faced，面對打者數。",
    "ER": "Earned Run，責失分。排除部分因守備失誤等因素造成的非自責失分。",
    "ERA": "Earned Run Average，防禦率。公式：ER × 9 / IP。代表投手每 9 局平均責失分，越低越好。",
    "WHIP": "Walks plus Hits per Inning Pitched，每局被上壘率。公式：(BB + H) / IP。衡量投手每局讓多少打者靠安打或保送上壘，越低通常越好。",
    "K-BB%": "三振率減保送率。公式：K% - BB%。常用來快速觀察投手壓制力與控球綜合表現，越高通常越好。",
    "NP/IP": "每局用球數。公式：NP / IP。越低通常代表投球效率較好。",
    "NP/BF": "每面對一名打者平均用球數。公式：NP / BF。",
    "IP/G": "場均投球局數。公式：IP / G。",
    "NP/G": "場均投球數。公式：NP / G。",
    "BF/G": "場均面對打者數。公式：BF / G。",
    "ER/G": "場均責失分。公式：ER / G。",
    "HB": "Hit Batter，投手造成觸身球的次數。",
    "HB/G": "場均觸身球。公式：HB / G。",

    # 每球與投球事件
    "pitches": "投球事件總數。通常只統計 type = PITCH 的 event。",
    "Pitches/G": "場均投球事件數。公式：pitches / G。",
    "strikes": "好球事件數。資料中 isStrike 為 True 的投球。",
    "balls": "壞球事件數。資料中 isBall 為 True 的投球。",
    "Strike%": "好球率。公式：strikes / pitches。代表投手把球投成好球或使打者形成好球結果的比例。",
    "Ball%": "壞球率。公式：balls / pitches。",
    "Whiff%": "揮空率。此 app 公式：pitchCode = SW 的球數 / pitches。代表投球造成揮棒落空的比例。",
    "InPlay%": "進場率。公式：inPlay / pitches。代表投球被打進場內形成擊球事件的比例。",
    "avgVelo": "平均球速。以清理後的有效球速計算；低於 60 或高於 170 km/h 的異常值會排除。",
    "maxVelo": "最快球速。以清理後的有效球速取最大值；低於 60 或高於 170 km/h 的異常值會排除。",
    "velocity": "清理後球速。資料單位通常為 km/h；若原始球速低於 60 或高於 170，視為異常值並顯示空值。",
    "velocityRaw": "原始球速，未清理。若看到 463、182 這類明顯不合理數字，通常是原始資料紀錄或轉換錯值。",
    "invalidVelocity": "是否為異常球速。True 表示原始球速低於 60 或高於 170，已從 avgVelo/maxVelo 統計中排除。",
    "invalidVelo": "異常球速筆數。原始球速低於 60 或高於 170 的投球會計入。",
    "pitchType": "球種代碼，例如 FF、SL、CH、CU 等。",
    "pitchTypeZh": "球種中文名稱。",
    "pitchCode": "單球結果代碼，例如 B 壞球、S 好球、F 界外、SW 揮空、H 擊入場內等。",
    "pitchCodeZh": "單球結果中文名稱。",
    "coordX": "清理後投球進壘點 X 座標，捕手視角左右位置。絕對值超過 220 的異常座標會排除。",
    "coordY": "清理後投球進壘點 Y 座標，高低位置。絕對值超過 220 的異常座標會排除。",
    "coordXRaw": "原始投球進壘點 X 座標，未清理。",
    "coordYRaw": "原始投球進壘點 Y 座標，未清理。",
    "invalidCoord": "是否為異常進壘點座標。True 表示 coordXRaw 或 coordYRaw 的絕對值超過 220，已從進壘點圖中排除。",

    # 打席與情境
    "inning": "局數。",
    "outs": "打席開始時的出局數。",
    "bases": "打席開始時的壘包狀態代碼。",
    "isRISP": "得點圈情境。二壘或三壘有人時為 True；RISP = Runner In Scoring Position。",
    "RISP_PA": "得點圈打席數。二壘或三壘有人時完成的打席數。",
    "RISP_AB": "得點圈打數。得點圈情境下的 AB。",
    "RISP_H": "得點圈安打。得點圈情境下的 H。",
    "RISP_RBI": "得點圈打點。得點圈情境下累積的 RBI。",
    "RISP_AVG": "得點圈打擊率。公式：得點圈 H / 得點圈 AB。",
    "RISP_OBP": "得點圈上壘率。公式：(H + BB + HBP) / (AB + BB + HBP + SF)，只計得點圈情境。",
    "RISP_SLG": "得點圈長打率。公式：TB / AB，只計得點圈情境。",
    "RISP_OPS": "得點圈 OPS。公式：RISP_OBP + RISP_SLG。",
    "basesLabel": "打席開始時的壘包狀態文字。",
    "endBases": "打席結束後的壘包狀態代碼。",
    "endBasesLabel": "打席結束後的壘包狀態文字。",
    "count": "球數，格式為壞球數-好球數。",
    "scoreBefore": "打席開始前比分。",
    "scoreAfter": "打席結束後比分。",
    "runsThisPA": "該打席攻方新增得分。公式：打席後攻方分數 - 打席前攻方分數。",
    "RBI": "打點。打者的打席結果使得分成立時，依棒球記錄規則計入。",
    "result": "打席結果代碼，例如 1B、2B、HR、SO、BB、GO、FO 等。",
    "resultZh": "打席結果中文名稱。",
    "pitchCount": "該打席實際投球事件數。",
    "pitchCodes": "該打席投球序列。",

    # 勝率/得分期望
    "homeWE": "主隊 Win Expectancy，主隊當下勝率期望。通常由局數、比分、出局數、壘包狀態等情境估計。",
    "WPA": "Win Probability Added，勝率貢獻。公式概念：事件後勝率期望 - 事件前勝率期望。正值代表提升球隊勝率，負值代表降低勝率；很適合看關鍵時刻。",
    "RE": "Run Expectancy，得分期望。代表在當下出局數與壘包狀態下，該半局後續平均可期待得幾分。",
    "RE24": "Run Expectancy 24，24 種壘包/出局狀態的得分期望變化。公式概念：打席後 RE - 打席前 RE + 該打席得分。衡量打席如何改變該半局的得分環境。",

    # 擊球品質
    "BIP": "Ball In Play，擊入場內球數。此 app 用有擊球落點/彈道的打席統計。",
    "BIP/G": "場均擊入場內球。公式：BIP / G。",
    "hardHit": "強勁擊球數。此 app 依野球革命 hardness = H 統計。",
    "HardHit%": "強勁擊球率。公式：hardHit / BIP。代表擊球品質中強擊球比例。",
    "groundBall": "滾地球數。此 app 依 trajectory = G 統計。",
    "lineDrive": "平飛球數。此 app 依 trajectory = L 統計。",
    "flyBall": "飛球數。此 app 依 trajectory = F 統計。",
    "popup": "內野高飛或小飛球數。此 app 依 trajectory = P 統計。",
    "GB%": "滾地球比例。公式：groundBall / BIP。",
    "LD%": "平飛球比例。公式：lineDrive / BIP。",
    "FB%": "飛球比例。公式：flyBall / BIP。",
    "Popup%": "高飛/小飛球比例。公式：popup / BIP。",
    "locationCode": "擊球落點代碼。依野球革命/Retrosheet 類型落點區域標記，例如 8、78D、9HR。",
    "trajectory": "擊球彈道代碼，例如 G 滾地、L 平飛、F 飛球、P 高飛。",
    "trajectoryZh": "擊球彈道中文。",
    "hardness": "擊球強度代碼，例如 S 弱、M 中、H 強。",
    "hardnessZh": "擊球強度中文。",

    # 跑壘
    "runnerName": "跑者姓名。",
    "runnerType": "跑壘事件代碼。",
    "runnerTypeZh": "跑壘事件中文名稱。",
    "isOut": "跑者是否在該事件中出局。",
    "scored": "跑者是否在該事件中得分。",
    "isRBI": "該得分是否計入打者打點。",
    "isER": "該得分是否為責失分。",
    "ERPitcherName": "責失歸屬投手。",
}


def get_stat_help(col: Any) -> Optional[str]:
    """回傳欄位說明；支援 /G、% 等衍生欄位。"""
    name = str(col)
    if name in STAT_HELP:
        return STAT_HELP[name]

    # 通用場均欄位
    if name.endswith("/G"):
        base = name[:-2]
        if base in STAT_HELP:
            return f"場均 {base}。公式：{base} / G。{STAT_HELP.get(base, '')}"
        return f"場均指標。公式：{base} / G。"

    # 通用百分比欄位
    if name.endswith("%") and name not in STAT_HELP:
        base = name[:-1]
        return f"{name}，比例型指標。通常公式為 {base} / 對應機會數。"

    return None


def build_column_config(df: pd.DataFrame) -> Dict[str, Any]:
    """讓 st.dataframe 欄名帶有中文 tooltip 說明。"""
    config: Dict[str, Any] = {}
    for col in df.columns:
        help_text = get_stat_help(col)
        if help_text:
            try:
                config[col] = st.column_config.Column(help=help_text)
            except Exception:
                # 舊版 Streamlit 若沒有泛用 Column，就退回文字欄位；表格仍可正常顯示。
                try:
                    config[col] = st.column_config.TextColumn(help=help_text)
                except Exception:
                    pass
    return config



# =========================
# Matplotlib 中文字型設定
# =========================

CHINESE_FONT_PROP: Optional[font_manager.FontProperties] = None


def configure_matplotlib_chinese_font() -> Optional[str]:
    """讓 Matplotlib 圖表正常顯示中文。

    這裡不內建或散布字型檔，只讀取你電腦裡已經存在的字型。
    Windows 會優先抓 Microsoft JhengHei / Microsoft YaHei / 細明體；
    也支援你自己在專案資料夾建立 fonts/ 後放入合法字型檔。
    """
    global CHINESE_FONT_PROP

    candidates: List[Path] = []

    win_font_dir = Path(r"C:/Windows/Fonts")
    if win_font_dir.exists():
        for name in [
            "msjh.ttc", "msjhbd.ttc", "msjhl.ttc",       # 微軟正黑體
            "mingliu.ttc", "mingliub.ttc",               # 細明體
            "msyh.ttc", "msyhbd.ttc", "msyhl.ttc",      # 微軟雅黑
            "simhei.ttf", "simsun.ttc",
            "NotoSansCJK-Regular.ttc", "NotoSansCJKtc-Regular.otf",
        ]:
            candidates.append(win_font_dir / name)
        for pattern in ["*Noto*Sans*CJK*", "*SourceHanSans*", "*JhengHei*", "*YaHei*", "*MingLiU*"]:
            candidates.extend(win_font_dir.glob(pattern))

    candidates.extend([
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/System/Library/Fonts/STHeiti Light.ttc"),
        Path("/Library/Fonts/Arial Unicode.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-TC-Regular.otf"),
        Path("/usr/share/fonts/truetype/arphic/uming.ttc"),
    ])

    local_fonts = Path.cwd() / "fonts"
    if local_fonts.exists():
        for pattern in ["*.ttf", "*.ttc", "*.otf"]:
            candidates.extend(local_fonts.glob(pattern))

    seen = set()
    for fp in candidates:
        try:
            fp = fp.resolve()
        except Exception:
            continue
        if fp in seen or not fp.exists():
            continue
        seen.add(fp)
        try:
            font_manager.fontManager.addfont(str(fp))
            CHINESE_FONT_PROP = font_manager.FontProperties(fname=str(fp))
            name = CHINESE_FONT_PROP.get_name()
            plt.rcParams.update({
                "font.family": name,
                "font.sans-serif": [name, "Microsoft JhengHei", "Microsoft YaHei", "Noto Sans CJK TC", "SimHei"],
                "axes.unicode_minus": False,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "svg.fonttype": "none",
            })
            return f"{name} ({fp.name})"
        except Exception:
            continue

    preferred = [
        "Microsoft JhengHei", "Microsoft YaHei", "MingLiU", "PMingLiU",
        "Noto Sans CJK TC", "Noto Sans CJK SC", "Source Han Sans TC",
        "PingFang TC", "Heiti TC", "SimHei", "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            CHINESE_FONT_PROP = font_manager.FontProperties(family=name)
            plt.rcParams["font.family"] = name
            plt.rcParams["font.sans-serif"] = [name] + list(plt.rcParams.get("font.sans-serif", []))
            plt.rcParams["axes.unicode_minus"] = False
            return name

    plt.rcParams["axes.unicode_minus"] = False
    return None


def font_kwargs() -> Dict[str, Any]:
    return {"fontproperties": CHINESE_FONT_PROP} if CHINESE_FONT_PROP is not None else {}


def apply_chinese_font_to_axes(ax: plt.Axes) -> None:
    if CHINESE_FONT_PROP is None:
        return
    ax.title.set_fontproperties(CHINESE_FONT_PROP)
    ax.xaxis.label.set_fontproperties(CHINESE_FONT_PROP)
    ax.yaxis.label.set_fontproperties(CHINESE_FONT_PROP)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(CHINESE_FONT_PROP)
    legend = ax.get_legend()
    if legend is not None:
        for text_obj in legend.get_texts():
            text_obj.set_fontproperties(CHINESE_FONT_PROP)
        if legend.get_title() is not None:
            legend.get_title().set_fontproperties(CHINESE_FONT_PROP)
    for text_obj in ax.texts:
        text_obj.set_fontproperties(CHINESE_FONT_PROP)



def legend_outside_right(
    ax: plt.Axes,
    fig: Optional[plt.Figure] = None,
    fontsize: int = 7,
    anchor_x: float = 1.02,
    anchor_y: float = 0.50,
    right: float = 0.76,
) -> None:
    """把圖例固定放到圖外右側，避免壓住柱狀圖或散點圖。"""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    legend = ax.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(anchor_x, anchor_y),
        fontsize=fontsize,
        frameon=True,
        borderaxespad=0.2,
        prop=CHINESE_FONT_PROP,
    )

    if CHINESE_FONT_PROP is not None:
        for text_obj in legend.get_texts():
            text_obj.set_fontproperties(CHINESE_FONT_PROP)

    if fig is not None:
        fig.subplots_adjust(right=right)


MATPLOTLIB_CHINESE_FONT = configure_matplotlib_chinese_font()



# =========================
# 球隊代表色
# =========================

TEAM_COLORS = {
    "中信兄弟": "#FFD204",
    "味全龍": "#E61B24",
    "統一獅": "#F88626",
    "統一7-ELEVEn獅": "#F88626",
    "統一7-ELEVEN獅": "#F88626",
    "統一7-11獅": "#F88626",
    "富邦悍將": "#1C3B8B",
    "樂天桃猿": "#D01026",
    "台鋼雄鷹": "#137A3B",
}

TEAM_COLOR_ALIASES = {
    "中信": "中信兄弟",
    "兄弟": "中信兄弟",
    "味全": "味全龍",
    "龍": "味全龍",
    "統一": "統一獅",
    "7-ELEVEn": "統一獅",
    "7-ELEVEN": "統一獅",
    "富邦": "富邦悍將",
    "悍將": "富邦悍將",
    "樂天": "樂天桃猿",
    "桃猿": "樂天桃猿",
    "台鋼": "台鋼雄鷹",
    "雄鷹": "台鋼雄鷹",
}

FALLBACK_COLORS = [
    "#6B7280", "#7C3AED", "#0891B2", "#EA580C", "#16A34A", "#BE123C",
    "#4338CA", "#0F766E", "#A16207", "#4B5563",
]


def canonical_team_name(team: Any) -> str:
    text = str(team or "").strip()
    if text in TEAM_COLORS:
        return text
    for key, canonical in TEAM_COLOR_ALIASES.items():
        if key in text:
            return canonical
    return text


def get_team_color(team: Any, index: int = 0) -> str:
    canonical = canonical_team_name(team)
    if canonical in TEAM_COLORS:
        return TEAM_COLORS[canonical]
    return FALLBACK_COLORS[index % len(FALLBACK_COLORS)]


def get_team_color_list(teams: Sequence[Any]) -> List[str]:
    return [get_team_color(team, i) for i, team in enumerate(teams)]



# =========================
# 中文對照表
# =========================

RESULT_ZH = {
    "1B": "一壘安打",
    "2B": "二壘安打",
    "3B": "三壘安打",
    "HR": "全壘打",
    "SO": "三振",
    "GO": "滾地出局",
    "FO": "飛球出局",
    "uBB": "四壞保送",
    "BB": "四壞保送",
    "IBB": "故意四壞",
    "HBP": "觸身球",
    "GIDP": "滾地雙殺",
    "DP": "雙殺打",
    "TP": "三殺打",
    "FC": "野手選擇",
    "E": "失誤上壘",
    "SF": "犧牲飛球",
    "SH": "犧牲觸擊",
    "SH_FC": "犧牲觸擊野選",
}

PITCH_CODE_ZH = {
    "S": "無揮棒好球",
    "SW": "揮棒落空",
    "B": "壞球",
    "F": "界外",
    "FT": "擦棒被捕",
    "FOUL_BUNT": "觸擊界外",
    "TRY_BUNT": "觸擊落空",
    "BUNT": "觸擊",
    "H": "打進場內",
}

PITCH_TYPE_ZH = {
    "FF": "四縫線速球",
    "SI": "伸卡／二縫線",
    "FC": "卡特球",
    "SL": "滑球",
    "CU": "曲球",
    "CH": "變速球",
    "FO": "指叉球",
}

TRAJECTORY_ZH = {
    "G": "滾地球",
    "L": "平飛球",
    "F": "飛球",
    "P": "內野高飛",
    "": "無擊球",
}

HARDNESS_ZH = {
    "S": "弱",
    "M": "中",
    "H": "強",
    "": "無擊球",
}

RUNNER_TYPE_ZH = {
    "PA": "打者跑壘",
    "ADVANCE": "跑者推進",
    "SB": "盜壘成功",
    "CS": "盜壘失敗",
    "CS_E": "盜壘失敗但失誤上壘",
    "PO": "牽制出局",
}

HIT_RESULTS = {"1B", "2B", "3B", "HR"}
WALK_RESULTS = {"uBB", "BB", "IBB"}
HBP_RESULTS = {"HBP"}
SAC_FLY_RESULTS = {"SF"}
SAC_BUNT_RESULTS = {"SH", "SH_FC"}
NON_AB_RESULTS = WALK_RESULTS | HBP_RESULTS | SAC_FLY_RESULTS | SAC_BUNT_RESULTS


# =========================
# 小工具函式
# =========================

def safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        if isinstance(value, str) and value.strip() == "":
            return default
        return int(float(value))
    except Exception:
        return default


def safe_float(value: Any) -> float:
    if value is None:
        return np.nan
    try:
        if isinstance(value, str) and value.strip() == "":
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def clean_velocity(value: Any) -> float:
    """清理球速欄位。

    野球革命 OpenData 偶爾會有人工紀錄或轉換錯值，例如 463、182 這類不合理球速。
    本 app 統計球速時只保留 VALID_VELO_MIN～VALID_VELO_MAX km/h。
    """
    v = safe_float(value)
    if pd.isna(v):
        return np.nan
    if v < VALID_VELO_MIN or v > VALID_VELO_MAX:
        return np.nan
    return float(v)


def invalid_velocity_flag(value: Any) -> bool:
    v = safe_float(value)
    if pd.isna(v):
        return False
    return bool(v < VALID_VELO_MIN or v > VALID_VELO_MAX)


def clean_coord(value: Any) -> float:
    """清理投球進壘點座標。

    座標絕對值過大時，通常是人工標點或座標轉換異常。
    清理後的 coordX/coordY 主要用於圖表；原始值保留在 coordXRaw/coordYRaw。
    """
    v = safe_float(value)
    if pd.isna(v):
        return np.nan
    if abs(v) > VALID_COORD_ABS_LIMIT:
        return np.nan
    return float(v)


def invalid_coord_flag(x: Any, y: Any) -> bool:
    x_val = safe_float(x)
    y_val = safe_float(y)
    if pd.isna(x_val) or pd.isna(y_val):
        return False
    return bool(abs(x_val) > VALID_COORD_ABS_LIMIT or abs(y_val) > VALID_COORD_ABS_LIMIT)


def div0(numerator: Any, denominator: Any) -> float:
    n = safe_float(numerator)
    d = safe_float(denominator)
    if pd.isna(n) or pd.isna(d) or d == 0:
        return np.nan
    return n / d


def fmt_rate(x: Any, digits: int = 3) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{float(x):.{digits}f}"


def fmt_batting_rate(x: Any, digits: int = 3) -> str:
    """棒球常用格式：.342、.000、1.403。"""
    if x is None or pd.isna(x):
        return "-"
    value = float(x)
    text = f"{value:.{digits}f}"
    if 0 <= value < 1:
        return text[1:]
    if -1 < value < 0:
        return "-" + text[2:]
    return text


def fmt_pct(x: Any, digits: int = 1) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{float(x) * 100:.{digits}f}%"


def fmt_ip_from_outs(outs: Any) -> str:
    total_outs = safe_int(outs)
    innings = total_outs // 3
    rem = total_outs % 3
    return f"{innings}.{rem}"


def fmt_num(x: Any, digits: int = 1) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{float(x):.{digits}f}"


def total_bases_from_box(row: pd.Series) -> int:
    h = safe_int(row.get("H"))
    doubles = safe_int(row.get("2B"))
    triples = safe_int(row.get("3B"))
    hr = safe_int(row.get("HR"))
    return h + doubles + 2 * triples + 3 * hr


def total_bases_from_result(result: Any) -> int:
    result = str(result)
    if result == "1B":
        return 1
    if result == "2B":
        return 2
    if result == "3B":
        return 3
    if result == "HR":
        return 4
    return 0


def bases_label(value: Any) -> str:
    n = safe_int(value)
    if n == 0:
        return "空壘"
    parts: List[str] = []
    if n & 1:
        parts.append("一壘")
    if n & 2:
        parts.append("二壘")
    if n & 4:
        parts.append("三壘")
    return "、".join(parts) if parts else str(value)


def score_sum(score_list: Any) -> int:
    if not isinstance(score_list, list):
        return 0
    return sum(safe_int(x) for x in score_list)


def add_basic_rate_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    for col in ["PA", "AB", "H", "BB", "HBP", "SF", "2B", "3B", "HR", "SO", "RBI"]:
        if col not in df.columns:
            df[col] = 0
    df["TB"] = df.apply(total_bases_from_box, axis=1)
    df["XBH"] = df["2B"] + df["3B"] + df["HR"]
    df["AVG"] = df.apply(lambda r: div0(r["H"], r["AB"]), axis=1)
    df["OBP"] = df.apply(lambda r: div0(r["H"] + r["BB"] + r["HBP"], r["AB"] + r["BB"] + r["HBP"] + r["SF"]), axis=1)
    df["SLG"] = df.apply(lambda r: div0(r["TB"], r["AB"]), axis=1)
    df["OPS"] = df["OBP"] + df["SLG"]
    df["ISO"] = df["SLG"] - df["AVG"]
    df["BB%"] = df.apply(lambda r: div0(r["BB"], r["PA"]), axis=1)
    df["K%"] = df.apply(lambda r: div0(r["SO"], r["PA"]), axis=1)
    df["BB/K"] = df.apply(lambda r: div0(r["BB"], r["SO"]), axis=1)
    return df


def add_pitcher_rate_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    for col in ["IPOuts", "NP", "BF", "H", "HR", "BB", "IBB", "HB", "SO", "R", "ER"]:
        if col not in df.columns:
            df[col] = 0
    df["IP"] = df["IPOuts"].apply(lambda x: safe_int(x) / 3 if safe_int(x) > 0 else np.nan)
    df["IP顯示"] = df["IPOuts"].apply(fmt_ip_from_outs)
    df["ERA"] = df.apply(lambda r: div0(r["ER"] * 9, r["IP"]), axis=1)
    df["WHIP"] = df.apply(lambda r: div0(r["BB"] + r["H"], r["IP"]), axis=1)
    df["K%"] = df.apply(lambda r: div0(r["SO"], r["BF"]), axis=1)
    df["BB%"] = df.apply(lambda r: div0(r["BB"], r["BF"]), axis=1)
    df["K-BB%"] = df.apply(lambda r: div0(r["SO"] - r["BB"], r["BF"]), axis=1)
    df["K/9"] = df.apply(lambda r: div0(r["SO"] * 9, r["IP"]), axis=1)
    df["BB/9"] = df.apply(lambda r: div0(r["BB"] * 9, r["IP"]), axis=1)
    df["H/9"] = df.apply(lambda r: div0(r["H"] * 9, r["IP"]), axis=1)
    df["HR/9"] = df.apply(lambda r: div0(r["HR"] * 9, r["IP"]), axis=1)
    df["NP/IP"] = df.apply(lambda r: div0(r["NP"], r["IP"]), axis=1)
    df["NP/BF"] = df.apply(lambda r: div0(r["NP"], r["BF"]), axis=1)
    return df


def clean_display_df(df: pd.DataFrame, cols: Optional[Sequence[str]] = None, rate_cols: Optional[Sequence[str]] = None, pct_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if cols is not None:
        cols = [c for c in cols if c in out.columns]
        out = out[cols]
    batting_rate_columns = {"AVG", "OBP", "SLG", "OPS", "ISO"}
    for c in rate_cols or []:
        if c in out.columns:
            if c in batting_rate_columns:
                out[c] = out[c].apply(fmt_batting_rate)
            else:
                out[c] = out[c].apply(fmt_rate)
    for c in pct_cols or []:
        if c in out.columns:
            out[c] = out[c].apply(fmt_pct)
    return out


def value_counts_table(df: pd.DataFrame, column: str, label_col: str = "項目", count_col: str = "次數") -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return pd.DataFrame(columns=[label_col, count_col, "比例"])
    s = df[column].fillna("").astype(str)
    table = s.value_counts(dropna=False).reset_index()
    table.columns = [label_col, count_col]
    total = table[count_col].sum()
    table["比例"] = table[count_col] / total if total else np.nan
    return table


# =========================
# 資料讀取
# =========================

def extract_games_from_payload(payload: Any) -> List[Dict[str, Any]]:
    games: List[Dict[str, Any]] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and "awayTeam" in item and "homeTeam" in item:
                games.append(item)
    elif isinstance(payload, dict):
        if "awayTeam" in payload and "homeTeam" in payload:
            games.append(payload)
        elif "games" in payload and isinstance(payload["games"], list):
            games.extend(extract_games_from_payload(payload["games"]))
    return games


def load_json_bytes(name: str, raw: bytes) -> List[Dict[str, Any]]:
    try:
        text = raw.decode("utf-8-sig")
        payload = json.loads(text)
        return extract_games_from_payload(payload)
    except Exception as exc:
        st.sidebar.warning(f"讀取 {name} 失敗：{exc}")
        return []


def discover_local_json_files() -> List[Path]:
    base = Path.cwd()
    patterns = ["*.json", "data/*.json", "Data/*.json", "opendata/*.json", "OpenData/*.json"]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(base.glob(pattern))
    # 避免讀到太奇怪的設定檔；真的不是比賽 JSON 也會在 parse 階段被濾掉
    return sorted(set(files), key=lambda p: str(p).lower())


def dedupe_games(games: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Dict[Tuple[Any, Any, Any, Any, Any], Dict[str, Any]] = {}
    for g in games:
        key = (
            g.get("seasonId", ""),
            g.get("seq", ""),
            g.get("date", ""),
            g.get("awayTeam", ""),
            g.get("homeTeam", ""),
        )
        seen[key] = g
    return sorted(seen.values(), key=lambda x: (str(x.get("seasonId", "")), safe_int(x.get("seq")), str(x.get("date", ""))))


def load_games_from_sources(uploaded_files: Optional[List[Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    games: List[Dict[str, Any]] = []
    source_names: List[str] = []

    if uploaded_files:
        for u in uploaded_files:
            raw = u.getvalue()
            parsed = load_json_bytes(u.name, raw)
            if parsed:
                games.extend(parsed)
                source_names.append(u.name)
    else:
        for path in discover_local_json_files():
            try:
                parsed = load_json_bytes(path.name, path.read_bytes())
            except Exception as exc:
                st.sidebar.warning(f"讀取 {path.name} 失敗：{exc}")
                parsed = []
            if parsed:
                games.extend(parsed)
                source_names.append(str(path))

    return dedupe_games(games), source_names


def local_source_fingerprint(files: Sequence[Path]) -> str:
    """用檔名、大小、mtime 建立本機 JSON 指紋；比重新 parse JSON 快很多。"""
    h = hashlib.sha256()
    h.update(b"cpbl-opendata-v4.3-local")
    for path in sorted(files, key=lambda p: str(p).lower()):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        h.update(str(path.resolve()).encode("utf-8", errors="ignore"))
        h.update(str(stat.st_size).encode())
        h.update(str(stat.st_mtime_ns).encode())
    return h.hexdigest()[:24]


def uploaded_source_fingerprint_from_meta(uploaded_files: Sequence[Any]) -> str:
    """用上傳檔 metadata 建立指紋，避免每次互動都重新讀整包檔案內容做 hash。

    如果你換了一份同名同大小但內容不同的檔案，按左側「清除本機快取」即可強制重建。
    """
    h = hashlib.sha256()
    h.update(b"cpbl-opendata-v4.3-uploaded-meta")
    for u in uploaded_files:
        h.update(str(getattr(u, "name", "")).encode("utf-8", errors="ignore"))
        h.update(str(getattr(u, "size", "")).encode())
        h.update(str(getattr(u, "type", "")).encode("utf-8", errors="ignore"))
    return h.hexdigest()[:24]


def cache_file_for_signature(signature: str) -> Path:
    return CACHE_DIR / f"{signature}.pkl"


def load_cached_dataframes(signature: str) -> Optional[Dict[str, pd.DataFrame]]:
    path = cache_file_for_signature(signature)
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            return None
        dfs = payload.get("dataframes")
        if not isinstance(dfs, dict):
            return None
        if not all(k in dfs for k in DATAFRAME_KEYS):
            return None
        return dfs
    except Exception:
        return None


def save_cached_dataframes(signature: str, dataframes: Dict[str, pd.DataFrame], source_names: Sequence[str]) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": "v4.3",
            "source_names": list(source_names),
            "dataframes": dataframes,
        }
        with cache_file_for_signature(signature).open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        st.sidebar.warning(f"快取寫入失敗，不影響使用：{exc}")


def get_memory_cached_dataframes(signature: str) -> Optional[Dict[str, pd.DataFrame]]:
    if st.session_state.get("active_data_signature") == signature:
        dfs = st.session_state.get("active_dataframes")
        if isinstance(dfs, dict) and all(k in dfs for k in DATAFRAME_KEYS):
            return dfs
    return None


def set_memory_cached_dataframes(signature: str, dataframes: Dict[str, pd.DataFrame]) -> None:
    st.session_state["active_data_signature"] = signature
    st.session_state["active_dataframes"] = dataframes


def load_dataframes_from_sources(uploaded_files: Optional[List[Any]]) -> Tuple[Dict[str, pd.DataFrame], List[str], str]:
    """先用記憶體快取，再用本機 pickle 快取；真的沒有快取才 parse JSON。"""
    if uploaded_files:
        source_names = [u.name for u in uploaded_files]
        signature = uploaded_source_fingerprint_from_meta(uploaded_files)

        memory_cached = get_memory_cached_dataframes(signature)
        if memory_cached is not None:
            return memory_cached, source_names, "memory"

        cached = load_cached_dataframes(signature)
        if cached is not None:
            set_memory_cached_dataframes(signature, cached)
            return cached, source_names, "hit"

        payloads = [(u.name, u.getvalue()) for u in uploaded_files]
        games: List[Dict[str, Any]] = []
        for name, raw in payloads:
            games.extend(load_json_bytes(name, raw))
        games = dedupe_games(games)
        if not games:
            return {}, source_names, "miss"
        dfs = build_dataframes(games)
        save_cached_dataframes(signature, dfs, source_names)
        set_memory_cached_dataframes(signature, dfs)
        return dfs, source_names, "miss"

    files = discover_local_json_files()
    source_names = [str(p) for p in files]
    if not files:
        return {}, [], "disabled"

    signature = local_source_fingerprint(files)

    memory_cached = get_memory_cached_dataframes(signature)
    if memory_cached is not None:
        return memory_cached, source_names, "memory"

    cached = load_cached_dataframes(signature)
    if cached is not None:
        set_memory_cached_dataframes(signature, cached)
        return cached, source_names, "hit"

    games: List[Dict[str, Any]] = []
    for path in files:
        try:
            games.extend(load_json_bytes(path.name, path.read_bytes()))
        except Exception as exc:
            st.sidebar.warning(f"讀取 {path.name} 失敗：{exc}")
    games = dedupe_games(games)
    if not games:
        return {}, source_names, "miss"

    dfs = build_dataframes(games)
    save_cached_dataframes(signature, dfs, source_names)
    set_memory_cached_dataframes(signature, dfs)
    return dfs, source_names, "miss"


# =========================
# 攤平成 DataFrame
# =========================

@st.cache_data(show_spinner="正在攤平 OpenData，整季資料第一次會比較久……")
def build_dataframes(games: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    game_rows: List[Dict[str, Any]] = []
    score_rows: List[Dict[str, Any]] = []
    batter_rows: List[Dict[str, Any]] = []
    pitcher_rows: List[Dict[str, Any]] = []
    pa_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []
    runner_rows: List[Dict[str, Any]] = []

    global_pa_id = 0
    global_event_id = 0

    for g in games:
        seq = safe_int(g.get("seq"))
        season = g.get("season", "")
        season_id = g.get("seasonId", "")
        date = g.get("date", "")
        stadium = g.get("stadium", "")
        away_team = g.get("awayTeam", "")
        home_team = g.get("homeTeam", "")
        away_team_id = g.get("awayTeamId", "")
        home_team_id = g.get("homeTeamId", "")
        away_runs = score_sum(g.get("awayScores", []))
        home_runs = score_sum(g.get("homeScores", []))
        winner = away_team if away_runs > home_runs else home_team if home_runs > away_runs else "和局"
        loser = home_team if away_runs > home_runs else away_team if home_runs > away_runs else "和局"

        game_rows.append(
            {
                "seasonId": season_id,
                "season": season,
                "G": seq,
                "date": date,
                "stadium": stadium,
                "awayTeamId": away_team_id,
                "awayTeam": away_team,
                "homeTeamId": home_team_id,
                "homeTeam": home_team,
                "awayRuns": away_runs,
                "homeRuns": home_runs,
                "winner": winner,
                "loser": loser,
                "score": f"{away_team} {away_runs} : {home_runs} {home_team}",
            }
        )

        for side, team, scores in [("away", away_team, g.get("awayScores", [])), ("home", home_team, g.get("homeScores", []))]:
            opponent = home_team if side == "away" else away_team
            for inning_idx, run in enumerate(scores, start=1):
                score_rows.append(
                    {
                        "G": seq,
                        "date": date,
                        "stadium": stadium,
                        "side": side,
                        "team": team,
                        "opponentTeam": opponent,
                        "inning": inning_idx,
                        "runs": safe_int(run),
                    }
                )

        for side, team, opponent in [("away", away_team, home_team), ("home", home_team, away_team)]:
            for row in g.get(f"{side}BatterBox", []) or []:
                r = dict(row)
                r.update(
                    {
                        "G": seq,
                        "date": date,
                        "stadium": stadium,
                        "side": side,
                        "team": team,
                        "opponentTeam": opponent,
                    }
                )
                batter_rows.append(r)

            for row in g.get(f"{side}PitcherBox", []) or []:
                r = dict(row)
                r.update(
                    {
                        "G": seq,
                        "date": date,
                        "stadium": stadium,
                        "side": side,
                        "team": team,
                        "opponentTeam": opponent,
                    }
                )
                pitcher_rows.append(r)

            for pa_order_in_game, pa in enumerate(g.get(f"{side}PAList", []) or [], start=1):
                global_pa_id += 1
                offense_team = team
                defense_team = opponent
                offense_score_before = safe_int(pa.get("awayScores")) if side == "away" else safe_int(pa.get("homeScores"))
                defense_score_before = safe_int(pa.get("homeScores")) if side == "away" else safe_int(pa.get("awayScores"))
                offense_score_after = safe_int(pa.get("endAwayScores")) if side == "away" else safe_int(pa.get("endHomeScores"))
                defense_score_after = safe_int(pa.get("endHomeScores")) if side == "away" else safe_int(pa.get("endAwayScores"))
                result = str(pa.get("result", "") or "")
                pa_ab = 0 if result in NON_AB_RESULTS else 1
                pa_h = 1 if result in HIT_RESULTS else 0
                pa_bb = 1 if result in WALK_RESULTS else 0
                pa_hbp = 1 if result in HBP_RESULTS else 0
                pa_sf = 1 if result in SAC_FLY_RESULTS else 0
                pa_sh = 1 if result in SAC_BUNT_RESULTS else 0
                tb = total_bases_from_result(result)
                pitch_codes = pa.get("pitchCodes") if isinstance(pa.get("pitchCodes"), list) else []
                events = pa.get("events") if isinstance(pa.get("events"), list) else []
                pitch_events = [e for e in events if isinstance(e, dict) and e.get("type") == "PITCH"]

                pa_row = {
                    "paId": global_pa_id,
                    "G": seq,
                    "date": date,
                    "stadium": stadium,
                    "side": side,
                    "offenseTeam": offense_team,
                    "defenseTeam": defense_team,
                    "inning": safe_int(pa.get("inning")),
                    "scored": bool(pa.get("scored")),
                    "batterName": pa.get("batterName", ""),
                    "batterHand": pa.get("batterHand", ""),
                    "pitcherName": pa.get("pitcherName", ""),
                    "pitcherHand": pa.get("pitcherHand", ""),
                    "catcherName": pa.get("catcherName", ""),
                    "paRound": safe_int(pa.get("paRound")),
                    "paOrder": safe_int(pa.get("paOrder")),
                    "paOrderInGame": pa_order_in_game,
                    "isPH": bool(pa.get("isPH")),
                    "awayScores": safe_int(pa.get("awayScores")),
                    "homeScores": safe_int(pa.get("homeScores")),
                    "scoreBefore": f"{safe_int(pa.get('awayScores'))}:{safe_int(pa.get('homeScores'))}",
                    "offenseScoreBefore": offense_score_before,
                    "defenseScoreBefore": defense_score_before,
                    "scoreDiffOffense": offense_score_before - defense_score_before,
                    "strikes": safe_int(pa.get("strikes")),
                    "balls": safe_int(pa.get("balls")),
                    "count": f"{safe_int(pa.get('balls'))}-{safe_int(pa.get('strikes'))}",
                    "outs": safe_int(pa.get("outs")),
                    "bases": safe_int(pa.get("bases")),
                    "basesLabel": bases_label(pa.get("bases")),
                    "homeWE": safe_float(pa.get("homeWE")),
                    "RE": safe_float(pa.get("RE")),
                    "pitchCodes": "-".join(str(x) for x in pitch_codes),
                    "pitchCount": len(pitch_events),
                    "result": result,
                    "resultZh": RESULT_ZH.get(result, result),
                    "RBI": safe_int(pa.get("RBI")),
                    "locationCode": str(pa.get("locationCode", "") or ""),
                    "trajectory": str(pa.get("trajectory", "") or ""),
                    "trajectoryZh": TRAJECTORY_ZH.get(str(pa.get("trajectory", "") or ""), str(pa.get("trajectory", "") or "")),
                    "hardness": str(pa.get("hardness", "") or ""),
                    "hardnessZh": HARDNESS_ZH.get(str(pa.get("hardness", "") or ""), str(pa.get("hardness", "") or "")),
                    "endAwayScores": safe_int(pa.get("endAwayScores")),
                    "endHomeScores": safe_int(pa.get("endHomeScores")),
                    "scoreAfter": f"{safe_int(pa.get('endAwayScores'))}:{safe_int(pa.get('endHomeScores'))}",
                    "offenseScoreAfter": offense_score_after,
                    "defenseScoreAfter": defense_score_after,
                    "runsThisPA": max(0, offense_score_after - offense_score_before),
                    "endOuts": safe_int(pa.get("endOuts")),
                    "endBases": safe_int(pa.get("endBases")),
                    "endBasesLabel": bases_label(pa.get("endBases")),
                    "WPA": safe_float(pa.get("WPA")),
                    "RE24": safe_float(pa.get("RE24")),
                    "AB_flag": pa_ab,
                    "H_flag": pa_h,
                    "BB_flag": pa_bb,
                    "HBP_flag": pa_hbp,
                    "SF_flag": pa_sf,
                    "SH_flag": pa_sh,
                    "SO_flag": 1 if result == "SO" else 0,
                    "HR_flag": 1 if result == "HR" else 0,
                    "2B_flag": 1 if result == "2B" else 0,
                    "3B_flag": 1 if result == "3B" else 0,
                    "TB": tb,
                    "isHit": result in HIT_RESULTS,
                    "isOnBase": result in HIT_RESULTS or result in WALK_RESULTS or result in HBP_RESULTS or result == "E",
                    "isBattedBall": str(pa.get("locationCode", "") or "") != "" or str(pa.get("trajectory", "") or "") != "",
                    "isRISP": (safe_int(pa.get("bases")) & 2 > 0) or (safe_int(pa.get("bases")) & 4 > 0),
                    "isLate": safe_int(pa.get("inning")) >= 7,
                    "isClose": abs(offense_score_before - defense_score_before) <= 3,
                    "isHighLeverage": abs(safe_float(pa.get("WPA"))) >= 0.05 if not pd.isna(safe_float(pa.get("WPA"))) else False,
                }
                pa_rows.append(pa_row)

                pitch_number = 0
                for event_order, event in enumerate(events, start=1):
                    if not isinstance(event, dict):
                        continue
                    global_event_id += 1
                    is_pitch = event.get("type") == "PITCH"
                    if is_pitch:
                        pitch_number += 1
                    event_row = {
                        "eventId": global_event_id,
                        "paId": global_pa_id,
                        "G": seq,
                        "date": date,
                        "stadium": stadium,
                        "side": side,
                        "offenseTeam": offense_team,
                        "defenseTeam": defense_team,
                        "inning": pa_row["inning"],
                        "eventOrder": event_order,
                        "pitchNumber": pitch_number if is_pitch else np.nan,
                        "paOrderInGame": pa_order_in_game,
                        "type": event.get("type", ""),
                        "inPlay": bool(event.get("inPlay")),
                        "isStrike": bool(event.get("isStrike")),
                        "isBall": bool(event.get("isBall")),
                        "pitcherName": event.get("pitcherName", pa.get("pitcherName", "")),
                        "catcherName": event.get("catcherName", pa.get("catcherName", "")),
                        "batterName": event.get("batterName", pa.get("batterName", "")),
                        "batterHand": pa_row["batterHand"],
                        "pitcherHand": pa_row["pitcherHand"],
                        "pitchCode": event.get("pitchCode", ""),
                        "pitchCodeZh": PITCH_CODE_ZH.get(str(event.get("pitchCode", "")), str(event.get("pitchCode", ""))),
                        "pitchType": event.get("pitchType", ""),
                        "pitchTypeZh": PITCH_TYPE_ZH.get(str(event.get("pitchType", "")), str(event.get("pitchType", ""))),
                        "velocityRaw": safe_float(event.get("velocity")),
                        "velocity": clean_velocity(event.get("velocity")),
                        "invalidVelocity": invalid_velocity_flag(event.get("velocity")),
                        "coordXRaw": safe_float(event.get("coordX")),
                        "coordYRaw": safe_float(event.get("coordY")),
                        "coordX": clean_coord(event.get("coordX")),
                        "coordY": clean_coord(event.get("coordY")),
                        "invalidCoord": invalid_coord_flag(event.get("coordX"), event.get("coordY")),
                        "paResult": result,
                        "paResultZh": RESULT_ZH.get(result, result),
                        "locationCode": pa_row["locationCode"],
                        "trajectory": pa_row["trajectory"],
                        "hardness": pa_row["hardness"],
                        "WPA": pa_row["WPA"],
                        "RE24": pa_row["RE24"],
                        "runnersCount": len(event.get("runners", []) or []),
                    }
                    event_rows.append(event_row)

                    for runner_order, runner in enumerate(event.get("runners", []) or [], start=1):
                        if not isinstance(runner, dict):
                            continue
                        runner_rows.append(
                            {
                                "eventId": global_event_id,
                                "paId": global_pa_id,
                                "G": seq,
                                "date": date,
                                "stadium": stadium,
                                "side": side,
                                "offenseTeam": offense_team,
                                "defenseTeam": defense_team,
                                "inning": pa_row["inning"],
                                "eventOrder": event_order,
                                "runnerOrder": runner_order,
                                "eventType": event.get("type", ""),
                                "pitcherName": event.get("pitcherName", pa.get("pitcherName", "")),
                                "batterName": event.get("batterName", pa.get("batterName", "")),
                                "paResult": result,
                                "runnerType": runner.get("type", ""),
                                "runnerTypeZh": RUNNER_TYPE_ZH.get(str(runner.get("type", "")), str(runner.get("type", ""))),
                                "runnerName": runner.get("runnerName", ""),
                                "isOut": bool(runner.get("isOut")),
                                "scored": bool(runner.get("scored")),
                                "isRBI": bool(runner.get("isRBI")),
                                "isER": bool(runner.get("isER")),
                                "ERPitcherName": runner.get("ERPitcherName"),
                                "WPA": pa_row["WPA"],
                                "RE24": pa_row["RE24"],
                            }
                        )

    dfs = {
        "games": pd.DataFrame(game_rows),
        "scores": pd.DataFrame(score_rows),
        "batters_game": pd.DataFrame(batter_rows),
        "pitchers_game": pd.DataFrame(pitcher_rows),
        "pa": pd.DataFrame(pa_rows),
        "events": pd.DataFrame(event_rows),
        "runners": pd.DataFrame(runner_rows),
    }

    # 型別整理
    numeric_batter_cols = ["PA", "AB", "R", "H", "RBI", "2B", "3B", "HR", "GIDP", "DP", "TP", "BB", "IBB", "HBP", "SO", "SH", "SF", "E", "SB", "CS"]
    if not dfs["batters_game"].empty:
        for c in numeric_batter_cols:
            if c in dfs["batters_game"].columns:
                dfs["batters_game"][c] = dfs["batters_game"][c].apply(safe_int)

    numeric_pitcher_cols = ["IPOuts", "NP", "BF", "H", "HR", "BB", "IBB", "HB", "SO", "R", "ER"]
    if not dfs["pitchers_game"].empty:
        for c in numeric_pitcher_cols:
            if c in dfs["pitchers_game"].columns:
                dfs["pitchers_game"][c] = dfs["pitchers_game"][c].apply(safe_int)

    return dfs



# =========================
# 球員索引 / 守位篩選
# =========================

def normalize_player_text(value: Any) -> str:
    """搜尋用：去掉空白並轉小寫。中文不受影響。"""
    return re.sub(r"\s+", "", str(value or "")).lower()


def join_unique(values: Iterable[Any], sep: str = " / ") -> str:
    cleaned: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in cleaned and text.lower() != "nan":
            cleaned.append(text)
    return sep.join(cleaned)


def normalize_player_metadata_columns(meta: pd.DataFrame) -> pd.DataFrame:
    """支援使用者自行提供 player_meta.csv / roster.csv 的中英文欄位。"""
    if meta.empty:
        return meta

    rename_map = {
        "姓名": "playerName",
        "球員": "playerName",
        "球員姓名": "playerName",
        "player": "playerName",
        "name": "playerName",
        "背號": "playerNumber",
        "球衣背號": "playerNumber",
        "number": "playerNumber",
        "no": "playerNumber",
        "隊伍": "team",
        "球隊": "team",
        "所屬球隊": "team",
        "teamName": "team",
        "守位": "position",
        "位置": "position",
        "守備位置": "position",
        "position": "position",
        "pos": "position",
    }

    out = meta.copy()
    new_cols = {}
    for c in out.columns:
        key = str(c).strip()
        new_cols[c] = rename_map.get(key, rename_map.get(key.lower(), key))
    out = out.rename(columns=new_cols)

    for c in ["playerName", "playerNumber", "team", "position"]:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip()
    return out


def load_optional_player_metadata() -> pd.DataFrame:
    """可選：讀取專案資料夾內的球員守位表。

    檔名支援：
    - player_meta.csv / player_metadata.csv / roster.csv / players.csv
    - player_meta.json / player_metadata.json / roster.json / players.json

    需要欄位至少有 playerName 或 姓名；若有 team/所屬球隊 與 position/守位，會優先使用。
    """
    candidates = [
        "player_meta.csv", "player_metadata.csv", "roster.csv", "players.csv",
        "player_meta.json", "player_metadata.json", "roster.json", "players.json",
    ]
    for name in candidates:
        path = Path.cwd() / name
        if not path.exists():
            continue
        try:
            if path.suffix.lower() == ".csv":
                meta = pd.read_csv(path, encoding="utf-8-sig")
            else:
                meta = pd.read_json(path)
            meta = normalize_player_metadata_columns(meta)
            if "playerName" in meta.columns:
                return meta
        except Exception:
            continue
    return pd.DataFrame()


def build_player_index(batters_game: pd.DataFrame, pitchers_game: pd.DataFrame, pa_df: pd.DataFrame) -> pd.DataFrame:
    """建立全域球員索引，用來做「所屬球隊 → 守位 → 姓名/背號」篩選。

    注意：野球革命這份 PA 資料本身沒有完整的每位野手守備位置。
    因此若沒有另外提供 roster/player_meta，程式只能從資料推得：
    投手、捕手、野手/打者、投打二刀流。
    """
    rows: List[Dict[str, Any]] = []

    if batters_game is not None and not batters_game.empty:
        for _, r in batters_game.iterrows():
            rows.append({
                "team": r.get("team", ""),
                "playerName": r.get("playerName", ""),
                "playerNumber": r.get("playerNumber", ""),
                "asBatter": True,
                "asPitcher": False,
                "asCatcher": False,
                "PA": safe_int(r.get("PA")),
                "AB": safe_int(r.get("AB")),
                "BF": 0,
                "games": 1,
            })

    if pitchers_game is not None and not pitchers_game.empty:
        for _, r in pitchers_game.iterrows():
            rows.append({
                "team": r.get("team", ""),
                "playerName": r.get("playerName", ""),
                "playerNumber": r.get("playerNumber", ""),
                "asBatter": False,
                "asPitcher": True,
                "asCatcher": False,
                "PA": 0,
                "AB": 0,
                "BF": safe_int(r.get("BF")),
                "games": 1,
            })

    if pa_df is not None and not pa_df.empty and "catcherName" in pa_df.columns:
        catcher_rows = pa_df[["defenseTeam", "catcherName", "G"]].dropna().copy()
        catcher_rows = catcher_rows[catcher_rows["catcherName"].astype(str).str.strip() != ""]
        for _, r in catcher_rows.iterrows():
            rows.append({
                "team": r.get("defenseTeam", ""),
                "playerName": r.get("catcherName", ""),
                "playerNumber": "",
                "asBatter": False,
                "asPitcher": False,
                "asCatcher": True,
                "PA": 0,
                "AB": 0,
                "BF": 0,
                "games": 1,
            })

    if not rows:
        return pd.DataFrame(columns=["team", "playerName", "playerNumber", "position", "inferredPosition", "roles", "PA", "AB", "BF", "searchKey"])

    raw = pd.DataFrame(rows)
    for c in ["team", "playerName", "playerNumber"]:
        raw[c] = raw[c].astype(str).str.strip()

    raw = raw[raw["playerName"] != ""].copy()

    grouped = raw.groupby(["team", "playerName"], dropna=False).agg(
        playerNumber=("playerNumber", join_unique),
        asBatter=("asBatter", "max"),
        asPitcher=("asPitcher", "max"),
        asCatcher=("asCatcher", "max"),
        PA=("PA", "sum"),
        AB=("AB", "sum"),
        BF=("BF", "sum"),
        games=("games", "sum"),
    ).reset_index()

    def infer_position(r: pd.Series) -> str:
        if bool(r.get("asPitcher")) and bool(r.get("asBatter")):
            return "投打二刀流"
        if bool(r.get("asPitcher")):
            return "投手"
        if bool(r.get("asCatcher")):
            return "捕手"
        return "野手/打者"

    def role_text(r: pd.Series) -> str:
        roles = []
        if bool(r.get("asBatter")):
            roles.append("打者")
        if bool(r.get("asPitcher")):
            roles.append("投手")
        if bool(r.get("asCatcher")):
            roles.append("捕手")
        return " / ".join(roles) if roles else "未知"

    grouped["inferredPosition"] = grouped.apply(infer_position, axis=1)
    grouped["roles"] = grouped.apply(role_text, axis=1)
    grouped["position"] = grouped["inferredPosition"]
    grouped["positionSource"] = "資料推估"

    # 可選 metadata 覆蓋守位
    meta = load_optional_player_metadata()
    if not meta.empty and "playerName" in meta.columns and "position" in meta.columns:
        use_cols = [c for c in ["team", "playerName", "playerNumber", "position"] if c in meta.columns]
        meta_small = meta[use_cols].copy()
        for c in use_cols:
            meta_small[c] = meta_small[c].astype(str).str.strip()

        if "team" in meta_small.columns:
            meta_team = meta_small.dropna(subset=["playerName"]).drop_duplicates(subset=["team", "playerName"], keep="first")
            grouped = grouped.merge(
                meta_team[["team", "playerName", "position"]].rename(columns={"position": "metaPosition"}),
                on=["team", "playerName"],
                how="left",
            )
        else:
            grouped["metaPosition"] = np.nan

        # 若 team 對不到，再用姓名對一次。
        missing = grouped["metaPosition"].isna() | (grouped["metaPosition"].astype(str).str.strip() == "")
        if missing.any():
            meta_name = meta_small.dropna(subset=["playerName"]).drop_duplicates(subset=["playerName"], keep="first")
            name_map = dict(zip(meta_name["playerName"], meta_name["position"]))
            grouped.loc[missing, "metaPosition"] = grouped.loc[missing, "playerName"].map(name_map)

        has_meta = grouped["metaPosition"].notna() & (grouped["metaPosition"].astype(str).str.strip() != "")
        grouped.loc[has_meta, "position"] = grouped.loc[has_meta, "metaPosition"].astype(str).str.strip()
        grouped.loc[has_meta, "positionSource"] = "player_meta/roster"
        grouped = grouped.drop(columns=["metaPosition"], errors="ignore")

    grouped["searchKey"] = (
        grouped["team"].astype(str) + " " +
        grouped["playerName"].astype(str) + " " +
        grouped["playerNumber"].astype(str) + " " +
        grouped["position"].astype(str) + " " +
        grouped["roles"].astype(str)
    ).apply(normalize_player_text)

    grouped = grouped.sort_values(["team", "position", "playerNumber", "playerName"]).reset_index(drop=True)
    return grouped


def filter_player_index(player_index: pd.DataFrame, teams: Sequence[str], positions: Sequence[str], query: str) -> pd.DataFrame:
    if player_index.empty:
        return player_index
    out = player_index.copy()

    if teams:
        out = out[out["team"].isin(list(teams))]

    if positions:
        out = out[out["position"].isin(list(positions))]

    query = str(query or "").strip()
    if query:
        terms = [normalize_player_text(t) for t in re.split(r"[,，\s]+", query) if normalize_player_text(t)]
        for term in terms:
            out = out[out["searchKey"].str.contains(re.escape(term), na=False)]

    return out


def key_mask(df: pd.DataFrame, team_col: str, name_col: str, keys: set[Tuple[str, str]]) -> pd.Series:
    if df.empty or not keys or team_col not in df.columns or name_col not in df.columns:
        return pd.Series(False, index=df.index)
    pairs = pd.Series(list(zip(df[team_col].astype(str), df[name_col].astype(str))), index=df.index)
    return pairs.isin(keys)



# =========================
# 統計表產生
# =========================

def aggregate_batters(batter_game: pd.DataFrame) -> pd.DataFrame:
    if batter_game.empty:
        return pd.DataFrame()
    group_cols = ["playerId", "playerNumber", "playerName", "team"]
    sum_cols = ["PA", "AB", "R", "H", "RBI", "2B", "3B", "HR", "GIDP", "DP", "TP", "BB", "IBB", "HBP", "SO", "SH", "SF", "E", "SB", "CS"]
    agg = batter_game.groupby(group_cols, dropna=False)[sum_cols].sum().reset_index()
    return add_basic_rate_stats(agg)


def aggregate_pitchers(pitcher_game: pd.DataFrame) -> pd.DataFrame:
    if pitcher_game.empty:
        return pd.DataFrame()
    group_cols = ["playerId", "playerNumber", "playerName", "team"]
    sum_cols = ["IPOuts", "NP", "BF", "H", "HR", "BB", "IBB", "HB", "SO", "R", "ER"]
    agg = pitcher_game.groupby(group_cols, dropna=False)[sum_cols].sum().reset_index()
    return add_pitcher_rate_stats(agg)


def aggregate_matchups(pa_df: pd.DataFrame) -> pd.DataFrame:
    if pa_df.empty:
        return pd.DataFrame()
    group_cols = ["batterName", "batterHand", "pitcherName", "pitcherHand", "offenseTeam", "defenseTeam"]
    agg = pa_df.groupby(group_cols, dropna=False).agg(
        PA=("paId", "count"),
        AB=("AB_flag", "sum"),
        H=("H_flag", "sum"),
        BB=("BB_flag", "sum"),
        HBP=("HBP_flag", "sum"),
        SF=("SF_flag", "sum"),
        SH=("SH_flag", "sum"),
        SO=("SO_flag", "sum"),
        RBI=("RBI", "sum"),
        HR=("HR_flag", "sum"),
        doubles=("2B_flag", "sum"),
        triples=("3B_flag", "sum"),
        TB=("TB", "sum"),
        pitches=("pitchCount", "sum"),
        WPA=("WPA", "sum"),
        RE24=("RE24", "sum"),
    ).reset_index()
    agg = agg.rename(columns={"doubles": "2B", "triples": "3B"})
    agg["AVG"] = agg.apply(lambda r: div0(r["H"], r["AB"]), axis=1)
    agg["OBP"] = agg.apply(lambda r: div0(r["H"] + r["BB"] + r["HBP"], r["AB"] + r["BB"] + r["HBP"] + r["SF"]), axis=1)
    agg["SLG"] = agg.apply(lambda r: div0(r["TB"], r["AB"]), axis=1)
    agg["OPS"] = agg["OBP"] + agg["SLG"]
    agg["K%"] = agg.apply(lambda r: div0(r["SO"], r["PA"]), axis=1)
    agg["BB%"] = agg.apply(lambda r: div0(r["BB"], r["PA"]), axis=1)
    return agg.sort_values(["PA", "OPS"], ascending=[False, False])


def add_risp_rate_stats(df: pd.DataFrame) -> pd.DataFrame:
    """為得點圈摘要表加上 AVG/OBP/SLG/OPS 等比例指標。"""
    if df.empty:
        return df
    out = df.copy()
    for col in ["PA", "AB", "H", "BB", "HBP", "SF", "2B", "3B", "HR", "SO", "RBI", "TB", "R"]:
        if col not in out.columns:
            out[col] = 0
    out["AVG"] = out.apply(lambda r: div0(r["H"], r["AB"]), axis=1)
    out["OBP"] = out.apply(lambda r: div0(r["H"] + r["BB"] + r["HBP"], r["AB"] + r["BB"] + r["HBP"] + r["SF"]), axis=1)
    out["SLG"] = out.apply(lambda r: div0(r["TB"], r["AB"]), axis=1)
    out["OPS"] = out["OBP"] + out["SLG"]
    out["K%"] = out.apply(lambda r: div0(r["SO"], r["PA"]), axis=1)
    out["BB%"] = out.apply(lambda r: div0(r["BB"], r["PA"]), axis=1)
    out["R/PA"] = out.apply(lambda r: div0(r["R"], r["PA"]), axis=1)
    return out


def risp_summary_from_pa(pa_df: pd.DataFrame, group_cols: Sequence[str], rename_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """依 PA 明細建立得點圈摘要。

    RISP = Runner In Scoring Position，也就是打席開始時二壘或三壘有人。
    """
    if pa_df.empty or "isRISP" not in pa_df.columns:
        return pd.DataFrame()

    risp = pa_df[pa_df["isRISP"]].copy()
    if risp.empty:
        return pd.DataFrame()

    agg = risp.groupby(list(group_cols), dropna=False).agg(
        PA=("paId", "count"),
        AB=("AB_flag", "sum"),
        H=("H_flag", "sum"),
        doubles=("2B_flag", "sum"),
        triples=("3B_flag", "sum"),
        HR=("HR_flag", "sum"),
        BB=("BB_flag", "sum"),
        HBP=("HBP_flag", "sum"),
        SF=("SF_flag", "sum"),
        SH=("SH_flag", "sum"),
        SO=("SO_flag", "sum"),
        RBI=("RBI", "sum"),
        TB=("TB", "sum"),
        R=("runsThisPA", "sum"),
        WPA=("WPA", "sum"),
        RE24=("RE24", "sum"),
    ).reset_index()

    agg = agg.rename(columns={"doubles": "2B", "triples": "3B"})
    if rename_map:
        agg = agg.rename(columns=rename_map)

    agg = add_risp_rate_stats(agg)
    return agg.sort_values(["PA", "OPS"], ascending=[False, False])


def team_game_counts(games_df: pd.DataFrame) -> pd.DataFrame:
    """回傳每隊出賽場數。用 game 表算，避免用 box score 時漏算沒有事件的場次。"""
    if games_df.empty:
        return pd.DataFrame(columns=["team", "G"])
    rows: List[Dict[str, Any]] = []
    for _, g in games_df.iterrows():
        rows.append({"team": g.get("awayTeam", ""), "G": safe_int(g.get("G"))})
        rows.append({"team": g.get("homeTeam", ""), "G": safe_int(g.get("G"))})
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["team", "G"])
    return out.groupby("team", dropna=False)["G"].nunique().reset_index()


def add_per_game_columns(df: pd.DataFrame, count_cols: Sequence[str], games_col: str = "G") -> pd.DataFrame:
    """替團隊總表加上 /G 欄位。"""
    if df.empty or games_col not in df.columns:
        return df
    out = df.copy()
    for col in count_cols:
        if col in out.columns:
            out[f"{col}/G"] = out.apply(lambda r: div0(r[col], r[games_col]), axis=1)
    return out


def team_batting_from_box(batter_game: pd.DataFrame) -> pd.DataFrame:
    if batter_game.empty:
        return pd.DataFrame()
    sum_cols = ["PA", "AB", "R", "H", "RBI", "2B", "3B", "HR", "GIDP", "DP", "TP", "BB", "IBB", "HBP", "SO", "SH", "SF", "E", "SB", "CS"]
    out = batter_game.groupby("team", dropna=False)[sum_cols].sum().reset_index()
    games = batter_game.groupby("team", dropna=False)["G"].nunique().reset_index(name="G") if "G" in batter_game.columns else pd.DataFrame()
    if not games.empty:
        out = out.merge(games, on="team", how="left")
    out = add_basic_rate_stats(out)
    out = add_per_game_columns(out, ["PA", "AB", "R", "H", "2B", "3B", "HR", "RBI", "BB", "SO", "SB", "CS", "GIDP"])
    return out


def team_pitching_from_box(pitcher_game: pd.DataFrame) -> pd.DataFrame:
    if pitcher_game.empty:
        return pd.DataFrame()
    sum_cols = ["IPOuts", "NP", "BF", "H", "HR", "BB", "IBB", "HB", "SO", "R", "ER"]
    out = pitcher_game.groupby("team", dropna=False)[sum_cols].sum().reset_index()
    games = pitcher_game.groupby("team", dropna=False)["G"].nunique().reset_index(name="G") if "G" in pitcher_game.columns else pd.DataFrame()
    if not games.empty:
        out = out.merge(games, on="team", how="left")
    out = add_pitcher_rate_stats(out)
    if "IPOuts" in out.columns and "G" in out.columns:
        out["IP/G"] = out.apply(lambda r: div0(safe_int(r["IPOuts"]) / 3, r["G"]), axis=1)
    out = add_per_game_columns(out, ["NP", "BF", "H", "HR", "BB", "HB", "SO", "R", "ER"])
    return out


def team_pitch_event_summary(events_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    """用 event 層級做團隊投球內容：球種、球速、好球、壞球、揮空、進場。"""
    if events_df.empty:
        return pd.DataFrame()
    pitches = events_df[events_df["type"] == "PITCH"].copy() if "type" in events_df.columns else pd.DataFrame()
    if pitches.empty:
        return pd.DataFrame()
    group = pitches.groupby("defenseTeam", dropna=False).agg(
        pitches=("eventId", "count"),
        strikes=("isStrike", "sum"),
        balls=("isBall", "sum"),
        whiffs=("pitchCode", lambda s: (s == "SW").sum()),
        inPlay=("inPlay", "sum"),
        avgVelo=("velocity", "mean"),
        maxVelo=("velocity", "max"),
        invalidVelo=("invalidVelocity", "sum"),
        invalidCoord=("invalidCoord", "sum"),
    ).reset_index().rename(columns={"defenseTeam": "team"})
    games = team_game_counts(games_df)
    if not games.empty:
        group = group.merge(games, on="team", how="left")
    group["Pitches/G"] = group.apply(lambda r: div0(r["pitches"], r["G"]), axis=1)
    group["Strike%"] = group.apply(lambda r: div0(r["strikes"], r["pitches"]), axis=1)
    group["Ball%"] = group.apply(lambda r: div0(r["balls"], r["pitches"]), axis=1)
    group["Whiff%"] = group.apply(lambda r: div0(r["whiffs"], r["pitches"]), axis=1)
    group["InPlay%"] = group.apply(lambda r: div0(r["inPlay"], r["pitches"]), axis=1)
    return group.sort_values("pitches", ascending=False)


def team_batted_ball_summary(pa_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    """用 PA 層級做團隊擊球品質：強勁擊球、滾飛、落點、BIP。"""
    if pa_df.empty or "isBattedBall" not in pa_df.columns:
        return pd.DataFrame()
    batted = pa_df[pa_df["isBattedBall"]].copy()
    if batted.empty:
        return pd.DataFrame()
    batted["hardHit"] = batted["hardness"].astype(str).eq("H")
    batted["groundBall"] = batted["trajectory"].astype(str).eq("G")
    batted["lineDrive"] = batted["trajectory"].astype(str).eq("L")
    batted["flyBall"] = batted["trajectory"].astype(str).eq("F")
    batted["popup"] = batted["trajectory"].astype(str).eq("P")
    group = batted.groupby("offenseTeam", dropna=False).agg(
        BIP=("paId", "count"),
        H=("H_flag", "sum"),
        HR=("HR_flag", "sum"),
        hardHit=("hardHit", "sum"),
        groundBall=("groundBall", "sum"),
        lineDrive=("lineDrive", "sum"),
        flyBall=("flyBall", "sum"),
        popup=("popup", "sum"),
        WPA=("WPA", "sum"),
        RE24=("RE24", "sum"),
    ).reset_index().rename(columns={"offenseTeam": "team"})
    games = team_game_counts(games_df)
    if not games.empty:
        group = group.merge(games, on="team", how="left")
    group["BIP/G"] = group.apply(lambda r: div0(r["BIP"], r["G"]), axis=1)
    group["HardHit%"] = group.apply(lambda r: div0(r["hardHit"], r["BIP"]), axis=1)
    group["GB%"] = group.apply(lambda r: div0(r["groundBall"], r["BIP"]), axis=1)
    group["LD%"] = group.apply(lambda r: div0(r["lineDrive"], r["BIP"]), axis=1)
    group["FB%"] = group.apply(lambda r: div0(r["flyBall"], r["BIP"]), axis=1)
    group["Popup%"] = group.apply(lambda r: div0(r["popup"], r["BIP"]), axis=1)
    group["WPA/G"] = group.apply(lambda r: div0(r["WPA"], r["G"]), axis=1)
    group["RE24/G"] = group.apply(lambda r: div0(r["RE24"], r["G"]), axis=1)
    return group.sort_values("HardHit%", ascending=False)


def pitch_summary(events_df: pd.DataFrame) -> pd.DataFrame:
    pitches = events_df[events_df["type"] == "PITCH"].copy() if not events_df.empty else pd.DataFrame()
    if pitches.empty:
        return pd.DataFrame()
    group = pitches.groupby(["pitcherName", "pitchType", "pitchTypeZh"], dropna=False).agg(
        pitches=("eventId", "count"),
        strikes=("isStrike", "sum"),
        balls=("isBall", "sum"),
        inPlay=("inPlay", "sum"),
        avgVelo=("velocity", "mean"),
        maxVelo=("velocity", "max"),
        invalidVelo=("invalidVelocity", "sum"),
        invalidCoord=("invalidCoord", "sum"),
        whiffs=("pitchCode", lambda s: (s == "SW").sum()),
        fouls=("pitchCode", lambda s: (s == "F").sum()),
    ).reset_index()
    group["Strike%"] = group.apply(lambda r: div0(r["strikes"], r["pitches"]), axis=1)
    group["Ball%"] = group.apply(lambda r: div0(r["balls"], r["pitches"]), axis=1)
    group["Whiff%"] = group.apply(lambda r: div0(r["whiffs"], r["pitches"]), axis=1)
    group["InPlay%"] = group.apply(lambda r: div0(r["inPlay"], r["pitches"]), axis=1)
    return group.sort_values(["pitches"], ascending=False)


# =========================
# 視覺化函式
# =========================

def plot_pitch_locations(df: pd.DataFrame, title: str = "投球進壘點") -> Optional[plt.Figure]:
    if df.empty or "coordX" not in df.columns or "coordY" not in df.columns:
        return None
    invalid_coord_n = int(df["invalidCoord"].sum()) if "invalidCoord" in df.columns else 0
    plot_df = df.dropna(subset=["coordX", "coordY"]).copy()
    if plot_df.empty:
        return None
    if invalid_coord_n:
        title = f"{title}（已排除 {invalid_coord_n} 筆異常座標）"

    original_n = len(plot_df)
    max_points = int(st.session_state.get("max_scatter_points", DEFAULT_MAX_SCATTER_POINTS))
    if original_n > max_points:
        plot_df = plot_df.sample(n=max_points, random_state=42).copy()
        title = f"{title}（抽樣 {max_points:,}/{original_n:,} 球）"

    fig, ax = plt.subplots(figsize=(4.9, 4.1), dpi=115)
    pitch_types = sorted(plot_df["pitchType"].fillna("").astype(str).unique()) if "pitchType" in plot_df.columns else [""]
    for pt in pitch_types:
        sub = plot_df[plot_df["pitchType"].fillna("").astype(str) == pt]
        label = f"{pt} {PITCH_TYPE_ZH.get(pt, '')}".strip()
        ax.scatter(sub["coordX"], sub["coordY"], s=24, alpha=0.65, label=label)

    ax.axvline(0, linewidth=1, alpha=0.5)
    ax.axhline(0, linewidth=1, alpha=0.5)
    approx_zone = patches.Rectangle((-70, -70), 140, 140, fill=False, linewidth=1.5, linestyle="--", alpha=0.8)
    ax.add_patch(approx_zone)
    ax.set_title(title, fontsize=12, **font_kwargs())
    ax.set_xlabel("coordX：捕手視角左右位置", **font_kwargs())
    ax.set_ylabel("coordY：高低位置", **font_kwargs())
    ax.set_xlim(-190, 190)
    ax.set_ylim(-170, 130)
    ax.grid(True, alpha=0.25)
    if len(pitch_types) <= 8:
        legend_outside_right(ax, fig=fig, fontsize=7, anchor_x=1.02, anchor_y=0.50, right=0.74)
    apply_chinese_font_to_axes(ax)
    fig.subplots_adjust(left=0.12, right=0.74, top=0.90, bottom=0.14)
    return fig


def plot_velocity_histogram(df: pd.DataFrame, title: str = "球速分布") -> Optional[plt.Figure]:
    if df.empty or "velocity" not in df.columns:
        return None
    v = df["velocity"].dropna()
    if v.empty:
        return None
    invalid_n = int(df["invalidVelocity"].sum()) if "invalidVelocity" in df.columns else 0
    if invalid_n:
        title = f"{title}（已排除 {invalid_n} 筆異常球速）"
    fig, ax = plt.subplots(figsize=(4.9, 3.0), dpi=115)
    ax.hist(v, bins=min(18, max(6, int(v.nunique()))), alpha=0.75)
    ax.set_title(title, fontsize=12, **font_kwargs())
    ax.set_xlabel("球速", **font_kwargs())
    ax.set_ylabel("顆數", **font_kwargs())
    ax.grid(True, alpha=0.25)
    apply_chinese_font_to_axes(ax)
    fig.tight_layout(pad=0.4)
    return fig


# 球場外野牆尺寸資料：單位為英呎。
# 來源以 CPBL 官方球場介紹為主；臺南官方頁面未在搜尋摘要中直接列尺寸時，採用公開資料常見標示 339/400/339。
BALLPARK_DIMENSIONS: Dict[str, Dict[str, Any]] = {
    "臺北大巨蛋": {
        "display": "臺北大巨蛋",
        "lf": 335, "lcf": 375, "cf": 400, "rcf": 375, "rf": 335,
        "surface": "人工草皮",
    },
    "台北大巨蛋": {
        "display": "臺北大巨蛋",
        "lf": 335, "lcf": 375, "cf": 400, "rcf": 375, "rf": 335,
        "surface": "人工草皮",
    },
    "臺中市洲際棒球場": {
        "display": "臺中市洲際棒球場",
        "lf": 325, "lcf": 370, "cf": 400, "rcf": 370, "rf": 325,
        "surface": "天然草皮",
    },
    "台中市洲際棒球場": {
        "display": "臺中市洲際棒球場",
        "lf": 325, "lcf": 370, "cf": 400, "rcf": 370, "rf": 325,
        "surface": "天然草皮",
    },
    "臺南市立棒球場": {
        "display": "臺南市立棒球場",
        "lf": 339, "lcf": 372, "cf": 400, "rcf": 372, "rf": 339,
        "surface": "天然草皮",
    },
    "台南市立棒球場": {
        "display": "臺南市立棒球場",
        "lf": 339, "lcf": 372, "cf": 400, "rcf": 372, "rf": 339,
        "surface": "天然草皮",
    },
}

DEFAULT_BALLPARK = {
    "display": "標準棒球場",
    "lf": 330, "lcf": 375, "cf": 400, "rcf": 375, "rf": 330,
    "surface": "草皮",
}


def get_ballpark_dimensions(stadium: Any) -> Dict[str, Any]:
    name = str(stadium or "").strip()
    if name in BALLPARK_DIMENSIONS:
        return BALLPARK_DIMENSIONS[name]
    for key, dims in BALLPARK_DIMENSIONS.items():
        if key and key in name:
            return dims
    return DEFAULT_BALLPARK


def polar_to_xy(radius: float, theta_degree: float) -> Tuple[float, float]:
    rad = math.radians(theta_degree)
    return float(radius * math.cos(rad)), float(radius * math.sin(rad))


def cubic_bezier_points(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    samples: int = 90,
    include_endpoint: bool = True,
) -> List[Tuple[float, float]]:
    """產生三次 Bezier 曲線點。用來讓外野牆變成真正連續的漂亮曲線。"""
    points: List[Tuple[float, float]] = []
    end = samples + 1 if include_endpoint else samples
    for i in range(end):
        t = i / samples
        u = 1 - t
        x = (u ** 3) * p0[0] + 3 * (u ** 2) * t * p1[0] + 3 * u * (t ** 2) * p2[0] + (t ** 3) * p3[0]
        y = (u ** 3) * p0[1] + 3 * (u ** 2) * t * p1[1] + 3 * u * (t ** 2) * p2[1] + (t ** 3) * p3[1]
        points.append((float(x), float(y)))
    return points


def wall_points_for_ballpark(stadium: Any) -> List[Tuple[float, float]]:
    """依球場外野距離建立平滑外野牆曲線。

    v4.3 改法：
    - 不再用角度分段插值，也不再用多邊形折線。
    - 右外野線 → 中外野、以及中外野 → 左外野線各用一段 cubic Bezier。
    - 兩段 Bezier 在 CF 交會時共用水平切線，所以外野頂部會很平順。
    """
    dims = get_ballpark_dimensions(stadium)
    rf = float(dims["rf"])
    cf = float(dims["cf"])
    lf = float(dims["lf"])

    rf_corner = polar_to_xy(rf, 45)
    cf_point = (0.0, cf)
    lf_corner = polar_to_xy(lf, 135)

    # 右外野牆：RF foul pole -> CF
    r_c1 = (rf_corner[0] * 0.83, rf_corner[1] + (cf - rf_corner[1]) * 0.30)
    r_c2 = (max(85.0, rf_corner[0] * 0.38), cf)

    # 左外野牆：CF -> LF foul pole
    l_c1 = (min(-85.0, lf_corner[0] * 0.38), cf)
    l_c2 = (lf_corner[0] * 0.83, lf_corner[1] + (cf - lf_corner[1]) * 0.30)

    right_wall = cubic_bezier_points(rf_corner, r_c1, r_c2, cf_point, samples=120, include_endpoint=False)
    left_wall = cubic_bezier_points(cf_point, l_c1, l_c2, lf_corner, samples=120, include_endpoint=True)
    return right_wall + left_wall


def outfield_wall_radius_at_angle(theta: float, stadium: Any = None) -> float:
    """從 Bezier 外野牆曲線反查某角度的外野牆距離。"""
    theta = max(45.0, min(float(theta), 135.0))
    wall = wall_points_for_ballpark(stadium)

    angle_radius: List[Tuple[float, float]] = []
    for x, y in wall:
        angle = math.degrees(math.atan2(y, x))
        radius = math.sqrt(x * x + y * y)
        if 45 <= angle <= 135:
            angle_radius.append((angle, radius))

    if not angle_radius:
        dims = get_ballpark_dimensions(stadium)
        return float(dims["cf"])

    angle_radius.sort(key=lambda p: p[0])
    angles = [p[0] for p in angle_radius]
    radii = [p[1] for p in angle_radius]
    return float(np.interp(theta, angles, radii))


def scaled_points(points: List[Tuple[float, float]], scale: float) -> List[Tuple[float, float]]:
    return [(x * scale, y * scale) for x, y in points]


def parse_location_code(code: Any) -> Tuple[str, str]:
    text = str(code or "").upper().strip()
    digits = "".join(re.findall(r"[1-9]", text))
    suffix = re.sub(r"[1-9]", "", text)
    return digits, suffix


def location_to_polar(code: Any, stadium: Any = None) -> Optional[Tuple[float, float]]:
    """把擊球落點代碼轉成球場極座標。

    半徑使用英呎邏輯：內野約 45–125 ft，外野淺區約 205 ft，深區接近該方向外野牆。
    角度遵守棒球場：一壘線 45°、中外野 90°、三壘線 135°。
    """
    text = str(code or "").upper().strip()
    if not text:
        return None
    digits, suffix = parse_location_code(text)
    if not digits:
        return None

    angle_map = {
        "1": 90, "2": 90,
        "3": 52, "4": 72, "6": 108, "5": 128,
        "9": 52, "8": 90, "7": 128,
    }
    angles = [angle_map[d] for d in digits if d in angle_map]
    if not angles:
        return None
    theta = float(np.mean(angles))

    has_outfield = any(d in digits for d in ["7", "8", "9"])
    has_infield = any(d in digits for d in ["3", "4", "5", "6"])

    # 該方向的外野牆距離，用來避免落點跑到牆外。
    wall_r = outfield_wall_radius_at_angle(theta, stadium)

    if digits == "2":
        r = 18
    elif digits == "1":
        r = 62
    elif has_outfield and has_infield:
        r = min(210, wall_r * 0.58)
    elif has_outfield:
        r = min(265, wall_r * 0.68)
    else:
        r = 105

    # S/D/XD/HR 決定深淺。
    if "HR" in suffix:
        r = wall_r + 8
    elif "XD" in suffix:
        r = wall_r * 0.90
    elif "D" in suffix:
        r = wall_r * 0.78
    elif "S" in suffix:
        r = r * 0.82
    elif "M" in suffix:
        r = r * 1.03
    elif "L" in suffix:
        r = r * 0.90

    if "F" in suffix:
        if theta < 90:
            theta -= 4
        elif theta > 90:
            theta += 4

    theta = max(45, min(float(theta), 135))
    r = max(10, min(float(r), wall_r + 12))
    return r, theta


def location_to_xy(code: Any, stadium: Any = None) -> Optional[Tuple[float, float]]:
    polar = location_to_polar(code, stadium=stadium)
    if polar is None:
        return None
    r, theta = polar
    return polar_to_xy(r, theta)


def draw_baseball_field(ax: plt.Axes, stadium: Any = None) -> None:
    """畫比較接近棒球場平面圖的 spray chart 底圖。

    重點：不再用單純扇形；外野牆依球場 LF/CF/RF 距離繪製，內野用 90 ft 壘間距離。
    """
    dims = get_ballpark_dimensions(stadium)
    wall = wall_points_for_ballpark(stadium)
    warning = scaled_points(wall, 0.965)

    home = (0.0, 0.0)
    # 草地邊界：本壘 → 右外野牆 → 中外野 → 左外野牆 → 本壘
    field_polygon = [home] + wall + [home]
    warning_polygon = wall + list(reversed(warning))

    ax.add_patch(patches.Polygon(field_polygon, closed=True, facecolor="#9bd5c6", edgecolor="#d9d9d9", linewidth=1.2, alpha=0.72, zorder=0))
    ax.add_patch(patches.Polygon(warning_polygon, closed=True, facecolor="#f3eee4", edgecolor="none", alpha=0.95, zorder=1))
    ax.plot([p[0] for p in wall], [p[1] for p in wall], color="#c7c7c7", linewidth=1.5, zorder=3)

    # Foul line：實際棒球場左右界線是 90 度夾角。
    rf_x, rf_y = polar_to_xy(float(dims["rf"]), 45)
    lf_x, lf_y = polar_to_xy(float(dims["lf"]), 135)
    ax.plot([0, rf_x], [0, rf_y], color="#d2d2d2", linewidth=1.0, zorder=4)
    ax.plot([0, lf_x], [0, lf_y], color="#d2d2d2", linewidth=1.0, zorder=4)

    # 內野菱形：壘間 90 ft。
    base = 90 / math.sqrt(2)
    first = (base, base)
    second = (0.0, 90 * math.sqrt(2))
    third = (-base, base)
    diamond = [home, first, second, third]

    # 內野紅土近似形狀：菱形外包一點，不蓋成誇張扇形。
    dirt_points = [
        (0, -10), (22, 8), (78, 52), (86, 74), (54, 108),
        (18, 132), (0, 140), (-18, 132), (-54, 108), (-86, 74), (-78, 52), (-22, 8)
    ]
    ax.add_patch(patches.Polygon(dirt_points, closed=True, facecolor="#c99a6b", edgecolor="none", alpha=0.62, zorder=2))
    ax.add_patch(patches.Polygon(diamond, closed=True, fill=False, edgecolor="white", linewidth=2.0, zorder=5))
    ax.add_patch(patches.Polygon(diamond, closed=True, fill=False, edgecolor="#bfbfbf", linewidth=0.8, zorder=6))

    # 壘包與投手丘
    ax.scatter([first[0], second[0], third[0]], [first[1], second[1], third[1]], marker="s", s=24, color="white", edgecolors="#bfbfbf", linewidths=0.8, zorder=7)
    ax.scatter([home[0]], [home[1]], marker="p", s=45, color="white", edgecolors="#bfbfbf", linewidths=0.8, zorder=7)
    ax.scatter([0], [60.5], marker="o", s=24, color="white", edgecolors="#d0d0d0", linewidths=0.8, zorder=7)

    # 草地內緣弧線，模擬 spray chart 常見底圖。
    for scale, alpha in [(0.68, 0.20), (0.84, 0.18)]:
        inner = scaled_points(wall, scale)
        ax.plot([p[0] for p in inner], [p[1] for p in inner], color="#b8b8b8", linewidth=0.7, alpha=alpha, zorder=2)

    # 距離標示
    label_specs = [
        (135, float(dims["lf"]), str(dims["lf"])),
        (90, float(dims["cf"]), str(dims["cf"])),
        (45, float(dims["rf"]), str(dims["rf"])),
    ]
    for theta, radius, label in label_specs:
        x, y = polar_to_xy(radius * 0.98, theta)
        ax.text(x, y + 8, label, ha="center", va="bottom", fontsize=8, color="#86c8c3", weight="bold", **font_kwargs())

    # 守位參考文字
    positions = {
        "P": (0, 60.5), "C": (0, -7), "1B": (64, 78), "2B": (24, 128),
        "SS": (-28, 128), "3B": (-64, 78), "LF": (-132, 215), "CF": (0, 278), "RF": (132, 215),
    }
    for label, (x, y) in positions.items():
        ax.text(x, y, label, ha="center", va="center", fontsize=7, color="#777777", alpha=0.85, **font_kwargs())

    # 球場名稱
    ax.text(0, -28, dims.get("display", "標準棒球場"), ha="center", va="top", fontsize=8, color="#777777", **font_kwargs())


def plot_batted_ball_map(pa_df: pd.DataFrame, title: str = "擊球落點圖") -> Optional[plt.Figure]:
    if pa_df.empty or "locationCode" not in pa_df.columns:
        return None

    stadium = None
    if "stadium" in pa_df.columns and not pa_df["stadium"].dropna().empty:
        stadium = pa_df["stadium"].dropna().astype(str).mode().iloc[0]

    plot_rows = []
    for _, row in pa_df.iterrows():
        xy = location_to_xy(row.get("locationCode"), stadium=stadium)
        if xy is None:
            continue
        plot_rows.append({**row.to_dict(), "x": xy[0], "y": xy[1]})

    if not plot_rows:
        return None

    plot_df = pd.DataFrame(plot_rows)
    fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=115)
    draw_baseball_field(ax, stadium=stadium)

    result_order = ["1B", "2B", "3B", "HR", "FO", "GO", "GIDP", "FC", "E", "SF", "SH", "SO"]
    present = [r for r in result_order if r in set(plot_df["result"].fillna("").astype(str))]
    others = sorted(set(plot_df["result"].fillna("").astype(str)) - set(present))

    for result in present + others:
        sub = plot_df[plot_df["result"].fillna("").astype(str) == result]
        if sub.empty:
            continue
        label = f"{result} {RESULT_ZH.get(str(result), '')}".strip()
        ax.scatter(
            sub["x"], sub["y"],
            s=36, alpha=0.86, edgecolors="#222222", linewidths=0.45,
            label=label, zorder=10,
        )

    # 落點代碼標籤：少量資料才標，避免整張圖糊掉。
    counts = plot_df.groupby("locationCode").size().reset_index(name="n")
    if len(counts) <= 30:
        for _, r in counts.iterrows():
            xy = location_to_xy(r["locationCode"], stadium=stadium)
            if xy:
                ax.text(
                    xy[0], xy[1] + 6, f"{r['locationCode']}({r['n']})",
                    fontsize=7, ha="center", va="bottom", color="#222222",
                    zorder=12, **font_kwargs(),
                )

    dims = get_ballpark_dimensions(stadium)
    ax.set_title(f"{title}｜{dims.get('display', '')}", fontsize=12, pad=8, **font_kwargs())
    ax.set_xlim(-300, 300)
    ax.set_ylim(-38, 430)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    legend_outside_right(ax, fig=fig, fontsize=7, anchor_x=1.02, anchor_y=0.54, right=0.73)

    apply_chinese_font_to_axes(ax)
    # 不用 tight_layout，避免外移圖例被裁掉；改用 subplots_adjust 預留右側圖例空間。
    fig.subplots_adjust(left=0.04, right=0.73, top=0.92, bottom=0.05)
    return fig



def plot_team_inning_runs(
    score_df: pd.DataFrame,
    title: str = "場均每局得分",
    highlighted_teams: Optional[Sequence[str]] = None,
    unchecked_alpha: float = 0.50,
) -> Optional[plt.Figure]:
    """依球隊代表色繪製「場均每局得分」。

    計算方式：
    每隊第 n 局總得分 / 該隊在目前篩選範圍內的出賽場數。
    highlighted_teams 有勾選的隊伍用 alpha=1；沒勾選用 unchecked_alpha。
    """
    if score_df.empty or not {"inning", "team", "runs"}.issubset(score_df.columns):
        return None

    total_runs = score_df.pivot_table(index="inning", columns="team", values="runs", aggfunc="sum", fill_value=0)
    if total_runs.empty:
        return None

    if "G" in score_df.columns:
        games_by_team = score_df.groupby("team")["G"].nunique()
    else:
        games_by_team = score_df.groupby("team").size()

    pivot = total_runs.copy().astype(float)
    for team in pivot.columns:
        denom = safe_float(games_by_team.get(team, np.nan))
        if pd.isna(denom) or denom == 0:
            pivot[team] = np.nan
        else:
            pivot[team] = pivot[team] / denom

    pivot = pivot.fillna(0)
    innings = pivot.index.astype(int).tolist()
    teams = list(pivot.columns)
    selected_set = set(highlighted_teams or teams)

    x = np.arange(len(innings), dtype=float)
    n = max(1, len(teams))
    width = min(0.78 / n, 0.18)

    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=120)

    for i, team in enumerate(teams):
        offset = (i - (n - 1) / 2) * width
        alpha = 1.0 if team in selected_set else unchecked_alpha
        ax.bar(
            x + offset,
            pivot[team].values,
            width=width,
            label=str(team),
            color=get_team_color(team, i),
            alpha=alpha,
        )

    ax.set_title(title, **font_kwargs())
    ax.set_xlabel("局數", **font_kwargs())
    ax.set_ylabel("場均得分", **font_kwargs())
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in innings])
    ax.grid(axis="y", alpha=0.25)
    legend_outside_right(ax, fig=fig, fontsize=8, anchor_x=1.02, anchor_y=0.50, right=0.76)
    apply_chinese_font_to_axes(ax)
    fig.subplots_adjust(left=0.10, right=0.76, top=0.90, bottom=0.16)
    return fig


def plot_rank_bar_by_team(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    team_col: str = "team",
    title: str = "排行榜",
    value_label: Optional[str] = None,
    top_n: Optional[int] = None,
) -> Optional[plt.Figure]:
    """排行榜：依球員所屬球隊套色。"""
    if df.empty or label_col not in df.columns or value_col not in df.columns:
        return None

    plot_df = df.copy()
    if top_n is not None:
        plot_df = plot_df.head(top_n)

    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[value_col])
    if plot_df.empty:
        return None

    plot_df = plot_df.iloc[::-1].copy()
    labels = plot_df[label_col].astype(str).tolist()
    values = plot_df[value_col].astype(float).tolist()
    teams = plot_df[team_col].astype(str).tolist() if team_col in plot_df.columns else [""] * len(plot_df)
    colors = get_team_color_list(teams)

    height = max(3.2, min(7.5, 0.36 * len(plot_df) + 1.2))
    fig, ax = plt.subplots(figsize=(6.4, height), dpi=120)
    bars = ax.barh(labels, values, color=colors, alpha=0.92)

    ax.set_title(title, **font_kwargs())
    ax.set_xlabel(value_label or value_col, **font_kwargs())
    ax.grid(axis="x", alpha=0.25)

    # 圖例：只顯示本圖出現的球隊，不重複。
    legend_handles = []
    used = []
    for team in teams:
        if team not in used:
            used.append(team)
            legend_handles.append(patches.Patch(color=get_team_color(team, len(used)-1), label=team))
    if legend_handles and len(legend_handles) <= 8:
        ax.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.50),
            fontsize=8,
            frameon=True,
            borderaxespad=0.2,
            prop=CHINESE_FONT_PROP,
        )
        fig.subplots_adjust(right=0.74)

    # 數值標籤
    x_min, x_max = ax.get_xlim()
    span = x_max - x_min if x_max != x_min else 1
    for bar, value in zip(bars, values):
        if value_col in {"AVG", "OBP", "SLG", "OPS", "ISO"}:
            label = fmt_batting_rate(value)
        elif value_col.endswith("%"):
            label = fmt_pct(value)
        else:
            label = fmt_num(value, 3 if abs(value) < 2 else 1)
        ax.text(
            bar.get_width() + span * 0.01,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=8,
            **font_kwargs(),
        )

    apply_chinese_font_to_axes(ax)
    fig.tight_layout(pad=0.6)
    return fig


def plot_team_timeline(
    timeline: pd.DataFrame,
    teams: Sequence[str],
    title: str = "累積 WPA 時間軸",
) -> Optional[plt.Figure]:
    """依球隊代表色繪製累積 WPA 折線。"""
    if timeline.empty or "PA序號" not in timeline.columns:
        return None

    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=120)
    plotted = False

    for i, team in enumerate(teams):
        col = f"{team}_cumWPA"
        if col not in timeline.columns:
            continue
        ax.plot(
            timeline["PA序號"],
            timeline[col],
            label=team,
            color=get_team_color(team, i),
            linewidth=2.2,
            alpha=0.95,
        )
        plotted = True

    if not plotted:
        return None

    ax.axhline(0, color="#999999", linewidth=0.8, alpha=0.7)
    ax.set_title(title, **font_kwargs())
    ax.set_xlabel("PA 序號", **font_kwargs())
    ax.set_ylabel("累積 WPA", **font_kwargs())
    ax.grid(True, alpha=0.25)
    legend_outside_right(ax, fig=fig, fontsize=8, anchor_x=1.02, anchor_y=0.50, right=0.76)
    apply_chinese_font_to_axes(ax)
    fig.subplots_adjust(left=0.10, right=0.76, top=0.90, bottom=0.16)
    return fig



# =========================
# UI 小元件
# =========================

def metric_row(items: Sequence[Tuple[str, Any, Optional[str]]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        col.metric(label, value, delta=delta)


def select_or_all(label: str, options: Sequence[Any], default: Optional[Sequence[Any]] = None, key: Optional[str] = None) -> List[Any]:
    options = list(options)
    if default is None:
        default = options
    return st.multiselect(label, options, default=list(default), key=key)


def show_dataframe(df: pd.DataFrame, height: Optional[int] = None, max_rows: Optional[int] = DEFAULT_MAX_DISPLAY_ROWS) -> None:
    """大型表格只先送前幾千列給前端，並在欄名上加中文數據說明 tooltip。"""
    if df is None:
        st.info("沒有資料。")
        return

    display_df = df
    if isinstance(max_rows, int) and max_rows > 0 and len(df) > max_rows:
        st.caption(f"資料共有 {len(df):,} 列；為了避免網頁卡住，畫面先顯示前 {max_rows:,} 列。需要完整資料可在原始資料表頁面另外匯出。")
        display_df = df.head(max_rows).copy()

    kwargs: Dict[str, Any] = {
        "use_container_width": True,
        "hide_index": True,
    }

    column_config = build_column_config(display_df)
    if column_config:
        kwargs["column_config"] = column_config

    if isinstance(height, int) and height > 0:
        kwargs["height"] = height
    st.dataframe(display_df, **kwargs)
    
    help_rows = []
    for col in display_df.columns:
        help_text = get_stat_help(col)
        if help_text:
            help_rows.append({
                "欄位": str(col),
                "說明": help_text,
            })

    if help_rows:
        with st.expander("📘 手機版：查看本表欄位說明"):
            st.dataframe(
                pd.DataFrame(help_rows),
                use_container_width=True,
                hide_index=True,
            )


def show_plot(fig: Optional[plt.Figure]) -> None:
    """用較小寬度顯示圖表，避免 Streamlit 把圖撐到整頁。"""
    if fig is None:
        return
    try:
        left, middle, right = st.columns([0.08, 0.62, 0.30])
        with middle:
            st.pyplot(fig, clear_figure=True, use_container_width=False)
    except TypeError:
        st.pyplot(fig, clear_figure=True)


# =========================
# 主程式資料準備
# =========================

st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

st.markdown(
    """
    <style>
    div[data-testid="stMetricValue"] {
        font-size: 1.65rem;
        line-height: 1.2;
        overflow: visible;
        white-space: nowrap;
    }
    div[data-baseweb="tag"]:has(span[title*="中信"]),
    div[data-baseweb="tag"][aria-label*="中信"],
    div[data-baseweb="tag"][title*="中信"] {
        background-color: #FFD204 !important;
        color: #1f2937 !important;
    }
    div[data-baseweb="tag"]:has(span[title*="味全"]),
    div[data-baseweb="tag"][aria-label*="味全"],
    div[data-baseweb="tag"][title*="味全"] {
        background-color: #E61B24 !important;
        color: white !important;
    }
    div[data-baseweb="tag"]:has(span[title*="統一"]),
    div[data-baseweb="tag"][aria-label*="統一"],
    div[data-baseweb="tag"][title*="統一"] {
        background-color: #F88626 !important;
        color: white !important;
    }
    div[data-baseweb="tag"]:has(span[title*="富邦"]),
    div[data-baseweb="tag"][aria-label*="富邦"],
    div[data-baseweb="tag"][title*="富邦"] {
        background-color: #1C3B8B !important;
        color: white !important;
    }
    div[data-baseweb="tag"]:has(span[title*="樂天"]),
    div[data-baseweb="tag"][aria-label*="樂天"],
    div[data-baseweb="tag"][title*="樂天"] {
        background-color: #D01026 !important;
        color: white !important;
    }
    div[data-baseweb="tag"]:has(span[title*="台鋼"]),
    div[data-baseweb="tag"][aria-label*="台鋼"],
    div[data-baseweb="tag"][title*="台鋼"] {
        background-color: #137A3B !important;
        color: white !important;
    }
    div[data-baseweb="tag"] svg {
        color: currentColor !important;
    }
    /* 讓圖表下方球隊 checkbox 維持單行，不要把統一7-ELEVEn獅拆成兩行。 */
    div[data-testid="stCheckbox"] label,
    div[data-testid="stCheckbox"] label div,
    div[data-testid="stCheckbox"] label p {
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }
    div[data-testid="stCheckbox"] {
        min-width: 170px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("資料來源")
    st.caption("版本：v4.3｜得點圈數據｜圖例外移不重疊｜記憶體快取｜PR表對齊")
    st.caption("提示：把滑鼠移到表格欄名上，可看到該數據的公式與用途。")
    st.caption(f"球速統計會排除低於 {VALID_VELO_MIN} 或高於 {VALID_VELO_MAX} km/h 的異常值；進壘點座標絕對值超過 {VALID_COORD_ABS_LIMIT} 也會排除。")
    if MATPLOTLIB_CHINESE_FONT:
        st.caption(f"圖表中文字型：{MATPLOTLIB_CHINESE_FONT}")
    else:
        st.warning("Matplotlib 沒找到中文字型。Windows 通常要有 Microsoft JhengHei；也可以在專案資料夾建立 fonts/ 放入合法中文字型檔。")
    uploaded_files = st.file_uploader(
        "上傳野球革命 JSON；不傳的話會自動讀取 app.py 同資料夾內的 JSON",
        type=["json"],
        accept_multiple_files=True,
    )
    if st.button("清除本機快取", use_container_width=True):
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
        st.cache_data.clear()
        st.success("已清除快取，下一次會重新讀取 JSON。")

all_dfs, source_names, cache_status = load_dataframes_from_sources(uploaded_files)

if not all_dfs or "games" not in all_dfs or all_dfs["games"].empty:
    st.error("目前沒有讀到可用的野球革命 JSON。把 OpenData JSON 放在 app.py 同一個資料夾，或從左邊上傳 JSON。")
    st.stop()

games_df = all_dfs["games"]
scores_df = all_dfs["scores"]
batters_game_df = all_dfs["batters_game"]
pitchers_game_df = all_dfs["pitchers_game"]
pa_df = all_dfs["pa"]
events_df = all_dfs["events"]
runners_df = all_dfs["runners"]


def add_month_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "date" not in df.columns:
        return df
    out = df.copy()
    dt = pd.to_datetime(out["date"], errors="coerce")
    out["dateParsed"] = dt
    out["month"] = dt.dt.strftime("%Y-%m")
    out["monthLabel"] = dt.dt.strftime("%Y年%m月")
    return out


games_df = add_month_columns(games_df)
scores_df = add_month_columns(scores_df)
batters_game_df = add_month_columns(batters_game_df)
pitchers_game_df = add_month_columns(pitchers_game_df)
pa_df = add_month_columns(pa_df)
events_df = add_month_columns(events_df)
runners_df = add_month_columns(runners_df)

player_index_df = build_player_index(batters_game_df, pitchers_game_df, pa_df)

with st.sidebar:
    st.success(f"已讀取 {len(games_df)} 場比賽")
    if cache_status == "memory":
        st.success("讀取模式：使用記憶體快取，這是最快模式。")
    elif cache_status == "hit":
        st.info("讀取模式：使用本機快取，速度會快很多。")
    elif cache_status == "miss":
        st.warning("讀取模式：第一次讀取或資料已更新，正在使用 JSON 重新建立快取。")
    with st.expander("讀到的檔案", expanded=False):
        for name in source_names:
            st.write(f"- {name}")

    st.header("全域篩選")
    st.caption("不選＝全部。改成表單後，點選條件不會每一下都重跑，按「套用篩選」才更新。")

    with st.form("global_filter_form"):
        team_options = sorted(set(games_df["awayTeam"].dropna().tolist() + games_df["homeTeam"].dropna().tolist()))
        selected_teams = st.multiselect("1. 所屬球隊（不選＝全部）", team_options, default=[])

        players_after_team = player_index_df[player_index_df["team"].isin(selected_teams)].copy() if selected_teams else player_index_df.copy()
        position_options = sorted(players_after_team["position"].dropna().astype(str).unique().tolist()) if not players_after_team.empty else []
        selected_positions = st.multiselect("2. 守位（不選＝全部）", position_options, default=[])

        player_query = st.text_input(
            "3. 球員名字 / 球衣背號",
            value="",
            placeholder="例：陳傑憲、24、宋、20",
            help="可輸入中文姓名、部分姓名或背號；多個條件可用空白或逗號分開。",
        )

        month_options = sorted(games_df["month"].dropna().astype(str).unique().tolist()) if "month" in games_df.columns else []
        month_label_map = dict(zip(games_df["month"].astype(str), games_df.get("monthLabel", games_df["month"]).astype(str))) if month_options else {}
        selected_months = st.multiselect(
            "4. 月份（不選＝全部）",
            month_options,
            default=[],
            format_func=lambda m: month_label_map.get(str(m), str(m)),
        )

        st.markdown("##### 進階：比賽範圍 / 效能")
        month_game_df = games_df.copy()
        if selected_months and "month" in month_game_df.columns:
            month_game_df = month_game_df[month_game_df["month"].isin(selected_months)]

        game_options = month_game_df["G"].dropna().astype(int).sort_values().unique().tolist()
        if game_options:
            g_min, g_max = min(game_options), max(game_options)
            if g_min == g_max:
                game_range = (g_min, g_max)
                st.caption(f"目前只有 G{g_min}")
            else:
                game_range = st.slider("比賽編號範圍", min_value=g_min, max_value=g_max, value=(g_min, g_max), step=1)
            selected_games = [g for g in game_options if game_range[0] <= g <= game_range[1]]
        else:
            selected_games = []

        game_text = st.text_input(
            "指定比賽（可選，例：1,2,15-20）",
            value="",
            help="有填這格時會覆蓋上面的範圍滑桿。",
        )

        def parse_game_selection(text_value: str, allowed_games: list[int]) -> list[int]:
            text_value = str(text_value or "").strip()
            if not text_value:
                return allowed_games
            chosen: set[int] = set()
            for part in re.split(r"[,，\\s]+", text_value):
                if not part:
                    continue
                if "-" in part:
                    left, right = part.split("-", 1)
                    if left.strip().isdigit() and right.strip().isdigit():
                        a, b = int(left), int(right)
                        if a > b:
                            a, b = b, a
                        chosen.update(range(a, b + 1))
                elif part.isdigit():
                    chosen.add(int(part))
            allowed_set = set(allowed_games)
            return sorted(g for g in chosen if g in allowed_set)

        selected_games = parse_game_selection(game_text, selected_games)

        max_scatter_points = st.slider(
            "投球散點圖最大顯示球數",
            min_value=1000,
            max_value=30000,
            value=int(st.session_state.get("max_scatter_points", DEFAULT_MAX_SCATTER_POINTS)),
            step=1000,
            help="整季資料投球數很多；超過上限會自動抽樣，避免圖表卡住。",
        )

        st.form_submit_button("套用篩選", use_container_width=True)

    st.session_state["max_scatter_points"] = int(max_scatter_points)
    st.caption(f"目前比賽數：{len(selected_games)} / {len(games_df)}")

    matched_player_index = filter_player_index(player_index_df, selected_teams, selected_positions, player_query)
    position_filter_active = bool(selected_positions)
    player_query_active = bool(str(player_query).strip())
    player_filter_active = position_filter_active or player_query_active

    if player_filter_active:
        if matched_player_index.empty:
            st.warning("目前球員篩選沒有符合的人。")
        else:
            st.caption(f"符合球員：{len(matched_player_index)} 人")
            with st.expander("查看符合的球員", expanded=False):
                st.dataframe(
                    matched_player_index[["team", "position", "playerNumber", "playerName", "roles", "positionSource"]].head(200),
                    use_container_width=True,
                    hide_index=True,
                )

    page = st.radio(
        "頁面",
        [
            "系列賽總覽",
            "球員索引",
            "球員查詢",
            "打者排行榜",
            "投手排行榜",
            "對戰組合",
            "投球分析",
            "擊球落點",
            "關鍵時刻 WPA/RE24",
            "跑壘與責失",
            "資料品質檢查",
            "原始資料表",
        ],
    )

# 套用全域比賽篩選
fgames = games_df[games_df["G"].isin(selected_games)].copy()
fscores = scores_df[scores_df["G"].isin(selected_games)].copy()
fbatters_game = batters_game_df[batters_game_df["G"].isin(selected_games)].copy()
fpitchers_game = pitchers_game_df[pitchers_game_df["G"].isin(selected_games)].copy()
fpa = pa_df[pa_df["G"].isin(selected_games)].copy()
fevents = events_df[events_df["G"].isin(selected_games)].copy()
frunners = runners_df[runners_df["G"].isin(selected_games)].copy()

# 1. 所屬球隊篩選
if selected_teams:
    fgames = fgames[(fgames["awayTeam"].isin(selected_teams)) | (fgames["homeTeam"].isin(selected_teams))] if not fgames.empty else fgames
    fbatters_game = fbatters_game[fbatters_game["team"].isin(selected_teams)] if not fbatters_game.empty else fbatters_game
    fpitchers_game = fpitchers_game[fpitchers_game["team"].isin(selected_teams)] if not fpitchers_game.empty else fpitchers_game
    fpa = fpa[(fpa["offenseTeam"].isin(selected_teams)) | (fpa["defenseTeam"].isin(selected_teams))] if not fpa.empty else fpa
    fevents = fevents[(fevents["offenseTeam"].isin(selected_teams)) | (fevents["defenseTeam"].isin(selected_teams))] if not fevents.empty else fevents
    frunners = frunners[(frunners["offenseTeam"].isin(selected_teams)) | (frunners["defenseTeam"].isin(selected_teams))] if not frunners.empty else frunners
    fscores = fscores[fscores["team"].isin(selected_teams)] if not fscores.empty else fscores

# 2–3. 守位 / 姓名 / 背號篩選
selected_player_keys = set(
    zip(matched_player_index["team"].astype(str), matched_player_index["playerName"].astype(str))
) if not matched_player_index.empty else set()

if player_filter_active:
    if selected_player_keys:
        if not fbatters_game.empty:
            fbatters_game = fbatters_game[key_mask(fbatters_game, "team", "playerName", selected_player_keys)]
        if not fpitchers_game.empty:
            fpitchers_game = fpitchers_game[key_mask(fpitchers_game, "team", "playerName", selected_player_keys)]
        if not fpa.empty:
            pa_mask = (
                key_mask(fpa, "offenseTeam", "batterName", selected_player_keys) |
                key_mask(fpa, "defenseTeam", "pitcherName", selected_player_keys) |
                key_mask(fpa, "defenseTeam", "catcherName", selected_player_keys)
            )
            fpa = fpa[pa_mask]
        if not fevents.empty:
            event_mask = (
                key_mask(fevents, "offenseTeam", "batterName", selected_player_keys) |
                key_mask(fevents, "defenseTeam", "pitcherName", selected_player_keys) |
                key_mask(fevents, "defenseTeam", "catcherName", selected_player_keys)
            )
            fevents = fevents[event_mask]
        if not frunners.empty:
            runner_mask = (
                key_mask(frunners, "offenseTeam", "batterName", selected_player_keys) |
                key_mask(frunners, "defenseTeam", "pitcherName", selected_player_keys) |
                key_mask(frunners, "offenseTeam", "runnerName", selected_player_keys)
            )
            frunners = frunners[runner_mask]

        # 若有篩到特定球員，game/score 也只保留有該球員出現的比賽。
        active_games: set[int] = set()
        for df in [fbatters_game, fpitchers_game, fpa, fevents, frunners]:
            if df is not None and not df.empty and "G" in df.columns:
                active_games.update(safe_int(g) for g in df["G"].dropna().unique())
        if active_games:
            fgames = fgames[fgames["G"].isin(active_games)]
            fscores = fscores[fscores["G"].isin(active_games)] if not fscores.empty else fscores
    else:
        fgames = fgames.iloc[0:0]
        fscores = fscores.iloc[0:0]
        fbatters_game = fbatters_game.iloc[0:0]
        fpitchers_game = fpitchers_game.iloc[0:0]
        fpa = fpa.iloc[0:0]
        fevents = fevents.iloc[0:0]
        frunners = frunners.iloc[0:0]


batters_agg = aggregate_batters(fbatters_game)
pitchers_agg = aggregate_pitchers(fpitchers_game)
matchups_agg = aggregate_matchups(fpa)

risp_batters_agg = risp_summary_from_pa(
    fpa,
    ["batterName", "offenseTeam"],
    {"batterName": "playerName", "offenseTeam": "team"},
)
risp_pitchers_agg = risp_summary_from_pa(
    fpa,
    ["pitcherName", "defenseTeam"],
    {"pitcherName": "playerName", "defenseTeam": "team"},
)
team_risp_bat = risp_summary_from_pa(
    fpa,
    ["offenseTeam"],
    {"offenseTeam": "team"},
)
team_risp_pitch = risp_summary_from_pa(
    fpa,
    ["defenseTeam"],
    {"defenseTeam": "team"},
)



# =========================
# PR 百分位排名
# =========================

def percentile_rank(value: Any, population: pd.Series, higher_is_better: bool = True) -> float:
    """計算 Percentile Rank。

    PR 80 = 這個指標約優於比較群 80% 球員。
    用 0.5 * ties 處理同分，避免大量同值時全部擠到 100。
    """
    v = safe_float(value)
    if pd.isna(v):
        return np.nan

    pop = pd.to_numeric(population, errors="coerce").dropna()
    if pop.empty:
        return np.nan

    if higher_is_better:
        better = (pop < v).sum()
    else:
        better = (pop > v).sum()

    ties = (pop == v).sum()
    return float((better + 0.5 * ties) / len(pop) * 100)


def pr_grade(pr: Any) -> str:
    p = safe_float(pr)
    if pd.isna(p):
        return "-"
    if p >= 90:
        return "頂尖"
    if p >= 75:
        return "優秀"
    if p >= 60:
        return "中上"
    if p >= 40:
        return "平均附近"
    if p >= 25:
        return "偏低"
    return "需加強"


def format_metric_for_pr(metric: str, value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    if metric in {"AVG", "OBP", "SLG", "OPS", "ISO"}:
        return fmt_batting_rate(value)
    if metric in {"BB%", "K%", "K-BB%"}:
        return fmt_pct(value)
    if metric in {"ERA", "WHIP", "NP/IP", "NP/BF", "K/9", "BB/9", "H/9", "HR/9", "WPA", "RE24"}:
        return fmt_rate(value, 3 if metric in {"WPA", "RE24"} else 2)
    if metric in {"PA", "AB", "H", "HR", "RBI", "TB", "XBH", "SO", "BB", "SB", "BF", "NP"}:
        return str(int(safe_float(value))) if not pd.isna(safe_float(value)) else "-"
    return fmt_rate(value, 3)


def build_pr_table(
    player_row: pd.Series,
    population_df: pd.DataFrame,
    specs: Sequence[Dict[str, Any]],
    minimum_col: str,
    minimum_value: int,
    title_prefix: str,
) -> pd.DataFrame:
    """建立球員 PR 表。"""
    if population_df.empty:
        return pd.DataFrame()

    player_name = str(player_row.get("playerName", ""))
    population = population_df.copy()
    if minimum_col in population.columns:
        population = population[pd.to_numeric(population[minimum_col], errors="coerce").fillna(0) >= minimum_value].copy()

    # 避免選手剛好低於門檻時完全沒比較；至少把本人放回比較群。
    if player_name and "playerName" in population_df.columns and player_name not in set(population.get("playerName", pd.Series(dtype=str)).astype(str)):
        self_rows = population_df[population_df["playerName"].astype(str) == player_name].copy()
        if not self_rows.empty:
            population = pd.concat([population, self_rows], ignore_index=True)

    rows: List[Dict[str, Any]] = []
    for spec in specs:
        metric = spec["metric"]
        if metric not in player_row.index or metric not in population.columns:
            continue
        value = player_row.get(metric)
        pr = percentile_rank(value, population[metric], higher_is_better=spec.get("higher", True))
        if pd.isna(pr):
            continue
        rows.append(
            {
                "類別": spec.get("category", title_prefix),
                "指標": metric,
                "數值": format_metric_for_pr(metric, value),
                "PR": round(pr, 1),
                "等級": pr_grade(pr),
                "比較群": f"{len(population)} 人｜{minimum_col}≥{minimum_value}",
                "解讀": spec.get("desc", ""),
            }
        )

    return pd.DataFrame(rows)


def build_batter_pr_table(player_row: pd.Series, batters_agg: pd.DataFrame) -> pd.DataFrame:
    player_pa = safe_int(player_row.get("PA"))
    min_pa = 30 if player_pa >= 30 else max(1, min(player_pa, 10))
    specs = [
        {"metric": "OPS", "category": "整體攻擊", "higher": True, "desc": "上壘率 + 長打率，越高代表整體攻擊火力越好。"},
        {"metric": "OBP", "category": "上壘", "higher": True, "desc": "避免出局並成功上壘的能力。"},
        {"metric": "SLG", "category": "長打", "higher": True, "desc": "每打數平均壘打數，越高代表長打破壞力越好。"},
        {"metric": "AVG", "category": "安打", "higher": True, "desc": "每打數形成安打的比例。"},
        {"metric": "ISO", "category": "長打", "higher": True, "desc": "SLG - AVG，單獨觀察純長打能力。"},
        {"metric": "BB%", "category": "選球", "higher": True, "desc": "保送 / PA，越高通常代表選球與上壘能力較好。"},
        {"metric": "K%", "category": "三振控制", "higher": False, "desc": "三振 / PA；打者端越低越好。"},
        {"metric": "HR", "category": "累積產量", "higher": True, "desc": "全壘打累積數，受出賽機會影響。"},
        {"metric": "TB", "category": "累積產量", "higher": True, "desc": "壘打數，衡量累積攻擊產出。"},
        {"metric": "RBI", "category": "累積產量", "higher": True, "desc": "打點累積數，會受棒次與壘上有人機會影響。"},
        {"metric": "WPA", "category": "關鍵貢獻", "higher": True, "desc": "勝率貢獻累積；越高代表對贏球情境幫助越多。"},
        {"metric": "RE24", "category": "得分環境", "higher": True, "desc": "打席對得分期望的累積貢獻。"},
    ]
    return build_pr_table(player_row, batters_agg, specs, "PA", min_pa, "打擊")


def build_pitcher_pr_table(player_row: pd.Series, pitchers_agg: pd.DataFrame) -> pd.DataFrame:
    player_bf = safe_int(player_row.get("BF"))
    min_bf = 30 if player_bf >= 30 else max(1, min(player_bf, 10))
    specs = [
        {"metric": "ERA", "category": "失分控制", "higher": False, "desc": "責失分 × 9 / IP；投手端越低越好。"},
        {"metric": "WHIP", "category": "上壘壓制", "higher": False, "desc": "(BB + H) / IP；每局讓打者靠安打或保送上壘的頻率，越低越好。"},
        {"metric": "K%", "category": "三振能力", "higher": True, "desc": "三振 / BF，越高代表製造三振能力越好。"},
        {"metric": "BB%", "category": "控球", "higher": False, "desc": "保送 / BF；投手端越低代表控球越穩。"},
        {"metric": "K-BB%", "category": "壓制力", "higher": True, "desc": "K% - BB%，同時考慮三振與保送，越高越好。"},
        {"metric": "K/9", "category": "三振能力", "higher": True, "desc": "每 9 局三振數。"},
        {"metric": "BB/9", "category": "控球", "higher": False, "desc": "每 9 局保送數；越低越好。"},
        {"metric": "H/9", "category": "被安打控制", "higher": False, "desc": "每 9 局被安打數；越低越好。"},
        {"metric": "HR/9", "category": "被長打控制", "higher": False, "desc": "每 9 局被全壘打數；越低越好。"},
        {"metric": "NP/IP", "category": "效率", "higher": False, "desc": "每局用球數；越低通常代表投球效率越好。"},
        {"metric": "NP/BF", "category": "效率", "higher": False, "desc": "每面對一名打者平均用球數；越低通常代表解決打者較有效率。"},
        {"metric": "SO", "category": "累積產量", "higher": True, "desc": "三振累積數，受投球局數與角色影響。"},
    ]
    return build_pr_table(player_row, pitchers_agg, specs, "BF", min_bf, "投球")


def show_pr_table(pr_df: pd.DataFrame, title: str, align_with_metric_row: bool = False) -> None:
    st.markdown(f"### {title}")
    if pr_df.empty:
        st.info("目前資料不足，無法建立 PR 表。")
        return

    display_df = pr_df.drop(columns=["等級"], errors="ignore").copy()

    # 個人總表左邊有一排 metric；右邊 PR 表加一點空白，讓兩邊表格頂端對齊。
    if align_with_metric_row:
        st.markdown('<div style="height:72px;"></div>', unsafe_allow_html=True)

    try:
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=300,
            column_config={
                "PR": st.column_config.ProgressColumn(
                    "PR",
                    help="Percentile Rank，百分位排名。PR 越高代表該項表現越好。",
                    min_value=0,
                    max_value=100,
                    format="%.1f",
                ),
                "解讀": st.column_config.TextColumn(width="large"),
            },
        )
    except Exception:
        show_dataframe(display_df, height=300, max_rows=None)


# =========================
# Page 1: 系列賽總覽
# =========================

if page == "系列賽總覽":
    st.subheader("系列賽總覽")

    total_games = len(fgames)
    total_pa = len(fpa)
    total_pitches = len(fevents[fevents["type"] == "PITCH"]) if not fevents.empty else 0
    total_runs = int(fscores["runs"].sum()) if not fscores.empty else 0
    metric_row(
        [
            ("比賽數", total_games, None),
            ("總得分", total_runs, None),
            ("總打席 PA", total_pa, None),
            ("總投球事件", total_pitches, None),
        ]
    )

    st.markdown("### 每場比分")
    show_dataframe(fgames[["G", "date", "stadium", "awayTeam", "awayRuns", "homeRuns", "homeTeam", "winner"]].sort_values("G"))

    col1, col2 = st.columns([1.1, 1])
    with col1:
        st.markdown("### 球隊得分與勝敗")
        team_rows: List[Dict[str, Any]] = []
        overview_teams = selected_teams if selected_teams else sorted(set(fgames["awayTeam"].tolist() + fgames["homeTeam"].tolist()))
        for team in overview_teams:
            gf = fgames[(fgames["awayTeam"] == team) | (fgames["homeTeam"] == team)]
            wins = int((gf["winner"] == team).sum())
            losses = int(((gf["winner"] != team) & (gf["winner"] != "和局")).sum())
            runs_for = 0
            runs_against = 0
            for _, g in gf.iterrows():
                if g["awayTeam"] == team:
                    runs_for += safe_int(g["awayRuns"])
                    runs_against += safe_int(g["homeRuns"])
                else:
                    runs_for += safe_int(g["homeRuns"])
                    runs_against += safe_int(g["awayRuns"])
            games_played = len(gf)
            team_rows.append(
                {
                    "team": team,
                    "G": games_played,
                    "W": wins,
                    "L": losses,
                    "W%": div0(wins, wins + losses),
                    "得分": runs_for,
                    "失分": runs_against,
                    "分差": runs_for - runs_against,
                    "得分/G": div0(runs_for, games_played),
                    "失分/G": div0(runs_against, games_played),
                    "分差/G": div0(runs_for - runs_against, games_played),
                }
            )
        team_overview = pd.DataFrame(team_rows)
        show_dataframe(
            clean_display_df(
                team_overview.sort_values(["W%", "分差/G"], ascending=[False, False]),
                rate_cols=["W%", "得分/G", "失分/G", "分差/G"],
            )
        )

    with col2:
        st.markdown("### 場均每局得分")
        if not fscores.empty:
            chart_team_options = sorted(fscores["team"].dropna().astype(str).unique().tolist())

            # 用 session_state 保存圖表顯示狀態；checkbox 放進 form，避免每點一次就重跑整個 app。
            for team in chart_team_options:
                state_key = f"inning_alpha_{canonical_team_name(team)}"
                if state_key not in st.session_state:
                    st.session_state[state_key] = True

            highlighted_teams = [
                team for team in chart_team_options
                if st.session_state.get(f"inning_alpha_{canonical_team_name(team)}", True)
            ]

            fig = plot_team_inning_runs(fscores, "場均每局得分", highlighted_teams=highlighted_teams, unchecked_alpha=0.10)
            if fig:
                show_plot(fig)

                st.caption("勾選＝不透明顯示；取消勾選＝保留 10% 透明度。選完後按「套用圖表顯示」，不會每點一次就重跑。")

                with st.form("inning_score_display_form"):
                    if len(chart_team_options) == 6:
                        checkbox_cols = st.columns([1, 1, 1, 1, 1, 1.55])
                    else:
                        checkbox_cols = st.columns(min(6, max(1, len(chart_team_options))))
                    for i, team in enumerate(chart_team_options):
                        state_key = f"inning_alpha_{canonical_team_name(team)}"
                        temp_key = f"{state_key}_pending"
                        with checkbox_cols[i % len(checkbox_cols)]:
                            st.checkbox(
                                team,
                                value=st.session_state.get(state_key, True),
                                key=temp_key,
                            )

                    submitted = st.form_submit_button("套用圖表顯示", use_container_width=True)

                if submitted:
                    for team in chart_team_options:
                        state_key = f"inning_alpha_{canonical_team_name(team)}"
                        temp_key = f"{state_key}_pending"
                        st.session_state[state_key] = st.session_state.get(temp_key, True)
                    st.rerun()
            else:
                st.info("沒有得分資料。")
        else:
            st.info("沒有得分資料。")


    st.markdown("### 團隊打擊")
    team_bat = team_batting_from_box(fbatters_game)
    show_dataframe(
        clean_display_df(
            team_bat.sort_values("OPS", ascending=False),
            cols=["team", "G", "PA", "PA/G", "AB", "R", "R/G", "H", "H/G", "2B", "2B/G", "3B", "3B/G", "HR", "HR/G", "RBI", "RBI/G", "BB", "BB/G", "SO", "SO/G", "SB", "SB/G", "CS", "CS/G", "AVG", "OBP", "SLG", "OPS", "BB%", "K%"],
            rate_cols=["PA/G", "R/G", "H/G", "2B/G", "3B/G", "HR/G", "RBI/G", "BB/G", "SO/G", "SB/G", "CS/G", "AVG", "OBP", "SLG", "OPS"],
            pct_cols=["BB%", "K%"],
        ),
        height=360,
    )

    st.markdown("### 團隊投球")
    team_pitch = team_pitching_from_box(fpitchers_game)
    show_dataframe(
        clean_display_df(
            team_pitch.sort_values("ERA"),
            cols=["team", "G", "IP顯示", "IP/G", "NP", "NP/G", "BF", "BF/G", "H", "H/G", "HR", "HR/G", "BB", "BB/G", "HB", "HB/G", "SO", "SO/G", "R", "R/G", "ER", "ER/G", "ERA", "WHIP", "K%", "BB%", "K-BB%", "NP/IP"],
            rate_cols=["IP/G", "NP/G", "BF/G", "H/G", "HR/G", "BB/G", "HB/G", "SO/G", "R/G", "ER/G", "ERA", "WHIP", "NP/IP"],
            pct_cols=["K%", "BB%", "K-BB%"],
        ),
        height=360,
    )

    st.markdown("### 團隊得點圈攻守")
    risp_col1, risp_col2 = st.columns(2)
    with risp_col1:
        st.markdown("#### 得點圈打擊")
        if team_risp_bat.empty:
            st.info("目前沒有得點圈打擊資料。")
        else:
            show_dataframe(
                clean_display_df(
                    team_risp_bat.sort_values("OPS", ascending=False),
                    cols=["team", "PA", "AB", "H", "2B", "3B", "HR", "RBI", "BB", "SO", "AVG", "OBP", "SLG", "OPS", "K%", "BB%", "WPA", "RE24"],
                    rate_cols=["AVG", "OBP", "SLG", "OPS", "WPA", "RE24"],
                    pct_cols=["K%", "BB%"],
                ),
                height=320,
            )
    with risp_col2:
        st.markdown("#### 得點圈投球 / 被打擊")
        if team_risp_pitch.empty:
            st.info("目前沒有得點圈投球資料。")
        else:
            show_dataframe(
                clean_display_df(
                    team_risp_pitch.sort_values("OPS", ascending=True),
                    cols=["team", "PA", "AB", "H", "2B", "3B", "HR", "R", "RBI", "BB", "SO", "AVG", "OBP", "SLG", "OPS", "K%", "BB%", "WPA", "RE24"],
                    rate_cols=["AVG", "OBP", "SLG", "OPS", "WPA", "RE24"],
                    pct_cols=["K%", "BB%"],
                ),
                height=320,
            )

    st.markdown("### 團隊擊球品質")
    team_batted = team_batted_ball_summary(fpa, fgames)
    if team_batted.empty:
        st.info("目前沒有可用的擊球品質資料。")
    else:
        show_dataframe(
            clean_display_df(
                team_batted.sort_values("HardHit%", ascending=False),
                cols=["team", "G", "BIP", "BIP/G", "H", "HR", "hardHit", "HardHit%", "groundBall", "GB%", "lineDrive", "LD%", "flyBall", "FB%", "popup", "Popup%", "WPA", "WPA/G", "RE24", "RE24/G"],
                rate_cols=["BIP/G", "WPA", "WPA/G", "RE24", "RE24/G"],
                pct_cols=["HardHit%", "GB%", "LD%", "FB%", "Popup%"],
            ),
            height=360,
        )

    st.markdown("### 團隊投球事件")
    team_pitch_event = team_pitch_event_summary(fevents, fgames)
    if team_pitch_event.empty:
        st.info("目前沒有可用的投球事件資料。")
    else:
        show_dataframe(
            clean_display_df(
                team_pitch_event.sort_values("Strike%", ascending=False),
                cols=["team", "G", "pitches", "Pitches/G", "strikes", "balls", "whiffs", "inPlay", "Strike%", "Ball%", "Whiff%", "InPlay%", "avgVelo", "maxVelo", "invalidVelo", "invalidCoord"],
                rate_cols=["Pitches/G", "avgVelo", "maxVelo"],
                pct_cols=["Strike%", "Ball%", "Whiff%", "InPlay%"],
            ),
            height=360,
        )



# =========================
# Page 2: 球員索引
# =========================

elif page == "球員索引":
    st.subheader("球員索引：所屬球隊 / 守位 / 姓名 / 背號")
    st.caption("若沒有另外提供 player_meta.csv 或 roster.csv，守位會依資料推估：投手、捕手、野手/打者、投打二刀流；野球革命 PA 內沒有完整一壘/二壘/游擊/外野守位。")

    idx_view = matched_player_index.copy()
    if idx_view.empty:
        st.warning("目前篩選沒有符合球員。")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("球員數", len(idx_view))
        c2.metric("球隊數", idx_view["team"].nunique())
        c3.metric("守位類別", idx_view["position"].nunique())
        c4.metric("有背號資料", int((idx_view["playerNumber"].astype(str).str.strip() != "").sum()))

        st.markdown("### 守位分布")
        pos_count = idx_view.groupby(["team", "position"], dropna=False).size().reset_index(name="球員數")
        show_dataframe(pos_count.sort_values(["team", "position"]))

        st.markdown("### 球員清單")
        show_dataframe(
            idx_view[["team", "position", "playerNumber", "playerName", "roles", "PA", "AB", "BF", "positionSource"]].sort_values(["team", "position", "playerNumber", "playerName"]),
            height=620,
        )


# =========================
# Page 2: 球員查詢
# =========================

elif page == "球員查詢":
    st.subheader("球員查詢")

    if not matched_player_index.empty:
        option_df = matched_player_index.copy()
        option_df["label"] = option_df.apply(
            lambda r: f"{r['playerName']}｜{r['team']}｜#{r['playerNumber']}｜{r['position']}".replace("#｜", "｜"),
            axis=1,
        )
        label_to_name = dict(zip(option_df["label"], option_df["playerName"]))
        labels = option_df["label"].tolist()
        selected_label = st.selectbox("選擇球員", labels)
        player = label_to_name[selected_label]
    else:
        names = sorted(set(batters_agg.get("playerName", pd.Series(dtype=str)).dropna().tolist()) |
                       set(pitchers_agg.get("playerName", pd.Series(dtype=str)).dropna().tolist()) |
                       set(fpa.get("batterName", pd.Series(dtype=str)).dropna().tolist()) |
                       set(fpa.get("pitcherName", pd.Series(dtype=str)).dropna().tolist()))
        if not names:
            st.warning("沒有球員資料。")
            st.stop()
        player = st.selectbox("選擇球員", names)

    player_info = matched_player_index[matched_player_index["playerName"] == player] if not matched_player_index.empty else pd.DataFrame()
    if not player_info.empty:
        info = player_info.iloc[0]
        st.markdown(f"## {player}　`{info['team']}` `{info['position']}` `#{info['playerNumber']}`")
    else:
        st.markdown(f"## {player}")
    bat_row = batters_agg[batters_agg["playerName"] == player] if not batters_agg.empty else pd.DataFrame()
    pit_row = pitchers_agg[pitchers_agg["playerName"] == player] if not pitchers_agg.empty else pd.DataFrame()

    if not bat_row.empty:
        row = bat_row.iloc[0]
        bat_col, pr_col = st.columns([1, 1])
        with bat_col:
            st.markdown("### 打擊總表")
            metric_row(
                [
                    ("PA", int(row["PA"]), None),
                    ("H / AB", f"{int(row['H'])}/{int(row['AB'])}", None),
                    ("AVG", fmt_batting_rate(row["AVG"]), None),
                    ("OBP", fmt_batting_rate(row["OBP"]), None),
                    ("SLG", fmt_batting_rate(row["SLG"]), None),
                    ("OPS", fmt_batting_rate(row["OPS"]), None),
                ]
            )
            show_dataframe(
                clean_display_df(
                    bat_row,
                    cols=["team", "playerNumber", "PA", "AB", "R", "H", "2B", "3B", "HR", "RBI", "BB", "HBP", "SO", "SB", "CS", "AVG", "OBP", "SLG", "OPS", "BB%", "K%", "WPA", "RE24"],
                    rate_cols=["AVG", "OBP", "SLG", "OPS"],
                    pct_cols=["BB%", "K%"],
                ),
                height=160,
            )

            st.markdown("### 得點圈打擊")
            player_risp_bat = risp_batters_agg[risp_batters_agg["playerName"] == player].copy() if not risp_batters_agg.empty else pd.DataFrame()
            if player_risp_bat.empty:
                st.info("目前沒有得點圈打擊資料。")
            else:
                show_dataframe(
                    clean_display_df(
                        player_risp_bat,
                        cols=["team", "PA", "AB", "H", "2B", "3B", "HR", "RBI", "BB", "SO", "AVG", "OBP", "SLG", "OPS", "K%", "BB%", "WPA", "RE24"],
                        rate_cols=["AVG", "OBP", "SLG", "OPS", "WPA", "RE24"],
                        pct_cols=["K%", "BB%"],
                    ),
                    height=190,
                )
        with pr_col:
            show_pr_table(build_batter_pr_table(row, batters_agg), "打擊 PR 表", align_with_metric_row=True)
            st.caption("PR 是 Percentile Rank。PR 80 代表該指標約優於比較群 80% 球員。")

        st.markdown("### 逐場打擊")
        per_game = add_basic_rate_stats(fbatters_game[fbatters_game["playerName"] == player].copy())
        show_dataframe(
            clean_display_df(
                per_game.sort_values("G"),
                cols=["G", "date", "stadium", "team", "opponentTeam", "order", "PA", "AB", "R", "H", "2B", "3B", "HR", "RBI", "BB", "SO", "AVG", "OBP", "SLG", "OPS"],
                rate_cols=["AVG", "OBP", "SLG", "OPS"],
            )
        )

    if not pit_row.empty:
        row = pit_row.iloc[0]
        pit_col, pr_col = st.columns([1, 1])
        with pit_col:
            st.markdown("### 投球總表")
            metric_row(
                [
                    ("IP", row["IP顯示"], None),
                    ("ERA", fmt_rate(row["ERA"], 2), None),
                    ("WHIP", fmt_rate(row["WHIP"], 2), None),
                    ("SO", int(row["SO"]), None),
                    ("K%", fmt_pct(row["K%"]), None),
                    ("NP", int(row["NP"]), None),
                ]
            )
            show_dataframe(
                clean_display_df(
                    pit_row,
                    cols=["team", "playerNumber", "IP顯示", "NP", "BF", "H", "HR", "BB", "HB", "SO", "R", "ER", "ERA", "WHIP", "K%", "BB%", "K-BB%", "NP/IP", "NP/BF"],
                    rate_cols=["ERA", "WHIP", "NP/IP", "NP/BF"],
                    pct_cols=["K%", "BB%", "K-BB%"],
                ),
                height=160,
            )

            st.markdown("### 得點圈投球 / 被打擊")
            player_risp_pitch = risp_pitchers_agg[risp_pitchers_agg["playerName"] == player].copy() if not risp_pitchers_agg.empty else pd.DataFrame()
            if player_risp_pitch.empty:
                st.info("目前沒有得點圈投球資料。")
            else:
                show_dataframe(
                    clean_display_df(
                        player_risp_pitch,
                        cols=["team", "PA", "AB", "H", "2B", "3B", "HR", "R", "RBI", "BB", "SO", "AVG", "OBP", "SLG", "OPS", "K%", "BB%", "WPA", "RE24"],
                        rate_cols=["AVG", "OBP", "SLG", "OPS", "WPA", "RE24"],
                        pct_cols=["K%", "BB%"],
                    ),
                    height=190,
                )
        with pr_col:
            show_pr_table(build_pitcher_pr_table(row, pitchers_agg), "投球 PR 表", align_with_metric_row=True)
            st.caption("投手 PR 中 ERA、WHIP、BB%、BB/9、H/9、HR/9、NP/IP 等是越低越好，已自動反向計算。")

        st.markdown("### 逐場投球")
        per_game_p = add_pitcher_rate_stats(fpitchers_game[fpitchers_game["playerName"] == player].copy())
        show_dataframe(
            clean_display_df(
                per_game_p.sort_values("G"),
                cols=["G", "date", "stadium", "team", "opponentTeam", "order", "IP顯示", "NP", "BF", "H", "HR", "BB", "HB", "SO", "R", "ER", "ERA", "WHIP"],
                rate_cols=["ERA", "WHIP"],
            )
        )

    player_pa_as_batter = fpa[fpa["batterName"] == player].copy() if not fpa.empty else pd.DataFrame()
    player_pa_as_pitcher = fpa[fpa["pitcherName"] == player].copy() if not fpa.empty else pd.DataFrame()

    tabs = st.tabs(["打席明細", "對戰投手/打者", "擊球落點", "投球分布", "跑壘事件"])

    with tabs[0]:
        if player_pa_as_batter.empty:
            st.info("這個球員在篩選範圍內沒有打席資料。")
        else:
            show_dataframe(
                clean_display_df(
                    player_pa_as_batter.sort_values(["G", "paOrderInGame"]),
                    cols=["G", "inning", "scoreBefore", "outs", "basesLabel", "count", "batterHand", "pitcherName", "pitcherHand", "pitchCodes", "pitchCount", "result", "resultZh", "RBI", "locationCode", "trajectoryZh", "hardnessZh", "scoreAfter", "WPA", "RE24"],
                    rate_cols=["WPA", "RE24"],
                ),
                height=420,
            )

    with tabs[1]:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 作為打者面對各投手")
            if player_pa_as_batter.empty:
                st.info("沒有打者對戰資料。")
            else:
                m1 = aggregate_matchups(player_pa_as_batter)
                show_dataframe(
                    clean_display_df(
                        m1,
                        cols=["pitcherName", "pitcherHand", "defenseTeam", "PA", "AB", "H", "2B", "3B", "HR", "BB", "SO", "RBI", "AVG", "OBP", "SLG", "OPS", "WPA", "RE24"],
                        rate_cols=["AVG", "OBP", "SLG", "OPS", "WPA", "RE24"],
                    )
                )
        with col_b:
            st.markdown("#### 作為投手面對各打者")
            if player_pa_as_pitcher.empty:
                st.info("沒有投手對戰資料。")
            else:
                m2 = aggregate_matchups(player_pa_as_pitcher)
                show_dataframe(
                    clean_display_df(
                        m2,
                        cols=["batterName", "batterHand", "offenseTeam", "PA", "AB", "H", "2B", "3B", "HR", "BB", "SO", "RBI", "AVG", "OBP", "SLG", "OPS", "WPA", "RE24"],
                        rate_cols=["AVG", "OBP", "SLG", "OPS", "WPA", "RE24"],
                    )
                )

    with tabs[2]:
        fig = plot_batted_ball_map(player_pa_as_batter[player_pa_as_batter["isBattedBall"]], f"{player} 擊球落點")
        if fig:
            show_plot(fig)
        else:
            st.info("沒有可畫的擊球落點。")

    with tabs[3]:
        p_events = fevents[(fevents["type"] == "PITCH") & (fevents["pitcherName"] == player)].copy()
        if p_events.empty:
            st.info("這個球員沒有投球事件資料。")
        else:
            col1, col2 = st.columns(2)
            with col1:
                ps = pitch_summary(p_events)
                show_dataframe(
                    clean_display_df(
                        ps,
                        cols=["pitchType", "pitchTypeZh", "pitches", "strikes", "balls", "whiffs", "inPlay", "avgVelo", "maxVelo", "Strike%", "Whiff%", "InPlay%"],
                        rate_cols=["avgVelo", "maxVelo"],
                        pct_cols=["Strike%", "Whiff%", "InPlay%"],
                    )
                )
            with col2:
                fig = plot_velocity_histogram(p_events, f"{player} 球速分布")
                if fig:
                    show_plot(fig)
            fig = plot_pitch_locations(p_events, f"{player} 進壘點")
            if fig:
                show_plot(fig)

    with tabs[4]:
        pr = frunners[(frunners["runnerName"] == player) | (frunners["batterName"] == player)].copy()
        if pr.empty:
            st.info("沒有跑壘資料。")
        else:
            show_dataframe(
                clean_display_df(
                    pr.sort_values(["G", "paId", "eventOrder"]),
                    cols=["G", "inning", "offenseTeam", "batterName", "paResult", "runnerTypeZh", "runnerName", "isOut", "scored", "isRBI", "isER", "ERPitcherName", "WPA", "RE24"],
                    rate_cols=["WPA", "RE24"],
                ),
                height=420,
            )


# =========================
# Page 3: 打者排行榜
# =========================

elif page == "打者排行榜":
    st.subheader("打者排行榜")
    if batters_agg.empty:
        st.warning("沒有打者資料。")
        st.stop()

    min_pa = st.slider("最低 PA", 0, int(max(1, batters_agg["PA"].max())), min(10, int(batters_agg["PA"].max())))
    sort_col = st.selectbox("排序指標", ["OPS", "AVG", "OBP", "SLG", "H", "HR", "RBI", "BB%", "K%", "PA", "TB", "XBH"], index=0)
    ascending = sort_col in ["K%"]
    rank_df = batters_agg[batters_agg["PA"] >= min_pa].sort_values(sort_col, ascending=ascending)

    st.markdown("### 排行榜")
    show_dataframe(
        clean_display_df(
            rank_df,
            cols=["team", "playerNumber", "playerName", "PA", "AB", "R", "H", "2B", "3B", "HR", "RBI", "BB", "HBP", "SO", "SB", "CS", "TB", "XBH", "AVG", "OBP", "SLG", "OPS", "ISO", "BB%", "K%", "BB/K"],
            rate_cols=["AVG", "OBP", "SLG", "OPS", "ISO", "BB/K"],
            pct_cols=["BB%", "K%"],
        ),
        height=520,
    )

    st.markdown("### 圖表")
    top_n = st.slider("圖表顯示前 N 名", 1, min(30, len(rank_df)), min(10, len(rank_df))) if len(rank_df) > 0 else 0
    if top_n:
        fig = plot_rank_bar_by_team(
            rank_df.head(top_n),
            label_col="playerName",
            value_col=sort_col,
            team_col="team",
            title=f"打者排行榜：{sort_col}",
            value_label=sort_col,
            top_n=top_n,
        )
        if fig:
            show_plot(fig)

    st.markdown("### 打者 WPA / RE24 來自 PA 明細")
    if not fpa.empty:
        clutch_bat = fpa.groupby(["batterName", "offenseTeam"], dropna=False).agg(
            PA=("paId", "count"),
            WPA=("WPA", "sum"),
            RE24=("RE24", "sum"),
            highLevPA=("isHighLeverage", "sum"),
            RISP_PA=("isRISP", "sum"),
            RISP_RBI=("RBI", lambda s: s[fpa.loc[s.index, "isRISP"]].sum() if len(s) else 0),
        ).reset_index().rename(columns={"batterName": "playerName", "offenseTeam": "team"})
        show_dataframe(
            clean_display_df(
                clutch_bat.sort_values("WPA", ascending=False),
                cols=["team", "playerName", "PA", "WPA", "RE24", "highLevPA", "RISP_PA", "RISP_RBI"],
                rate_cols=["WPA", "RE24"],
            )
        )


# =========================
# Page 4: 投手排行榜
# =========================

elif page == "投手排行榜":
    st.subheader("投手排行榜")
    if pitchers_agg.empty:
        st.warning("沒有投手資料。")
        st.stop()

    max_outs = int(max(1, pitchers_agg["IPOuts"].max()))
    min_outs = st.slider("最低出局數 IPOuts", 0, max_outs, min(3, max_outs))
    sort_col = st.selectbox("排序指標", ["ERA", "WHIP", "K%", "K-BB%", "SO", "IPOuts", "NP/IP", "NP/BF", "BB%", "H/9"], index=0)
    ascending = sort_col in ["ERA", "WHIP", "NP/IP", "NP/BF", "BB%", "H/9"]
    rank_df = pitchers_agg[pitchers_agg["IPOuts"] >= min_outs].sort_values(sort_col, ascending=ascending)

    show_dataframe(
        clean_display_df(
            rank_df,
            cols=["team", "playerNumber", "playerName", "IP顯示", "IPOuts", "NP", "BF", "H", "HR", "BB", "IBB", "HB", "SO", "R", "ER", "ERA", "WHIP", "K%", "BB%", "K-BB%", "K/9", "BB/9", "H/9", "HR/9", "NP/IP", "NP/BF"],
            rate_cols=["ERA", "WHIP", "K/9", "BB/9", "H/9", "HR/9", "NP/IP", "NP/BF"],
            pct_cols=["K%", "BB%", "K-BB%"],
        ),
        height=520,
    )

    st.markdown("### 圖表")
    top_n = st.slider("圖表顯示前 N 名", 1, min(30, len(rank_df)), min(10, len(rank_df))) if len(rank_df) > 0 else 0
    if top_n:
        fig = plot_rank_bar_by_team(
            rank_df.head(top_n),
            label_col="playerName",
            value_col=sort_col,
            team_col="team",
            title=f"投手排行榜：{sort_col}",
            value_label=sort_col,
            top_n=top_n,
        )
        if fig:
            show_plot(fig)

    st.markdown("### 投手球種摘要")
    p_events = fevents[fevents["type"] == "PITCH"].copy() if not fevents.empty else pd.DataFrame()
    if p_events.empty:
        st.info("沒有投球事件資料。")
    else:
        ps = pitch_summary(p_events)
        show_dataframe(
            clean_display_df(
                ps,
                cols=["pitcherName", "pitchType", "pitchTypeZh", "pitches", "strikes", "balls", "whiffs", "fouls", "inPlay", "avgVelo", "maxVelo", "Strike%", "Ball%", "Whiff%", "InPlay%"],
                rate_cols=["avgVelo", "maxVelo"],
                pct_cols=["Strike%", "Ball%", "Whiff%", "InPlay%"],
            ),
            height=420,
        )


# =========================
# Page 5: 對戰組合
# =========================

elif page == "對戰組合":
    st.subheader("對戰組合：Batter vs Pitcher")
    if fpa.empty:
        st.warning("沒有 PA 資料。")
        st.stop()

    mode = st.selectbox(
        "顯示模式",
        ["指定打者 vs 指定投手", "此打者面對所有投手", "此投手面對所有打者"],
        key="matchup_mode",
    )

    def build_matchup_candidate_df(role: str) -> pd.DataFrame:
        """建立對戰頁面用的候選球員表，支援球隊、守位、姓名/背號篩選。"""
        if role == "batter":
            base = fpa[["offenseTeam", "batterName", "batterHand"]].drop_duplicates().copy()
            base = base.rename(columns={"offenseTeam": "team", "batterName": "playerName", "batterHand": "hand"})
            default_position = "野手/打者"
        else:
            base = fpa[["defenseTeam", "pitcherName", "pitcherHand"]].drop_duplicates().copy()
            base = base.rename(columns={"defenseTeam": "team", "pitcherName": "playerName", "pitcherHand": "hand"})
            default_position = "投手"

        base["team"] = base["team"].astype(str).str.strip()
        base["playerName"] = base["playerName"].astype(str).str.strip()
        base["hand"] = base["hand"].astype(str).str.strip()
        base = base[base["playerName"] != ""].copy()

        meta_cols = ["team", "playerName", "playerNumber", "position", "roles", "positionSource"]
        if "player_index_df" in globals() and not player_index_df.empty:
            meta = player_index_df[[c for c in meta_cols if c in player_index_df.columns]].drop_duplicates(subset=["team", "playerName"]).copy()
            base = base.merge(meta, on=["team", "playerName"], how="left")

        if "playerNumber" not in base.columns:
            base["playerNumber"] = ""
        if "position" not in base.columns:
            base["position"] = default_position
        if "roles" not in base.columns:
            base["roles"] = "打者" if role == "batter" else "投手"

        base["playerNumber"] = base["playerNumber"].fillna("").astype(str).str.strip()
        base["position"] = base["position"].fillna(default_position).astype(str).str.strip()
        base.loc[base["position"] == "", "position"] = default_position
        base["roles"] = base["roles"].fillna("").astype(str).str.strip()

        base["searchKey"] = (
            base["team"].astype(str) + " " +
            base["playerName"].astype(str) + " " +
            base["playerNumber"].astype(str) + " " +
            base["position"].astype(str) + " " +
            base["hand"].astype(str)
        ).apply(normalize_player_text)

        return base.sort_values(["team", "position", "playerNumber", "playerName"]).reset_index(drop=True)

    def filter_candidate_df(df: pd.DataFrame, teams: Sequence[str], positions: Sequence[str], keyword: str) -> pd.DataFrame:
        out = df.copy()
        if teams:
            out = out[out["team"].isin(list(teams))]
        if positions:
            out = out[out["position"].isin(list(positions))]
        keyword = str(keyword or "").strip()
        if keyword:
            terms = [normalize_player_text(t) for t in re.split(r"[,，\s]+", keyword) if normalize_player_text(t)]
            for term in terms:
                out = out[out["searchKey"].str.contains(re.escape(term), na=False)]
        return out.reset_index(drop=True)

    def make_player_label(row: pd.Series, role_name: str) -> str:
        number = str(row.get("playerNumber", "") or "").strip()
        number_part = f"#{number}｜" if number else ""
        hand = str(row.get("hand", "") or "").strip()
        hand_part = f"｜{hand}{'打' if role_name == '打者' else '投'}" if hand else ""
        return f"{row.get('playerName', '')}｜{row.get('team', '')}｜{number_part}{row.get('position', '')}{hand_part}"

    batter_candidates_all = build_matchup_candidate_df("batter")
    pitcher_candidates_all = build_matchup_candidate_df("pitcher")

    st.markdown("#### 對戰篩選")
    st.caption("這裡的篩選只影響本頁對戰組合；改成表單後，點很多篩選條件也不會每一下都重跑。")

    with st.form("matchup_filter_form"):
        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:
            st.markdown("##### 打者")
            batter_team_options = sorted(batter_candidates_all["team"].dropna().astype(str).unique().tolist())
            batter_teams = st.multiselect(
                "打者所屬球隊（不選＝全部）",
                batter_team_options,
                default=[],
                key="matchup_batter_teams",
            )
            batter_after_team = batter_candidates_all[batter_candidates_all["team"].isin(batter_teams)] if batter_teams else batter_candidates_all
            batter_position_options = sorted(batter_after_team["position"].dropna().astype(str).unique().tolist())
            batter_positions = st.multiselect(
                "打者守位（不選＝全部）",
                batter_position_options,
                default=[],
                key="matchup_batter_positions",
            )
            batter_keyword = st.text_input(
                "打者名字 / 背號",
                placeholder="例：陳傑憲、24、宋、20",
                key="matchup_batter_keyword",
            )

        with filter_col2:
            st.markdown("##### 投手")
            pitcher_team_options = sorted(pitcher_candidates_all["team"].dropna().astype(str).unique().tolist())
            pitcher_teams = st.multiselect(
                "投手所屬球隊（不選＝全部）",
                pitcher_team_options,
                default=[],
                key="matchup_pitcher_teams",
            )
            pitcher_after_team = pitcher_candidates_all[pitcher_candidates_all["team"].isin(pitcher_teams)] if pitcher_teams else pitcher_candidates_all
            pitcher_position_options = sorted(pitcher_after_team["position"].dropna().astype(str).unique().tolist())
            pitcher_positions = st.multiselect(
                "投手守位（不選＝全部）",
                pitcher_position_options,
                default=[],
                key="matchup_pitcher_positions",
            )
            pitcher_keyword = st.text_input(
                "投手名字 / 背號",
                placeholder="例：德保拉、46、伍、18",
                key="matchup_pitcher_keyword",
            )

        st.form_submit_button("套用對戰篩選", use_container_width=True)

    batter_candidates = filter_candidate_df(batter_candidates_all, batter_teams, batter_positions, batter_keyword)
    pitcher_candidates = filter_candidate_df(pitcher_candidates_all, pitcher_teams, pitcher_positions, pitcher_keyword)

    if batter_candidates.empty:
        st.warning("找不到符合條件的打者。")
        st.stop()
    if pitcher_candidates.empty:
        st.warning("找不到符合條件的投手。")
        st.stop()

    select_col1, select_col2 = st.columns(2)

    with select_col1:
        batter_candidates = batter_candidates.copy()
        batter_candidates["label"] = batter_candidates.apply(lambda r: make_player_label(r, "打者"), axis=1)
        batter_label_to_name = dict(zip(batter_candidates["label"], batter_candidates["playerName"]))
        batter_label = st.selectbox("選擇打者", batter_candidates["label"].tolist(), key="matchup_batter_select")
        batter = batter_label_to_name[batter_label]

    with select_col2:
        pitcher_select_pool = pitcher_candidates.copy()
        if mode == "指定打者 vs 指定投手":
            faced_pitchers = set(fpa[fpa["batterName"] == batter]["pitcherName"].dropna().astype(str).tolist())
            narrowed = pitcher_select_pool[pitcher_select_pool["playerName"].astype(str).isin(faced_pitchers)].copy()
            if not narrowed.empty:
                pitcher_select_pool = narrowed
                st.caption("投手清單已自動縮小為此打者實際對戰過的投手。")
            else:
                st.caption("目前投手篩選沒有此打者實際對戰過的人，暫時顯示所有符合投手。")

        pitcher_select_pool["label"] = pitcher_select_pool.apply(lambda r: make_player_label(r, "投手"), axis=1)
        pitcher_label_to_name = dict(zip(pitcher_select_pool["label"], pitcher_select_pool["playerName"]))
        pitcher_label = st.selectbox("選擇投手", pitcher_select_pool["label"].tolist(), key="matchup_pitcher_select")
        pitcher = pitcher_label_to_name[pitcher_label]

    if mode == "指定打者 vs 指定投手":
        matchup_pa = fpa[(fpa["batterName"] == batter) & (fpa["pitcherName"] == pitcher)].copy()
        title = f"{batter} vs {pitcher}"
    elif mode == "此打者面對所有投手":
        valid_pitchers = set(pitcher_candidates["playerName"].astype(str).tolist())
        matchup_pa = fpa[(fpa["batterName"] == batter) & (fpa["pitcherName"].astype(str).isin(valid_pitchers))].copy()
        title = f"{batter} 面對篩選投手"
    else:
        valid_batters = set(batter_candidates["playerName"].astype(str).tolist())
        matchup_pa = fpa[(fpa["pitcherName"] == pitcher) & (fpa["batterName"].astype(str).isin(valid_batters))].copy()
        title = f"{pitcher} 面對篩選打者"

    st.markdown(f"### {title}")
    if matchup_pa.empty:
        st.info("這個篩選沒有對戰資料。")
        st.stop()

    m_agg = aggregate_matchups(matchup_pa)
    pa_n = len(matchup_pa)
    ab_n = int(matchup_pa["AB_flag"].sum())
    h_n = int(matchup_pa["H_flag"].sum())
    bb_n = int(matchup_pa["BB_flag"].sum())
    so_n = int(matchup_pa["SO_flag"].sum())
    tb_n = int(matchup_pa["TB"].sum())
    avg = div0(h_n, ab_n)
    obp = div0(h_n + bb_n + int(matchup_pa["HBP_flag"].sum()), ab_n + bb_n + int(matchup_pa["HBP_flag"].sum()) + int(matchup_pa["SF_flag"].sum()))
    slg = div0(tb_n, ab_n)
    ops = obp + slg if not pd.isna(obp) and not pd.isna(slg) else np.nan

    metric_row(
        [
            ("PA", pa_n, None),
            ("H / AB", f"{h_n}/{ab_n}", None),
            ("AVG", fmt_batting_rate(avg), None),
            ("OBP", fmt_batting_rate(obp), None),
            ("SLG", fmt_batting_rate(slg), None),
            ("OPS", fmt_batting_rate(ops), None),
        ]
    )

    st.markdown("### 對戰彙總")
    show_dataframe(
        clean_display_df(
            m_agg,
            cols=["batterName", "batterHand", "offenseTeam", "pitcherName", "pitcherHand", "defenseTeam", "PA", "AB", "H", "2B", "3B", "HR", "BB", "HBP", "SO", "RBI", "pitches", "AVG", "OBP", "SLG", "OPS", "WPA", "RE24"],
            rate_cols=["AVG", "OBP", "SLG", "OPS", "WPA", "RE24"],
        )
    )

    st.markdown("### 打席明細")
    show_dataframe(
        clean_display_df(
            matchup_pa.sort_values(["G", "paOrderInGame"]),
            cols=["G", "inning", "scoreBefore", "outs", "basesLabel", "count", "batterName", "batterHand", "pitcherName", "pitcherHand", "catcherName", "pitchCodes", "pitchCount", "result", "resultZh", "RBI", "locationCode", "trajectoryZh", "hardnessZh", "scoreAfter", "WPA", "RE24"],
            rate_cols=["WPA", "RE24"],
        ),
        height=360,
    )

    st.markdown("### 每球資料")
    matchup_events = fevents[fevents["paId"].isin(matchup_pa["paId"])].copy()
    pitch_only = matchup_events[matchup_events["type"] == "PITCH"].copy()
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if pitch_only.empty:
            st.info("沒有投球事件。")
        else:
            pc = value_counts_table(pitch_only, "pitchType", "球種", "顆數")
            pc["球種中文"] = pc["球種"].map(PITCH_TYPE_ZH).fillna(pc["球種"])
            show_dataframe(clean_display_df(pc[["球種", "球種中文", "顆數", "比例"]], pct_cols=["比例"]))
    with col_b:
        if pitch_only.empty:
            st.info("沒有可畫球速。")
        else:
            fig = plot_velocity_histogram(pitch_only, f"{title} 球速分布")
            if fig:
                show_plot(fig)

    fig = plot_pitch_locations(pitch_only, f"{title} 進壘點")
    if fig:
        show_plot(fig)

    show_dataframe(
        clean_display_df(
            matchup_events.sort_values(["G", "paOrderInGame", "eventOrder"]),
            cols=["G", "inning", "paOrderInGame", "eventOrder", "pitchNumber", "type", "batterName", "pitcherName", "pitchCode", "pitchCodeZh", "pitchType", "pitchTypeZh", "velocity", "velocityRaw", "invalidVelocity", "coordX", "coordY", "coordXRaw", "coordYRaw", "invalidCoord", "inPlay", "paResult", "locationCode", "trajectory", "hardness"],
            rate_cols=["velocity", "coordX", "coordY"],
        ),
        height=420,
    )



# =========================
# Page 6: 投球分析
# =========================

elif page == "投球分析":
    st.subheader("投球分析")
    pitch_events = fevents[fevents["type"] == "PITCH"].copy() if not fevents.empty else pd.DataFrame()
    if pitch_events.empty:
        st.warning("沒有投球事件資料。")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        pitcher_sel = st.multiselect("投手", sorted(pitch_events["pitcherName"].dropna().unique().tolist()), default=[])
    with col2:
        pitch_type_sel = st.multiselect("球種", sorted(pitch_events["pitchType"].dropna().unique().tolist()), default=[])
    with col3:
        pitch_code_sel = st.multiselect("投球結果", sorted(pitch_events["pitchCode"].dropna().unique().tolist()), default=[])

    pf = pitch_events.copy()
    if pitcher_sel:
        pf = pf[pf["pitcherName"].isin(pitcher_sel)]
    if pitch_type_sel:
        pf = pf[pf["pitchType"].isin(pitch_type_sel)]
    if pitch_code_sel:
        pf = pf[pf["pitchCode"].isin(pitch_code_sel)]

    total = len(pf)
    strikes = int(pf["isStrike"].sum()) if total else 0
    balls = int(pf["isBall"].sum()) if total else 0
    whiffs = int((pf["pitchCode"] == "SW").sum()) if total else 0
    inplay = int(pf["inPlay"].sum()) if total else 0
    avg_velo = pf["velocity"].mean() if total else np.nan
    max_velo = pf["velocity"].max() if total else np.nan
    metric_row(
        [
            ("投球數", total, None),
            ("Strike%", fmt_pct(div0(strikes, total)), None),
            ("Ball%", fmt_pct(div0(balls, total)), None),
            ("Whiff%", fmt_pct(div0(whiffs, total)), None),
            ("平均球速", fmt_num(avg_velo, 1), None),
            ("最快球速", fmt_num(max_velo, 1), None),
        ]
    )

    tab1, tab2, tab3, tab4 = st.tabs(["球種", "結果", "進壘點", "原始投球表"])
    with tab1:
        ps = pitch_summary(pf)
        show_dataframe(
            clean_display_df(
                ps,
                cols=["pitcherName", "pitchType", "pitchTypeZh", "pitches", "strikes", "balls", "whiffs", "fouls", "inPlay", "avgVelo", "maxVelo", "Strike%", "Ball%", "Whiff%", "InPlay%"],
                rate_cols=["avgVelo", "maxVelo"],
                pct_cols=["Strike%", "Ball%", "Whiff%", "InPlay%"],
            ),
            height=520,
        )
        if not ps.empty:
            st.bar_chart(ps.groupby("pitchType")["pitches"].sum().sort_values(ascending=False))

    with tab2:
        pc = value_counts_table(pf, "pitchCode", "投球結果", "顆數")
        pc["中文"] = pc["投球結果"].map(PITCH_CODE_ZH).fillna(pc["投球結果"])
        show_dataframe(clean_display_df(pc[["投球結果", "中文", "顆數", "比例"]], pct_cols=["比例"]))
        st.bar_chart(pc.set_index("投球結果")[["顆數"]])

    with tab3:
        col_a, col_b = st.columns([1, 1])
        with col_a:
            fig = plot_pitch_locations(pf, "投球進壘點")
            if fig:
                show_plot(fig)
            else:
                st.info("沒有 coordX / coordY 可畫。")
        with col_b:
            fig = plot_velocity_histogram(pf, "球速分布")
            if fig:
                show_plot(fig)
            else:
                st.info("沒有球速可畫。")

    with tab4:
        show_dataframe(
            clean_display_df(
                pf.sort_values(["G", "paOrderInGame", "eventOrder"]),
                cols=["G", "inning", "offenseTeam", "defenseTeam", "batterName", "batterHand", "pitcherName", "pitcherHand", "catcherName", "pitchNumber", "pitchCode", "pitchCodeZh", "pitchType", "pitchTypeZh", "velocity", "coordX", "coordY", "inPlay", "paResult", "paResultZh", "locationCode", "trajectory", "hardness", "WPA", "RE24"],
                rate_cols=["velocity", "coordX", "coordY", "WPA", "RE24"],
            ),
            height=560,
        )


# =========================
# Page 7: 擊球落點
# =========================

elif page == "擊球落點":
    st.subheader("擊球落點與擊球品質")
    batted = fpa[fpa["isBattedBall"]].copy() if not fpa.empty else pd.DataFrame()
    if batted.empty:
        st.warning("沒有擊球落點資料。")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        batter_sel = st.multiselect("打者", sorted(batted["batterName"].dropna().unique().tolist()), default=[])
    with col2:
        pitcher_sel = st.multiselect("投手", sorted(batted["pitcherName"].dropna().unique().tolist()), default=[])
    with col3:
        traj_sel = st.multiselect("彈道", sorted(batted["trajectory"].dropna().unique().tolist()), default=[])
    with col4:
        hard_sel = st.multiselect("強度", sorted(batted["hardness"].dropna().unique().tolist()), default=[])

    bf = batted.copy()
    if batter_sel:
        bf = bf[bf["batterName"].isin(batter_sel)]
    if pitcher_sel:
        bf = bf[bf["pitcherName"].isin(pitcher_sel)]
    if traj_sel:
        bf = bf[bf["trajectory"].isin(traj_sel)]
    if hard_sel:
        bf = bf[bf["hardness"].isin(hard_sel)]

    metric_row(
        [
            ("擊球 PA", len(bf), None),
            ("安打", int(bf["H_flag"].sum()), None),
            ("全壘打", int(bf["HR_flag"].sum()), None),
            ("強勁擊球", int((bf["hardness"] == "H").sum()), None),
            ("滾地球", int((bf["trajectory"] == "G").sum()), None),
            ("飛球/平飛", int(bf["trajectory"].isin(["F", "L"]).sum()), None),
        ]
    )

    tab1, tab2, tab3 = st.tabs(["落點圖", "分布表", "明細"])
    with tab1:
        fig = plot_batted_ball_map(bf, "擊球落點圖")
        if fig:
            show_plot(fig)
        else:
            st.info("目前篩選沒有可畫的落點。")

    with tab2:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("#### 落點代碼")
            loc = value_counts_table(bf, "locationCode", "落點", "次數")
            show_dataframe(clean_display_df(loc, pct_cols=["比例"]))
        with col_b:
            st.markdown("#### 彈道")
            traj = value_counts_table(bf, "trajectory", "彈道", "次數")
            traj["中文"] = traj["彈道"].map(TRAJECTORY_ZH).fillna(traj["彈道"])
            show_dataframe(clean_display_df(traj[["彈道", "中文", "次數", "比例"]], pct_cols=["比例"]))
        with col_c:
            st.markdown("#### 強度")
            hard = value_counts_table(bf, "hardness", "強度", "次數")
            hard["中文"] = hard["強度"].map(HARDNESS_ZH).fillna(hard["強度"])
            show_dataframe(clean_display_df(hard[["強度", "中文", "次數", "比例"]], pct_cols=["比例"]))

    with tab3:
        show_dataframe(
            clean_display_df(
                bf.sort_values(["G", "paOrderInGame"]),
                cols=["G", "inning", "scoreBefore", "outs", "basesLabel", "batterName", "batterHand", "pitcherName", "pitcherHand", "result", "resultZh", "RBI", "locationCode", "trajectory", "trajectoryZh", "hardness", "hardnessZh", "pitchCodes", "WPA", "RE24"],
                rate_cols=["WPA", "RE24"],
            ),
            height=560,
        )


# =========================
# Page 8: 關鍵時刻 WPA/RE24
# =========================

elif page == "關鍵時刻 WPA/RE24":
    st.subheader("關鍵時刻：WPA / RE24")
    if fpa.empty:
        st.warning("沒有 PA 資料。")
        st.stop()

    metric_row(
        [
            ("高槓桿 PA |WPA| ≥ .050", int(fpa["isHighLeverage"].sum()), None),
            ("得點圈 PA", int(fpa["isRISP"].sum()), None),
            ("七局後 PA", int(fpa["isLate"].sum()), None),
            ("三分差內 PA", int(fpa["isClose"].sum()), None),
        ]
    )

    tab1, tab2, tab3, tab4 = st.tabs(["WPA 排行", "RE24 排行", "得點圈", "逐 PA 時間軸"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 對進攻方最有利")
            top = fpa.sort_values("WPA", ascending=False).head(20)
            show_dataframe(
                clean_display_df(
                    top,
                    cols=["G", "inning", "scoreBefore", "outs", "basesLabel", "offenseTeam", "batterName", "pitcherName", "result", "resultZh", "RBI", "scoreAfter", "WPA", "RE24", "pitchCodes", "locationCode"],
                    rate_cols=["WPA", "RE24"],
                ),
                height=520,
            )
        with col2:
            st.markdown("#### 對進攻方最傷")
            low = fpa.sort_values("WPA", ascending=True).head(20)
            show_dataframe(
                clean_display_df(
                    low,
                    cols=["G", "inning", "scoreBefore", "outs", "basesLabel", "offenseTeam", "batterName", "pitcherName", "result", "resultZh", "RBI", "scoreAfter", "WPA", "RE24", "pitchCodes", "locationCode"],
                    rate_cols=["WPA", "RE24"],
                ),
                height=520,
            )

    with tab2:
        st.markdown("#### RE24 前 30")
        top_re = fpa.sort_values("RE24", ascending=False).head(30)
        show_dataframe(
            clean_display_df(
                top_re,
                cols=["G", "inning", "scoreBefore", "outs", "basesLabel", "offenseTeam", "batterName", "pitcherName", "result", "resultZh", "RBI", "scoreAfter", "RE", "RE24", "WPA", "pitchCodes", "locationCode"],
                rate_cols=["RE", "RE24", "WPA"],
            ),
            height=560,
        )

    with tab3:
        risp = fpa[fpa["isRISP"]].copy()
        if risp.empty:
            st.info("沒有得點圈 PA。")
        else:
            st.markdown("#### 得點圈打者表現")
            risp_hit = risp.groupby(["batterName", "offenseTeam"], dropna=False).agg(
                PA=("paId", "count"),
                AB=("AB_flag", "sum"),
                H=("H_flag", "sum"),
                RBI=("RBI", "sum"),
                BB=("BB_flag", "sum"),
                SO=("SO_flag", "sum"),
                TB=("TB", "sum"),
                WPA=("WPA", "sum"),
                RE24=("RE24", "sum"),
            ).reset_index().rename(columns={"batterName": "playerName", "offenseTeam": "team"})
            risp_hit["AVG"] = risp_hit.apply(lambda r: div0(r["H"], r["AB"]), axis=1)
            risp_hit["SLG"] = risp_hit.apply(lambda r: div0(r["TB"], r["AB"]), axis=1)
            show_dataframe(
                clean_display_df(
                    risp_hit.sort_values(["RBI", "WPA"], ascending=[False, False]),
                    cols=["team", "playerName", "PA", "AB", "H", "RBI", "BB", "SO", "AVG", "SLG", "WPA", "RE24"],
                    rate_cols=["AVG", "SLG", "WPA", "RE24"],
                ),
                height=360,
            )

            st.markdown("#### 得點圈 PA 明細")
            show_dataframe(
                clean_display_df(
                    risp.sort_values(["G", "paOrderInGame"]),
                    cols=["G", "inning", "scoreBefore", "outs", "basesLabel", "offenseTeam", "batterName", "pitcherName", "result", "resultZh", "RBI", "scoreAfter", "WPA", "RE24", "pitchCodes"],
                    rate_cols=["WPA", "RE24"],
                ),
                height=420,
            )

    with tab4:
        timeline = fpa.sort_values(["G", "paOrderInGame"]).copy()
        timeline["PA序號"] = range(1, len(timeline) + 1)
        for team in sorted(timeline["offenseTeam"].dropna().unique()):
            timeline[f"{team}_cumWPA"] = (timeline["WPA"] * (timeline["offenseTeam"] == team)).cumsum()
        wpa_cols = [c for c in timeline.columns if c.endswith("_cumWPA")]
        if wpa_cols:
            timeline_teams = [c.replace("_cumWPA", "") for c in wpa_cols]
            fig = plot_team_timeline(timeline, timeline_teams, "累積 WPA 時間軸")
            if fig:
                show_plot(fig)
        show_dataframe(
            clean_display_df(
                timeline,
                cols=["PA序號", "G", "inning", "scoreBefore", "outs", "basesLabel", "offenseTeam", "batterName", "pitcherName", "result", "resultZh", "scoreAfter", "WPA", "RE24"],
                rate_cols=["WPA", "RE24"],
            ),
            height=420,
        )


# =========================
# Page 9: 跑壘與責失
# =========================

elif page == "跑壘與責失":
    st.subheader("跑壘、得分、RBI、責失紀錄")
    if frunners.empty:
        st.warning("沒有 runner 資料。")
        st.stop()

    metric_row(
        [
            ("跑壘事件", len(frunners), None),
            ("跑者得分", int(frunners["scored"].sum()), None),
            ("跑者出局", int(frunners["isOut"].sum()), None),
            ("算 RBI 的得分", int(frunners["isRBI"].sum()), None),
            ("責失得分", int(frunners["isER"].sum()), None),
        ]
    )

    tab1, tab2, tab3, tab4 = st.tabs(["跑壘類型", "得分與 RBI", "責失歸屬", "明細"])

    with tab1:
        rt = value_counts_table(frunners, "runnerType", "跑壘類型", "次數")
        rt["中文"] = rt["跑壘類型"].map(RUNNER_TYPE_ZH).fillna(rt["跑壘類型"])
        show_dataframe(clean_display_df(rt[["跑壘類型", "中文", "次數", "比例"]], pct_cols=["比例"]))
        st.bar_chart(rt.set_index("中文")[["次數"]])

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 得分跑者")
            scored = frunners[frunners["scored"]].groupby(["runnerName", "offenseTeam"], dropna=False).size().reset_index(name="得分次數")
            show_dataframe(scored.sort_values("得分次數", ascending=False))
        with col2:
            st.markdown("#### RBI 來源")
            rbi = frunners[frunners["isRBI"]].groupby(["batterName", "offenseTeam"], dropna=False).size().reset_index(name="RBI runners")
            show_dataframe(rbi.sort_values("RBI runners", ascending=False))

    with tab3:
        er = frunners[frunners["isER"]].copy()
        if er.empty:
            st.info("沒有責失跑者資料。")
        else:
            er_sum = er.groupby(["ERPitcherName", "defenseTeam"], dropna=False).size().reset_index(name="責失跑者數")
            show_dataframe(er_sum.sort_values("責失跑者數", ascending=False))
            show_dataframe(
                clean_display_df(
                    er.sort_values(["G", "paId", "eventOrder"]),
                    cols=["G", "inning", "offenseTeam", "defenseTeam", "batterName", "paResult", "runnerTypeZh", "runnerName", "scored", "isRBI", "ERPitcherName", "WPA", "RE24"],
                    rate_cols=["WPA", "RE24"],
                ),
                height=420,
            )

    with tab4:
        show_dataframe(
            clean_display_df(
                frunners.sort_values(["G", "paId", "eventOrder", "runnerOrder"]),
                cols=["G", "inning", "offenseTeam", "defenseTeam", "batterName", "pitcherName", "paResult", "runnerType", "runnerTypeZh", "runnerName", "isOut", "scored", "isRBI", "isER", "ERPitcherName", "WPA", "RE24"],
                rate_cols=["WPA", "RE24"],
            ),
            height=620,
        )


# =========================
# Page 10: 資料品質檢查
# =========================

elif page == "資料品質檢查":
    st.subheader("資料品質檢查 / 整季資料檢查")
    st.caption("整季 360 場進來後，先看這頁可以快速確認資料有沒有讀錯、缺球速、缺座標或重複。")

    total_games = len(fgames)
    total_pa = len(fpa)
    total_events = len(fevents)
    pitch_events = fevents[fevents["type"] == "PITCH"].copy() if not fevents.empty and "type" in fevents.columns else pd.DataFrame()
    total_pitches = len(pitch_events)
    missing_velo = int(pitch_events["velocity"].isna().sum()) if not pitch_events.empty and "velocity" in pitch_events.columns else 0
    missing_coord = int(pitch_events[["coordX", "coordY"]].isna().any(axis=1).sum()) if not pitch_events.empty and {"coordX", "coordY"}.issubset(pitch_events.columns) else 0
    invalid_velo = int(pitch_events["invalidVelocity"].sum()) if not pitch_events.empty and "invalidVelocity" in pitch_events.columns else 0
    invalid_coord = int(pitch_events["invalidCoord"].sum()) if not pitch_events.empty and "invalidCoord" in pitch_events.columns else 0

    metric_row(
        [
            ("比賽數", total_games, None),
            ("PA", total_pa, None),
            ("event", total_events, None),
            ("投球事件", total_pitches, None),
            ("缺/異常球速", missing_velo, None),
            ("異常座標", invalid_coord, None),
        ]
    )

    st.markdown("### 球隊 / 比賽概況")
    if not fgames.empty:
        team_games = []
        for team in sorted(set(fgames["awayTeam"].dropna().tolist() + fgames["homeTeam"].dropna().tolist())):
            team_games.append({
                "team": team,
                "games": int(((fgames["awayTeam"] == team) | (fgames["homeTeam"] == team)).sum()),
                "homeGames": int((fgames["homeTeam"] == team).sum()),
                "awayGames": int((fgames["awayTeam"] == team).sum()),
            })
        show_dataframe(pd.DataFrame(team_games).sort_values("games", ascending=False))
    else:
        st.info("目前沒有比賽資料。")

    st.markdown("### 缺值比例")
    quality_rows = []
    if not pitch_events.empty:
        quality_rows.extend([
            {"欄位": "velocity", "總數": total_pitches, "缺值": missing_velo, "缺值比例": div0(missing_velo, total_pitches)},
            {"欄位": "invalidVelocity", "總數": total_pitches, "缺值": invalid_velo, "缺值比例": div0(invalid_velo, total_pitches)},
            {"欄位": "coordX/coordY", "總數": total_pitches, "缺值": missing_coord, "缺值比例": div0(missing_coord, total_pitches)},
            {"欄位": "invalidCoord", "總數": total_pitches, "缺值": invalid_coord, "缺值比例": div0(invalid_coord, total_pitches)},
        ])
    if not fpa.empty:
        missing_location = int((fpa["locationCode"].astype(str).str.strip() == "").sum()) if "locationCode" in fpa.columns else 0
        missing_wpa = int(fpa["WPA"].isna().sum()) if "WPA" in fpa.columns else 0
        quality_rows.extend([
            {"欄位": "locationCode", "總數": len(fpa), "缺值": missing_location, "缺值比例": div0(missing_location, len(fpa))},
            {"欄位": "WPA", "總數": len(fpa), "缺值": missing_wpa, "缺值比例": div0(missing_wpa, len(fpa))},
        ])
    if quality_rows:
        show_dataframe(clean_display_df(pd.DataFrame(quality_rows), pct_cols=["缺值比例"]))
    else:
        st.info("目前沒有足夠資料可檢查。")

    st.markdown("### 已清理的明顯異常資料")
    if pitch_events.empty:
        st.info("目前沒有投球事件可檢查。")
    else:
        anomalies = []
        if "invalidVelocity" in pitch_events.columns:
            bad_v = pitch_events[pitch_events["invalidVelocity"]].copy()
            if not bad_v.empty:
                tmp = bad_v[["G", "date", "defenseTeam", "pitcherName", "batterName", "pitchType", "pitchCode", "velocityRaw", "velocity"]].copy()
                tmp["異常類型"] = "球速異常"
                anomalies.append(tmp)
        if "invalidCoord" in pitch_events.columns:
            bad_c = pitch_events[pitch_events["invalidCoord"]].copy()
            if not bad_c.empty:
                tmp = bad_c[["G", "date", "defenseTeam", "pitcherName", "batterName", "pitchType", "pitchCode", "coordXRaw", "coordYRaw", "coordX", "coordY"]].copy()
                tmp["異常類型"] = "座標異常"
                anomalies.append(tmp.head(200))
        if anomalies:
            anomaly_df = pd.concat(anomalies, ignore_index=True, sort=False)
            show_dataframe(anomaly_df, height=360, max_rows=400)
        else:
            st.success("目前沒有偵測到明顯球速或座標異常。")

    st.markdown("### 可能重複的比賽")
    if not games_df.empty:
        dup_cols = ["seasonId", "G", "date", "awayTeam", "homeTeam"]
        dup = games_df[games_df.duplicated(subset=dup_cols, keep=False)] if set(dup_cols).issubset(games_df.columns) else pd.DataFrame()
        if dup.empty:
            st.success("目前沒有偵測到重複比賽。")
        else:
            show_dataframe(dup.sort_values(["G", "date"]), height=360)


# =========================
# Page 11: 原始資料表
# =========================

elif page == "原始資料表":
    st.subheader("原始資料表 / Debug 用")
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["game", "score", "batterBox", "pitcherBox", "PA", "event", "runner"])
    with tab1:
        show_dataframe(fgames, height=600)
    with tab2:
        show_dataframe(fscores, height=600)
    with tab3:
        show_dataframe(fbatters_game, height=600)
    with tab4:
        show_dataframe(fpitchers_game, height=600)
    with tab5:
        show_dataframe(fpa, height=600)
    with tab6:
        show_dataframe(fevents, height=600)
    with tab7:
        show_dataframe(frunners, height=600)


# =========================
# 頁尾
# =========================

st.caption(
    "提示：如果你之後換成 2024 整季資料，直接把整季 OpenData JSON 放進同資料夾或從左側上傳；這份 app.py 會自動讀取、攤平、去重並套用相同分析架構。"
)
