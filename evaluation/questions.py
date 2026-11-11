"""评测集：30 条问题，覆盖单跳 / 多跳 / 跨页对比 / 拒答四类。

语料是两份年报：
    1225251805.pdf                苏宁环球 2025 年年报（A股，简体，185 页）
    H2_AN202604091821090996_1.pdf 腾讯控股 2025 年年报（港股，繁体，282 页）

字段说明：
    expect_points  预期答案要点，外层是"要点组"，内层是该要点的多种写法。
                   命中组数 / 总组数 = 要点覆盖率，≥60% 视为要点达标。
                   同义词要覆盖数字的常见写法（带/不带千分位、换成"亿"）。
    expect_pages   预期来源页码，1-based，与 PDF 阅读器页码一致。
    expect_files   预期来源文件。多文档场景下页码会撞车（两份都有第 7 页），
                   因此判分必须按 (文件, 页码) 配对，不能只看页码。
    computed       答案含计算步骤（如“两者相差多少个百分点”），要点是算出来的值，
                   原文里当然搜不到。标记后 check_questions.py 不再校验要点是否出现在原文，
                   只校验页码。
    refuse         该题在文档中无答案，正确行为是明确说明找不到，而不是编造。

出题原则：
    - 答案必须能从原文直接读出或经一步计算得出，不含主观判断。
    - 拒绝题必须确认文档真的没有该信息（不是"藏得深"，而是根本没披露）。
    - 繁体文档的要点同时收录繁简两种写法，避免判分因字形差异误伤。
"""

SUNING = "1225251805.pdf"
TENCENT = "H2_AN202604091821090996_1.pdf"

QUESTIONS = [
    # ==================== 单跳（8 条）：一次检索即可定位到答案 ====================
    {
        "id": "S1",
        "category": "单跳",
        "question": "苏宁环球2025年实现的营业收入是多少元？同比增减多少？",
        "expect_points": [
            ["1,387,372,693", "1387372693", "13.87亿", "13.9亿", "138,737.27", "138737.27万"],
            ["34.59", "34.6%", "下降34.6", "减少34.6"],
        ],
        "expect_pages": [7],
        "expect_files": [SUNING],
        "refuse": False,
    },
    {
        "id": "S2",
        "category": "单跳",
        "question": "苏宁环球2025年归属于上市公司股东的净利润是多少？同比变化多少？",
        "expect_points": [
            ["18,595,226.85", "18595226", "1,859.52万", "1859.52万", "1,859万"],
            ["88.74", "88.7"],
        ],
        "expect_pages": [7],
        "expect_files": [SUNING],
        "refuse": False,
    },
    {
        "id": "S3",
        "category": "单跳",
        "question": "腾讯控股2025年全年的收入是多少？同比增长多少？",
        "expect_points": [
            ["751,766", "751766", "7,518亿", "7518亿", "7517.66", "7,517.66"],
            ["14%", "增长14", "14.0%", "增幅14"],
        ],
        "expect_pages": [4, 9],
        "expect_files": [TENCENT],
        "refuse": False,
    },
    {
        "id": "S4",
        "category": "单跳",
        "question": "腾讯控股2025年本公司权益持有人应占盈利是多少？同比增长多少？",
        "expect_points": [
            ["2,248亿", "2248亿", "224,842", "224842", "2,248.42"],
            ["16%", "增长16", "16.0%"],
        ],
        "expect_pages": [4, 11, 5],
        "expect_files": [TENCENT],
        "refuse": False,
    },
    {
        "id": "S5",
        "category": "单跳",
        "question": "苏宁环球2025年末的总资产是多少？比上年末增减多少？",
        "expect_points": [
            ["13,119,276,644.67", "13119276644", "131.19亿", "131.2亿", "131.1亿"],
            ["8.74", "8.7"],
        ],
        "expect_pages": [7],
        "expect_files": [SUNING],
        "refuse": False,
    },
    {
        "id": "S6",
        "category": "单跳",
        "question": "腾讯控股董事会建议就2025年度派发多少末期股息？",
        "expect_points": [
            ["5.30港元", "5.3港元", "5.30港幣", "5.3港幣", "每股5.3"],
        ],
        "expect_pages": [27],
        "expect_files": [TENCENT],
        "refuse": False,
    },
    {
        "id": "S7",
        "category": "单跳",
        "question": "苏宁环球的股票代码是多少？在哪家证券交易所上市？",
        "expect_points": [
            ["000718", "000,718"],
            ["深圳证券交易所", "深交所"],
        ],
        "expect_pages": [6],
        "expect_files": [SUNING],
        "refuse": False,
    },
    {
        "id": "S8",
        "category": "单跳",
        "question": "腾讯控股2025年年报的核数师是哪家机构？",
        "expect_points": [
            ["羅兵咸永道", "罗兵咸永道", "普华永道", "pricewaterhouse", "pwc"],
        ],
        "expect_pages": [3],
        "expect_files": [TENCENT],
        "refuse": False,
    },

    # ==================== 多跳（8 条）：需要 2 次以上取数或一步计算 ====================
    {
        "id": "M1",
        "category": "多跳",
        "question": "苏宁环球2025年四个季度中，哪个季度亏损金额最大？该季度营业收入是多少？",
        "expect_points": [
            ["第四季度", "四季度", "q4", "第4季度"],
            ["81,590,427.98", "81590427", "8,159.04万", "8159万", "8,159万"],
            ["225,476,511.49", "225476511", "22,547.65万", "22,547万", "2.25亿"],
        ],
        "expect_pages": [8],
        "expect_files": [SUNING],
        "refuse": False,
    },
    {
        "id": "M2",
        "category": "多跳",
        "question": "苏宁环球2025年房地产业务收入占营业收入的比重是多少？该业务收入同比变化多少？",
        "expect_points": [
            ["68.67", "68.7%", "68.7％"],
            ["42.43", "42.4"],
        ],
        "expect_pages": [18],
        "expect_files": [SUNING],
        "refuse": False,
    },
    {
        "id": "M3",
        "category": "多跳",
        "question": "腾讯控股2025年的毛利是多少？毛利率相比2024年提升多少个百分点？",
        "expect_points": [
            ["4,226亿", "4226亿", "422,593", "422593", "4,225.93"],
            ["3个百分点", "3pp", "3.0个百分点", "三个百分点"],
        ],
        "expect_pages": [8, 10],
        "expect_files": [TENCENT],
        "computed": True,
        "refuse": False,
    },
    {
        "id": "M4",
        "category": "多跳",
        "question": "腾讯控股2025年三大业务分部中，毛利率最高的是哪个分部？该分部毛利率是多少？",
        "expect_points": [
            ["增值服务", "增值服務"],
            ["60%", "60.0%"],
        ],
        "expect_pages": [10],
        "expect_files": [TENCENT],
        "refuse": False,
    },
    {
        "id": "M5",
        "category": "多跳",
        "question": "苏宁环球2025年经营活动产生的现金流量净额是多少？同比变化多少？",
        "expect_points": [
            ["223,948,311.25", "223948311", "22,394.83万", "2.24亿", "22,394万"],
            ["57.16", "57.2"],
        ],
        "expect_pages": [7, 21],
        "expect_files": [SUNING],
        "refuse": False,
    },
    {
        "id": "M6",
        "category": "多跳",
        "question": "腾讯控股2025年一般及行政开支增长的主要原因是什么？",
        "expect_points": [
            ["研發", "研发", "ai"],
            ["股份酬金", "40亿", "40億"],
        ],
        "expect_pages": [11],
        "expect_files": [TENCENT],
        "refuse": False,
    },
    {
        "id": "M7",
        "category": "多跳",
        "question": "苏宁环球2025年度是否派发现金红利？说明了什么理由？",
        "expect_points": [
            ["不派发", "不分配", "不派发现金", "不进行现金分红", "不派发现金股利", "不送红股"],
        ],
        "expect_pages": [2, 41],
        "expect_files": [SUNING],
        "refuse": False,
    },
    {
        "id": "M8",
        "category": "多跳",
        "question": "腾讯控股2025年的所得税开支是多少？同比增长多少？",
        "expect_points": [
            # 管理层讨论写“475 亿”，报表附注写“47,448 百万”，两者是同一数的不同精度写法
            ["475亿", "475億", "47,500", "474.48", "47,448", "47448"],
            # 不能只写 "5%"："25%" 包含 "5%"，会被误命中
            ["5.4%", "5.40%", "增长5%", "同比增长5"],
        ],
        # 11 是管理层讨论，204/205 是所得税附注；两条路都合法，不应只认一条
        "expect_pages": [11, 204, 205],
        "expect_files": [TENCENT],
        "computed": True,
        "refuse": False,
    },

    # ==================== 跨页对比（7 条）：跨页取数、跨文档对比或计算 ====================
    {
        "id": "C1",
        "category": "跨页对比",
        "question": "腾讯控股2025年收入相比2024年增加了多少亿元？",
        "expect_points": [
            ["915亿", "915.09", "91,509", "91509", "约915", "914.9"],
        ],
        "expect_pages": [4, 9],
        "expect_files": [TENCENT],
        "computed": True,
        "refuse": False,
    },
    {
        "id": "C2",
        "category": "跨页对比",
        "question": "苏宁环球2025年营业收入同比下降34.59%，腾讯控股同期增长14%，两者增速相差多少个百分点？",
        "expect_points": [
            ["48.59", "48.6", "48.5", "49个百分点", "约49个"],
        ],
        "expect_pages": [7, 9],
        "expect_files": [SUNING, TENCENT],
        "computed": True,
        "refuse": False,
    },
    {
        "id": "C3",
        "category": "跨页对比",
        "question": "苏宁环球2025年末总资产相比2024年末减少约多少亿元？",
        "expect_points": [
            ["12.56", "12.6亿", "12.5亿", "1,255,670,623", "1255670623", "约12.5"],
        ],
        # 7 是主要会计数据表，66/67 是合并资产负债表；后者是原始数据，算出的结果更可信
        "expect_pages": [7, 66, 67],
        "expect_files": [SUNING],
        "computed": True,
        "refuse": False,
    },
    {
        "id": "C4",
        "category": "跨页对比",
        "question": "2025年腾讯控股的营业收入大约是苏宁环球的多少倍？",
        "expect_points": [
            ["542", "541", "约542倍", "541.9", "540倍", "约540"],
        ],
        "expect_pages": [7, 9],
        "expect_files": [SUNING, TENCENT],
        "computed": True,
        "refuse": False,
    },
    {
        "id": "C5",
        "category": "跨页对比",
        "question": "两家公司2025年末的员工人数相差约多少倍？",
        # 腾讯年报给了两个员工口径，两个答案都对，只认一个会把更严谨的回答判成错：
        #   p81 集团口径      115,849 人（2024：110,558）  → 115,849 / 909 ≈ 127.4 倍
        #   p99 本公司正式员工 58,812 男 + 28,600 女 = 87,412 → 87,412 / 909 ≈ 96.2 倍
        # 注意 p99 原文没有“87,412”这个字面量，是两个数相加的结果，
        # 所以关键词里收 58,812 / 28,600 而不是 87,412（后者会让题库校验误报）。
        "expect_points": [
            ["127", "127.4", "约127倍", "115,849", "115849",
             "96.2", "96.1", "96倍", "约96倍", "58,812", "58812", "28,600", "28600"],
            ["909"],
        ],
        # 99 是企业管治报告里的雇员性别构成，正是第二个口径的来源
        "expect_pages": [39, 81, 99],
        "expect_files": [SUNING, TENCENT],
        "refuse": False,
    },
    {
        "id": "C6",
        "category": "跨页对比",
        "question": "腾讯控股2025年非国际财务报告准则下的本公司权益持有人应占盈利，比国际财务报告准则下的金额高多少亿元？",
        "expect_points": [
            ["347", "348", "34,784", "34784", "347.84", "347.8"],
        ],
        "expect_pages": [4, 11, 5],
        "expect_files": [TENCENT],
        "computed": True,
        "refuse": False,
    },
    {
        "id": "C7",
        "category": "跨页对比",
        "question": "苏宁环球2025年分地区收入中，哪个地区同比降幅最大？降幅是多少？",
        "expect_points": [
            ["吉林"],
            ["48.94", "48.9"],
        ],
        "expect_pages": [18],
        "expect_files": [SUNING],
        "refuse": False,
    },

    # ==================== 拒答（7 条）：文档确实没有，正确行为是说明找不到 ====================
    {
        "id": "R1",
        "category": "拒答",
        # 年报"研发投入"章节勾选的是"不适用"，等于披露了"没有研发投入"，
        # 所以"未披露 / 不适用"都算合格，见 refuse_allow
        "question": "苏宁环球2025年的研发投入具体金额是多少？",
        "expect_points": [],
        "expect_pages": [],
        "expect_files": [SUNING],
        "refuse": True,
    },
    {
        "id": "R2",
        "category": "拒答",
        # 港股年报，不存在 A 股口径的每股收益
        "question": "腾讯控股2025年的A股每股收益是多少？",
        "expect_points": [],
        "expect_pages": [],
        "expect_files": [TENCENT],
        "refuse": True,
    },
    {
        "id": "R3",
        "category": "拒答",
        # 分地区收入只有南京、芜湖、宜兴、上海、吉林、无锡等国内地区
        "question": "苏宁环球2025年在海外有哪些房地产项目？",
        "expect_points": [],
        "expect_pages": [],
        "expect_files": [SUNING],
        "refuse": True,
    },
    {
        "id": "R4",
        "category": "拒答",
        # 年报只覆盖到 2025 年度，不含 2026 年任何期间数据
        "question": "苏宁环球2026年第一季度的营业收入是多少？",
        "expect_points": [],
        "expect_pages": [],
        "expect_files": [SUNING],
        "refuse": True,
    },
    {
        "id": "R5",
        "category": "拒答",
        # 分季度指标只有营业收入、净利润、扣非净利润、经营现金流，没有毛利率
        "question": "苏宁环球2025年第三季度的毛利率是多少？",
        "expect_points": [],
        "expect_pages": [],
        "expect_files": [SUNING],
        "refuse": True,
    },
    {
        "id": "R6",
        "category": "拒答",
        # 两家都只披露人数与学历/专业构成，未披露年龄分布
        "question": "这两家公司2025年末员工的平均年龄分别是多少岁？",
        "expect_points": [],
        "expect_pages": [],
        "expect_files": [SUNING, TENCENT],
        "refuse": True,
    },
    {
        "id": "R7",
        "category": "拒答",
        # 互联网公司年报没有"土地储备"这一口径，这是把地产口径套到腾讯身上的陷阱
        "question": "腾讯控股2025年末的土地储备面积是多少平方米？",
        "expect_points": [],
        "expect_pages": [],
        "expect_files": [TENCENT],
        "refuse": True,
    },
]
