from pathlib import Path


# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_FILE = PROJECT_ROOT / "data.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Excel parsing
DEFAULT_SHEET_INDEX = 0
DEFAULT_HEADER_ROW = 1

# Labels
THEME_MONETARY = "货币政策"
THEME_ECONOMIC = "经济形势"
THEME_UNKNOWN = "未分类"

POLICY_TONES = ["宽松", "稳健", "从紧"]
ECON_TONES = ["正面", "中性", "负面"]

# Phrase extraction
MIN_PHRASE_CHARS = 2
MAX_PHRASE_CHARS = 37
MIN_PHRASE_TOKENS = 1
MAX_PHRASE_TOKENS = 10

# Phrase quality filters
MIN_CJK_CHAR_COUNT = 1
MAX_DIGIT_RATIO = 0.6
MAX_ASCII_RATIO = 0.7

# Dictionary filtering
DICTIONARY_PROB_THRESHOLD = 0.5
MIN_PHRASE_SUPPORT = 2
MIN_DICT_PHRASE_CHARS = 3
MAX_DICT_EVENT_COVERAGE = 0.30

# Theme scoring
THEME_KEYWORD_WEIGHT = 3.0
THEME_CONTEXT_WEIGHT = 1.0
THEME_SCORE_GAP_TO_ASSIGN = 0.5

# Length bins for phrase distribution
LENGTH_BINS = [(2, 5), (6, 10), (11, 20), (21, 37)]

# Boundary/function words used to suppress noisy phrases
BOUNDARY_STOPWORDS = {
    "的",
    "了",
    "和",
    "与",
    "及",
    "并",
    "且",
    "或",
    "等",
    "将",
    "把",
    "由",
    "对",
    "在",
    "于",
    "按",
    "从",
    "向",
    "就",
    "而",
    "为",
    "是",
    "有",
    "要",
    "也",
    "都",
    "仍",
    "还",
    "但",
    "并且",
    "以及",
    "通过",
    "继续",
    "当前",
    "今年",
    "明年",
    "其中",
    "方面",
    "有关",
    "这个",
    "那个",
    "我们",
    "你们",
    "他们",
    "进行",
    "落实",
    "推进",
    "加强",
    "保持",
}

PHRASE_STOPWORDS = {
    "进一步",
    "有关方面",
    "有关部门",
    "下一步",
    "与此同时",
    "一方面",
    "另一方面",
    "总体上",
    "总体来看",
    "总的来看",
    "从而",
    "其中",
    "当前",
    "持续",
    "继续",
    "今年以来",
    "工作会议",
    "中央经济工作会议",
}

DICTIONARY_EXACT_BLACKLIST = {
    "周小川",
    "易纲",
    "潘功胜",
    "朱鹤新",
    "刘国强",
    "郭树清",
    "新华社",
    "中国新闻网",
    "记者",
    "记者采访",
    "专访",
    "发布会",
    "论坛",
    "年会",
    "两会",
    "新年致辞",
    "工作会议",
    "中央经济工作会议",
    "国务院",
    "今年以来",
}

MONETARY_GATE_KEYWORDS = [
    "货币",
    "货币政策",
    "宽松",
    "稳健",
    "从紧",
    "紧缩",
    "利率",
    "降准",
    "降息",
    "加息",
    "准备金",
    "逆周期",
    "跨周期",
    "流动性",
    "信贷",
    "融资",
    "再贷款",
    "再贴现",
    "汇率",
]

ECONOMIC_GATE_KEYWORDS = [
    "经济",
    "增长",
    "GDP",
    "就业",
    "物价",
    "通胀",
    "CPI",
    "PPI",
    "需求",
    "供给",
    "风险",
    "压力",
    "挑战",
    "复苏",
    "下行",
    "回升",
    "向好",
    "放缓",
    "稳定",
    "高质量发展",
    "稳增长",
]

# Theme keywords
MONETARY_KEYWORDS = [
    "货币政策",
    "利率",
    "政策利率",
    "存款准备金",
    "准备金率",
    "降准",
    "降息",
    "加息",
    "公开市场",
    "逆回购",
    "流动性",
    "信贷",
    "再贷款",
    "再贴现",
    "融资成本",
    "贷款",
    "社融",
    "广义货币",
    "狭义货币",
    "货币供应量",
    "汇率",
    "人民币汇率",
    "MLF",
    "LPR",
]

ECONOMIC_KEYWORDS = [
    "经济",
    "增长",
    "GDP",
    "就业",
    "物价",
    "通胀",
    "通货膨胀",
    "CPI",
    "PPI",
    "需求",
    "供给",
    "风险",
    "挑战",
    "压力",
    "复苏",
    "下行",
    "预期",
    "产能",
    "消费",
    "投资",
    "出口",
    "收入",
    "利润",
    "失业",
]

# Economic sentiment lexicon
ECON_POSITIVE_KEYWORDS = [
    "回升",
    "向好",
    "改善",
    "增长",
    "平稳",
    "稳定",
    "复苏",
    "强劲",
    "扩大",
    "积极",
    "韧性",
    "恢复",
]

ECON_NEGATIVE_KEYWORDS = [
    "下行",
    "放缓",
    "风险",
    "压力",
    "冲击",
    "不确定",
    "困难",
    "挑战",
    "疲弱",
    "波动",
    "收缩",
    "失业",
    "萎缩",
]
